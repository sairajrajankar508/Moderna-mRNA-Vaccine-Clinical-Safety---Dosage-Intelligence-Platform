"""Module 3: retrieval index over the mRNA safety guideline document.

Covers FR-3.1 to FR-3.5.

Pipeline: guideline text -> sentence-aware chunks (300-500 tokens, ~50 token
overlap) -> dense embeddings -> persistent ChromaDB collection -> top-k
semantic search with latency logging.

Choices worth explaining
------------------------
* Chunking. LangChain's RecursiveCharacterTextSplitter enforces a maximum
  chunk size but not a minimum, so the last piece of a section can come out
  at 40 tokens, which breaks FR-3.1. ``chunk_text`` packs whole sentences
  into chunks of at least 300 and at most 500 tokens, carries the last
  sentences of each chunk into the next one as the overlap, and grows the
  final chunk backwards if it would otherwise be too short.
* Token counting. Tokens are counted as words and punctuation marks
  (regex ``\\w+|[^\\w\\s]``). That is close to, but a little below, what a
  sub-word tokenizer counts, so chunks target ~350 tokens to stay well
  inside the embedding model's input limit.
* Embedding model. The default is ``BAAI/bge-small-en-v1.5`` through
  LangChain's HuggingFaceEmbeddings. The PRD template uses
  ``all-MiniLM-L6-v2``, but that model truncates input at 256 word-pieces,
  so most of a 300-500 token chunk would never be embedded. bge-small reads
  up to 512 and is about the same size (33M parameters). Change it with the
  EMBEDDING_MODEL environment variable.
* Offline fallbacks. If sentence-transformers or the model download is
  unavailable, a deterministic hashing embedder is used; if chromadb is not
  installed, a small NumPy store is used. Both print a warning, and the
  index records which backend built it. The real submission should run with
  both proper backends installed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    GUIDELINES_TXT,
    LOGS_DIR,
    OUTPUTS_DIR,
    RETRIEVAL_LOG_PATH,
    ensure_output_dirs,
)

MIN_CHUNK_TOKENS = 300
MAX_CHUNK_TOKENS = 500
TARGET_CHUNK_TOKENS = 350
OVERLAP_TOKENS = 50
DEFAULT_K = 3
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")
SECTION_PATTERN = re.compile(r"^Section \d+:.*")

# (question, section the answer is in, phrase that must appear in the top chunk for a hit).
# TEST_QUESTIONS reuse the guideline's own wording; STRESS_QUESTIONS paraphrase it
# with different vocabulary, which separates keyword overlap from real semantic matching.
TEST_QUESTIONS = [
    ("What are the storage requirements below -20C?", "Section 1", "-25C and -15C"),
    ("How long can a thawed vial stay in the refrigerator before it must be discarded?", "Section 1", "up to 30 days"),
    ("What dose range applies to adults aged 12 to 64?", "Section 2", "Ages 12 to 64"),
    ("When does a dose need secondary clinician review?", "Section 2", "Doses above 100 micrograms"),
    ("Should dosing be deferred for a patient with an elevated inflammatory biomarker?", "Section 2", "dosing deferred"),
    ("What are the absolute contraindications for an mRNA product?", "Section 3", "anaphylaxis"),
    ("How long should patients be monitored after administration?", "Section 3", "at least 15 minutes"),
    ("Within how many hours must a moderate adverse event be documented?", "Section 4", "within 24 hours"),
]
STRESS_QUESTIONS = [
    ("Can a vial be put back in the freezer after it has thawed?", "Section 1", "never be refrozen"),
    ("What should I do if the liquid in a vial looks cloudy?", "Section 1", "cloudiness"),
    ("What happens to a batch if the cold chain fails during shipping?", "Section 1", "quarantined"),
    ("Is a severe allergy to the first shot a reason to skip the next one?", "Section 3", "anaphylaxis"),
    ("When must a reaction be escalated to a safety review?", "Section 4", "hospitalization"),
]


# ---------------------------------------------------------------------------
# FR-3.1  Chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: int
    text: str
    n_tokens: int
    sections: list[str]
    overlap_with_previous: int = 0  # tokens shared with the previous chunk


def count_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def split_units(text: str) -> list[tuple[str, str]]:
    """Split text into sentence-level units, each tagged with its section heading.

    Paragraphs are split on blank lines, then on sentence-ending punctuation.
    Headings and bullet points stay as their own units.
    """
    units, section = [], "Front matter"
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if SECTION_PATTERN.match(paragraph):
            section = paragraph.split(":")[0]
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(-])", paragraph):
            if sentence.strip():
                units.append((sentence.strip(), section))
    return units


def _split_long_unit(unit: str, max_tokens: int) -> list[str]:
    """Break a single sentence longer than max_tokens into word windows."""
    words = unit.split()
    pieces, current = [], []
    for word in words:
        if current and count_tokens(" ".join(current + [word])) > max_tokens:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(
    text: str,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """Pack whole sentences into overlapping chunks of min_tokens..max_tokens tokens."""
    units = []
    for sentence, section in split_units(text):
        for piece in _split_long_unit(sentence, max_tokens - overlap_tokens):
            units.append((piece, section, count_tokens(piece)))
    if not units:
        return []

    total = sum(n for _, _, n in units)
    if total <= max_tokens:  # the whole document fits in one chunk
        return [_make_chunk(0, units, 0, len(units), 0)]

    spans = []  # (start, end, overlap) indices into units, end exclusive
    start, overlap = 0, 0
    while True:
        end, size = start, 0
        while end < len(units) and (size < target_tokens or end - start == 0) and size + units[end][2] <= max_tokens:
            size += units[end][2]
            end += 1
        spans.append([start, end, overlap])
        if end >= len(units):
            break
        # Carry trailing sentences of this chunk into the next one until ~overlap_tokens are shared.
        next_start, carried = end, 0
        while next_start > start + 1 and carried < overlap_tokens:
            next_start -= 1
            carried += units[next_start][2]
        start, overlap = next_start, carried

    # Grow a too-short final chunk backwards (more overlap) until it reaches min_tokens.
    last = spans[-1]
    size = sum(units[i][2] for i in range(last[0], last[1]))
    while size < min_tokens and last[0] > 0 and size + units[last[0] - 1][2] <= max_tokens:
        last[0] -= 1
        size += units[last[0]][2]
    if len(spans) > 1:
        prev_end = spans[-2][1]
        last[2] = sum(units[i][2] for i in range(last[0], prev_end)) if last[0] < prev_end else 0

    return [_make_chunk(i, units, s, e, o) for i, (s, e, o) in enumerate(spans)]


def _make_chunk(chunk_id: int, units, start: int, end: int, overlap: int) -> Chunk:
    selected = units[start:end]
    sections = list(dict.fromkeys(section for _, section, _ in selected))
    text = "\n".join(sentence for sentence, _, _ in selected)
    return Chunk(chunk_id, text, count_tokens(text), sections, overlap)


def describe_chunks(chunks: list[Chunk]) -> str:
    lines = [f"{len(chunks)} chunks (allowed size {MIN_CHUNK_TOKENS}-{MAX_CHUNK_TOKENS} tokens, target overlap ~{OVERLAP_TOKENS}):"]
    for c in chunks:
        lines.append(f"  chunk {c.chunk_id}: {c.n_tokens:>3} tokens, overlap with previous {c.overlap_with_previous:>3}, sections: {', '.join(c.sections)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FR-3.2  Embeddings
# ---------------------------------------------------------------------------

class HashingEmbeddings:
    """Offline fallback: hashed bag of words + word bigrams, L2-normalized.

    Deterministic across runs (uses md5, not Python's randomized hash()).
    It only captures word overlap, not meaning, so retrieval quality is
    lower than a real sentence-embedding model.
    """

    name = "hashing-bow-512 (offline fallback)"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        vec = np.zeros(self.dim)
        for feat in features:
            vec[int(hashlib.md5(feat.encode()).hexdigest(), 16) % self.dim] += 1.0
        vec = np.log1p(vec)
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class LangChainEmbeddings:
    """Wraps a LangChain embeddings object and records the model name.

    (LangChain embedding classes are Pydantic models, so a ``name``
    attribute cannot simply be attached to them.)
    """

    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)


def get_embedding_model(backend: str | None = None):
    """Return an object with embed_documents/embed_query and a ``name`` attribute.

    backend: "huggingface", "hashing", or "auto" (default; env EMBEDDING_BACKEND).
    """
    backend = (backend or os.getenv("EMBEDDING_BACKEND", "auto")).lower()
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    if backend == "hashing":
        return HashingEmbeddings()
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        model = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})
        return LangChainEmbeddings(model, model_name)
    except Exception as err:  # ImportError, or OSError/HTTP errors when the model cannot be downloaded
        if backend == "huggingface":
            raise
        warnings.warn(f"Sentence-embedding model unavailable ({type(err).__name__}: {err}). "
                      "Falling back to the offline hashing embedder.", stacklevel=2)
        return HashingEmbeddings()


# ---------------------------------------------------------------------------
# FR-3.3  Persistent vector store
# ---------------------------------------------------------------------------

class ChromaStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    backend = "chromadb"

    def __init__(self, persist_dir: Path, collection_name: str):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.name = collection_name
        self.collection = self._open_collection()

    def _open_collection(self):
        # Chroma 1.x sets the distance via `configuration`; older releases used collection metadata.
        try:
            return self.client.get_or_create_collection(self.name, configuration={"hnsw": {"space": "cosine"}})
        except (TypeError, ValueError):
            return self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:  # collection did not exist yet
            pass
        self.collection = self._open_collection()

    def add(self, ids, texts, embeddings, metadatas) -> None:
        self.collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def count(self) -> int:
        return self.collection.count()

    def query(self, embedding, k: int):
        res = self.collection.query(query_embeddings=[embedding], n_results=k,
                                    include=["documents", "metadatas", "embeddings"])
        # Embeddings are unit length, so their dot product is the cosine similarity, whatever
        # distance function the collection uses internally (cosine and L2 give the same ranking).
        query_vec = np.asarray(embedding, dtype=float)
        sims = [float(np.dot(query_vec, np.asarray(e, dtype=float))) for e in res["embeddings"][0]]
        return list(zip(res["documents"][0], sims, res["metadatas"][0]))


class NumpyStore:
    """Offline fallback with the same interface, saved as .npz + .json."""

    backend = "numpy (offline fallback)"

    def __init__(self, persist_dir: Path, collection_name: str):
        self.path = Path(persist_dir) / f"{collection_name}.npz"
        self.meta_path = Path(persist_dir) / f"{collection_name}.json"
        self.embeddings = np.zeros((0, 0))
        self.texts, self.metadatas = [], []
        if self.path.exists() and self.meta_path.exists():
            self.embeddings = np.load(self.path)["embeddings"]
            saved = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.texts, self.metadatas = saved["texts"], saved["metadatas"]

    def reset(self) -> None:
        self.embeddings, self.texts, self.metadatas = np.zeros((0, 0)), [], []

    def add(self, ids, texts, embeddings, metadatas) -> None:
        new = np.asarray(embeddings, dtype=float)
        self.embeddings = new if self.embeddings.size == 0 else np.vstack([self.embeddings, new])
        self.texts += list(texts)
        self.metadatas += list(metadatas)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, embeddings=self.embeddings)
        self.meta_path.write_text(json.dumps({"texts": self.texts, "metadatas": self.metadatas}), encoding="utf-8")

    def count(self) -> int:
        return len(self.texts)

    def query(self, embedding, k: int):
        sims = self.embeddings @ np.asarray(embedding, dtype=float)
        order = np.argsort(-sims)[:k]
        return [(self.texts[i], float(sims[i]), self.metadatas[i]) for i in order]


def open_store(persist_dir: Path = CHROMA_DIR, collection_name: str = CHROMA_COLLECTION):
    try:
        return ChromaStore(persist_dir, collection_name)
    except ImportError:
        warnings.warn("chromadb is not installed; using the NumPy fallback store. "
                      "Install chromadb to meet FR-3.3.", stacklevel=2)
    except Exception as err:  # e.g. old chromadb releases fail to import on Python 3.14
        warnings.warn(f"chromadb failed to start ({type(err).__name__}: {err}); using the NumPy fallback store. "
                      "Upgrade with: pip install -U \"chromadb>=1.5.9\"", stacklevel=2)
    return NumpyStore(persist_dir, collection_name)


# ---------------------------------------------------------------------------
# Index build / load and FR-3.4 / FR-3.5 querying
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    question: str
    chunks: list[str]
    similarities: list[float]
    metadatas: list[dict]
    latency_s: float
    k: int = DEFAULT_K
    sections: list[str] = field(default_factory=list)


class GuidelineIndex:
    """A built or reloaded index: a vector store plus the embedder that built it."""

    def __init__(self, store, embedder, manifest: dict, log_path: Path | None = RETRIEVAL_LOG_PATH):
        self.store = store
        self.embedder = embedder
        self.manifest = manifest
        self.log_path = log_path

    def query(self, question: str, k: int = DEFAULT_K) -> RetrievalResult:
        """FR-3.4 + FR-3.5: top-k chunks ordered by similarity, with latency in seconds."""
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        k = min(k, self.store.count())
        start = time.perf_counter()
        hits = self.store.query(self.embedder.embed_query(question), k)
        latency = time.perf_counter() - start
        hits.sort(key=lambda h: h[1], reverse=True)
        result = RetrievalResult(
            question=question,
            chunks=[h[0] for h in hits],
            similarities=[float(h[1]) for h in hits],
            metadatas=[h[2] for h in hits],
            latency_s=latency,
            k=k,
            sections=[h[2].get("sections", "") for h in hits],
        )
        self._log(result)
        return result

    def _log(self, result: RetrievalResult) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.log_path.exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["timestamp", "question", "k", "latency_s", "top_sections", "top_similarity", "embedding_model", "store"])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"), result.question, result.k,
                f"{result.latency_s:.6f}", result.sections[0] if result.sections else "",
                f"{result.similarities[0]:.4f}" if result.similarities else "",
                self.manifest.get("embedding_model"), self.manifest.get("store_backend"),
            ])


def query_guidelines(index: GuidelineIndex, question: str, k: int = DEFAULT_K):
    """PRD-style helper: returns (list of k chunk strings, latency in seconds)."""
    result = index.query(question, k)
    return result.chunks, result.latency_s


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest_path(persist_dir: Path) -> Path:
    return Path(persist_dir) / "index_manifest.json"


def build_index(doc_path: Path = GUIDELINES_TXT, persist_dir: Path = CHROMA_DIR, force: bool = False,
                embedder=None, log_path: Path | None = RETRIEVAL_LOG_PATH) -> GuidelineIndex:
    """Chunk, embed and persist the document, or reuse an up-to-date persisted index."""
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    embedder = embedder or get_embedding_model()
    store = open_store(persist_dir)
    doc_hash = _file_hash(doc_path)

    manifest_file = _manifest_path(persist_dir)
    if manifest_file.exists() and not force:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        up_to_date = (
            manifest.get("doc_sha256") == doc_hash
            and manifest.get("embedding_model") == embedder.name
            and manifest.get("store_backend") == store.backend
            and manifest.get("n_chunks") == store.count()
        )
        if up_to_date:
            print(f"Reusing persisted index ({manifest['n_chunks']} chunks, no re-embedding).")
            return GuidelineIndex(store, embedder, manifest, log_path)

    chunks = chunk_text(Path(doc_path).read_text(encoding="utf-8"))
    print(describe_chunks(chunks))
    start = time.perf_counter()
    embeddings = embedder.embed_documents([c.text for c in chunks])
    embed_seconds = time.perf_counter() - start

    store.reset()
    store.add(
        ids=[f"chunk-{c.chunk_id}" for c in chunks],
        texts=[c.text for c in chunks],
        embeddings=[list(map(float, e)) for e in embeddings],
        metadatas=[{"chunk_id": c.chunk_id, "n_tokens": c.n_tokens, "sections": ", ".join(c.sections)} for c in chunks],
    )
    manifest = {
        "doc_path": Path(doc_path).name,
        "doc_sha256": doc_hash,
        "n_chunks": len(chunks),
        "n_embeddings": len(embeddings),
        "embedding_dim": len(embeddings[0]),
        "embedding_model": embedder.name,
        "store_backend": store.backend,
        "collection": CHROMA_COLLECTION,
        "chunk_tokens": [c.n_tokens for c in chunks],
        "overlap_tokens": [c.overlap_with_previous for c in chunks],
        "chunking": {"min": MIN_CHUNK_TOKENS, "max": MAX_CHUNK_TOKENS, "target": TARGET_CHUNK_TOKENS, "overlap": OVERLAP_TOKENS},
        "embedding_seconds": round(embed_seconds, 3),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Embedded {len(embeddings)} chunks with {embedder.name} into {store.backend} ({embed_seconds:.2f}s).")
    return GuidelineIndex(store, embedder, manifest, log_path)


def load_index(persist_dir: Path = CHROMA_DIR, embedder=None, log_path: Path | None = RETRIEVAL_LOG_PATH) -> GuidelineIndex:
    """Reopen a persisted index in a fresh process without re-embedding anything."""
    manifest_file = _manifest_path(persist_dir)
    if not manifest_file.exists():
        raise FileNotFoundError("No persisted index found. Run 'python main.py index' first.")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    embedder = embedder or get_embedding_model()
    if embedder.name != manifest["embedding_model"]:
        raise RuntimeError(f"Index was built with {manifest['embedding_model']} but the current embedder is "
                           f"{embedder.name}. Rebuild with 'python main.py index --rebuild'.")
    store = open_store(persist_dir)
    if store.count() != manifest["n_chunks"]:
        raise RuntimeError("Persisted store is incomplete; rebuild with 'python main.py index --rebuild'.")
    return GuidelineIndex(store, embedder, manifest, log_path)


def evaluate_questions(index: GuidelineIndex, questions, k: int = DEFAULT_K) -> list[dict]:
    """Query each (question, section, phrase); a hit means the phrase is in the top-1 chunk."""
    rows = []
    for question, section, phrase in questions:
        result = index.query(question, k)
        rank = next((i + 1 for i, c in enumerate(result.chunks) if phrase.lower() in c.lower()), None)
        first_line = next((ln for ln in result.chunks[0].splitlines() if len(ln) > 40), result.chunks[0][:120])
        rows.append({
            "question": question, "expected": section, "phrase": phrase,
            "top_sections": result.sections[0], "hit": rank == 1, "answer_rank": rank,
            "similarity": result.similarities[0], "latency_s": result.latency_s,
            "snippet": first_line[:100] + ("..." if len(first_line) > 100 else ""),
        })
    return rows


def _table(rows: list[dict]) -> list[str]:
    lines = ["| # | Question | Answer phrase | Top chunk covers | Hit@1 | Answer rank | Similarity | Top chunk starts with | Latency (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        rank = r["answer_rank"] or f"not in top {DEFAULT_K}"
        lines.append(f"| {i} | {r['question']} | \"{r['phrase']}\" | {r['top_sections']} | {'yes' if r['hit'] else '**NO**'} | "
                     f"{rank} | {r['similarity']:.3f} | {r['snippet'].replace('|', '/')} | {r['latency_s']:.4f} |")
    return lines


def run_test_queries(index: GuidelineIndex, k: int = DEFAULT_K, path: Path = OUTPUTS_DIR / "rag_test_queries.md") -> dict:
    """Run both question sets and write a markdown report (Week 3 Definition of Done)."""
    results = {"standard": evaluate_questions(index, TEST_QUESTIONS, k), "stress": evaluate_questions(index, STRESS_QUESTIONS, k)}
    m = index.manifest
    lines = [
        "# RAG retrieval test queries",
        "",
        f"Embedding model: `{m['embedding_model']}`  |  Store: `{m['store_backend']}`  |  Chunks: {m['n_chunks']} "
        f"({', '.join(map(str, m['chunk_tokens']))} tokens)  |  k = {k}",
        "",
        "A **hit** means the sentence that answers the question is inside the top-ranked chunk.",
        "",
    ]
    for name, title, note in (
        ("standard", "Standard questions", "Worded like the guideline itself."),
        ("stress", "Paraphrase stress test", "Same facts, different vocabulary (e.g. 'put back in the freezer' vs 'refrozen')."),
    ):
        rows = results[name]
        lines += [f"## {title}: {sum(r['hit'] for r in rows)}/{len(rows)} hits", "", note, "", *_table(rows), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return results


def run(rebuild: bool = False) -> GuidelineIndex:
    ensure_output_dirs()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    index = build_index(force=rebuild)
    results = run_test_queries(index)
    for name, rows in results.items():
        print(f"\n{name.capitalize()} questions: {sum(r['hit'] for r in rows)}/{len(rows)} hits (answer sentence in top chunk)")
        for r in rows:
            status = "hit " if r["hit"] else f"MISS (answer rank: {r['answer_rank'] or 'not retrieved'})"
            print(f"  {status:<32} {r['latency_s']:.4f}s  {r['question']}")
    print("\nFull tables saved to outputs/rag_test_queries.md")
    return index


if __name__ == "__main__":
    import sys

    run(rebuild="--rebuild" in sys.argv)
