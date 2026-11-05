"""Module 3 tests: FR-3.1 to FR-3.5.

These use the offline hashing embedder so they run quickly without a model
download. When chromadb is installed the real ChromaDB store is used, so
FR-3.3 is tested against ChromaDB itself.
"""

import contextlib
import csv
import io
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import helpers

import rag_index as rag
from config import GUIDELINES_TXT


class TestFR31Chunking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = GUIDELINES_TXT.read_text(encoding="utf-8")
        cls.chunks = rag.chunk_text(cls.text)

    def test_all_chunks_within_300_to_500_tokens(self):
        self.assertGreaterEqual(len(self.chunks), 2)
        for c in self.chunks:
            with self.subTest(chunk=c.chunk_id, tokens=c.n_tokens):
                self.assertGreaterEqual(c.n_tokens, rag.MIN_CHUNK_TOKENS)
                self.assertLessEqual(c.n_tokens, rag.MAX_CHUNK_TOKENS)
                self.assertEqual(c.n_tokens, rag.count_tokens(c.text))

    def test_consecutive_chunks_overlap(self):
        for prev, cur in zip(self.chunks, self.chunks[1:]):
            with self.subTest(chunk=cur.chunk_id):
                self.assertGreaterEqual(cur.overlap_with_previous, 30)
                shared_first_line = cur.text.splitlines()[0]
                self.assertIn(shared_first_line, prev.text)
        # Every chunk except possibly the last (grown backwards to reach 300) is close to 50.
        for c in self.chunks[1:-1]:
            self.assertLessEqual(c.overlap_with_previous, 100)

    def test_no_text_lost(self):
        joined = "\n".join(c.text for c in self.chunks)
        for sentence, _ in rag.split_units(self.text):
            self.assertIn(sentence, joined)

    def test_works_on_other_lengths(self):
        sentence = "The vial must be inspected for particulate matter before use. "
        for repeats in (40, 120, 300):
            chunks = rag.chunk_text(sentence * repeats)
            for c in chunks:
                self.assertLessEqual(c.n_tokens, rag.MAX_CHUNK_TOKENS)
                if len(chunks) > 1:
                    self.assertGreaterEqual(c.n_tokens, rag.MIN_CHUNK_TOKENS)


class IndexTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # Windows: Chroma may hold files open
        cls.dir = Path(cls.tmp.name) / "store"
        cls.log = Path(cls.tmp.name) / "latency.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            cls.index = rag.build_index(GUIDELINES_TXT, cls.dir, force=True, embedder=rag.HashingEmbeddings(), log_path=cls.log)

    @classmethod
    def tearDownClass(cls):
        del cls.index
        try:  # release Chroma's cached client so its files can be deleted
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        cls.tmp.cleanup()


class TestFR32Embeddings(IndexTestCase):
    def test_one_embedding_per_chunk(self):
        n_chunks = len(rag.chunk_text(GUIDELINES_TXT.read_text(encoding="utf-8")))
        self.assertEqual(self.index.manifest["n_chunks"], n_chunks)
        self.assertEqual(self.index.manifest["n_embeddings"], n_chunks)
        self.assertEqual(self.index.store.count(), n_chunks)

    def test_embedder_is_deterministic_and_normalized(self):
        e = rag.HashingEmbeddings()
        a, b = e.embed_query("storage temperature"), e.embed_query("storage temperature")
        self.assertEqual(a, b)
        self.assertAlmostEqual(sum(x * x for x in a), 1.0, places=6)


class TestFR33Persistence(IndexTestCase):
    def test_rebuild_is_skipped_when_up_to_date(self):
        with contextlib.redirect_stdout(io.StringIO()):
            again = rag.build_index(GUIDELINES_TXT, self.dir, embedder=rag.HashingEmbeddings(), log_path=None)
        self.assertEqual(again.manifest["built_at"], self.index.manifest["built_at"])

    def test_fresh_process_can_query_without_reembedding(self):
        script = textwrap.dedent(f"""
            import json, sys, warnings
            warnings.filterwarnings("ignore")
            sys.path.insert(0, {str(helpers.SRC)!r})
            import rag_index as rag
            index = rag.load_index({str(self.dir)!r}, embedder=rag.HashingEmbeddings(), log_path=None)
            result = index.query("What are the storage requirements below -20C?", 3)
            print(json.dumps({{"n": len(result.chunks), "top": result.sections[0], "built_at": index.manifest["built_at"]}}))
        """)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        data = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(data["n"], 3)
        self.assertIn("Section 1", data["top"])
        self.assertEqual(data["built_at"], self.index.manifest["built_at"])

    def test_wrong_embedder_is_refused(self):
        class Other(rag.HashingEmbeddings):
            name = "something-else"
        with self.assertRaises(RuntimeError):
            rag.load_index(self.dir, embedder=Other(), log_path=None)


class TestFR34TopK(IndexTestCase):
    def test_k_chunks_ordered_by_similarity_for_test_questions(self):
        for question, expected, _phrase in rag.TEST_QUESTIONS:
            with self.subTest(question=question):
                result = self.index.query(question, k=3)
                self.assertEqual(len(result.chunks), 3)
                self.assertTrue(all(isinstance(c, str) and c for c in result.chunks))
                self.assertEqual(result.similarities, sorted(result.similarities, reverse=True))
                self.assertIn(expected, result.sections[0])
        self.assertGreaterEqual(len(rag.TEST_QUESTIONS), 5)

    def test_answer_phrases_exist_in_guideline(self):
        text = GUIDELINES_TXT.read_text(encoding="utf-8").lower()
        for _, _, phrase in rag.TEST_QUESTIONS + rag.STRESS_QUESTIONS:
            self.assertIn(phrase.lower(), text)

    def test_evaluation_report_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.md"
            results = rag.run_test_queries(self.index, path=path)
            self.assertEqual(len(results["standard"]), len(rag.TEST_QUESTIONS))
            self.assertIn("| Question |", path.read_text(encoding="utf-8"))

    def test_prd_helper_signature(self):
        chunks, latency = rag.query_guidelines(self.index, "What dose needs clinician review?")
        self.assertEqual(len(chunks), rag.DEFAULT_K)
        self.assertGreater(latency, 0)

    def test_custom_k(self):
        self.assertEqual(len(self.index.query("monitoring after injection", k=2).chunks), 2)

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            self.index.query("   ")


class TestFR35Latency(IndexTestCase):
    def test_latency_recorded_for_every_query(self):
        before = self._rows()
        questions = [q for q, _, _ in rag.TEST_QUESTIONS]
        for q in questions:
            result = self.index.query(q)
            self.assertGreater(result.latency_s, 0)
            self.assertLess(result.latency_s, 5)
        rows = self._rows()[len(before):]
        self.assertEqual(len(rows), len(questions))
        for row in rows:
            self.assertTrue(0 < float(row["latency_s"]) < 5)

    def _rows(self):
        if not self.log.exists():
            return []
        with open(self.log, encoding="utf-8") as f:
            return list(csv.DictReader(f))


if __name__ == "__main__":
    unittest.main()
