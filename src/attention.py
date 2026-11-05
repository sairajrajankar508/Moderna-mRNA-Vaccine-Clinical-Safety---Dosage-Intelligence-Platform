"""Module 2: scaled dot-product self-attention over an mRNA sequence.

Covers FR-2.1 to FR-2.5, plus the optional multi-head bonus.

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

Everything is implemented by hand in NumPy. A PyTorch version of the same
formula (``self_attention_torch``) is included as an independent cross-check:
given the same weights it must produce the same matrix.

Two deliberate differences from the PRD template
------------------------------------------------
1. The Q/K/V weight matrices are created once from a seeded generator and
   reused, instead of being re-drawn inside every call. Re-drawing them means
   two calls on the same sequence give two different heatmaps.
2. The weights use Xavier-style scaling (std = 1/sqrt(embed_dim)) instead of
   ``* 0.1``. With ``* 0.1`` every Q.K score is close to 0, so softmax
   returns an almost flat 1/seq_len everywhere and the heatmap shows nothing.
   Xavier scaling keeps the scores around unit variance, which is what the
   1/sqrt(d_k) factor in the formula assumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    ATTENTION_HEATMAP_PATH,
    ATTENTION_META_PATH,
    ATTENTION_WEIGHTS_PATH,
    RANDOM_SEED,
    ensure_output_dirs,
)

VOCAB = ["A", "C", "G", "U"]
EMBED_DIM = 8
MIN_LEN, MAX_LEN = 5, 50
DEFAULT_SEQUENCE = "ACGUACGUACGGCUA"  # 15 bases, the PRD's sample sequence

_rng = np.random.default_rng(RANDOM_SEED)
# One embedding vector per nucleotide. Fixed by the seed, so results are reproducible.
EMBEDDING_TABLE = {base: _rng.standard_normal(EMBED_DIM) for base in VOCAB}


class InvalidSequenceError(ValueError):
    """Raised for sequences with characters outside {A, C, G, U} or a bad length."""


@dataclass
class ProjectionWeights:
    W_q: np.ndarray
    W_k: np.ndarray
    W_v: np.ndarray


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def validate_sequence(sequence: str, check_length: bool = True) -> str:
    """Return the sequence in upper case, or raise InvalidSequenceError (FR-2.2).

    Lower-case input is accepted and upper-cased. Anything else outside
    {A, C, G, U} is rejected, including T (DNA thymine, which is U in RNA).
    """
    if not isinstance(sequence, str) or not sequence:
        raise InvalidSequenceError("Sequence must be a non-empty string of A, C, G, U.")
    seq = sequence.strip().upper()
    bad = [(i + 1, ch) for i, ch in enumerate(seq) if ch not in VOCAB]
    if bad:
        shown = ", ".join(f"'{ch}' at position {pos}" for pos, ch in bad[:5])
        hint = " (T is DNA; mRNA uses U)" if any(ch == "T" for _, ch in bad) else ""
        raise InvalidSequenceError(f"Invalid nucleotide(s): {shown}{hint}. Allowed: A, C, G, U.")
    if check_length and not MIN_LEN <= len(seq) <= MAX_LEN:
        raise InvalidSequenceError(f"Sequence length must be {MIN_LEN}-{MAX_LEN} bases; got {len(seq)}.")
    return seq


def encode_sequence(sequence: str, positional: bool = False) -> np.ndarray:
    """FR-2.1: turn 'ACGU...' into a (seq_len, EMBED_DIM) matrix of stacked embeddings.

    Only the alphabet is checked here, so short toy inputs such as 'ACGU' work
    (the PRD's own acceptance example is 4 bases long). The 5-50 length rule
    is applied by ``analyze_sequence``, the entry point used by the pipeline.

    positional=True adds the sinusoidal positional encoding from the original
    Transformer paper. Without it, attention depends only on which base sits
    at a position, not where it sits, so identical bases get identical rows.
    """
    seq = validate_sequence(sequence, check_length=False)
    X = np.stack([EMBEDDING_TABLE[base] for base in seq])
    if positional:
        X = X + sinusoidal_positions(len(seq), EMBED_DIM)
    return X


def sinusoidal_positions(seq_len: int, dim: int) -> np.ndarray:
    """PE[pos, 2i] = sin(pos / 10000^(2i/dim)),  PE[pos, 2i+1] = cos(same)."""
    positions = np.arange(seq_len)[:, None]
    rates = 1.0 / np.power(10000.0, np.arange(0, dim, 2) / dim)
    pe = np.zeros((seq_len, dim))
    pe[:, 0::2] = np.sin(positions * rates)
    pe[:, 1::2] = np.cos(positions * rates)
    return pe


# ---------------------------------------------------------------------------
# Attention maths
# ---------------------------------------------------------------------------

def init_projection_weights(embed_dim: int = EMBED_DIM, d_k: int = EMBED_DIM, seed: int = RANDOM_SEED) -> ProjectionWeights:
    """Random W_q, W_k, W_v of shape (embed_dim, d_k), Xavier-scaled."""
    rng = np.random.default_rng(seed + 1)
    scale = 1.0 / np.sqrt(embed_dim)
    return ProjectionWeights(*(rng.standard_normal((embed_dim, d_k)) * scale for _ in range(3)))


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Subtracting the row max avoids overflow in exp()."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def self_attention(X: np.ndarray, weights: ProjectionWeights | None = None):
    """FR-2.3: single-head scaled dot-product attention.

    X: (seq_len, embed_dim). Returns (attention_weights (seq_len, seq_len),
    output (seq_len, d_k)). Each row i of attention_weights says how much
    position i attends to every position j, and sums to 1 (FR-2.4).
    """
    if weights is None:
        weights = init_projection_weights(X.shape[1])
    Q = X @ weights.W_q
    K = X @ weights.W_k
    V = X @ weights.W_v
    d_k = K.shape[1]
    scores = (Q @ K.T) / np.sqrt(d_k)  # (seq_len, seq_len) raw similarity
    attention_weights = softmax(scores)
    output = attention_weights @ V  # weighted blend of value vectors
    return attention_weights, output


def multi_head_attention(X: np.ndarray, n_heads: int = 2, seed: int = RANDOM_SEED):
    """Bonus: run n_heads independent attention heads and concatenate their outputs.

    Each head has d_k = embed_dim / n_heads, so the concatenated output has
    the same width as the input, as in a standard Transformer layer.
    Returns (list of per-head weight matrices, output (seq_len, embed_dim)).
    """
    embed_dim = X.shape[1]
    if embed_dim % n_heads:
        raise ValueError(f"embed_dim {embed_dim} is not divisible by n_heads {n_heads}")
    d_k = embed_dim // n_heads
    head_weights, head_outputs = [], []
    for h in range(n_heads):
        w, out = self_attention(X, init_projection_weights(embed_dim, d_k, seed=seed + 10 * (h + 1)))
        head_weights.append(w)
        head_outputs.append(out)
    return head_weights, np.concatenate(head_outputs, axis=1)


def self_attention_torch(X: np.ndarray, weights: ProjectionWeights):
    """The same formula in PyTorch, used to cross-check the NumPy version.

    Raises ImportError if PyTorch is not installed.
    """
    import torch

    Xt = torch.tensor(X, dtype=torch.float64)
    Wq, Wk, Wv = (torch.tensor(w, dtype=torch.float64) for w in (weights.W_q, weights.W_k, weights.W_v))
    Q, K, V = Xt @ Wq, Xt @ Wk, Xt @ Wv
    scores = (Q @ K.T) / (K.shape[1] ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return attn.numpy(), (attn @ V).numpy()


# ---------------------------------------------------------------------------
# Interpretation and plotting
# ---------------------------------------------------------------------------

def base_pair_attention(attention_weights: np.ndarray, sequence: str) -> dict:
    """Mean weight from a query of one base type to a single key of another, e.g. {'A->G': 0.09}.

    Using the mean per key (not the total) keeps bases that appear more often
    from looking more important just because there are more of them. Compare
    each value with 1/seq_len, the weight every pair would get if attention
    were uniform.
    """
    seq = np.array(list(sequence))
    summary = {}
    for src in VOCAB:
        for dst in VOCAB:
            rows, cols = seq == src, seq == dst
            if rows.any() and cols.any():
                summary[f"{src}->{dst}"] = float(attention_weights[np.ix_(rows, cols)].mean())
    return dict(sorted(summary.items(), key=lambda kv: kv[1], reverse=True))


def attention_figure(attention_weights: np.ndarray, sequence: str, title: str | None = None):
    """FR-2.5: labelled heatmap figure; ticks show position number and base, e.g. '3 G'."""
    n = len(sequence)
    labels = [f"{i + 1} {base}" for i, base in enumerate(sequence)]
    fig, ax = plt.subplots(figsize=(max(7, n * 0.5), max(5.5, n * 0.42)))
    im = ax.imshow(attention_weights, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Key position (attending TO)")
    ax.set_ylabel("Query position (attending FROM)")
    ax.set_title(title or f"mRNA self-attention weights ({n} bases)")
    if n <= 20:
        for i in range(n):
            for j in range(n):
                value = attention_weights[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                        color="black" if value > attention_weights.max() * 0.6 else "white")
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    return fig


def plot_attention(attention_weights: np.ndarray, sequence: str, save_path=ATTENTION_HEATMAP_PATH, title=None):
    """Save the heatmap from attention_figure() as a PNG."""
    fig = attention_figure(attention_weights, sequence, title)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def analyze_sequence(sequence: str = DEFAULT_SEQUENCE, positional: bool = False, save: bool = True) -> dict:
    """Pipeline entry point: validate (incl. 5-50 length), attend, check, plot, save."""
    seq = validate_sequence(sequence, check_length=True)
    X = encode_sequence(seq, positional=positional)
    weights = init_projection_weights(X.shape[1])
    attention_weights, output = self_attention(X, weights)
    if not np.allclose(attention_weights.sum(axis=1), 1.0):
        raise AssertionError("Softmax rows must sum to 1")

    result = {
        "sequence": seq,
        "attention_weights": attention_weights,
        "output": output,
        "base_pair_attention": base_pair_attention(attention_weights, seq),
    }
    if save:
        ensure_output_dirs()
        np.save(ATTENTION_WEIGHTS_PATH, attention_weights)
        ATTENTION_META_PATH.write_text(json.dumps({
            "sequence": seq,
            "embed_dim": EMBED_DIM,
            "positional_encoding": positional,
            "seed": RANDOM_SEED,
            "row_sums_all_one": True,
            "base_pair_attention": result["base_pair_attention"],
        }, indent=2), encoding="utf-8")
        plot_attention(attention_weights, seq)
    return result


def run(sequence: str = DEFAULT_SEQUENCE) -> dict:
    result = analyze_sequence(sequence)
    w = result["attention_weights"]
    print(f"Sequence: {result['sequence']} ({len(result['sequence'])} bases, embed_dim={EMBED_DIM})")
    print(f"Attention matrix shape: {w.shape}; every row sums to 1: {np.allclose(w.sum(axis=1), 1.0)}")
    print(f"Max weight {w.max():.3f}, min weight {w.min():.3f}, uniform would be {1 / len(w):.3f}")
    print(f"Strongest base-to-base attention (mean weight per query/key pair; uniform = {1 / len(w):.3f}):")
    for pair, value in list(result["base_pair_attention"].items())[:4]:
        print(f"  {pair}: {value:.3f}")
    try:
        torch_w, _ = self_attention_torch(encode_sequence(result["sequence"]), init_projection_weights())
        print(f"PyTorch cross-check matches NumPy: {np.allclose(torch_w, w)}")
    except (ImportError, OSError) as err:  # OSError: a broken torch install on Windows (DLL load failure)
        print(f"PyTorch unavailable ({type(err).__name__}); skipped the cross-check.")
    print(f"Saved heatmap to outputs/figures/{ATTENTION_HEATMAP_PATH.name}")
    return result


if __name__ == "__main__":
    import sys

    try:
        run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SEQUENCE)
    except InvalidSequenceError as err:
        sys.exit(f"Invalid sequence: {err}")
