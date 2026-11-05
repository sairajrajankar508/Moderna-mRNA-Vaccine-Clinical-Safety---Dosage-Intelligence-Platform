"""Module 2 tests: FR-2.1 to FR-2.5 plus the multi-head bonus."""

import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401

import numpy as np
from PIL import Image

import attention as att


class TestFR21Encoding(unittest.TestCase):
    def test_prd_example_shape(self):
        self.assertEqual(att.encode_sequence("ACGU").shape, (4, att.EMBED_DIM))

    def test_shape_matches_length(self):
        for n in (5, 15, 30, 50):
            seq = "ACGU" * 13
            self.assertEqual(att.encode_sequence(seq[:n]).shape, (n, att.EMBED_DIM))

    def test_same_base_same_embedding(self):
        X = att.encode_sequence("AAGA")
        np.testing.assert_array_equal(X[0], X[1])
        np.testing.assert_array_equal(X[0], X[3])

    def test_lower_case_accepted(self):
        np.testing.assert_array_equal(att.encode_sequence("acgu"), att.encode_sequence("ACGU"))

    def test_length_limits_enforced_by_pipeline_entry(self):
        with self.assertRaises(att.InvalidSequenceError):
            att.analyze_sequence("ACGU", save=False)  # 4 < 5
        with self.assertRaises(att.InvalidSequenceError):
            att.analyze_sequence("A" * 51, save=False)
        att.analyze_sequence("ACGUA", save=False)
        att.analyze_sequence("A" * 50, save=False)


class TestFR22InvalidInput(unittest.TestCase):
    def test_prd_example_rejected(self):
        with self.assertRaises(att.InvalidSequenceError):
            att.encode_sequence("ACGX")

    def test_other_invalid_inputs(self):
        for bad in ("ACGT", "AC GU", "ACG-U", "12345", "", None):
            with self.subTest(sequence=bad), self.assertRaises(att.InvalidSequenceError):
                att.encode_sequence(bad)

    def test_dna_thymine_gets_a_hint(self):
        with self.assertRaises(att.InvalidSequenceError) as ctx:
            att.encode_sequence("ACGT")
        self.assertIn("U", str(ctx.exception))


class TestFR23AndFR24Attention(unittest.TestCase):
    def test_15_bases_gives_15x15(self):
        weights, output = att.self_attention(att.encode_sequence("ACGUACGUACGGCUA"))
        self.assertEqual(weights.shape, (15, 15))
        self.assertEqual(output.shape, (15, att.EMBED_DIM))

    def test_rows_sum_to_one_for_many_sequences(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            seq = "".join(rng.choice(att.VOCAB, size=rng.integers(5, 51)))
            weights, _ = att.self_attention(att.encode_sequence(seq))
            self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))
            self.assertTrue((weights > 0).all())

    def test_matches_formula_by_hand(self):
        X = att.encode_sequence("ACGUA")
        w = att.init_projection_weights()
        Q, K, V = X @ w.W_q, X @ w.W_k, X @ w.W_v
        scores = Q @ K.T / np.sqrt(K.shape[1])
        expected = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
        weights, output = att.self_attention(X, w)
        np.testing.assert_allclose(weights, expected)
        np.testing.assert_allclose(output, expected @ V)

    def test_softmax_is_numerically_stable(self):
        result = att.softmax(np.array([[1000.0, 1001.0, 1002.0]]))
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_deterministic(self):
        a, _ = att.self_attention(att.encode_sequence("ACGUACGU"))
        b, _ = att.self_attention(att.encode_sequence("ACGUACGU"))
        np.testing.assert_array_equal(a, b)

    def test_positional_encoding_breaks_symmetry(self):
        plain, _ = att.self_attention(att.encode_sequence("AAAAA"))
        positional, _ = att.self_attention(att.encode_sequence("AAAAA", positional=True))
        self.assertTrue(np.allclose(plain, 0.2))  # identical tokens -> uniform attention
        self.assertFalse(np.allclose(positional, 0.2))

    def test_multi_head_bonus(self):
        X = att.encode_sequence("ACGUACGUACGGCUA")
        heads, output = att.multi_head_attention(X, n_heads=4)
        self.assertEqual(len(heads), 4)
        self.assertEqual(output.shape, (15, att.EMBED_DIM))
        for w in heads:
            self.assertTrue(np.allclose(w.sum(axis=1), 1.0))
        self.assertFalse(np.allclose(heads[0], heads[1]))

    def test_pytorch_cross_check(self):
        try:
            import torch  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("PyTorch not installed or failed to load")
        X = att.encode_sequence("ACGUACGUACGGCUA")
        w = att.init_projection_weights()
        np.testing.assert_allclose(att.self_attention_torch(X, w)[0], att.self_attention(X, w)[0])


class TestFR25Heatmap(unittest.TestCase):
    def test_axes_labelled_with_positions(self):
        seq = "ACGUACGUACGGCUA"
        weights, _ = att.self_attention(att.encode_sequence(seq))
        fig = att.attention_figure(weights, seq)
        ax = fig.axes[0]
        self.assertTrue(ax.get_xlabel())
        self.assertTrue(ax.get_ylabel())
        expected = [f"{i + 1} {b}" for i, b in enumerate(seq)]
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], expected)
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()], expected)

    def test_png_written_and_opens(self):
        seq = "ACGUACGUACGGCUA"
        weights, _ = att.self_attention(att.encode_sequence(seq))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attention_heatmap.png"
            att.plot_attention(weights, seq, path)
            with Image.open(path) as im:
                im.verify()


if __name__ == "__main__":
    unittest.main()
