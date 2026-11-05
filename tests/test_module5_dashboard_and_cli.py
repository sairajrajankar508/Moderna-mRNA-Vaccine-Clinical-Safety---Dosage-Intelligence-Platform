"""Module 5 tests (FR-5.1 to FR-5.3) and command-line smoke tests."""

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import helpers  # noqa: F401

import numpy as np
import pandas as pd
from PIL import Image

import dashboard


def synthetic_inputs(seed=0, gen_latency=0.002):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 100)
    weights = rng.random((8, 8))
    weights /= weights.sum(axis=1, keepdims=True)
    return {
        "predictions": pd.DataFrame({"y_true": y, "score_logistic_regression": np.clip(y * 0.3 + rng.random(100) * 0.7, 0, 1),
                                     "score_random_forest": rng.random(100)}),
        "model_meta": {"selected_model": "logistic_regression"},
        "attention": weights,
        "sequence": "ACGUACGU",
        "retrieval": pd.DataFrame({"latency_s": rng.random(10) * 0.01, "embedding_model": "m", "store": "s"}),
        "generation": pd.DataFrame({"latency_s": [gen_latency] * 5, "provider": "p"}),
    }


class TestFR51ThreePanels(unittest.TestCase):
    def test_exactly_three_titled_panels(self):
        fig = dashboard.create_dashboard_figure(synthetic_inputs())
        titled = [ax for ax in fig.axes if ax.get_title()]
        self.assertEqual(len(titled), 3)
        self.assertEqual([ax.get_title()[:2] for ax in titled], ["A.", "B.", "C."])
        self.assertIn("ROC", titled[0].get_title())
        self.assertIn("attention", titled[1].get_title())
        self.assertIn("latency", titled[2].get_title())

    def test_log_scale_used_for_slow_llm(self):
        fig = dashboard.create_dashboard_figure(synthetic_inputs(gen_latency=3.0))
        self.assertEqual(fig.axes[2].get_yscale(), "log")


class TestFR52DrivenByData(unittest.TestCase):
    def _png_hash(self, inputs, path):
        dashboard.build_dashboard(path, inputs=inputs)
        return hashlib.md5(Path(path).read_bytes()).hexdigest()

    def test_changing_inputs_changes_the_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self._png_hash(synthetic_inputs(0), Path(tmp) / "a.png")
            b = self._png_hash(synthetic_inputs(1), Path(tmp) / "b.png")
            c = self._png_hash(synthetic_inputs(0), Path(tmp) / "c.png")
            self.assertNotEqual(a, b)
            self.assertEqual(a, c)

    def test_attention_panel_shows_saved_matrix(self):
        inputs = synthetic_inputs()
        fig = dashboard.create_dashboard_figure(inputs)
        np.testing.assert_array_equal(fig.axes[1].images[0].get_array(), inputs["attention"])

    def test_missing_module_output_is_reported(self):
        from unittest import mock
        with mock.patch.object(dashboard, "TEST_PREDICTIONS_PATH", Path("does_not_exist.csv")):
            with self.assertRaises(FileNotFoundError) as ctx:
                dashboard.load_inputs()
        self.assertIn("main.py train", str(ctx.exception))


class TestFR53Resolution(unittest.TestCase):
    def test_png_dpi_at_least_150(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dashboard.png"
            dashboard.build_dashboard(path, inputs=synthetic_inputs())
            with Image.open(path) as im:
                dpi = im.info["dpi"]
            self.assertGreaterEqual(round(dpi[0]), 150)
            self.assertGreaterEqual(round(dpi[1]), 150)


class TestCommandLine(unittest.TestCase):
    def run_cli(self, *args):
        import main
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main.main(list(args))
        return code, buffer.getvalue()

    def test_no_arguments_prints_help(self):
        code, out = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("assess", out)

    def test_invalid_sequence_exits_cleanly(self):
        code, out = self.run_cli("attention", "ACGX")
        self.assertEqual(code, 1)
        self.assertIn("Invalid nucleotide", out)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(helpers.ROOT))
    unittest.main()
