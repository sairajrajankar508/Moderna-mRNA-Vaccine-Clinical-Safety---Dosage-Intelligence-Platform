"""Module 1 tests: FR-1.1 to FR-1.7."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import helpers  # noqa: F401

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

import data_pipeline as dp
import risk_model as rm
from config import RAW_CSV


class TestFR11ColumnValidation(unittest.TestCase):
    def test_each_missing_column_is_named_in_the_error(self):
        raw = pd.read_csv(RAW_CSV, keep_default_na=False, na_values=[""])
        with tempfile.TemporaryDirectory() as tmp:
            for column in dp.REQUIRED_COLUMNS:
                path = Path(tmp) / f"without_{column}.csv"
                raw.drop(columns=[column]).to_csv(path, index=False)
                with self.subTest(column=column):
                    with self.assertRaises(dp.MissingColumnsError) as ctx:
                        dp.load_raw(path)
                    self.assertIn(column, str(ctx.exception))
                    self.assertEqual(ctx.exception.missing, [column])

    def test_full_file_loads(self):
        df = dp.load_raw()
        self.assertEqual(df.shape, (500, 7))

    def test_unexpected_category_is_rejected(self):
        raw = pd.read_csv(RAW_CSV, keep_default_na=False, na_values=[""])
        raw.loc[0, "Prior_Reaction_History"] = "Severe"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            raw.to_csv(path, index=False)
            with self.assertRaises(dp.InvalidCategoryError):
                dp.load_raw(path)


class TestFR12Imputation(unittest.TestCase):
    def setUp(self):
        self.raw = dp.load_raw()
        self.clean, self.log = dp.clean(self.raw)

    def test_no_missing_values_after_cleaning(self):
        self.assertEqual(int(self.clean.isnull().sum().sum()), 0)
        self.assertTrue((self.clean.isnull().sum() == 0).all())

    def test_text_none_is_a_category_not_missing(self):
        # 291 real "None" values; pandas' default reader would have treated them as missing.
        self.assertEqual(int(self.raw["Prior_Reaction_History"].isnull().sum()), 17)
        self.assertEqual(int((self.raw["Prior_Reaction_History"] == "None").sum()), 291)

    def test_median_and_mode_used(self):
        median = self.raw["Biomarker_Level"].median()
        mode = self.raw["Prior_Reaction_History"].mode()[0]
        gaps_bio = self.raw["Biomarker_Level"].isnull()
        gaps_hist = self.raw["Prior_Reaction_History"].isnull()
        self.assertTrue(np.allclose(self.clean.loc[gaps_bio, "Biomarker_Level"], median))
        self.assertTrue((self.clean.loc[gaps_hist, "Prior_Reaction_History"] == mode).all())
        self.assertEqual(self.log["Biomarker_Level"]["filled"], 20)
        self.assertEqual(self.log["Prior_Reaction_History"]["filled"], 17)
        self.assertEqual(len(self.clean), 500)


class TestModelTraining(unittest.TestCase):
    """Trains once (about 15 seconds) and checks FR-1.3 to FR-1.7."""

    @classmethod
    def setUpClass(cls):
        cls.df, _ = dp.clean(dp.load_raw())
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = rm.split_data(cls.df)
        cls.trained = rm.train_models(cls.X_train, cls.y_train)
        cls.evaluations = {n: rm.evaluate(t["model"], cls.X_test, cls.y_test) for n, t in cls.trained.items()}

    # FR-1.3
    def test_scaled_columns_have_zero_mean_unit_std_on_training_set(self):
        for name, t in self.trained.items():
            stats = rm.scaled_training_stats(t["model"], self.X_train)
            self.assertEqual(set(stats), set(dp.NUMERIC_COLS))
            for col, s in stats.items():
                with self.subTest(model=name, column=col):
                    self.assertAlmostEqual(s["mean"], 0.0, places=6)
                    self.assertAlmostEqual(s["std"], 1.0, places=6)

    # FR-1.4
    def test_split_is_80_20_stratified_and_reproducible(self):
        X_tr2, X_te2, _, y_te2 = rm.split_data(self.df)
        self.assertEqual(list(self.X_train.index), list(X_tr2.index))
        self.assertEqual(list(self.X_test.index), list(X_te2.index))
        self.assertEqual((len(self.X_train), len(self.X_test)), (400, 100))
        overall = self.df["Reaction_Score"].mean()
        self.assertAlmostEqual(self.y_test.mean(), overall, delta=0.01)
        self.assertAlmostEqual(self.y_train.mean(), overall, delta=0.01)

    # FR-1.5
    def test_both_models_trained_on_same_data(self):
        self.assertEqual(set(self.trained), {"logistic_regression", "random_forest"})
        for t in self.trained.values():
            check_is_fitted(t["model"])
        self.assertEqual(type(self.trained["logistic_regression"]["model"].named_steps["model"]).__name__, "LogisticRegression")
        self.assertEqual(type(self.trained["random_forest"]["model"].named_steps["model"]).__name__, "RandomForestClassifier")

    def test_performance_is_believable(self):
        for ev in self.evaluations.values():
            self.assertGreater(ev["roc_auc"], 0.6)
            self.assertLess(ev["roc_auc"], 0.95)

    # FR-1.6 + FR-1.7
    def test_artifacts_saved_and_model_reloads_identically(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            patches = {
                "METRICS_DIR": tmp, "FIGURES_DIR": tmp, "MODEL_PATH": tmp / "risk_model.joblib",
                "MODEL_META_PATH": tmp / "meta.json", "TEST_PREDICTIONS_PATH": tmp / "preds.csv",
                "ensure_output_dirs": lambda: None,
            }
            with mock.patch.multiple(rm, **patches):
                best = rm.select_best(self.evaluations)
                rm.save_artifacts(self.trained, self.evaluations, best, self.X_train, self.X_test, self.y_test)

            for name in ("logistic_regression", "random_forest"):
                report = (tmp / f"{name}_classification_report.txt").read_text(encoding="utf-8")
                self.assertIn("precision", report)
                self.assertIn("recall", report)
                self.assertIn("f1-score", report)
                cm = pd.read_csv(tmp / f"{name}_confusion_matrix.csv", index_col=0)
                self.assertEqual(int(cm.values.sum()), len(self.y_test))
                self.assertTrue((tmp / f"{name}_confusion_matrix.png").exists())

            # FR-1.7: highest F1 selected, saved, and reloads with identical predictions.
            f1s = {n: e["f1"] for n, e in self.evaluations.items()}
            self.assertEqual(f1s[best], max(f1s.values()))
            meta = json.loads((tmp / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["selected_model"], best)
            reloaded = joblib.load(tmp / "risk_model.joblib")
            saved = pd.read_csv(tmp / "preds.csv")
            np.testing.assert_allclose(reloaded.predict_proba(self.X_test)[:, 1], saved["score_selected"].values)
            np.testing.assert_array_equal(reloaded.predict(self.X_test), saved["pred_selected"].values)

    def test_predict_risk_on_raw_record(self):
        model = self.trained["logistic_regression"]["model"]
        result = rm.predict_risk(model, {"Age": 58, "Biomarker_Level": 6.2, "Pre_Existing_Conditions": 2,
                                         "Vaccine_Dose_mcg": 120.0, "Prior_Reaction_History": "Mild"})
        self.assertIn(result.risk_label, ("High", "Low"))
        self.assertGreaterEqual(result.risk_probability, 0.0)
        self.assertLessEqual(result.risk_probability, 1.0)
        self.assertEqual(result.risk_label == "High", result.risk_probability >= rm.RISK_THRESHOLD)

    def test_reaction_history_is_ordinal(self):
        encoder = self.trained["logistic_regression"]["model"].named_steps["preprocess"].named_transformers_["reaction"]
        self.assertEqual(list(encoder.categories_[0]), ["None", "Mild", "Moderate"])


if __name__ == "__main__":
    unittest.main()
