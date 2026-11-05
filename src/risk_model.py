"""Module 1b: train, compare and save the adverse-reaction risk classifier.

Covers FR-1.3 to FR-1.7.

Design choices
--------------
* Preprocessing lives inside a scikit-learn ``Pipeline``. The scaler is fitted
  on the training split only (the PRD template fits it on the whole file,
  which leaks test-set statistics into training). Saving the whole pipeline
  also means Module 4 can score a raw patient record without re-implementing
  the scaling.
* Prior_Reaction_History is encoded as an ordered number
  (None=0 < Mild=1 < Moderate=2). ``LabelEncoder`` would sort alphabetically
  (Mild=0, Moderate=1, None=2) and put "None" above "Moderate", which makes
  no sense for a severity scale.
* Each model gets a small 5-fold grid search on the training split so the
  comparison is between reasonably tuned models, not defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_META_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_PREDICTIONS_PATH,
    ensure_output_dirs,
)
from data_pipeline import (
    CATEGORICAL_COL,
    COUNT_COL,
    NUMERIC_COLS,
    REACTION_LEVELS,
    TARGET_COL,
    load_raw,
    clean,
)

FEATURE_COLS = NUMERIC_COLS + [COUNT_COL, CATEGORICAL_COL]
RISK_THRESHOLD = 0.5  # probability at or above this is labelled "High"
MODEL_DISPLAY_NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}


@dataclass
class RiskResult:
    """Module 1 output consumed by Module 4."""

    risk_label: str  # "High" or "Low"
    risk_probability: float  # P(Reaction_Score == 1)
    model_name: str


# ---------------------------------------------------------------------------
# Features and split
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Scale the 3 continuous columns, ordinal-encode reaction history, pass the count through."""
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), NUMERIC_COLS),
            ("count", "passthrough", [COUNT_COL]),
            ("reaction", OrdinalEncoder(categories=[REACTION_LEVELS]), [CATEGORICAL_COL]),
        ]
    )


def split_data(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """FR-1.4: stratified 80/20 split with a fixed seed. Patient_ID becomes the index."""
    indexed = df.set_index("Patient_ID")
    X = indexed[FEATURE_COLS]
    y = indexed[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def scaled_training_stats(model: Pipeline, X_train: pd.DataFrame) -> dict:
    """FR-1.3 check: mean and std of the standardized columns on the training set."""
    scaler = model.named_steps["preprocess"].named_transformers_["scale"]
    scaled = scaler.transform(X_train[NUMERIC_COLS])
    return {
        col: {"mean": float(scaled[:, i].mean()), "std": float(scaled[:, i].std())}
        for i, col in enumerate(NUMERIC_COLS)
    }


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def _candidate_models(seed: int) -> dict[str, tuple[Pipeline, dict]]:
    """Each candidate: an untrained pipeline plus a small hyper-parameter grid."""
    logistic = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    forest = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)),
    ])
    return {
        "logistic_regression": (logistic, {"model__C": [0.01, 0.1, 1.0, 10.0]}),
        "random_forest": (forest, {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 5, 10]}),
    }


def train_models(X_train: pd.DataFrame, y_train: pd.Series, seed: int = RANDOM_SEED) -> dict:
    """FR-1.5: fit Logistic Regression and Random Forest on the same training data.

    Returns {name: {"model": fitted Pipeline, "best_params": dict, "cv_f1": float}}.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    trained = {}
    for name, (pipeline, grid) in _candidate_models(seed).items():
        search = GridSearchCV(pipeline, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        trained[name] = {
            "model": search.best_estimator_,
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "cv_f1": float(search.best_score_),
        }
    return trained


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """FR-1.6: classification report, confusion matrix, F1 and ROC-AUC on the test set."""
    scores = model.predict_proba(X_test)[:, 1]
    preds = (scores >= RISK_THRESHOLD).astype(int)
    return {
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "report_text": classification_report(y_test, preds, target_names=["Low risk (0)", "High risk (1)"], digits=3),
        "report_dict": classification_report(y_test, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "scores": scores,
        "preds": preds,
    }


def select_best(evaluations: dict) -> str:
    """FR-1.7: pick the model with the highest test-set F1 (class 1 = High risk).

    On an exact tie the Logistic Regression wins because it is simpler and its
    coefficients are directly interpretable.
    """
    order = ["logistic_regression", "random_forest"]
    return max(order, key=lambda name: (round(evaluations[name]["f1"], 6), -order.index(name)))


def interpret(model: Pipeline) -> pd.DataFrame:
    """Feature weights: LR coefficients (on scaled inputs) or RF impurity importances."""
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values, kind = estimator.coef_[0], "coefficient"
    else:
        values, kind = estimator.feature_importances_, "importance"
    out = pd.DataFrame({"feature": FEATURE_COLS, kind: values})
    return out.reindex(out[kind].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Saving artifacts
# ---------------------------------------------------------------------------

def _save_confusion_matrix(cm: list, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ConfusionMatrixDisplay(np.array(cm), display_labels=["Low", "High"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Actual risk")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_report_figure(text: str, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11)
    ax.text(0, 1, text, family="monospace", fontsize=9, va="top")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def save_artifacts(trained: dict, evaluations: dict, best: str, X_train, X_test, y_test) -> dict:
    """Write reports, confusion matrices, the chosen model and its metadata to outputs/."""
    ensure_output_dirs()
    comparison = []
    for name, ev in evaluations.items():
        label = MODEL_DISPLAY_NAMES[name]
        (METRICS_DIR / f"{name}_classification_report.txt").write_text(ev["report_text"], encoding="utf-8")
        (METRICS_DIR / f"{name}_classification_report.json").write_text(
            json.dumps(ev["report_dict"], indent=2), encoding="utf-8"
        )
        pd.DataFrame(ev["confusion_matrix"], index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(
            METRICS_DIR / f"{name}_confusion_matrix.csv"
        )
        _save_confusion_matrix(ev["confusion_matrix"], f"{label}: confusion matrix", FIGURES_DIR / f"{name}_confusion_matrix.png")
        comparison.append({
            "model": name,
            "test_f1": round(ev["f1"], 4),
            "test_roc_auc": round(ev["roc_auc"], 4),
            "test_accuracy": round(ev["report_dict"]["accuracy"], 4),
            "cv_f1": round(trained[name]["cv_f1"], 4),
            "best_params": json.dumps(trained[name]["best_params"]),
            "selected": name == best,
        })
    pd.DataFrame(comparison).to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    # The chosen model's figures go to outputs/figures/ (Week 1 Definition of Done).
    best_label = MODEL_DISPLAY_NAMES[best]
    _save_confusion_matrix(evaluations[best]["confusion_matrix"], f"Final model ({best_label})", FIGURES_DIR / "final_model_confusion_matrix.png")
    _save_report_figure(evaluations[best]["report_text"], f"Final model ({best_label}): classification report", FIGURES_DIR / "final_model_classification_report.png")

    model = trained[best]["model"]
    joblib.dump(model, MODEL_PATH)
    interpret(model).to_csv(METRICS_DIR / "final_model_feature_weights.csv", index=False)

    # Test-set predictions from both models: used by the dashboard ROC panel and the reload test.
    predictions = pd.DataFrame({"Patient_ID": X_test.index, "y_true": y_test.values})
    for name, ev in evaluations.items():
        predictions[f"score_{name}"] = ev["scores"]
    predictions["score_selected"] = evaluations[best]["scores"]
    predictions["pred_selected"] = evaluations[best]["preds"]
    predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    meta = {
        "selected_model": best,
        "selection_rule": "highest test-set F1 for the High-risk class",
        "risk_threshold": RISK_THRESHOLD,
        "feature_columns": FEATURE_COLS,
        "reaction_encoding": {level: i for i, level in enumerate(REACTION_LEVELS)},
        "test_metrics": {n: {"f1": e["f1"], "roc_auc": e["roc_auc"]} for n, e in evaluations.items()},
        "scaled_training_stats": scaled_training_stats(model, X_train),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_seed": RANDOM_SEED,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# Inference (used by Module 4)
# ---------------------------------------------------------------------------

def load_model(path=MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run 'python main.py train' first.")
    return joblib.load(path)


def predict_risk(model: Pipeline, patient: dict) -> RiskResult:
    """Score one raw patient record (unscaled values, reaction history as text)."""
    row = pd.DataFrame([{col: patient[col] for col in FEATURE_COLS}])
    probability = float(model.predict_proba(row)[0, 1])
    label = "High" if probability >= RISK_THRESHOLD else "Low"
    meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8")) if MODEL_META_PATH.exists() else {}
    return RiskResult(label, probability, meta.get("selected_model", type(model.named_steps["model"]).__name__))


def run(df: pd.DataFrame | None = None) -> dict:
    """Full Module 1 training run. Returns the saved metadata."""
    if df is None:
        df, _ = clean(load_raw())
    X_train, X_test, y_train, y_test = split_data(df)
    trained = train_models(X_train, y_train)
    evaluations = {name: evaluate(t["model"], X_test, y_test) for name, t in trained.items()}

    for name, ev in evaluations.items():
        print(f"\n--- {MODEL_DISPLAY_NAMES[name]} (best params {trained[name]['best_params']}) ---")
        print(ev["report_text"])
        print(f"Confusion matrix [[TN FP] [FN TP]]: {ev['confusion_matrix']}")
        print(f"Test F1 = {ev['f1']:.3f}   ROC-AUC = {ev['roc_auc']:.3f}")

    best = select_best(evaluations)
    meta = save_artifacts(trained, evaluations, best, X_train, X_test, y_test)
    print(f"\nSelected model: {MODEL_DISPLAY_NAMES[best]} (test F1 {evaluations[best]['f1']:.3f})")
    print(f"Saved {MODEL_PATH.name}, reports and confusion matrices to outputs/")
    return meta


if __name__ == "__main__":
    from data_pipeline import run as run_data_pipeline

    run(run_data_pipeline())
