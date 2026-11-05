"""Module 5: one figure with three panels built from the other modules' saved outputs.

Covers FR-5.1 to FR-5.3.

  Panel A  ROC curve of the risk classifier     <- outputs/test_predictions.csv
  Panel B  mRNA self-attention heatmap           <- outputs/attention_weights.npy + attention_meta.json
  Panel C  RAG retrieval vs report-generation latency <- outputs/logs/*.csv

Nothing is hard-coded: if a module has not been run, its input file is
missing and the dashboard stops with a message saying which step to run.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from config import (
    ATTENTION_META_PATH,
    ATTENTION_WEIGHTS_PATH,
    DASHBOARD_PATH,
    GENERATION_LOG_PATH,
    MODEL_META_PATH,
    RETRIEVAL_LOG_PATH,
    TEST_PREDICTIONS_PATH,
    ensure_output_dirs,
)

DPI = 200  # FR-5.3 asks for at least 150
NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}


def _require(path, step: str):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing. Run '{step}' first.")
    return path


def load_inputs() -> dict:
    """Read every module output the dashboard needs."""
    predictions = pd.read_csv(_require(TEST_PREDICTIONS_PATH, "python main.py train"))
    meta = json.loads(_require(MODEL_META_PATH, "python main.py train").read_text(encoding="utf-8"))
    weights = np.load(_require(ATTENTION_WEIGHTS_PATH, "python main.py attention"))
    attention_meta = json.loads(_require(ATTENTION_META_PATH, "python main.py attention").read_text(encoding="utf-8"))
    retrieval = pd.read_csv(_require(RETRIEVAL_LOG_PATH, "python main.py index"))
    generation = pd.read_csv(_require(GENERATION_LOG_PATH, "python main.py report"))

    # Logs accumulate across runs; use the rows from the most recent configuration only.
    last = retrieval.iloc[-1]
    retrieval = retrieval[(retrieval["embedding_model"] == last["embedding_model"]) & (retrieval["store"] == last["store"])]
    generation = generation[generation["provider"] == generation.iloc[-1]["provider"]]
    return {"predictions": predictions, "model_meta": meta, "attention": weights,
            "sequence": attention_meta["sequence"], "retrieval": retrieval, "generation": generation}


def panel_roc(ax, predictions: pd.DataFrame, selected: str) -> None:
    for name in NAMES:
        column = f"score_{name}"
        if column not in predictions:
            continue
        fpr, tpr, _ = roc_curve(predictions["y_true"], predictions[column])
        is_selected = name == selected
        ax.plot(fpr, tpr, lw=2.2 if is_selected else 1.4, ls="-" if is_selected else "--",
                color="#1F3A5F" if is_selected else "#8A9BB0",
                label=f"{NAMES[name]}{' (selected)' if is_selected else ''}  AUC = {auc(fpr, tpr):.2f}")
    ax.plot([0, 1], [0, 1], ls=":", color="gray", label="Chance  AUC = 0.50")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.set_title("A. Risk classifier ROC curve (test set, n=%d)" % len(predictions))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)


def panel_attention(fig, ax, weights: np.ndarray, sequence: str) -> None:
    im = ax.imshow(weights, cmap="viridis")
    labels = [f"{i + 1}{b}" for i, b in enumerate(sequence)]
    ax.set_xticks(range(len(sequence)))
    ax.set_yticks(range(len(sequence)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Key position (attending TO)")
    ax.set_ylabel("Query position (attending FROM)")
    ax.set_title(f"B. mRNA self-attention weights ({len(sequence)} bases)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention weight")


def panel_latency(ax, retrieval: pd.DataFrame, generation: pd.DataFrame) -> None:
    stages = {
        f"RAG retrieval\nper query (n={len(retrieval)})": retrieval["latency_s"].astype(float),
        f"Report generation\nper report (n={len(generation)})": generation["latency_s"].astype(float),
    }
    means = [s.mean() for s in stages.values()]
    stds = [s.std(ddof=0) for s in stages.values()]
    bars = ax.bar(list(stages), means, yerr=stds, capsize=6, color=["#0F6E6E", "#1F3A5F"], width=0.55)
    # A real LLM call takes seconds while retrieval takes milliseconds; use a log axis
    # when the two differ by more than 50x so both bars stay readable.
    log_scale = max(means) / max(min(means), 1e-9) > 50
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(min(means) / 3, max(m + s for m, s in zip(means, stds)) * 4)
    else:
        ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.3)
    for bar, value, sd in zip(bars, means, stds):
        label = f"{value * 1000:.2f} ms" if value < 1 else f"{value:.2f} s"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, value + sd), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=9, weight="bold")
    ax.set_title("C. Pipeline stage latency (mean +/- sd)")
    ax.set_ylabel("Seconds" + (" (log scale)" if log_scale else ""))
    provider = generation["provider"].iloc[-1]
    embedder = retrieval["embedding_model"].iloc[-1]
    ax.text(0.5, -0.2, f"Embedder: {embedder}\nGenerator: {provider}", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color="#555555")


def create_dashboard_figure(inputs: dict):
    """Build the 3-panel figure from a dict shaped like load_inputs()'s result."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.12, 0.95]})
    panel_roc(axes[0], inputs["predictions"], inputs["model_meta"]["selected_model"])
    panel_attention(fig, axes[1], inputs["attention"], inputs["sequence"])
    panel_latency(axes[2], inputs["retrieval"], inputs["generation"])
    fig.suptitle("mRNA Vaccine Clinical Safety & Dosage Intelligence Platform: results dashboard", fontsize=14, weight="bold")
    fig.supxlabel("Student engineering project on synthetic data - not real clinical guidance.",
                  fontsize=8, color="#777777")
    return fig


def build_dashboard(save_path=DASHBOARD_PATH, inputs: dict | None = None) -> str:
    """Load the saved module outputs (unless given) and save the dashboard PNG."""
    fig = create_dashboard_figure(inputs or load_inputs())
    ensure_output_dirs()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    return str(save_path)


def run() -> str:
    path = build_dashboard()
    print(f"Saved 3-panel dashboard to outputs/figures/{DASHBOARD_PATH.name} at {DPI} DPI")
    return path


if __name__ == "__main__":
    run()
