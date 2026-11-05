"""Module 1c: exploratory data analysis figures (Week 1 Definition of Done).

Produces four figures in outputs/figures/:
  eda_distributions.png        histograms of the numeric features, split by risk class
  eda_correlation_heatmap.png  Pearson correlations, including the target
  eda_class_balance.png        how many Low vs High risk patients there are
  eda_reaction_history.png     share of High-risk patients per prior-reaction level
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR, ensure_output_dirs
from data_pipeline import CATEGORICAL_COL, COUNT_COL, NUMERIC_COLS, REACTION_LEVELS, TARGET_COL

RISK_PALETTE = {0: "#3B7EA1", 1: "#C8553D"}
RISK_NAMES = {0: "Low risk", 1: "High risk"}


def _finish(fig, path, close: bool):
    """Save the figure; keep it open (close=False) so a notebook can display it inline."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if close:
        plt.close(fig)
    return path


def plot_distributions(df: pd.DataFrame, path=FIGURES_DIR / "eda_distributions.png", close: bool = True):
    cols = NUMERIC_COLS + [COUNT_COL]
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 3.8))
    labelled = df.assign(Risk=df[TARGET_COL].map(RISK_NAMES))
    palette = {RISK_NAMES[k]: v for k, v in RISK_PALETTE.items()}
    for ax, col in zip(axes, cols):
        discrete = col == COUNT_COL
        sns.histplot(data=labelled, x=col, hue="Risk", palette=palette, ax=ax,
                     discrete=discrete, bins=None if discrete else 20, alpha=0.55, legend=ax is axes[-1])
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("")
    fig.suptitle("Feature distributions by reaction-risk class", fontsize=13)
    return _finish(fig, path, close)


def plot_correlation(df: pd.DataFrame, path=FIGURES_DIR / "eda_correlation_heatmap.png", close: bool = True):
    encoded = df.assign(**{CATEGORICAL_COL: df[CATEGORICAL_COL].map({v: i for i, v in enumerate(REACTION_LEVELS)})})
    corr = encoded[NUMERIC_COLS + [COUNT_COL, CATEGORICAL_COL, TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation matrix (reaction history encoded None=0, Mild=1, Moderate=2)", fontsize=10)
    return _finish(fig, path, close)


def plot_class_balance(df: pd.DataFrame, path=FIGURES_DIR / "eda_class_balance.png", close: bool = True):
    counts = df[TARGET_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([RISK_NAMES[i] for i in counts.index], counts.values, color=[RISK_PALETTE[i] for i in counts.index])
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 4, f"{n} ({n / len(df):.0%})", ha="center")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_ylabel("Patients")
    ax.set_title("Class balance of Reaction_Score")
    return _finish(fig, path, close)


def plot_reaction_history(df: pd.DataFrame, path=FIGURES_DIR / "eda_reaction_history.png", close: bool = True):
    rate = df.groupby(CATEGORICAL_COL)[TARGET_COL].agg(["mean", "count"]).reindex(REACTION_LEVELS)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(rate.index, rate["mean"], color="#C8553D")
    for i, (share, n) in enumerate(zip(rate["mean"], rate["count"])):
        ax.text(i, share + 0.02, f"{share:.0%}\n(n={n})", ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of High-risk patients")
    ax.set_xlabel("Prior reaction history")
    ax.set_title("High-risk rate by prior reaction")
    return _finish(fig, path, close)


def run(df: pd.DataFrame) -> list:
    """Generate all EDA figures from a clean DataFrame."""
    ensure_output_dirs()
    paths = [plot_distributions(df), plot_correlation(df), plot_class_balance(df), plot_reaction_history(df)]
    print(f"Saved {len(paths)} EDA figures to outputs/figures/")
    return paths


if __name__ == "__main__":
    from data_pipeline import clean, load_raw

    run(clean(load_raw())[0])
