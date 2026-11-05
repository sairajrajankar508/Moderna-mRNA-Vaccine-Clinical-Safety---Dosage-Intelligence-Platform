"""Shared paths and constants for every module.

All paths are built from the repository root, so the scripts work no matter
which folder you launch them from (``python main.py``, ``python src/attention.py``,
or the test runner).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load settings from .env before any module reads environment variables.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv not installed: environment variables still work
    pass

# ---- Input data (provided with the project) ---------------------------------
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "patient_records.csv"
CLEAN_CSV = DATA_DIR / "patient_records_clean.csv"
GUIDELINES_TXT = DATA_DIR / "mrna_safety_guidelines.txt"

# ---- Generated outputs ------------------------------------------------------
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Module 1
MODEL_PATH = OUTPUTS_DIR / "risk_model.joblib"
MODEL_META_PATH = OUTPUTS_DIR / "risk_model_meta.json"
TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"

# Module 2
ATTENTION_WEIGHTS_PATH = OUTPUTS_DIR / "attention_weights.npy"
ATTENTION_META_PATH = OUTPUTS_DIR / "attention_meta.json"
ATTENTION_HEATMAP_PATH = FIGURES_DIR / "attention_heatmap.png"

# Module 3
CHROMA_DIR = OUTPUTS_DIR / "chroma_store"
CHROMA_COLLECTION = "mrna_safety_guidelines"
RETRIEVAL_LOG_PATH = LOGS_DIR / "retrieval_latency.csv"

# Module 4
GENERATION_LOG_PATH = LOGS_DIR / "report_generation_latency.csv"

# Module 5
DASHBOARD_PATH = FIGURES_DIR / "dashboard.png"

# ---- Reproducibility --------------------------------------------------------
RANDOM_SEED = 42


def ensure_output_dirs() -> None:
    """Create every output folder if it does not exist yet."""
    for folder in (OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR, METRICS_DIR, LOGS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
