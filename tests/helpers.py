"""Shared test setup: put src/ on the import path and silence the fallback warnings."""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

warnings.filterwarnings("ignore", message=".*fallback.*")
warnings.filterwarnings("ignore", message=".*unavailable.*")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
