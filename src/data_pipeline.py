"""Module 1a: load, validate, profile and clean ``patient_records.csv``.

Covers FR-1.1 (column validation) and FR-1.2 (imputation).

A note on reading the CSV
-------------------------
By default pandas treats the text ``None`` as a missing value. In this dataset
``None`` is a real category of ``Prior_Reaction_History`` ("no prior
reaction", 291 patients). Reading the file with the defaults would turn those
291 patients into blanks and the mode-imputation step would then relabel them
all as ``Mild``. ``load_raw`` therefore tells pandas that only empty cells are
missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from config import CLEAN_CSV, RAW_CSV

REQUIRED_COLUMNS = [
    "Patient_ID",
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
    "Prior_Reaction_History",
    "Reaction_Score",
]
NUMERIC_COLS = ["Age", "Biomarker_Level", "Vaccine_Dose_mcg"]  # standardized (FR-1.3)
COUNT_COL = "Pre_Existing_Conditions"  # small integer count 0-4, used as-is
CATEGORICAL_COL = "Prior_Reaction_History"
TARGET_COL = "Reaction_Score"

# Ordered from least to most severe; the order matters for ordinal encoding.
REACTION_LEVELS = ["None", "Mild", "Moderate"]


class MissingColumnsError(ValueError):
    """Raised when the patient CSV lacks one or more required columns."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        names = ", ".join(missing)
        super().__init__(f"patient_records.csv is missing required column(s): {names}")


class InvalidCategoryError(ValueError):
    """Raised when Prior_Reaction_History holds a value outside REACTION_LEVELS."""


def validate_columns(df: pd.DataFrame) -> None:
    """FR-1.1: raise MissingColumnsError naming every missing required column."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise MissingColumnsError(missing)


def load_raw(path: str | Path = RAW_CSV) -> pd.DataFrame:
    """Read the CSV, treating only empty cells as missing, then validate it."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"Patient_ID": str})
    validate_columns(df)

    present = df[CATEGORICAL_COL].dropna()
    unknown = sorted(set(present) - set(REACTION_LEVELS))
    if unknown:
        raise InvalidCategoryError(
            f"{CATEGORICAL_COL} contains unexpected value(s) {unknown}; "
            f"allowed values are {REACTION_LEVELS}"
        )
    return df


def profile(df: pd.DataFrame) -> dict:
    """Summarize shape, dtypes, missing values and basic statistics."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "describe": df.describe().round(3).to_dict(),
        "class_balance": df[TARGET_COL].value_counts(normalize=True).round(3).to_dict(),
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """FR-1.2: impute missing values and return (clean_df, imputation_log).

    Why impute instead of dropping rows: 37 of 500 rows have a gap. Dropping
    them would throw away 7% of an already small dataset, and the gaps are in
    a single feature per row, so the rest of each record is still useful.

    - Biomarker_Level -> median. The column is right-skewed (median 4.8,
      max 15.8), so the median is a more typical value than the mean.
    - Prior_Reaction_History -> mode, the standard choice for a categorical.

    The other numeric columns have no gaps in the provided file; they get the
    same median rule so the function stays safe on other copies of the data.
    A row without a target label cannot be used for training, so it is dropped.
    """
    df = df.copy()
    log: dict = {"rows_in": len(df)}

    no_label = df[TARGET_COL].isnull()
    if no_label.any():
        log["rows_dropped_missing_target"] = int(no_label.sum())
        df = df.loc[~no_label]

    for col in NUMERIC_COLS + [COUNT_COL]:
        n_missing = int(df[col].isnull().sum())
        if n_missing:
            median = float(df[col].median())
            df[col] = df[col].fillna(median)
            log[col] = {"strategy": "median", "value": median, "filled": n_missing}

    n_missing = int(df[CATEGORICAL_COL].isnull().sum())
    if n_missing:
        mode = df[CATEGORICAL_COL].mode()[0]
        df[CATEGORICAL_COL] = df[CATEGORICAL_COL].fillna(mode)
        log[CATEGORICAL_COL] = {"strategy": "mode", "value": mode, "filled": n_missing}

    df[COUNT_COL] = df[COUNT_COL].astype(int)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    log["rows_out"] = len(df)
    return df.reset_index(drop=True), log


def run(raw_path: str | Path = RAW_CSV, clean_path: str | Path = CLEAN_CSV) -> pd.DataFrame:
    """Load, profile, clean and save the dataset. Returns the clean DataFrame."""
    df = load_raw(raw_path)
    stats = profile(df)
    print(f"Loaded {stats['shape'][0]} rows x {stats['shape'][1]} columns")
    print("Missing values before cleaning:")
    for col, count in stats["missing"].items():
        print(f"  {col:<24} {count:>4}  ({stats['missing_pct'][col]}%)")

    df_clean, log = clean(df)
    for col in (c for c in log if isinstance(log[c], dict)):
        entry = log[col]
        print(f"Imputed {entry['filled']} missing {col} values with the {entry['strategy']} ({entry['value']})")

    remaining = int(df_clean.isnull().sum().sum())
    print(f"Missing values after cleaning: {remaining}")
    df_clean.to_csv(clean_path, index=False)
    print(f"Saved clean dataset to {Path(clean_path).name}")
    return df_clean


if __name__ == "__main__":
    try:
        run()
    except (MissingColumnsError, InvalidCategoryError) as err:
        sys.exit(f"Data validation failed: {err}")
