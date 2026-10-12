from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import FEATURE_SETS, LEAKAGE_COLUMNS, TARGET_COLUMN


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the CVD dataset and normalize column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. See data/README.md for download instructions."
        )

    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply target-independent cleaning safe to run before the train/test split."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"Required target column is missing: {TARGET_COLUMN!r}")

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).copy()
    cleaned[TARGET_COLUMN] = (
        cleaned[TARGET_COLUMN].astype(str).str.strip().str.upper()
    )
    return cleaned


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def prepare_features(
    df: pd.DataFrame,
    feature_set: str = "all",
) -> tuple[pd.DataFrame, pd.Series]:
    """Create X and y without fitting any preprocessing on the full dataset.

    The continuous ``CVD Risk Score`` is intentionally removed from the feature
    matrix because the categorical target ``CVD Risk Level`` is closely tied to
    that score. Keeping it would make the classification task vulnerable to
    target leakage.
    """
    validate_columns(df, [TARGET_COLUMN])

    y = df[TARGET_COLUMN].copy()

    if feature_set == "all_clean":
        X = prepare_all_clean_features(df)
    elif feature_set == "all":
        drop_columns = [TARGET_COLUMN] + [
            column for column in LEAKAGE_COLUMNS if column in df.columns
        ]
        X = df.drop(columns=drop_columns).copy()
    elif feature_set in FEATURE_SETS:
        selected = FEATURE_SETS[feature_set]
        validate_columns(df, selected)
        X = df[selected].copy()
    else:
        choices = ["all", "all_clean", *FEATURE_SETS.keys()]
        raise ValueError(
            f"Unknown feature set {feature_set!r}. Choose one of: {choices}."
        )

    if X.empty:
        raise ValueError("Feature matrix is empty after column selection.")

    return X, y


def prepare_all_clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic *row-wise* cleanup; learns no statistics from the dataset.

    - Keep recorded BMI untouched: mass/height and BMI are inconsistent for many rows.
    - Prefer numeric blood pressure; recover missing readings from the text field.
    - Prefer height in metres; recover missing values from centimetres.
    - Treat negative estimated LDL as missing; fold-trained imputation is later.
    - Remove the redundant text blood pressure and height-in-cm columns.

    No rows are removed, no medians or category frequencies are estimated here.
    """
    drop = [TARGET_COLUMN, *LEAKAGE_COLUMNS]
    X = df.drop(columns=[name for name in drop if name in df.columns]).copy()

    if "Blood Pressure (mmHg)" in X.columns:
        pressure = X["Blood Pressure (mmHg)"].astype("string").str.extract(
            r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$"
        )
        for column, part in (("Systolic BP", 0), ("Diastolic BP", 1)):
            recovered = pd.to_numeric(pressure[part], errors="coerce")
            if column not in X.columns:
                X[column] = recovered
            else:
                original = pd.to_numeric(X[column], errors="coerce")
                X[column] = original.fillna(recovered)
        X = X.drop(columns=["Blood Pressure (mmHg)"])

    if "Height (cm)" in X.columns:
        recovered = pd.to_numeric(X["Height (cm)"], errors="coerce") / 100.0
        if "Height (m)" not in X.columns:
            X["Height (m)"] = recovered
        else:
            original = pd.to_numeric(X["Height (m)"], errors="coerce")
            X["Height (m)"] = original.fillna(recovered)
        X = X.drop(columns=["Height (cm)"])

    if "Estimated LDL (mg/dL)" in X.columns:
        ldl = pd.to_numeric(X["Estimated LDL (mg/dL)"], errors="coerce")
        X["Estimated LDL (mg/dL)"] = ldl.mask(ldl < 0)

    return X


def prepare_prediction_features(
    df: pd.DataFrame, feature_set: str, required_columns: list[str]
) -> pd.DataFrame:
    """Apply the same deterministic cleanup when running saved-model inference."""
    X = prepare_all_clean_features(df) if feature_set == "all_clean" else df
    validate_columns(X, required_columns)
    return X.loc[:, required_columns].copy()
