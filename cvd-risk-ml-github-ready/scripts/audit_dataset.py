"""Reproducible data-quality audit for the original CVD dataset.

No imputation is performed: derived-field consistency is tested only on
records where all operands are present. The audit writes aggregate statistics,
not patient-level records.

Run from the repository root:
    python scripts/audit_dataset.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.config import DEFAULT_DATA_PATH, REPORTS_DIR, TARGET_COLUMN
from cvd_risk.data import load_dataset


SCORE_INPUTS = [
    "BMI",
    "Systolic BP",
    "Total Cholesterol (mg/dL)",
    "Estimated LDL (mg/dL)",
    "HDL (mg/dL)",
    "Diabetes Status",
]
BMI_INPUTS = ["BMI", "Weight (kg)", "Height (m)"]


def audit_dataset(df: pd.DataFrame) -> dict:
    summary: dict = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
    }

    if TARGET_COLUMN in df.columns:
        summary["target_counts"] = {
            str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().items()
        }

    required = [*SCORE_INPUTS, "CVD Risk Score"]
    if all(col in df.columns for col in required):
        complete = df.dropna(subset=required)
        diabetes = complete["Diabetes Status"].astype(str).str.upper().eq("Y")
        reconstructed = (
            0.2
            + 2 * diabetes.astype(int)
            + complete["BMI"] / 5
            + complete["Systolic BP"] / 20
            + complete["Total Cholesterol (mg/dL)"] / 75
            + complete["Estimated LDL (mg/dL)"] / 150
            + complete["HDL (mg/dL)"] / 150
        )
        err = np.abs(complete["CVD Risk Score"] - reconstructed)
        summary["score_formula_check"] = {
            "complete_rows": int(len(complete)),
            "mean_absolute_error": float(err.mean()) if len(err) else None,
            "max_absolute_error": float(err.max()) if len(err) else None,
            "share_absolute_error_le_0_01": float((err <= 0.01).mean()) if len(err) else None,
            "formula": (
                "0.2 + 2*(Diabetes Status == Y) + BMI/5 + Systolic BP/20"
                " + Total Cholesterol/75 + Estimated LDL/150 + HDL/150"
            ),
        }

    if all(col in df.columns for col in BMI_INPUTS):
        complete = df.dropna(subset=BMI_INPUTS)
        valid = complete["Height (m)"] > 0
        complete = complete.loc[valid]
        expected = complete["Weight (kg)"] / complete["Height (m)"].pow(2)
        error = (complete["BMI"] - expected).abs()
        summary["bmi_consistency"] = {
            "complete_rows": int(len(complete)),
            "rows_absolute_difference_gt_0_5": int((error > 0.5).sum()),
            "fraction_absolute_difference_gt_0_5": float((error > 0.5).mean()) if len(error) else None,
            "max_absolute_difference": float(error.max()) if len(error) else None,
        }

    if "Estimated LDL (mg/dL)" in df.columns:
        summary["negative_estimated_ldl_count"] = int(
            (df["Estimated LDL (mg/dL)"] < 0).sum()
        )

    if "Blood Pressure (mmHg)" in df.columns:
        summary["unique_blood_pressure_strings"] = int(
            df["Blood Pressure (mmHg)"].nunique(dropna=True)
        )
        if {"Systolic BP", "Diastolic BP"}.issubset(df.columns):
            split = df["Blood Pressure (mmHg)"].str.split("/", n=1, expand=True)
            systolic = pd.to_numeric(split[0], errors="coerce")
            diastolic = pd.to_numeric(split[1], errors="coerce")
            bp_comparison = {}
            for label, parsed, column in [
                ("systolic", systolic, "Systolic BP"),
                ("diastolic", diastolic, "Diastolic BP"),
            ]:
                mask = parsed.notna() & df[column].notna()
                bp_comparison[label] = {
                    "pairs_compared": int(mask.sum()),
                    "exact_match_fraction": float((parsed.loc[mask] == df.loc[mask, column]).mean()) if mask.any() else None,
                }
            summary["blood_pressure_redundancy"] = bp_comparison

    if {"CVD Risk Score", TARGET_COLUMN}.issubset(df.columns):
        summary["risk_score_ranges_by_level"] = {
            str(label): {
                "min": float(group.min()),
                "max": float(group.max()),
                "count_not_missing": int(group.count()),
            }
            for label, group in df.groupby(TARGET_COLUMN)["CVD Risk Score"]
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit original CVD data quality and derived fields")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "dataset_audit.json")
    args = parser.parse_args()
    report = audit_dataset(load_dataset(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nAudit saved: {args.output}")


if __name__ == "__main__":
    main()
