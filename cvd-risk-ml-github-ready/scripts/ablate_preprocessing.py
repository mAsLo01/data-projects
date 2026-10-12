"""Isolate preprocessing changes for MLP with paired train-only cross-validation.

Run from the project root after `python -m pip install -e .`:
    python scripts/ablate_preprocessing.py

The fixed, stratified holdout is never used for training or evaluation here.
All row-wise transformations use no training-dependent statistics; median/mode
imputation and scaling remain inside the scikit-learn Pipeline and CV folds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from cvd_risk.config import CV_SPLITS, DEFAULT_DATA_PATH, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from cvd_risk.data import clean_dataset, load_dataset, prepare_features
from cvd_risk.modeling import pipeline_for_model

# Single-step variants show independent effects; leave-one-out variants expose
# possible interactions with the other cleaning steps.
CONFIGURATIONS: dict[str, tuple[str, ...]] = {
    "original_all": (),
    "drop_bp_text_only": ("drop_bp",),
    "recover_bp_then_drop_text": ("recover_bp",),
    "all_clean_without_bp": ("recover_height", "mask_ldl"),
    "all_clean_without_height": ("recover_bp", "mask_ldl"),
    "all_clean_without_ldl": ("recover_bp", "recover_height"),
    "all_clean": ("recover_bp", "recover_height", "mask_ldl"),
}


def transform_features(raw_features: pd.DataFrame, steps: tuple[str, ...]) -> pd.DataFrame:
    """Apply independent, deterministic transformations without fitting."""
    x = raw_features.copy()
    for step in steps:
        if step == "drop_bp":
            x = x.drop(columns=["Blood Pressure (mmHg)"], errors="ignore")
        elif step == "recover_bp":
            if "Blood Pressure (mmHg)" in x:
                pressure = x["Blood Pressure (mmHg)"].astype("string").str.extract(
                    r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$"
                )
                for name, part in (("Systolic BP", 0), ("Diastolic BP", 1)):
                    recovered = pd.to_numeric(pressure[part], errors="coerce")
                    if name in x:
                        current = pd.to_numeric(x[name], errors="coerce")
                        x[name] = current.fillna(recovered)
                    else:
                        x[name] = recovered
                x = x.drop(columns=["Blood Pressure (mmHg)"])
        elif step == "drop_height":
            x = x.drop(columns=["Height (cm)"], errors="ignore")
        elif step == "recover_height":
            if "Height (cm)" in x:
                recovered = pd.to_numeric(x["Height (cm)"], errors="coerce") / 100.0
                if "Height (m)" in x:
                    current = pd.to_numeric(x["Height (m)"], errors="coerce")
                    x["Height (m)"] = current.fillna(recovered)
                else:
                    x["Height (m)"] = recovered
                x = x.drop(columns=["Height (cm)"])
        elif step == "mask_ldl":
            if "Estimated LDL (mg/dL)" in x:
                ldl = pd.to_numeric(x["Estimated LDL (mg/dL)"], errors="coerce")
                x["Estimated LDL (mg/dL)"] = ldl.mask(ldl < 0)
        else:
            raise ValueError(f"Unknown preprocessing step: {step}")
    return x


def run(data_path: Path, random_state: int, output_dir: Path) -> None:
    raw = clean_dataset(load_dataset(data_path))
    train_idx, _holdout_idx = train_test_split(
        np.arange(len(raw)),
        test_size=TEST_SIZE,
        stratify=raw["CVD Risk Level"],
        random_state=random_state,
    )
    train_df = raw.iloc[train_idx].reset_index(drop=True)
    x_original, y = prepare_features(train_df, "all")
    folds = list(
        StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_state)
        .split(x_original, y)
    )
    labels = sorted(y.unique())
    records: list[dict[str, object]] = []
    classwise: list[dict[str, object]] = []

    for name, steps in CONFIGURATIONS.items():
        x = transform_features(x_original, steps)
        if name == "all_clean":
            # Ensure this standalone ablation exactly matches the project's
            # real preprocessing, rather than evaluating a different recipe.
            clean_x, _ = prepare_features(train_df, "all_clean")
            assert_frame_equal(x, clean_x, check_dtype=True)
        oof = pd.Series(index=y.index, dtype="object")
        for fold_number, (fit_idx, valid_idx) in enumerate(folds, start=1):
            print(f"  {name}: fold {fold_number}", flush=True)
            model = pipeline_for_model("mlp", random_state)
            # Avoid BLAS oversubscription on small tabular ML workloads.
            with threadpool_limits(limits=1, user_api="blas"):
                model.fit(x.iloc[fit_idx], y.iloc[fit_idx])
                pred = model.predict(x.iloc[valid_idx])
            oof.iloc[valid_idx] = pred
            records.append(
                {
                    "experiment": name,
                    "fold": fold_number,
                    "n_features_input": x.shape[1],
                    "macro_f1": f1_score(y.iloc[valid_idx], pred, average="macro"),
                    "balanced_accuracy": balanced_accuracy_score(y.iloc[valid_idx], pred),
                }
            )
        per_label = classification_report(
            y, oof, labels=labels, output_dict=True, zero_division=0
        )
        for label in labels:
            classwise.append(
                {
                    "experiment": name,
                    "class": label,
                    "precision": per_label[label]["precision"],
                    "recall": per_label[label]["recall"],
                    "f1": per_label[label]["f1-score"],
                    "support": per_label[label]["support"],
                }
            )
        print(f"Finished: {name} ({x.shape[1]} input columns)", flush=True)

    detailed = pd.DataFrame(records)
    summary = (
        detailed.groupby("experiment", sort=False)
        .agg(
            n_features_input=("n_features_input", "first"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
        )
        .reset_index()
    )
    fold_pivot = detailed.pivot(index="fold", columns="experiment", values="macro_f1")
    summary["macro_f1_delta_vs_original"] = summary["experiment"].map(
        lambda name: (fold_pivot[name] - fold_pivot["original_all"]).mean()
    )
    class_report = pd.DataFrame(classwise)
    low_scores = class_report.loc[class_report["class"] == "LOW", ["experiment", "f1", "recall"]]
    low_scores = low_scores.rename(columns={"f1": "low_f1_oof", "recall": "low_recall_oof"})
    summary = summary.merge(low_scores, on="experiment")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "preprocessing_ablation_summary.csv", index=False)
    detailed.to_csv(output_dir / "preprocessing_ablation_folds.csv", index=False)
    class_report.to_csv(output_dir / "preprocessing_ablation_per_class.csv", index=False)
    print("\nPaired CV results (TRAIN ONLY, holdout NOT evaluated):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    run(args.data, args.random_state, args.output_dir)


if __name__ == "__main__":
    main()
