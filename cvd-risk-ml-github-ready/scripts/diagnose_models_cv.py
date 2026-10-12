"""Paired, training-only CV diagnostics for cleaned and original feature sets.

This script does NOT train on or evaluate the holdout test set.

Run from project root:
    python scripts/diagnose_models_cv.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from cvd_risk.config import (
    CV_SPLITS,
    DEFAULT_DATA_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from cvd_risk.data import clean_dataset, load_dataset, prepare_features
from cvd_risk.modeling import pipeline_for_model

CONFIGS = (
    ("mlp_all", "mlp", "all"),
    ("mlp_all_clean", "mlp", "all_clean"),
    ("logreg_all_clean", "logistic_regression", "all_clean"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    df = clean_dataset(load_dataset(args.data))
    indices = np.arange(len(df))
    train_idx, _ = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["CVD Risk Level"],
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    y = train_df["CVD Risk Level"]
    labels = sorted(y.unique())
    folds = list(
        StratifiedKFold(
            n_splits=CV_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ).split(train_df, y)
    )

    fold_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    matrices: list[pd.DataFrame] = []

    for experiment, model_name, feature_set in CONFIGS:
        X, _ = prepare_features(train_df, feature_set)
        oof = pd.Series(index=y.index, dtype="object")

        for fold, (fit_idx, valid_idx) in enumerate(folds, 1):
            pipe = pipeline_for_model(model_name, RANDOM_STATE)
            pipe.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            pred = pipe.predict(X.iloc[valid_idx])
            oof.iloc[valid_idx] = pred
            fold_rows.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "macro_f1": f1_score(y.iloc[valid_idx], pred, average="macro"),
                    "balanced_accuracy": balanced_accuracy_score(
                        y.iloc[valid_idx], pred
                    ),
                }
            )

        report = classification_report(
            y, oof, labels=labels, output_dict=True, zero_division=0
        )
        for label in labels:
            metrics = report[label]
            per_class_rows.append(
                {
                    "experiment": experiment,
                    "class": label,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1-score"],
                    "support": int(metrics["support"]),
                }
            )

        matrix = pd.DataFrame(
            confusion_matrix(y, oof, labels=labels),
            index=[f"true_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )
        matrix.insert(0, "experiment", experiment)
        matrices.append(matrix.reset_index(names="true_class"))

    frame = pd.DataFrame(fold_rows)
    summary = (
        frame.groupby("experiment", sort=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
        )
        .reset_index()
    )
    pivot = frame.pivot(index="fold", columns="experiment", values="macro_f1")
    delta = pivot["mlp_all_clean"] - pivot["mlp_all"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORTS_DIR / "diagnostic_cv_folds.csv", index=False)
    summary.to_csv(REPORTS_DIR / "diagnostic_cv_summary.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(
        REPORTS_DIR / "diagnostic_cv_per_class.csv", index=False
    )
    pd.concat(matrices).to_csv(
        REPORTS_DIR / "diagnostic_cv_confusion_matrices.csv", index=False
    )

    print("Paired five-fold CV, training portion only (holdout untouched):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nMLP macro F1 gain, all_clean - all, per fold:")
    print(delta.round(4).to_string())
    print(f"Mean gain: {delta.mean():.4f}")
    print("\nClasswise metrics from out-of-fold predictions:")
    print(
        pd.DataFrame(per_class_rows).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print(f"\nCSV reports saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
