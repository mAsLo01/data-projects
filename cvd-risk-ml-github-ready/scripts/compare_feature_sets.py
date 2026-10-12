"""Paired, train-only feature ablation for one fixed estimator."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from cvd_risk.config import (
    CV_SPLITS, DEFAULT_DATA_PATH, RANDOM_STATE, REPORTS_DIR, TEST_SIZE,
)
from cvd_risk.data import clean_dataset, load_dataset, prepare_features
from cvd_risk.modeling import pipeline_for_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    data = clean_dataset(load_dataset(args.data))
    # The same training subset and same CV folds across configurations.
    rows = np.arange(len(data))
    train_idx, _holdout_idx = train_test_split(
        rows, test_size=TEST_SIZE, random_state=args.random_state,
        stratify=data["CVD Risk Level"],
    )
    train_df = data.iloc[train_idx].reset_index(drop=True)
    y_train = train_df["CVD Risk Level"]
    folds = list(StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=args.random_state,
    ).split(train_df, y_train))

    records = []
    for feature_set in ("two", "core", "all", "all_clean"):
        X, y = prepare_features(train_df, feature_set)
        for fold_index, (fold_train, fold_valid) in enumerate(folds, start=1):
            model = pipeline_for_model("logistic_regression", args.random_state)
            model.fit(X.iloc[fold_train], y.iloc[fold_train])
            predictions = model.predict(X.iloc[fold_valid])
            records.append({
                "feature_set": feature_set,
                "fold": fold_index,
                "macro_f1": f1_score(y.iloc[fold_valid], predictions, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(
                    y.iloc[fold_valid], predictions
                ),
            })

    report = pd.DataFrame.from_records(records)
    summary = (report.groupby("feature_set", sort=False)
               .agg(mean_macro_f1=("macro_f1", "mean"),
                    std_macro_f1=("macro_f1", "std"),
                    mean_balanced_accuracy=("balanced_accuracy", "mean"))
               .reset_index())
    reference = report.loc[report.feature_set == "all", ["fold", "macro_f1"]]
    reference = reference.set_index("fold")["macro_f1"]
    summary["mean_delta_macro_f1_vs_all"] = summary.feature_set.map(
        lambda name: (report.loc[report.feature_set == name]
                      .set_index("fold")["macro_f1"] - reference).mean()
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORTS_DIR / "feature_ablation_cv_folds.csv", index=False)
    summary.to_csv(REPORTS_DIR / "feature_ablation_cv_summary.csv", index=False)
    print("Paired 5-fold CV on training portion only (holdout NOT evaluated):")
    print(summary.to_string(index=False, float_format=lambda val: f"{val:.4f}"))
    print(f"Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
