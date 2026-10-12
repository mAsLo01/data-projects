"""Train-only cross-validated permutation importance for the cleaned MLP.

Run from the project root, after installing it (`python -m pip install -e .`):

    python scripts/permutation_importance_cv.py
    python scripts/permutation_importance_cv.py --repeats 5

The same fixed 80/20 stratified split and five validation folds are used as
in scripts/diagnose_models_cv.py. The 20% holdout is never inspected.

Importance is the drop in macro F1 when ONE ORIGINAL INPUT COLUMN is shuffled
in a held-out validation fold. All imputation/encoding/scaling is fitted only
on that fold's training partition through the sklearn Pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a GUI window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from cvd_risk.config import (
    CV_SPLITS,
    DEFAULT_DATA_PATH,
    FIGURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from cvd_risk.data import clean_dataset, load_dataset, prepare_features
from cvd_risk.modeling import pipeline_for_model


def summarize_importances(fold_details: pd.DataFrame) -> pd.DataFrame:
    """Summarize across validation folds (the unit of replication).

    Averaging within each fold first prevents repeat permutations being
    mistakenly treated as independent cross-validation folds.
    """
    grouped = (
        fold_details.groupby("feature", sort=False)
        .agg(
            mean_f1_drop=("importance_mean", "mean"),
            std_f1_drop_between_folds=("importance_mean", "std"),
            min_fold_drop=("importance_mean", "min"),
            max_fold_drop=("importance_mean", "max"),
            positive_folds=("importance_mean", lambda s: int((s > 0).sum())),
            n_folds=("importance_mean", "size"),
        )
        .reset_index()
        .sort_values("mean_f1_drop", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def save_chart(summary: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    best = summary.head(top_n).iloc[::-1].copy()
    # Across-fold dispersion is descriptive, not a confidence interval.
    errors = best["std_f1_drop_between_folds"].fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, max(5, 0.42 * len(best) + 1)))
    ax.barh(
        best["feature"],
        best["mean_f1_drop"],
        xerr=errors,
        capsize=2,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Decrease in validation macro F1 after shuffling a feature")
    ax.set_title("MLP / all_clean: permutation importance (five CV folds)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(data: Path, repeats: int) -> None:
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")

    df = clean_dataset(load_dataset(data))
    indices = np.arange(len(df))
    train_idx, _holdout_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=df["CVD Risk Level"],
        random_state=RANDOM_STATE,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    X, y = prepare_features(train_df, feature_set="all_clean")
    cv = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    fold_details: list[dict[str, object]] = []
    repeated_details: list[dict[str, object]] = []
    print(f"Rows in train/CV partition: {len(X)}; input features: {X.shape[1]}")
    print(f"CV folds: {CV_SPLITS}; permutation repeats per fold: {repeats}")
    print("Holdout test excluded. Model: MLP, features: all_clean.\n")

    for fold, (fit_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        model = pipeline_for_model("mlp", random_state=RANDOM_STATE)
        X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
        model.fit(X_fit, y_fit)
        baseline_f1 = f1_score(y_valid, model.predict(X_valid), average="macro")

        importance = permutation_importance(
            model,
            X_valid,
            y_valid,
            n_repeats=repeats,
            random_state=RANDOM_STATE + fold,
            scoring="f1_macro",
            n_jobs=1,
        )
        for col, feature in enumerate(X.columns):
            values = importance.importances[col]
            fold_details.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "baseline_macro_f1": baseline_f1,
                    "importance_mean": float(values.mean()),
                    "importance_std_permutations": float(values.std(ddof=1))
                    if repeats > 1 else 0.0,
                }
            )
            for rep, delta in enumerate(values, start=1):
                repeated_details.append(
                    {
                        "fold": fold,
                        "feature": feature,
                        "repeat": rep,
                        "baseline_macro_f1": baseline_f1,
                        "f1_drop": float(delta),
                        "permuted_macro_f1": float(baseline_f1 - delta),
                    }
                )
        print(f"Fold {fold}/{CV_SPLITS}: baseline macro F1 = {baseline_f1:.4f}", flush=True)

    fold_frame = pd.DataFrame(fold_details)
    summary = summarize_importances(fold_frame)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(REPORTS_DIR / "mlp_permutation_importance_cv_folds.csv", index=False)
    pd.DataFrame(repeated_details).to_csv(
        REPORTS_DIR / "mlp_permutation_importance_cv_repeats.csv", index=False
    )
    summary.to_csv(REPORTS_DIR / "mlp_permutation_importance_cv_summary.csv", index=False)
    save_chart(summary, FIGURES_DIR / "mlp_permutation_importance_cv.png")

    print("\nPermutation importance, original input columns, ordered by mean macro-F1 drop:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved CSV reports to {REPORTS_DIR}")
    print(f"Saved chart to {FIGURES_DIR / 'mlp_permutation_importance_cv.png'}")
    print("Note: correlated/redundant features can have small individual importance.")
    print("These associations do not establish clinical effects or causality.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Permutations of each original input feature per validation fold (default: 3)",
    )
    args = parser.parse_args()
    run(args.data, args.repeats)


if __name__ == "__main__":
    main()
