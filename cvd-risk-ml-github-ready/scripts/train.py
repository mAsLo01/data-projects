from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from cvd_risk.config import (
    DEFAULT_DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from cvd_risk.data import clean_dataset, load_dataset, prepare_features
from cvd_risk.evaluation import (
    evaluate_classifier,
    save_classification_report,
    save_confusion_matrix,
    save_metrics,
    save_multiclass_roc,
)
from cvd_risk.modeling import (
    compare_models_cv,
    pipeline_for_model,
    select_best_model_name,
    tune_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CVD risk classifiers.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--feature-set",
        choices=["two", "core", "all", "all_clean"],
        default="all",
        help="two: BMI+cholesterol; core: +systolic BP; all: all non-leaky features",
    )
    parser.add_argument(
        "--cv-only", action="store_true",
        help="Compute CV results only; never evaluate or overwrite the holdout model.",
    )
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a small grid search for the best cross-validation model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(args.data)
    df = clean_dataset(raw_df)
    X, y = prepare_features(df, feature_set=args.feature_set)

    print(f"Rows after duplicate removal / target cleanup: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print("Class distribution:")
    print(y.value_counts(normalize=True).sort_index().round(4))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    cv_results = compare_models_cv(
        X_train,
        y_train,
        random_state=args.random_state,
    )
    cv_path = REPORTS_DIR / f"cv_results_{args.feature_set}.csv"
    cv_results.to_csv(cv_path, index=False)
    print("\nCross-validation results:")
    print(cv_results.to_string(index=False))

    best_name = select_best_model_name(cv_results)
    print(f"\nSelected model by mean macro F1: {best_name}")

    if args.cv_only:
        print("\nCV-only mode: holdout predictions and model artifacts were not generated.")
        return

    best_params: dict[str, object] | None = None
    if args.tune:
        search = tune_pipeline(
            best_name,
            X_train,
            y_train,
            random_state=args.random_state,
        )
        model = search.best_estimator_
        best_params = search.best_params_
        print("Best tuning parameters:", best_params)
        print(f"Tuned CV macro F1: {search.best_score_:.4f}")
        pd.DataFrame(search.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].sort_values("rank_test_score").to_csv(
            REPORTS_DIR / f"tuning_results_{args.feature_set}.csv", index=False
        )
    else:
        model = pipeline_for_model(best_name, random_state=args.random_state)
        model.fit(X_train, y_train)

    metrics = evaluate_classifier(model, X_test, y_test)

    metrics_path = REPORTS_DIR / f"test_metrics_{args.feature_set}.json"
    report_path = REPORTS_DIR / f"classification_report_{args.feature_set}.csv"
    confusion_path = FIGURES_DIR / f"confusion_matrix_{args.feature_set}.png"
    roc_path = FIGURES_DIR / f"roc_{args.feature_set}.png"

    save_metrics(metrics, metrics_path)
    save_classification_report(metrics, report_path)
    save_confusion_matrix(model, X_test, y_test, confusion_path)
    save_multiclass_roc(model, X_test, y_test, roc_path)

    model_path = MODELS_DIR / f"cvd_risk_{args.feature_set}.joblib"
    bundle = {
        "model": model,
        "feature_columns": list(X.columns),
        "target_column": y.name,
        "feature_set": args.feature_set,
        "classes": [str(label) for label in model.classes_],
    }
    joblib.dump(bundle, model_path)

    metadata = {
        "dataset": str(args.data),
        "rows": int(len(df)),
        "feature_set": args.feature_set,
        "feature_columns": list(X.columns),
        "best_cv_model": best_name,
        "best_params": best_params,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "test_metrics": {
            key: value
            for key, value in metrics.items()
            if key != "classification_report"
        },
    }
    metadata_path = REPORTS_DIR / f"metadata_{args.feature_set}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nHoldout metrics:")
    for key, value in metadata["test_metrics"].items():
        print(f"  {key}: {value}")
    print(f"\nSaved model: {model_path}")
    print(f"Saved reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
