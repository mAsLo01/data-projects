"""Fit the pre-selected MLP/all_clean pipeline on all labeled data.

This is a deployment-style fit after experimental model selection.
It does not compute test metrics or suggest clinical applicability.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from cvd_risk.config import DEFAULT_DATA_PATH, MODELS_DIR, RANDOM_STATE
from cvd_risk.data import clean_dataset, load_dataset, prepare_features
from cvd_risk.modeling import pipeline_for_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output", type=Path, default=MODELS_DIR / "cvd_risk_all_clean_final.joblib"
    )
    args = parser.parse_args()

    data = clean_dataset(load_dataset(args.data))
    X, y = prepare_features(data, "all_clean")
    model = pipeline_for_model("mlp", random_state=RANDOM_STATE)
    model.fit(X, y)
    bundle = {
        "model": model,
        "feature_columns": list(X.columns),
        "target_column": y.name,
        "feature_set": "all_clean",
        "classes": [str(label) for label in model.classes_],
        "training_rows": len(X),
        "evaluation_note": (
            "Refit on all available labeled data; do not interpret training performance "
            "as an unbiased generalization estimate. Not for clinical use."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)
    print(f"Saved final MLP/all_clean model trained on {len(X)} rows: {args.output}")
    print("No holdout was evaluated. This is NOT a medically validated predictor.")


if __name__ == "__main__":
    main()
