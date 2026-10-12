from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from cvd_risk.data import prepare_prediction_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CVD risk level for CSV rows.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    df = pd.read_csv(args.input)
    df.columns = [str(column).strip() for column in df.columns]

    X = prepare_prediction_features(
        df, bundle.get("feature_set", "all"), feature_columns
    )
    result = df.copy()
    result["predicted_cvd_risk_level"] = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        for index, class_name in enumerate(model.classes_):
            result[f"probability_{class_name}"] = probabilities[:, index]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
