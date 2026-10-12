from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvd_risk.config import DEFAULT_DATA_PATH, REPORTS_DIR, TARGET_COLUMN
from cvd_risk.data import clean_dataset, load_dataset
from cvd_risk.statistics import compare_feature_between_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected statistical tests.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--feature", default="BMI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_dataset(load_dataset(args.data))
    result = compare_feature_between_classes(df, args.feature, TARGET_COLUMN)

    path = REPORTS_DIR / f"statistical_test_{args.feature.lower().replace(' ', '_')}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
