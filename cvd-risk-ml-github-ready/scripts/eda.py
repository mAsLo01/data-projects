from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cvd_risk.config import DEFAULT_DATA_PATH, FIGURES_DIR, REPORTS_DIR, TARGET_COLUMN
from cvd_risk.data import clean_dataset, load_dataset
from cvd_risk.statistics import spearman_correlations


PLOT_FEATURES = [
    "BMI",
    "Total Cholesterol (mg/dL)",
    "Systolic BP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA artifacts.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(args.data)
    df = clean_dataset(raw)

    missing = df.isna().sum().sort_values(ascending=False)
    missing.to_csv(REPORTS_DIR / "missing_values.csv", header=["missing_count"])

    df.describe(include="all").T.to_csv(REPORTS_DIR / "descriptive_statistics.csv")

    class_distribution = (
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .rename_axis(TARGET_COLUMN)
        .to_frame("count")
    )
    class_distribution["share"] = class_distribution["count"] / len(df)
    class_distribution.to_csv(REPORTS_DIR / "class_distribution.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    class_distribution["count"].plot(kind="bar", ax=ax)
    ax.set_title("CVD risk class distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel(TARGET_COLUMN)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)

    for feature in PLOT_FEATURES:
        if feature not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x=feature, hue=TARGET_COLUMN, kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature} by risk level")
        fig.tight_layout()
        filename = feature.lower().replace(" ", "_").replace("/", "_")
        fig.savefig(FIGURES_DIR / f"distribution_{filename}.png", dpi=160)
        plt.close(fig)

    correlations = spearman_correlations(df)
    correlations.to_csv(REPORTS_DIR / "spearman_correlations.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlations, ax=ax, center=0)
    ax.set_title("Spearman correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "spearman_correlation_matrix.png", dpi=160)
    plt.close(fig)

    print(f"EDA reports saved to: {REPORTS_DIR}")
    print(f"EDA figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
