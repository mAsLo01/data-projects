from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, levene, mannwhitneyu, shapiro
from statsmodels.stats.multitest import multipletests


def spearman_correlations(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    return numeric.corr(method="spearman")


def compare_feature_between_classes(
    df: pd.DataFrame,
    feature: str,
    target: str,
) -> dict[str, object]:
    """Run robust group comparison for a numeric feature.

    The original coursework used Tukey HSD after Kruskal-Wallis. This version
    keeps the non-parametric logic consistent: if Kruskal-Wallis rejects the
    global null, pairwise Mann-Whitney tests are followed by Holm correction.
    """
    grouped = {
        label: values[feature].dropna().to_numpy()
        for label, values in df.groupby(target)
    }
    grouped = {label: values for label, values in grouped.items() if len(values) >= 3}

    if len(grouped) < 2:
        raise ValueError("At least two non-empty classes are required.")

    normality = {
        label: float(shapiro(values).pvalue)
        for label, values in grouped.items()
    }
    levene_p = float(levene(*grouped.values()).pvalue)
    kruskal_result = kruskal(*grouped.values())

    pairwise_rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    pairs = list(combinations(grouped.keys(), 2))

    for left, right in pairs:
        result = mannwhitneyu(grouped[left], grouped[right], alternative="two-sided")
        raw_p_values.append(float(result.pvalue))
        pairwise_rows.append(
            {
                "class_a": left,
                "class_b": right,
                "u_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )

    if raw_p_values:
        rejected, corrected, _, _ = multipletests(raw_p_values, method="holm")
        for row, p_adj, reject in zip(pairwise_rows, corrected, rejected):
            row["p_value_holm"] = float(p_adj)
            row["reject_h0"] = bool(reject)

    return {
        "feature": feature,
        "normality_p_values": normality,
        "levene_p_value": levene_p,
        "kruskal_statistic": float(kruskal_result.statistic),
        "kruskal_p_value": float(kruskal_result.pvalue),
        "pairwise_mann_whitney_holm": pairwise_rows,
    }
