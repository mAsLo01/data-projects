"""Synthetic datasets used in the tree-based classification experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FeatureArray = NDArray[np.float64]
LabelArray = NDArray[np.int64]


def generate_normal_dataset(
    samples_per_class: int,
    rng: np.random.Generator,
) -> tuple[FeatureArray, LabelArray, FeatureArray, FeatureArray]:
    """Generate two partially overlapping normally distributed classes."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    means = np.array(
        [
            [0.0, 2.0, 3.0],
            [4.0, 6.0, 7.0],
        ]
    )
    standard_deviations = np.array(
        [
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
        ]
    )

    class_0 = rng.normal(
        loc=means[0],
        scale=standard_deviations[0],
        size=(samples_per_class, means.shape[1]),
    )
    class_1 = rng.normal(
        loc=means[1],
        scale=standard_deviations[1],
        size=(samples_per_class, means.shape[1]),
    )

    features = np.vstack((class_0, class_1))
    labels = np.concatenate(
        (
            np.zeros(samples_per_class, dtype=int),
            np.ones(samples_per_class, dtype=int),
        )
    )
    return features, labels, class_0, class_1


def generate_nonlinear_dataset(
    samples_per_class: int,
    rng: np.random.Generator,
    noise_standard_deviation: float = 0.5,
) -> tuple[FeatureArray, LabelArray, FeatureArray, FeatureArray]:
    """Generate the nonlinear two-class dataset from variant 9."""
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2")
    if noise_standard_deviation < 0:
        raise ValueError("noise_standard_deviation cannot be negative")

    curve_parameter = np.linspace(0, 2 * np.pi, samples_per_class)
    line_parameter = np.linspace(1, 1.5 * np.pi, samples_per_class)

    class_0 = np.column_stack(
        (
            curve_parameter,
            4 * np.sin(curve_parameter + 1),
        )
    )
    class_1 = np.column_stack(
        (
            line_parameter,
            -3.5 * line_parameter + 17,
        )
    )

    class_0 += rng.normal(
        loc=0.0,
        scale=noise_standard_deviation,
        size=class_0.shape,
    )
    class_1 += rng.normal(
        loc=0.0,
        scale=noise_standard_deviation,
        size=class_1.shape,
    )

    features = np.vstack((class_0, class_1))
    labels = np.concatenate(
        (
            np.zeros(samples_per_class, dtype=int),
            np.ones(samples_per_class, dtype=int),
        )
    )
    return features, labels, class_0, class_1
