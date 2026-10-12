"""Utilities for generating synthetic binary-classification datasets."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FeatureArray = NDArray[np.float64]
LabelArray = NDArray[np.int64]


def generate_normal_dataset(
    means: FeatureArray,
    standard_deviations: FeatureArray,
    samples_per_class: int,
    rng: np.random.Generator,
) -> tuple[FeatureArray, LabelArray, FeatureArray, FeatureArray]:
    """Generate two classes with independent normally distributed features.

    Parameters
    ----------
    means:
        Array with shape ``(2, n_features)`` containing class means.
    standard_deviations:
        Array with shape ``(2, n_features)`` containing positive feature
        standard deviations for both classes.
    samples_per_class:
        Number of generated objects in each class.
    rng:
        NumPy random generator used to make the experiment reproducible.
    """
    means = np.asarray(means, dtype=float)
    standard_deviations = np.asarray(standard_deviations, dtype=float)

    if means.ndim != 2 or means.shape[0] != 2:
        raise ValueError("means must have shape (2, n_features)")
    if standard_deviations.shape != means.shape:
        raise ValueError("standard_deviations must have the same shape as means")
    if np.any(standard_deviations <= 0):
        raise ValueError("all standard deviations must be positive")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

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
    """Generate the two-dimensional nonlinear dataset from variant 9."""
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2")
    if noise_standard_deviation < 0:
        raise ValueError("noise_standard_deviation cannot be negative")

    curve_parameter = np.linspace(0, 2 * np.pi, samples_per_class)
    line_parameter = np.linspace(1, 1.5 * np.pi, samples_per_class)

    class_0 = np.column_stack(
        (
            curve_parameter,
            2 * curve_parameter * np.sin(curve_parameter),
        )
    )
    class_1 = np.column_stack(
        (
            line_parameter,
            -5 * line_parameter + 25,
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
