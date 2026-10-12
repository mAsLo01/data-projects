"""Synthetic datasets used in the neural-network experiment."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_normal_dataset(
    samples_per_class: int,
    rng: np.random.Generator,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Generate two three-dimensional normally distributed classes."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    means = np.array([[0.0, 2.0, 3.0], [4.0, 6.0, 7.0]])
    standard_deviations = np.full_like(means, 2.0)

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
    targets = np.concatenate(
        (
            np.zeros(samples_per_class, dtype=np.int64),
            np.ones(samples_per_class, dtype=np.int64),
        )
    )
    return features, targets, class_0, class_1
