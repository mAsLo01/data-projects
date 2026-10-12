"""Generate, split and visualize synthetic classification datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from data_generation import (
    generate_nonlinear_dataset,
    generate_normal_dataset,
)


RANDOM_STATE = 42
SAMPLES_PER_CLASS = 1_000
TEST_SIZE = 0.3
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def save_feature_histograms(
    class_0: np.ndarray,
    class_1: np.ndarray,
    dataset_name: str,
) -> None:
    """Save one class-distribution histogram for every feature."""
    for feature_index in range(class_0.shape[1]):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            class_0[:, feature_index],
            bins="auto",
            alpha=0.7,
            label="Class 0",
        )
        ax.hist(
            class_1[:, feature_index],
            bins="auto",
            alpha=0.7,
            label="Class 1",
        )
        ax.set(
            title=f"{dataset_name}: feature {feature_index + 1}",
            xlabel="Feature value",
            ylabel="Frequency",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / f"{dataset_name}_feature_{feature_index + 1}_histogram.png",
            dpi=150,
        )
        plt.close(fig)


def save_scatter_plot(
    class_0: np.ndarray,
    class_1: np.ndarray,
    feature_indices: tuple[int, int],
    dataset_name: str,
) -> None:
    """Save a two-dimensional projection of the generated classes."""
    x_index, y_index = feature_indices

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        class_0[:, x_index],
        class_0[:, y_index],
        marker=".",
        alpha=0.7,
        label="Class 0",
    )
    ax.scatter(
        class_1[:, x_index],
        class_1[:, y_index],
        marker=".",
        alpha=0.7,
        label="Class 1",
    )
    ax.set(
        title=f"{dataset_name}: class distribution",
        xlabel=f"Feature {x_index + 1}",
        ylabel=f"Feature {y_index + 1}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{dataset_name}_scatter.png", dpi=150)
    plt.close(fig)


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible stratified training and test subsets."""
    return train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def print_split_summary(
    dataset_name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Print the shapes and class counts of a train/test split."""
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)

    print(f"\n{dataset_name}")
    print(f"Train features: {x_train.shape}; class counts: {train_counts.tolist()}")
    print(f"Test features:  {x_test.shape}; class counts: {test_counts.tolist()}")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    normal_means = np.array(
        [
            [0.0, 2.0, 3.0],
            [3.0, 5.0, 1.0],
        ]
    )
    normal_standard_deviations = np.array(
        [
            [2.0, 1.0, 2.0],
            [1.0, 2.0, 1.0],
        ]
    )

    normal_features, normal_labels, normal_class_0, normal_class_1 = (
        generate_normal_dataset(
            means=normal_means,
            standard_deviations=normal_standard_deviations,
            samples_per_class=SAMPLES_PER_CLASS,
            rng=np.random.default_rng(RANDOM_STATE),
        )
    )
    normal_split = split_dataset(normal_features, normal_labels)
    print_split_summary("Normal dataset", *normal_split)
    save_feature_histograms(normal_class_0, normal_class_1, "normal")
    save_scatter_plot(
        normal_class_0,
        normal_class_1,
        feature_indices=(0, 2),
        dataset_name="normal",
    )

    nonlinear_features, nonlinear_labels, nonlinear_class_0, nonlinear_class_1 = (
        generate_nonlinear_dataset(
            samples_per_class=SAMPLES_PER_CLASS,
            rng=np.random.default_rng(RANDOM_STATE + 1),
        )
    )
    nonlinear_split = split_dataset(nonlinear_features, nonlinear_labels)
    print_split_summary("Nonlinear dataset", *nonlinear_split)
    save_feature_histograms(nonlinear_class_0, nonlinear_class_1, "nonlinear")
    save_scatter_plot(
        nonlinear_class_0,
        nonlinear_class_1,
        feature_indices=(0, 1),
        dataset_name="nonlinear",
    )

    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
