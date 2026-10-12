"""Evaluation and visualization helpers for binary classifiers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_models(
    models: dict[str, ClassifierMixin],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit models and calculate train/test metrics without reusing test data."""
    rows: list[dict[str, float | str]] = []
    probabilities: dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)

        train_prediction = model.predict(x_train)
        test_prediction = model.predict(x_test)
        test_probability = model.predict_proba(x_test)[:, 1]
        true_negative, false_positive, false_negative, true_positive = (
            confusion_matrix(y_test, test_prediction, labels=[0, 1]).ravel()
        )

        probabilities[model_name] = test_probability
        rows.append(
            {
                "Model": model_name,
                "Train Accuracy": accuracy_score(y_train, train_prediction),
                "Test Accuracy": accuracy_score(y_test, test_prediction),
                "Test Precision": precision_score(
                    y_test,
                    test_prediction,
                    zero_division=0,
                ),
                "Test Sensitivity": recall_score(
                    y_test,
                    test_prediction,
                    zero_division=0,
                ),
                "Test Specificity": true_negative
                / (true_negative + false_positive),
                "Test F1": f1_score(y_test, test_prediction, zero_division=0),
                "Test ROC AUC": roc_auc_score(y_test, test_probability),
            }
        )

    return pd.DataFrame(rows), probabilities


def save_dataset_scatter(
    class_0: np.ndarray,
    class_1: np.ndarray,
    dataset_name: str,
    output_directory: Path,
) -> None:
    """Save a two-dimensional projection of a generated dataset."""
    y_feature_index = 1 if class_0.shape[1] == 2 else 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        class_0[:, 0],
        class_0[:, y_feature_index],
        marker=".",
        alpha=0.65,
        label="Class 0",
    )
    ax.scatter(
        class_1[:, 0],
        class_1[:, y_feature_index],
        marker=".",
        alpha=0.65,
        label="Class 1",
    )
    ax.set(
        title=f"{dataset_name}: generated classes",
        xlabel="Feature 1",
        ylabel=f"Feature {y_feature_index + 1}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / f"{dataset_name}_dataset.png", dpi=150)
    plt.close(fig)


def save_roc_curves(
    probabilities: dict[str, np.ndarray],
    y_test: np.ndarray,
    dataset_name: str,
    output_directory: Path,
) -> None:
    """Save ROC curves for every evaluated classifier."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name, probability in probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test,
            probability,
        )
        score = roc_auc_score(y_test, probability)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC={score:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set(
        title=f"{dataset_name}: ROC curves",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / f"{dataset_name}_roc_curves.png", dpi=150)
    plt.close(fig)


def save_accuracy_comparison(
    results: pd.DataFrame,
    dataset_name: str,
    output_directory: Path,
) -> None:
    """Compare training and test accuracy to expose overfitting."""
    positions = np.arange(len(results))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        positions - width / 2,
        results["Train Accuracy"],
        width,
        label="Train",
    )
    ax.bar(
        positions + width / 2,
        results["Test Accuracy"],
        width,
        label="Test",
    )
    ax.set(
        title=f"{dataset_name}: train and test accuracy",
        ylabel="Accuracy",
        xticks=positions,
        xticklabels=results["Model"],
        ylim=(0.75, 1.01),
    )
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_directory / f"{dataset_name}_accuracy_comparison.png",
        dpi=150,
    )
    plt.close(fig)
