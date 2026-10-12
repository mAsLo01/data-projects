"""Train and evaluate a NumPy neural network on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from data_generation import generate_normal_dataset
from neural_network import NeuralNetwork


RANDOM_STATE = 42
SAMPLES_PER_CLASS = 1_000
TEST_SIZE = 0.30
EPOCHS = 1_000
FIGURES_DIR = Path("figures")
RESULTS_PATH = Path("results.json")


def save_dataset_plot(
    class_0: np.ndarray,
    class_1: np.ndarray,
    output_path: Path,
) -> None:
    """Save a projection of the generated three-dimensional dataset."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(class_0[:, 0], class_0[:, 1], s=14, alpha=0.55, label="Class 0")
    ax.scatter(class_1[:, 0], class_1[:, 1], s=14, alpha=0.55, label="Class 1")
    ax.set(
        title="Generated classes: projection onto features 1 and 2",
        xlabel="Feature 1",
        ylabel="Feature 2",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_training_history(model: NeuralNetwork, output_path: Path) -> None:
    """Save loss and training-accuracy curves."""
    epochs = np.arange(1, len(model.loss_history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, model.loss_history, color="tab:blue")
    axes[0].set(
        title="Binary cross-entropy during training",
        xlabel="Epoch",
        ylabel="Loss",
    )
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, model.accuracy_history, color="tab:orange")
    axes[1].set(
        title="Training accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        ylim=(0.5, 1.0),
    )
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    roc_auc: float,
    output_path: Path,
) -> None:
    """Save the ROC curve computed on the held-out test set."""
    false_positive_rate, true_positive_rate, _ = roc_curve(
        targets,
        probabilities,
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Neural network (AUC={roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set(
        title="ROC curve on the test set",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    """Save the test confusion matrix."""
    matrix = confusion_matrix(targets, predictions)
    display = ConfusionMatrixDisplay(matrix, display_labels=["Class 0", "Class 1"])
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set_title("Confusion matrix on the test set")
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=160)
    plt.close(display.figure_)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    features, targets, class_0, class_1 = generate_normal_dataset(
        samples_per_class=SAMPLES_PER_CLASS,
        rng=rng,
    )
    features_train, features_test, targets_train, targets_test = train_test_split(
        features,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    model = NeuralNetwork(
        n_hidden=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    model.fit(
        features_train,
        targets_train,
        epochs=EPOCHS,
        verbose_every=100,
    )

    train_predictions = model.predict(features_train)
    test_probabilities = model.predict_proba(features_test)
    test_predictions = model.predict(features_test)

    metrics = {
        "train_accuracy": accuracy_score(targets_train, train_predictions),
        "test_accuracy": accuracy_score(targets_test, test_predictions),
        "test_precision": precision_score(targets_test, test_predictions),
        "test_recall": recall_score(targets_test, test_predictions),
        "test_f1": f1_score(targets_test, test_predictions),
        "test_roc_auc": roc_auc_score(targets_test, test_probabilities),
    }

    RESULTS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    save_dataset_plot(class_0, class_1, FIGURES_DIR / "dataset.png")
    save_training_history(model, FIGURES_DIR / "training_history.png")
    save_roc_curve(
        targets_test,
        test_probabilities,
        metrics["test_roc_auc"],
        FIGURES_DIR / "roc_curve.png",
    )
    save_confusion_matrix(
        targets_test,
        test_predictions,
        FIGURES_DIR / "confusion_matrix.png",
    )

    print("\nFinal metrics")
    print("-" * 34)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:>20}: {metric_value:.4f}")
    print(f"\nMetrics saved to: {RESULTS_PATH.resolve()}")
    print(f"Figures saved to: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
