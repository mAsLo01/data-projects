from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    auc,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def evaluate_classifier(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X_test)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        classes = model.classes_
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = None

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report
    return metrics


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def save_classification_report(metrics: dict[str, Any], path: str | Path) -> None:
    report = metrics["classification_report"]
    report_df = pd.DataFrame(report).T
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(path, index=True)


def save_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        model.predict(X_test),
        ax=ax,
    )
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_multiclass_roc(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    path: str | Path,
) -> None:
    if not hasattr(model, "predict_proba"):
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    probabilities = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)
    y_binary = label_binarize(y_test, classes=classes)

    if len(classes) == 2:
        y_binary = np.column_stack([1 - y_binary.ravel(), y_binary.ravel()])

    fig, ax = plt.subplots(figsize=(7, 6))
    for index, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_binary[:, index], probabilities[:, index])
        class_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={class_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-rest ROC curves")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
