"""Compare decision-tree and random-forest classifiers on synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_generation import generate_nonlinear_dataset, generate_normal_dataset
from evaluation import (
    evaluate_models,
    save_accuracy_comparison,
    save_dataset_scatter,
    save_roc_curves,
)


RANDOM_STATE = 42
SAMPLES_PER_CLASS = 1_000
TEST_SIZE = 0.3
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"


def tune_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[GridSearchCV, GridSearchCV]:
    """Tune both classifiers using training-set cross-validation only."""
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    tree_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [1, 5, 10, 20],
        },
        scoring="roc_auc",
        cv=cross_validation,
        n_jobs=-1,
    )
    forest_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_leaf": [1, 5],
        },
        scoring="roc_auc",
        cv=cross_validation,
        n_jobs=-1,
    )

    tree_search.fit(x_train, y_train)
    forest_search.fit(x_train, y_train)
    return tree_search, forest_search


def run_experiment(
    dataset_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    class_0: np.ndarray,
    class_1: np.ndarray,
) -> pd.DataFrame:
    """Tune, evaluate and visualize classifiers for one dataset."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    tree_search, forest_search = tune_models(x_train, y_train)
    models = {
        "Decision Tree (default)": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Decision Tree (tuned)": tree_search.best_estimator_,
        "Random Forest (default)": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Random Forest (tuned)": forest_search.best_estimator_,
    }

    results, probabilities = evaluate_models(
        models=models,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )
    results.insert(0, "Dataset", dataset_name)

    print(f"\n{'=' * 88}")
    print(dataset_name.upper())
    print(f"{'=' * 88}")
    print(
        "Best decision-tree parameters:",
        tree_search.best_params_,
        f"(CV ROC AUC={tree_search.best_score_:.4f})",
    )
    print(
        "Best random-forest parameters:",
        forest_search.best_params_,
        f"(CV ROC AUC={forest_search.best_score_:.4f})",
    )
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    save_dataset_scatter(class_0, class_1, dataset_name, FIGURES_DIR)
    save_roc_curves(probabilities, y_test, dataset_name, FIGURES_DIR)
    save_accuracy_comparison(results, dataset_name, FIGURES_DIR)

    return results


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    normal_data = generate_normal_dataset(
        samples_per_class=SAMPLES_PER_CLASS,
        rng=np.random.default_rng(RANDOM_STATE),
    )
    nonlinear_data = generate_nonlinear_dataset(
        samples_per_class=SAMPLES_PER_CLASS,
        rng=np.random.default_rng(RANDOM_STATE + 1),
    )

    results = pd.concat(
        [
            run_experiment("normal", *normal_data),
            run_experiment("nonlinear", *nonlinear_data),
        ],
        ignore_index=True,
    )
    results.to_csv(RESULTS_PATH, index=False)

    print(f"\nMetrics saved to: {RESULTS_PATH}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
