from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import CV_SPLITS, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                selector(dtype_include="number"),
            ),
            (
                "categorical",
                categorical_pipeline,
                selector(dtype_exclude="number"),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_estimators(random_state: int = RANDOM_STATE) -> Mapping[str, object]:
    return {
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-3,
            max_iter=1500,
            early_stopping=False,
            random_state=random_state,
        ),
    }


def build_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def compare_models_cv(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
    n_splits: int = CV_SPLITS,
) -> pd.DataFrame:
    """Compare candidate models using stratified cross-validation on train data."""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "roc_auc_ovr_macro": "roc_auc_ovr",
    }

    rows: list[dict[str, float | str]] = []
    for name, estimator in candidate_estimators(random_state).items():
        pipeline = build_pipeline(estimator)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        row: dict[str, float | str] = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=1))
        rows.append(row)

    result = pd.DataFrame(rows)
    return result.sort_values("mean_f1_macro", ascending=False).reset_index(drop=True)


def select_best_model_name(cv_results: pd.DataFrame) -> str:
    """Pick the strongest non-dummy model using macro F1."""
    candidates = cv_results[cv_results["model"] != "dummy_baseline"]
    if candidates.empty:
        raise ValueError("No trainable model found in cross-validation results.")
    return str(candidates.iloc[0]["model"])


def pipeline_for_model(name: str, random_state: int = RANDOM_STATE) -> Pipeline:
    estimators = candidate_estimators(random_state)
    if name not in estimators:
        raise ValueError(f"Unknown model: {name}")
    return build_pipeline(estimators[name])


def tune_pipeline(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
    n_splits: int = CV_SPLITS,
) -> GridSearchCV:
    """Run a deliberately small grid search for the selected model."""
    param_grids = {
        "logistic_regression": {
            "model__C": [0.1, 1.0, 10.0],
        },
        "decision_tree": {
            "model__max_depth": [3, 5, 8, None],
            "model__min_samples_leaf": [1, 5, 10],
        },
        "random_forest": {
            "model__max_depth": [None, 8, 12],
            "model__min_samples_leaf": [1, 3, 5],
            "model__max_features": ["sqrt", 0.7],
        },
        "mlp": {
            "model__hidden_layer_sizes": [(32,), (64, 32)],
            "model__alpha": [1e-4, 1e-3, 1e-2],
        },
    }

    if name not in param_grids:
        raise ValueError(f"No tuning grid is defined for {name!r}.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    search = GridSearchCV(
        estimator=pipeline_for_model(name, random_state),
        param_grid=param_grids[name],
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return search
