import pandas as pd

from cvd_risk.modeling import pipeline_for_model


def test_pipeline_handles_numeric_and_categorical_features() -> None:
    X = pd.DataFrame(
        {
            "BMI": [20, 21, 22, 30, 31, 32, 26, 27, 28],
            "Cholesterol": [140, 150, 160, 240, 250, 260, 190, 200, 210],
            "Sex": ["F", "M", "F", "M", "F", "M", "F", "M", "F"],
        }
    )
    y = pd.Series(
        ["LOW", "LOW", "LOW", "HIGH", "HIGH", "HIGH", "INTERMEDIARY", "INTERMEDIARY", "INTERMEDIARY"]
    )

    pipeline = pipeline_for_model("logistic_regression", random_state=42)
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset(set(y))
