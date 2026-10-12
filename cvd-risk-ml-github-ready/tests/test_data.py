import pandas as pd

from cvd_risk.data import (
    clean_dataset, prepare_all_clean_features, prepare_features, prepare_prediction_features
)


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BMI": [20.0, 25.0, 30.0, 30.0],
            "Total Cholesterol (mg/dL)": [150.0, 200.0, 250.0, 250.0],
            "Systolic BP": [110, 125, 145, 145],
            "Sex": ["F", "M", "F", "F"],
            "CVD Risk Score": [10.0, 40.0, 80.0, 80.0],
            "CVD Risk Level": ["LOW", "INTERMEDIARY", "HIGH", "HIGH"],
        }
    )


def test_clean_dataset_removes_duplicates() -> None:
    cleaned = clean_dataset(sample_dataframe())
    assert len(cleaned) == 3


def test_all_features_remove_target_and_leakage_score() -> None:
    cleaned = clean_dataset(sample_dataframe())
    X, y = prepare_features(cleaned, feature_set="all")

    assert "CVD Risk Level" not in X.columns
    assert "CVD Risk Score" not in X.columns
    assert len(X) == len(y)


def test_core_feature_set() -> None:
    cleaned = clean_dataset(sample_dataframe())
    X, _ = prepare_features(cleaned, feature_set="core")
    assert list(X.columns) == ["BMI", "Total Cholesterol (mg/dL)", "Systolic BP"]


def test_all_clean_removes_redundancy_and_recovers_missing_values() -> None:
    source = pd.DataFrame(
        {
            "CVD Risk Level": ["HIGH", "LOW"],
            "CVD Risk Score": [18.0, 11.0],
            "Blood Pressure (mmHg)": ["140/90", "112/68"],
            "Systolic BP": [float("nan"), 112.0],
            "Diastolic BP": [90.0, float("nan")],
            "Height (m)": [float("nan"), 1.67],
            "Height (cm)": [170.0, 167.0],
            "Estimated LDL (mg/dL)": [-4.0, 45.0],
            "BMI": [22.0, 24.0],
        }
    )
    X = prepare_all_clean_features(source)
    assert "Blood Pressure (mmHg)" not in X
    assert "Height (cm)" not in X
    assert "CVD Risk Score" not in X
    assert "CVD Risk Level" not in X
    assert X.loc[0, "Systolic BP"] == 140.0
    assert X.loc[1, "Diastolic BP"] == 68.0
    assert X.loc[0, "Height (m)"] == 1.7
    assert pd.isna(X.loc[0, "Estimated LDL (mg/dL)"])
    assert X.loc[0, "BMI"] == 22.0  # never overwrite inconsistent BMI
    assert source.loc[0, "Estimated LDL (mg/dL)"] == -4.0  # source unchanged


def test_prediction_uses_same_cleaning() -> None:
    X = prepare_all_clean_features(
        pd.DataFrame({
            "Blood Pressure (mmHg)": ["150/95"],
            "Systolic BP": [float("nan")],
            "Diastolic BP": [float("nan")],
            "Height (m)": [float("nan")],
            "Height (cm)": [180.0],
        })
    )
    features = prepare_prediction_features(
        pd.DataFrame({
            "Blood Pressure (mmHg)": ["150/95"],
            "Systolic BP": [float("nan")],
            "Diastolic BP": [float("nan")],
            "Height (cm)": [180.0],
        }),
        "all_clean", list(X.columns),
    )
    assert features.loc[0, "Systolic BP"] == 150.0
    assert features.loc[0, "Height (m)"] == 1.8
