from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DEFAULT_DATA_PATH = RAW_DATA_DIR / "CVD Dataset.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET_COLUMN = "CVD Risk Level"
LEAKAGE_COLUMNS = ["CVD Risk Score"]

FEATURE_SETS = {
    "two": ["BMI", "Total Cholesterol (mg/dL)"],
    "core": ["BMI", "Total Cholesterol (mg/dL)", "Systolic BP"],
}

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
