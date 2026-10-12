from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from cvd_risk.config import RAW_DATA_DIR


DATASET_SLUG = "ahmeduzaki/cardiovascular-disease-risk-assessment-dataset"


def main() -> None:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError(
            "Kaggle CLI is not installed. Install project dependencies and configure "
            "your Kaggle credentials first."
        )

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            kaggle,
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-p",
            str(RAW_DATA_DIR),
            "--unzip",
        ],
        check=True,
    )
    print(f"Dataset downloaded to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
