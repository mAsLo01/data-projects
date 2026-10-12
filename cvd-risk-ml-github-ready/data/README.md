# Data

The raw dataset is intentionally not stored in the repository.

Dataset source:

`ahmeduzaki/cardiovascular-disease-risk-assessment-dataset` on Kaggle.

Expected file location:

```text
data/raw/CVD Dataset.csv
```

You can either download it manually or configure the Kaggle CLI and run:

```bash
python scripts/download_data.py
```

Do not commit `kaggle.json` or any API token to GitHub.
