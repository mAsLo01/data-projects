# Dataset audit and interpretation

Run `python scripts/audit_dataset.py` to reproduce this audit using `data/raw/CVD Dataset.csv`.

- 1,529 rows, 22 columns, 0 exact duplicates.
- Class counts: HIGH 728, INTERMEDIARY 581, LOW 220.
- 1,146 rows have complete inputs for the reconstructed numerical risk score.
- `CVD Risk Score` can be almost exactly reconstructed as:

  ```text
  0.2 + 2 * (Diabetes Status == Y) + BMI/5 + Systolic BP/20
      + Total Cholesterol/75 + Estimated LDL/150 + HDL/150
  ```

  Maximum absolute difference: 0.0004 among these complete records. The small remainder is consistent with numerical rounding. No independence of the numerical score from these predictors is established.
- 449 / 1,328 rows with complete anthropometric measurements differ from weight/(height in m)^2 by >0.5 units of BMI. We **did not** guess which measurement is correct.
- 13 values of `Estimated LDL (mg/dL)` are negative; in `all_clean` only these values are converted to missing, with imputation fitted within folds.
- 1,254 unique string BP entries. Where both string and numeric BP components are present, they match. `all_clean` recovers the missing numeric components row-wise and removes the string category.
- Risk level labels do not correspond to disjoint numerical risk-score ranges. The method of assigning the three categories is not independently established from the released file.

The source listing is [Kaggle](https://www.kaggle.com/datasets/ahmeduzaki/cardiovascular-disease-risk-assessment-dataset). Provenance/clinical claims in the source metadata have not been independently verified. This project predicts supplied labels, not independently observed future patient outcomes. It is not for clinical use.
