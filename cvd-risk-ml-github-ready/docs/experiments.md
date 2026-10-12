# Experimental method and findings

## Pipeline and metrics

`clean_dataset` drops exact duplicates and rows with missing target, and standardizes target strings; `prepare_features` builds the selected feature matrix, excluding `CVD Risk Score`. The deterministic `all_clean` transformations are performed independently per row without learning dataset-level parameters. Every statistical imputer, standard scaler, and categorical encoder is fitted inside a `Pipeline` in the training portion of each CV fold.

The same 80/20 stratified train/holdout split (`random_state=42`) was used throughout. The 80% train subset was evaluated using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. **Macro F1** was the criterion for comparing model configurations; accuracy, balanced accuracy, weighted F1, and OvR macro ROC-AUC were auxiliary metrics. Standard deviations across five folds are descriptive variability, not confidence intervals.

## Model selection (train-side CV)

The original `all` inputs have 20 predictors. Logistic regression was top by mean CV macro F1 (0.521), although Random Forest had the highest macro ROC-AUC (0.788). The `all_clean` features have 18 predictors; **MLP** was top by mean CV macro F1 (0.552), compared to logistic regression (0.519) and Random Forest (0.479). Source files: `results/cv_results_all.csv`, `results/cv_results_all_clean.csv`.

The selected MLP is `MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=1500, random_state=42)` without early stopping. These parameters were not discovered through a full MLP hyperparameter search. The small hyperparameter search previously performed concerned logistic regression only; the chosen logistic parameters were `C=1.0`, `class_weight='balanced'`.

## Paired preprocessing experiment

On the same folds, MLP / `all` averaged macro F1 = 0.47269 (SD 0.02469), and MLP / `all_clean` averaged 0.55225 (SD 0.02166), for a **+0.07956** change. Each fold improved. Out-of-fold F1 for LOW was **0.18710 → 0.31858**, with recall **0.16477 → 0.30682**. This is still a limited minority-class performance.

Most improvement was explained by dropping the original high-cardinality categorical BP string, while retaining numerical BP information:

| Experiment | Mean CV macro F1 | Difference to `all` |
|---|---:|---:|
| Original all | 0.47269 | 0.00000 |
| Drop string BP only | 0.53624 | +0.06355 |
| Recover BP then drop string BP | 0.53835 | +0.06566 |
| Clean without BP operation | 0.48716 | +0.01447 |
| Clean without height operation | 0.52545 | +0.05276 |
| Clean without LDL operation | 0.54778 | +0.07508 |
| All clean | **0.55225** | **+0.07956** |

Because cleanup operations can interact, differences are descriptive; the experiment does not attribute unique causal shares to individual steps. Results: `results/preprocessing_ablation_*.csv`.

## Permutation importance

Each original feature column is permuted separately in the validation portion of each CV fold, three times by default. The difference from the unpermuted macro F1 is averaged within a fold and then across folds. Correlation between features may mask importance; ranking is model- and dataset-dependent. Do not interpret as medically causal risk factors.

Top six: Family History of CVD (0.06693), Estimated LDL (0.05472), Smoking Status (0.05461), Physical Activity Level (0.04778), Age (0.04097), Diabetes Status (0.03656). Each had positive average importance in five of five folds. Source: `results/mlp_permutation_importance_cv_summary.csv`.

## Limitations

The dataset's category labeling procedure cannot be independently validated from its provided columns. `CVD Risk Score` is itself a nearly exact derived formula of included predictors. BMI is inconsistent with other anthropometric variables for many rows, and some estimated LDL values are negative. These findings preclude clinical claims, even if a holdout F1 is numerically satisfactory. Reported validation is internal CV, not external cohort validation. The use of the original holdout earlier in model development may introduce adaptive selection bias; never describe it as independently untouched throughout the project.
