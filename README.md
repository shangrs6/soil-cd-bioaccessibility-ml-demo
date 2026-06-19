# Interpretable Machine Learning for Soil Cadmium Bioaccessibility Prediction

This repository contains a preliminary data-driven demo for predicting gastric-phase cadmium (Cd) bioaccessibility in Chinese soils and extending the prediction results toward bioaccessibility-informed risk interpretation.

## Background

Soil Cd pollution risk is often evaluated using total Cd concentration. However, total Cd alone may not fully represent exposure-related risk because Cd bioaccessibility is affected by soil physicochemical properties, source types, and in vitro digestion methods.

This project explores whether machine learning models can improve Cd bioaccessibility prediction beyond a total-Cd-only baseline, and further investigates whether hyperparameter tuning can improve the best-performing model.

## Research Questions

1. Can total Cd concentration alone predict gastric-phase Cd bioaccessibility?
2. Do soil physicochemical properties, digestion methods, and soil source types improve prediction performance?
3. Which variables are most important for predicting Cd bioaccessibility?
4. Can hyperparameter-tuned Random Forest and XGBoost models outperform default-parameter models?
5. Can predicted bioaccessibility be extended toward bioaccessibility-informed Cd risk interpretation?

## Hypotheses

**H1: Soil physicochemical properties provide additional explanatory power for Cd bioaccessibility beyond total Cd concentration.**  
This hypothesis is tested by comparing total-Cd-only baseline models with full-feature models that include soil properties such as pH, SOM, texture, Fe, and log-transformed total Cd.

**H2: Soil source type and in vitro digestion method contribute to variability in measured Cd bioaccessibility.**  
This hypothesis is tested using group-level summary statistics and Kruskal-Wallis tests for BA_GP across Method and Type groups. The corresponding results are saved in `results/h2_method_type_kruskal_results.csv`.

**H3: Machine learning models integrating soil properties and methodological variables outperform total-Cd-only baseline models in predicting Cd bioaccessibility.**  
This hypothesis is evaluated through cross-validated model comparison among Ridge Regression, Random Forest, Gradient Boosting, and XGBoost models. The comparison between total-Cd-only and full-feature models is saved in `results/model_comparison_baseline_vs_full.csv`. A systematic feature group ablation study (Total-Cd-only → Soil-properties-only → Cd + Soil → Cd + Soil + Method → Full-feature) is saved in `results/feature_group_ablation_results.csv`.

**H4: Predicted Cd bioaccessibility can be extended toward bioaccessibility-informed pollution risk interpretation.**  
This hypothesis is explored by calculating gastric-phase bioaccessible Cd as `T_Cd × BA_GP / 100` and summarizing relative risk groups by soil source type. The risk-extension outputs are saved in `results/bioaccessible_cd_risk_data.csv` and `results/bioaccessible_cd_risk_by_type.csv`.

## Dataset

This project uses a publicly available dataset from Zenodo:

**Dataset:** Compiled dataset on soil cadmium bioaccessibility, related properties, and concentration across China  
**Authors:** Jianghao Cao and Youya Zhou  
**DOI:** 10.5281/zenodo.18171038  
**Source:** https://zenodo.org/records/18171038  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

The original dataset is provided as an Excel file named `Data-Cd.xlsx`. According to the Zenodo record, it contains multiple worksheets, including:

- `Original_Data`: original literature-extracted soil Cd bioaccessibility data;
- `Processed_Data`: cleaned and imputed data for modeling;
- `Soil_Properties`: descriptive statistics and spatial distribution characteristics of soil properties across major geographical regions of China;
- `Cd_Content_Data`: provincial soil Cd concentration data and statistical summaries;
- `Cd_Source`: references and sources corresponding to provincial Cd concentration data.

In this demo, `Processed_Data` is used as the main dataset for gastric-phase Cd bioaccessibility (`BA_GP`) prediction, while `Cd_Source` is used as a supporting dataset for describing total Cd contamination patterns across different soil source types.

The raw Excel file is not included in this repository. Users can download it from the Zenodo record above if they want to reproduce the full preprocessing workflow. Processed datasets used in this demo are provided in `data/processed_data/`.

## Workflow

1. Data cleaning and feature engineering
2. Exploratory data analysis
3. Feature group ablation study (five feature group configurations × four models)
4. Random Forest hyperparameter tuning via RandomizedSearchCV
5. XGBoost hyperparameter tuning via RandomizedSearchCV
6. Evaluation of tuned Random Forest and XGBoost models
7. Final model interpretation: permutation importance and partial dependence plots
8. Bioaccessibility-informed risk extension

## Configuration

All feature definitions, feature group configurations, and cross-validation settings are centralized in `config.yaml`:

- `data.processed_file`: path to the processed dataset
- `target`: prediction target (`BA_GP`)
- `features.numeric`: numeric predictors (`log_T_Cd`, `pH`, `SOM`, `Clay`, `Silt`, `Sand`, `Fe`)
- `features.categorical`: categorical predictors (`Method`, `Type`)
- `feature_groups`: five feature group configurations used in the ablation study
- `cv`: cross-validation settings (5-fold, 10 repeats, random seed 42)

## Code Description

This repository contains both Python scripts and Jupyter notebooks. The scripts are used for reproducible data preprocessing, exploratory analysis, model tuning, and model interpretation, while the notebooks are used for model development, comparison, and risk-extension analysis.

### Scripts

- `scripts/01_data_processed.py`  
  Cleans the raw Excel dataset, standardizes variable names, converts numerical columns, constructs log-transformed variables such as `log_T_Cd`, and exports processed datasets to `data/processed_data/`.

- `scripts/02_eda.py`  
  Performs exploratory data analysis for the processed bioaccessibility dataset and the Cd_source dataset. It generates distribution plots, Method/Type group comparisons, Cd source-type boxplots, correlation matrices, and scatter plots. Output figures are saved to `figures/`.

- `scripts/03_feature_group_ablation.py`  
  Runs a systematic feature group ablation study across five feature group configurations (Total-Cd-only, Soil-properties-only, Cd + Soil, Cd + Soil + Method, Full-feature) and four models (Ridge Regression, Random Forest, Gradient Boosting, XGBoost). Uses 5-fold cross-validation with 10 repeats. Feature groups and CV settings are read from `config.yaml`. Results are saved to `results/feature_group_ablation_results.csv`.

- `scripts/04_rf_tuning.py`  
  Tunes Random Forest hyperparameters using `RandomizedSearchCV` with 20 iterations and 5-fold cross-validation repeated 5 times. Searches over `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features`. Saves tuning results to `results/rf_tuning_results.csv`, best parameters to `results/best_rf_params.json`, and a summary (including best MAE) to `results/rf_tuning_summary.json`.

- `scripts/05_evaluate_tuned_rf.py`  
  Evaluates the tuned Random Forest (loaded from `results/best_rf_params.json`) using 5-fold cross-validation with 10 repeats. Reports MAE, RMSE, and R². Results are saved to `results/tuned_rf_evaluation.csv`.

- `scripts/06_xgb_tuning.py`  
  Tunes XGBoost hyperparameters using `RandomizedSearchCV` with 30 iterations and 5-fold cross-validation repeated 5 times. Searches over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, and `reg_lambda`. Saves tuning results to `results/xgb_tuning_results.csv`, best parameters to `results/best_xgb_params.json`, and a summary to `results/xgb_tuning_summary.json`.

- `scripts/07_evaluate_tuned_xgb.py`  
  Evaluates the tuned XGBoost model (loaded from `results/best_xgb_params.json`) using 5-fold cross-validation with 10 repeats. Reports MAE, RMSE, and R². Results are saved to `results/tuned_xgb_evaluation.csv`.

- `scripts/08_final_model_interpretation.py`  
  Fits the tuned Random Forest on an 80/20 train-test split and computes permutation importance (30 repeats) on the test set. Generates partial dependence plots for `log_T_Cd`, `Fe`, `SOM`, and `pH`. Saves permutation importance to `results/final_tuned_rf_permutation_importance.csv` and figures to `figures/`.

### Notebooks

- `notebooks/09_model_ba_gp.ipynb`  
  Builds and evaluates machine learning models for predicting gastric-phase Cd bioaccessibility (`BA_GP`).  
  This notebook includes:
  - total-Cd-only baseline models;
  - full-feature model comparison (Ridge Regression, Random Forest, Gradient Boosting, XGBoost);
  - 5-fold cross-validation;
  - baseline vs full-feature comparison;
  - Kruskal-Wallis tests for Method and Type effects;
  - permutation importance;
  - partial dependence plots;
  - 80/20 train-test split for observed vs predicted visualization.

- `notebooks/10_risk_extension.ipynb`  
  Extends the BA_GP prediction results toward bioaccessibility-informed Cd risk interpretation.  
  This notebook calculates gastric-phase bioaccessible Cd using:

  `Bioaccessible Cd_GP = T_Cd × BA_GP / 100`

  It then compares total Cd and bioaccessible Cd, constructs Low / Medium / High relative risk groups, and summarizes risk patterns by soil source type.

### Results

- `results/feature_group_ablation_results.csv`  
  Cross-validated performance (MAE, RMSE, R²) across five feature group configurations and four models. Shows that adding soil properties and methodological variables progressively improves prediction performance.

- `results/model_comparison_baseline_vs_full.csv`  
  Model comparison table between total-Cd-only baseline models and full-feature models.

- `results/model_performance_ba_gp.csv`  
  Cross-validated performance of full-feature BA_GP prediction models.

- `results/rf_tuning_results.csv`  
  Full RandomizedSearchCV result table for Random Forest tuning (20 candidate parameter sets).

- `results/best_rf_params.json`  
  Best Random Forest hyperparameters: `n_estimators=300`, `max_depth=20`, `min_samples_leaf=1`, `max_features="sqrt"`.

- `results/rf_tuning_summary.json`  
  Summary of Random Forest tuning including best MAE (≈15.84).

- `results/tuned_rf_evaluation.csv`  
  Cross-validated evaluation of the tuned Random Forest: MAE ≈ 15.81, RMSE ≈ 19.97, R² ≈ 0.462 (5-fold × 10 repeats, n=190).

- `results/xgb_tuning_results.csv`  
  Full RandomizedSearchCV result table for XGBoost tuning (30 candidate parameter sets).

- `results/best_xgb_params.json`  
  Best XGBoost hyperparameters: `n_estimators=100`, `max_depth=3`, `learning_rate=0.1`, `subsample=0.7`, `colsample_bytree=1.0`, `reg_alpha=1`, `reg_lambda=1`.

- `results/xgb_tuning_summary.json`  
  Summary of XGBoost tuning including best MAE (≈15.87).

- `results/tuned_xgb_evaluation.csv`  
  Cross-validated evaluation of the tuned XGBoost: MAE ≈ 16.08, RMSE ≈ 20.45, R² ≈ 0.433 (5-fold × 10 repeats, n=190).

- `results/final_tuned_rf_permutation_importance.csv`  
  Permutation importance of the tuned Random Forest on the hold-out test set (30 repeats). Top predictors: Fe, SOM, log_T_Cd, Clay, Method.

- `results/feature_importance_ba_gp.csv`  
  Permutation importance results from the notebook-based model.

- `results/h2_method_type_kruskal_results.csv`  
  Kruskal-Wallis test results for BA_GP differences across Method and Type groups.

- `results/bioaccessible_cd_risk_data.csv`  
  Sample-level risk-extension output, including observed and predicted bioaccessible Cd.

- `results/bioaccessible_cd_risk_by_type.csv`  
  Summary of bioaccessibility-informed relative risk groups by soil source type.

## How to Run

This project can be reproduced from the processed datasets provided in `data/processed_data/`.  
The raw Excel file is not included in this repository due to data ownership and citation considerations, so users can start directly from the processed data and model notebooks.

### 1. Clone the repository

```bash
git clone https://github.com/shangrs6/soil-cd-bioaccessibility-ml-demo.git
cd soil-cd-bioaccessibility-ml-demo
```

### 2. Install dependencies

It is recommended to create a virtual environment before installing the required packages.

```bash
pip install -r requirements.txt
```

The main packages used in this project include:

```text
pandas
numpy
matplotlib
scikit-learn
scipy
openpyxl
xgboost
pyyaml
jupyter
```

### 3. Recommended running order

The full workflow runs in the following order:

```text
scripts/01_data_processed.py       # data cleaning (requires raw Excel)
scripts/02_eda.py                  # exploratory data analysis
scripts/03_feature_group_ablation.py   # feature group ablation study
scripts/04_rf_tuning.py            # Random Forest hyperparameter tuning
scripts/05_evaluate_tuned_rf.py    # evaluate tuned Random Forest
scripts/06_xgb_tuning.py           # XGBoost hyperparameter tuning
scripts/07_evaluate_tuned_xgb.py   # evaluate tuned XGBoost
scripts/08_final_model_interpretation.py  # permutation importance and PDP
notebooks/09_model_ba_gp.ipynb     # modeling and comparison notebook
notebooks/10_risk_extension.ipynb  # risk extension notebook
```

Because the raw Excel file is not included in this public repository, `scripts/01_data_processed.py` is provided mainly to document the original data-cleaning workflow. Users who clone this repository can start directly from the processed datasets:

```text
data/processed_data/processed_bioaccessibility.csv
data/processed_data/cd_source_cleaned.csv
data/processed_data/cd_source_analysis.csv
```

Therefore, the recommended reproducible starting point is `scripts/03_feature_group_ablation.py` (or `notebooks/09_model_ba_gp.ipynb` if you prefer the notebook workflow).

### 4. Run the feature group ablation study

```bash
python scripts/03_feature_group_ablation.py
```

This script evaluates five feature group configurations × four models using cross-validation. Results are saved to `results/feature_group_ablation_results.csv`.

### 5. Run hyperparameter tuning

```bash
python scripts/04_rf_tuning.py     # Random Forest tuning
python scripts/06_xgb_tuning.py    # XGBoost tuning
```

These scripts run `RandomizedSearchCV` and save the best parameters and tuning summaries to `results/`.

### 6. Evaluate tuned models

```bash
python scripts/05_evaluate_tuned_rf.py
python scripts/07_evaluate_tuned_xgb.py
```

These scripts load the best parameters from `results/best_rf_params.json` and `results/best_xgb_params.json` and evaluate the tuned models with repeated cross-validation.

### 7. Run final model interpretation

```bash
python scripts/08_final_model_interpretation.py
```

This script fits the tuned Random Forest on an 80/20 train-test split, computes permutation importance, and generates partial dependence plots for `log_T_Cd`, `Fe`, `SOM`, and `pH`. Outputs are saved to `results/` and `figures/`.

### 8. Run the modeling notebook

Open and run:

```text
notebooks/09_model_ba_gp.ipynb
```

This notebook performs:

- total-Cd-only baseline modeling;
- full-feature model comparison;
- 5-fold cross-validation;
- Ridge Regression, Random Forest, Gradient Boosting, and XGBoost modeling;
- baseline vs full-feature comparison;
- Kruskal-Wallis tests for Method and Type effects;
- permutation importance analysis;
- partial dependence plot analysis.

Main outputs are saved to:

```text
results/
figures/
```

### 9. Run the risk-extension notebook

After running the modeling notebook, open and run:

```text
notebooks/10_risk_extension.ipynb
```

This notebook extends Cd bioaccessibility prediction toward risk interpretation by calculating:

```text
Bioaccessible Cd_GP = T_Cd × BA_GP / 100
```

It then compares total Cd with bioaccessible Cd, constructs Low / Medium / High relative risk groups, and summarizes risk patterns by soil source type.

Main outputs are saved to:

```text
results/bioaccessible_cd_risk_data.csv
results/bioaccessible_cd_risk_by_type.csv
figures/
```

### 10. Optional: regenerate EDA figures

If users want to regenerate the exploratory analysis figures, run:

```bash
python scripts/02_eda.py
```

This script reads the processed datasets from `data/processed_data/` and saves figures to:

```text
figures/
```

### 11. Output folders

- `data/processed_data/` stores processed datasets used for modeling and analysis.
- `figures/` stores exploratory analysis, model performance, interpretation, and risk-extension figures.
- `results/` stores model comparison tables, tuning results, feature importance, and risk-extension outputs.

### 12. Notes

- This demo is designed as a sample-level machine learning workflow, not a national-scale spatial prediction model.
- The dataset does not include precise geographic coordinates.
- The risk groups are relative categories for demonstration and should not be interpreted as regulatory risk thresholds.
- Feature importance and partial dependence results indicate model-based associations rather than causal effects.

## Models

The following models were compared:

- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost (added in this version)

Model performance was evaluated using cross-validated MAE, RMSE, and R².

### Tuned Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Tuned Random Forest | 15.81 | 19.97 | 0.462 |
| Tuned XGBoost | 16.08 | 20.45 | 0.433 |

(5-fold cross-validation, 10 repeats, n=190)

## Preliminary Findings

Total-Cd-only models showed limited predictive ability for BA_GP. The feature group ablation study confirmed that adding soil physicochemical properties substantially improved prediction, and further adding `Method` and `Type` provided additional gains. Full-feature models achieved the best performance, with Random Forest and XGBoost outperforming Ridge Regression and Gradient Boosting.

After hyperparameter tuning, the tuned Random Forest achieved MAE ≈ 15.81 and R² ≈ 0.462, slightly outperforming the tuned XGBoost (MAE ≈ 16.08, R² ≈ 0.433).

Permutation importance analysis on the tuned Random Forest identified Fe, SOM, log_T_Cd, Clay, and Method as the most important predictors of gastric-phase Cd bioaccessibility.

## Risk Extension

Bioaccessible Cd in the gastric phase was calculated as:

`Bioaccessible Cd_GP = T_Cd × BA_GP / 100`

This indicator was used to explore how Cd bioaccessibility prediction can be connected to pollution risk interpretation. Samples were grouped into Low / Medium / High relative risk categories and summarized by soil source type.

## Limitations

- The dataset does not include precise geographic coordinates.
- This demo does not perform national-scale spatial prediction.
- Literature-derived data may contain methodological heterogeneity.
- Feature importance and PDP results are model-based associations, not causal effects.
- Risk groups are relative categories for demonstration, not regulatory thresholds.

## Repository Structure

```text
config.yaml
data/
figures/
notebooks/
scripts/
results/
README.md
requirements.txt
```

## Author

Rongshuo Shang  
Undergraduate Student, Agricultural Resources and Environment  
Nanjing Agricultural University
