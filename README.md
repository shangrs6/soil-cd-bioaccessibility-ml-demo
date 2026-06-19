# Interpretable Machine Learning for Soil Cadmium Bioaccessibility Prediction

This repository contains a preliminary data-driven demo for predicting gastric-phase cadmium (Cd) bioaccessibility in Chinese soils and extending the prediction results toward bioaccessibility-informed risk interpretation.

## Background

Soil Cd pollution risk is often evaluated using total Cd concentration. However, total Cd alone may not fully represent exposure-related risk because Cd bioaccessibility is affected by soil physicochemical properties, source types, and in vitro digestion methods.

This project explores whether machine learning models can improve Cd bioaccessibility prediction beyond a total-Cd-only baseline, and further investigates whether hyperparameter tuning can improve the best-performing model.

## Research Questions

1. Can total Cd concentration alone predict gastric-phase Cd bioaccessibility?
2. Do soil physicochemical properties, digestion methods, and soil source types improve prediction performance?
3. Which variables are most important for predicting Cd bioaccessibility?
4. Can predicted bioaccessibility be extended toward bioaccessibility-informed Cd risk interpretation?

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

This project contains two independent execution paths that share the same processed datasets but do not exchange outputs with each other.

**Script pipeline (scripts/04 → 05 → 06 → 07 → 08, strictly sequential within each model branch):**

1. Data cleaning and feature engineering (`scripts/01`, requires raw Excel)
2. Exploratory data analysis (`scripts/02`, independent)
3. Random Forest tuning (`scripts/04`) → produces `best_rf_params.json`
4. Evaluate tuned Random Forest (`scripts/05`) → reads `best_rf_params.json`
5. XGBoost tuning (`scripts/06`) → produces `best_xgb_params.json`
6. Evaluate tuned XGBoost (`scripts/07`) → reads `best_xgb_params.json`
7. Final model interpretation (`scripts/08`) → reads `best_rf_params.json`

**Optional standalone experiment (no dependencies, can run at any time):**

- Feature group ablation study (`scripts/03`) — reads only processed data and `config.yaml`; its output is not used by any other script

**Notebook path (independent of the script pipeline):**

- Model development and comparison (`notebooks/09`) — reads processed data directly; does not depend on any script output
- Risk extension (`notebooks/10`) — reads processed data directly; requires `notebooks/09` to be run first within the same notebook session

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

- `scripts/03_feature_group_ablation.py` *(standalone exploratory experiment)*  
  Runs a systematic feature group ablation study across five feature group configurations (Total-Cd-only, Soil-properties-only, Cd + Soil, Cd + Soil + Method, Full-feature) and four models (Ridge Regression, Random Forest, Gradient Boosting, XGBoost). Uses 5-fold cross-validation with 10 repeats. Feature groups and CV settings are read from `config.yaml`. Results are saved to `results/feature_group_ablation_results.csv`.  
  **This script is independent of the tuning pipeline (scripts/04–08) and does not produce outputs consumed by any other script. It can be run in isolation at any point after processed data is available.**

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

- `notebooks/09_model_ba_gp.ipynb` *(independent of script pipeline)*  
  Builds and evaluates machine learning models for predicting gastric-phase Cd bioaccessibility (`BA_GP`). Reads processed data directly from `data/processed_data/`; does not depend on any output from scripts/03–08.  
  This notebook includes:
  - total-Cd-only baseline models;
  - full-feature model comparison (Ridge Regression, Random Forest, Gradient Boosting, XGBoost);
  - 5-fold cross-validation;
  - baseline vs full-feature comparison;
  - Kruskal-Wallis tests for Method and Type effects;
  - permutation importance;
  - partial dependence plots;
  - 80/20 train-test split for observed vs predicted visualization.

- `notebooks/10_risk_extension.ipynb` *(depends only on notebooks/09 within the same session)*  
  Extends the BA_GP prediction results toward bioaccessibility-informed Cd risk interpretation. Reads processed data directly from `data/processed_data/`; does not depend on any output from scripts/03–08. Requires `notebooks/09_model_ba_gp.ipynb` to have been run first in the same notebook session (uses the fitted model object in memory).  
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
The raw Excel file is not included in this repository due to data ownership and citation considerations, so users can start directly from the processed data.

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

---

### Path A: Script pipeline (scripts/04 → 05 → 06 → 07 → 08)

This path runs the tuning and interpretation pipeline. Steps within each model branch are strictly sequential due to file dependencies. `scripts/03` is an independent experiment described in Path B.

#### A-1. Optional: regenerate EDA figures

```bash
python scripts/02_eda.py
```

Reads processed datasets from `data/processed_data/` and saves figures to `figures/`. Independent of all other scripts.

#### A-2. Random Forest hyperparameter tuning

```bash
python scripts/04_rf_tuning.py
```

Runs `RandomizedSearchCV` (20 iterations, 5-fold × 5 repeats). Saves best parameters to `results/best_rf_params.json` and full tuning results to `results/rf_tuning_results.csv`.

#### A-3. Evaluate tuned Random Forest

```bash
python scripts/05_evaluate_tuned_rf.py
```

**Requires:** `results/best_rf_params.json` (produced by A-2).  
Evaluates the tuned Random Forest with 5-fold × 10 repeats cross-validation. Results saved to `results/tuned_rf_evaluation.csv`.

#### A-4. XGBoost hyperparameter tuning

```bash
python scripts/06_xgb_tuning.py
```

Runs `RandomizedSearchCV` (30 iterations, 5-fold × 5 repeats). Saves best parameters to `results/best_xgb_params.json` and full tuning results to `results/xgb_tuning_results.csv`. Independent of A-2/A-3 and can be run in parallel.

#### A-5. Evaluate tuned XGBoost

```bash
python scripts/07_evaluate_tuned_xgb.py
```

**Requires:** `results/best_xgb_params.json` (produced by A-4).  
Evaluates the tuned XGBoost with 5-fold × 10 repeats cross-validation. Results saved to `results/tuned_xgb_evaluation.csv`.

#### A-6. Final model interpretation

```bash
python scripts/08_final_model_interpretation.py
```

**Requires:** `results/best_rf_params.json` (produced by A-2).  
Fits the tuned Random Forest on an 80/20 train-test split, computes permutation importance (30 repeats), and generates partial dependence plots for `log_T_Cd`, `Fe`, `SOM`, and `pH`. Outputs saved to `results/final_tuned_rf_permutation_importance.csv` and `figures/`.

**Dependency summary for Path A:**

```text
scripts/02_eda.py                              # independent
scripts/04_rf_tuning.py                        # independent
  └─ scripts/05_evaluate_tuned_rf.py           # requires best_rf_params.json
  └─ scripts/08_final_model_interpretation.py  # requires best_rf_params.json
scripts/06_xgb_tuning.py                       # independent of RF branch
  └─ scripts/07_evaluate_tuned_xgb.py          # requires best_xgb_params.json
```

---

### Path B: Standalone exploratory experiment (scripts/03)

```bash
python scripts/03_feature_group_ablation.py
```

Runs a feature group ablation study across five configurations (Total-Cd-only → Soil-properties-only → Cd + Soil → Cd + Soil + Method → Full-feature) and four models. Reads only `data/processed_data/processed_bioaccessibility.csv` and `config.yaml`. Its output (`results/feature_group_ablation_results.csv`) is not consumed by any other script. Can be run independently at any time after processed data is available, with no dependency on Path A or Path C.

---

### Path C: Notebook path (independent of scripts/03–08)

Both notebooks read processed data directly from `data/processed_data/` and do not depend on any output produced by the script pipeline.

#### C-1. Run the modeling notebook

```text
notebooks/09_model_ba_gp.ipynb
```

This notebook performs:

- total-Cd-only baseline modeling;
- full-feature model comparison (Ridge Regression, Random Forest, Gradient Boosting, XGBoost);
- 5-fold cross-validation;
- baseline vs full-feature comparison;
- Kruskal-Wallis tests for Method and Type effects;
- permutation importance analysis;
- partial dependence plot analysis.

Main outputs are saved to `results/` and `figures/`.

#### C-2. Run the risk-extension notebook

```text
notebooks/10_risk_extension.ipynb
```

**Requires:** `notebooks/09_model_ba_gp.ipynb` to have been run first in the same notebook session (uses the fitted model object held in memory).  
Calculates gastric-phase bioaccessible Cd as:

```text
Bioaccessible Cd_GP = T_Cd × BA_GP / 100
```

Constructs Low / Medium / High relative risk groups and summarizes risk patterns by soil source type. Main outputs are saved to `results/bioaccessible_cd_risk_data.csv`, `results/bioaccessible_cd_risk_by_type.csv`, and `figures/`.

---

### Output folders

- `data/processed_data/` stores processed datasets used for modeling and analysis.
- `figures/` stores exploratory analysis, model performance, interpretation, and risk-extension figures.
- `results/` stores model comparison tables, tuning results, feature importance, and risk-extension outputs.

### Notes

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
