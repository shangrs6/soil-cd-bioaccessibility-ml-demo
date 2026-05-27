# Interpretable Machine Learning for Soil Cadmium Bioaccessibility Prediction

This repository contains a preliminary data-driven demo for predicting gastric-phase cadmium (Cd) bioaccessibility in Chinese soils and extending the prediction results toward bioaccessibility-informed risk interpretation.

## Background

Soil Cd pollution risk is often evaluated using total Cd concentration. However, total Cd alone may not fully represent exposure-related risk because Cd bioaccessibility is affected by soil physicochemical properties, source types, and in vitro digestion methods.

This project explores whether machine learning models can improve Cd bioaccessibility prediction beyond a total-Cd-only baseline.

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
This hypothesis is evaluated through cross-validated model comparison among Ridge Regression, Random Forest, and Gradient Boosting models. The comparison between total-Cd-only and full-feature models is saved in `results/model_comparison_baseline_vs_full.csv`.

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
3. Total-Cd-only baseline modeling
4. Full-feature machine learning model comparison
5. Method and Type group-difference analysis
6. Permutation importance and partial dependence plots
7. Bioaccessibility-informed risk extension

## Code Description

This repository contains both Python scripts and Jupyter notebooks. The scripts are used for reproducible data preprocessing and exploratory analysis, while the notebooks are used for model development, interpretation, and risk-extension analysis.

### Scripts

- `scripts/01_data_processed.py`  
  Cleans the raw Excel dataset, standardizes variable names, converts numerical columns, constructs log-transformed variables such as `log_T_Cd`, and exports processed datasets to `data/processed_data/`.

- `scripts/02_eda.py`  
  Performs exploratory data analysis for the processed bioaccessibility dataset and the Cd_source dataset. It generates distribution plots, Method/Type group comparisons, Cd source-type boxplots, correlation matrices, and scatter plots. Output figures are saved to `figures/`.

### Notebooks

- `notebooks/03_model_ba_gp.ipynb`  
  Builds and evaluates machine learning models for predicting gastric-phase Cd bioaccessibility (`BA_GP`).  
  This notebook includes:
  - total-Cd-only baseline models;
  - full-feature model comparison;
  - Ridge Regression, Random Forest, and Gradient Boosting models;
  - 5-fold cross-validation;
  - baseline vs full-feature comparison;
  - Kruskal-Wallis tests for Method and Type effects;
  - permutation importance;
  - partial dependence plots.

- `notebooks/04_risk_extension.ipynb`  
  Extends the BA_GP prediction results toward bioaccessibility-informed Cd risk interpretation.  
  This notebook calculates gastric-phase bioaccessible Cd using:

  `Bioaccessible Cd_GP = T_Cd × BA_GP / 100`

  It then compares total Cd and bioaccessible Cd, constructs relative risk groups, and summarizes risk patterns by soil source type.

### Results

- `results/model_comparison_baseline_vs_full.csv`  
  Model comparison table between total-Cd-only baseline models and full-feature models.

- `results/model_performance_ba_gp.csv`  
  Cross-validated performance of full-feature BA_GP prediction models.

- `results/feature_importance_ba_gp.csv`  
  Permutation importance results for key predictors.

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
jupyter
```

### 3. Recommended running order

The suggested workflow is:

```text
scripts/01_data_processed.py
scripts/02_eda.py
notebooks/03_model_ba_gp.ipynb
notebooks/04_risk_extension.ipynb
```

However, because the raw Excel file is not included in this public repository, `scripts/01_data_processed.py` is provided mainly to document the original data-cleaning workflow. Users who clone this repository can start directly from the processed datasets:

```text
data/processed_data/processed_bioaccessibility.csv
data/processed_data/cd_source_cleaned.csv
data/processed_data/cd_source_analysis.csv
```

Therefore, the recommended reproducible starting point is:

```text
notebooks/03_model_ba_gp.ipynb
```

### 4. Run the modeling notebook

Open and run:

```text
notebooks/03_model_ba_gp.ipynb
```

This notebook performs:

- total-Cd-only baseline modeling;
- full-feature model comparison;
- 5-fold cross-validation;
- Ridge Regression, Random Forest, and Gradient Boosting modeling;
- baseline vs full-feature comparison;
- Kruskal-Wallis tests for Method and Type effects;
- permutation importance analysis;
- partial dependence plot analysis.

Main outputs are saved to:

```text
results/
figures/
```

### 5. Run the risk-extension notebook

After running the modeling notebook, open and run:

```text
notebooks/04_risk_extension.ipynb
```

This notebook extends Cd bioaccessibility prediction toward risk interpretation by calculating:

```text
Bioaccessible Cd_GP = T_Cd × BA_GP / 100
```

It then compares total Cd with bioaccessible Cd, constructs relative risk groups, and summarizes risk patterns by soil source type.

Main outputs are saved to:

```text
results/bioaccessible_cd_risk_data.csv
results/bioaccessible_cd_risk_by_type.csv
figures/
```

### 6. Optional: regenerate EDA figures

If users want to regenerate the exploratory analysis figures, run:

```bash
python scripts/02_eda.py
```

This script reads the processed datasets from `data/processed_data/` and saves figures to:

```text
figures/
```

### 7. Output folders

- `data/processed_data/` stores processed datasets used for modeling and analysis.
- `figures/` stores exploratory analysis, model performance, interpretation, and risk-extension figures.
- `results/` stores model comparison tables, statistical test outputs, feature importance results, and risk-extension outputs.

### 8. Notes

- This demo is designed as a sample-level machine learning workflow, not a national-scale spatial prediction model.
- The dataset does not include precise geographic coordinates.
- The risk groups are relative categories for demonstration and should not be interpreted as regulatory risk thresholds.
- Feature importance and partial dependence results indicate model-based associations rather than causal effects.

## Models

The following models were compared:

- Ridge Regression
- Random Forest
- Gradient Boosting

Model performance was evaluated using cross-validated MAE, RMSE, and R².

## Preliminary Findings

Total-Cd-only models showed limited predictive ability for BA_GP. Full-feature models incorporating soil physicochemical properties, Method, and Type achieved better performance, with Random Forest showing the best overall performance.

Model interpretation suggested that Fe, SOM, log-transformed total Cd, Method, and Type were important predictors of gastric-phase Cd bioaccessibility.

## Risk Extension

Bioaccessible Cd in the gastric phase was calculated as:

`Bioaccessible Cd_GP = T_Cd × BA_GP / 100`

This indicator was used to explore how Cd bioaccessibility prediction can be connected to pollution risk interpretation.

## Limitations

- The dataset does not include precise geographic coordinates.
- This demo does not perform national-scale spatial prediction.
- Literature-derived data may contain methodological heterogeneity.
- Feature importance and PDP results are model-based associations, not causal effects.
- Risk groups are relative categories for demonstration, not regulatory thresholds.

## Repository Structure

```text
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