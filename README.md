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

## Dataset

This project uses two processed datasets:

1. **Processed_Data**
   - Main modeling dataset for BA_GP prediction.
   - Variables include pH, SOM, soil texture, Fe, total Cd, digestion method, soil source type, and BA_GP.

2. **Cd_source**
   - Supporting dataset for describing total Cd contamination patterns across soil source types.
   - It is not used for BA_GP prediction because it lacks bioaccessibility and soil physicochemical variables.

## Workflow

1. Data cleaning and feature engineering
2. Exploratory data analysis
3. Total-Cd-only baseline modeling
4. Full-feature machine learning model comparison
5. Method and Type group-difference analysis
6. Permutation importance and partial dependence plots
7. Bioaccessibility-informed risk extension

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