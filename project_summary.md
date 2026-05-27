# Project Summary

## Title

Interpretable Machine Learning for Predicting Soil Cadmium Bioaccessibility from Heterogeneous Environmental Data

## Motivation

Total Cd concentration alone may not adequately represent exposure-related risk. This demo explores whether machine learning models can predict gastric-phase Cd bioaccessibility and support bioaccessibility-informed risk interpretation.

## Data Source

The dataset used in this demo was obtained from the public Zenodo record:

**Compiled dataset on soil cadmium bioaccessibility, related properties, and concentration across China**  
Authors: Jianghao Cao and Youya Zhou  
DOI: 10.5281/zenodo.18171038  
Source: https://zenodo.org/records/18171038  
License: CC BY 4.0

The original dataset is provided as `Data-Cd.xlsx` and includes soil Cd bioaccessibility data, soil physicochemical properties, soil Cd concentration data, and corresponding literature sources across China. In this demo, `Processed_Data` was used for BA_GP prediction, while `Cd_Source` was used for supporting EDA and background analysis.

The raw Excel file is not included in this repository. Processed datasets and analysis outputs are provided for demonstration and reproducibility.

## Methods

- Data cleaning and log transformation
- EDA and group comparisons
- Total-Cd-only baseline models
- Full-feature Ridge, Random Forest, and Gradient Boosting models
- 5-fold cross-validation
- Permutation importance and PDP
- Bioaccessible Cd risk extension

## Key Results

- Total-Cd-only baseline showed weak predictive ability.
- Full-feature Random Forest achieved the best overall performance.
- Method and Type showed significant differences in BA_GP.
- Fe, SOM, log_T_Cd, Method, and Type were important predictors.
- Bioaccessible Cd was calculated to connect BA_GP prediction with risk interpretation.

## Limitations

- No coordinates; not a spatial prediction model.
- Literature-derived data are heterogeneous.
- Model interpretation does not imply causality.
- Risk groups are relative and for demonstration only.

## Future Work

- Add spatial covariates if coordinates become available.
- Expand the dataset with more soil properties and contamination history.
- Add SHAP interpretation.
- Extend from sample-level prediction to regional-scale risk modeling.