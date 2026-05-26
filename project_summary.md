# Project Summary

## Title

Interpretable Machine Learning for Predicting Soil Cadmium Bioaccessibility from Heterogeneous Environmental Data

## Motivation

Total Cd concentration alone may not adequately represent exposure-related risk. This demo explores whether machine learning models can predict gastric-phase Cd bioaccessibility and support bioaccessibility-informed risk interpretation.

## Data

- Main dataset: Processed_Data for BA_GP modeling
- Supporting dataset: Cd_source for total Cd contamination background

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