# Project Summary

## Title

Interpretable Machine Learning for Predicting Soil Cadmium Bioaccessibility from Heterogeneous Environmental Data

## Motivation

Total Cd concentration alone may not adequately represent exposure-related risk. This demo explores whether machine learning models that integrate soil physicochemical properties, digestion method, and soil source type can predict gastric-phase Cd bioaccessibility (BA_GP) more accurately than a total-Cd-only baseline, and whether hyperparameter tuning yields further improvement. Predicted BA_GP is then used to compute bioaccessible Cd concentrations and construct relative risk groups by soil source type.

## Data Source

The dataset used in this demo was obtained from the public Zenodo record:

**Compiled dataset on soil cadmium bioaccessibility, related properties, and concentration across China**  
Authors: Jianghao Cao and Youya Zhou  
DOI: 10.5281/zenodo.18171038  
Source: https://zenodo.org/records/18171038  
License: CC BY 4.0

The original dataset is provided as `Data-Cd.xlsx` and includes soil Cd bioaccessibility data, soil physicochemical properties, soil Cd concentration data, and corresponding literature sources across China. In this demo, `Processed_Data` was used for BA_GP prediction (n = 190 samples after dropping rows with missing target), while `Cd_Source` was used for supporting EDA and background analysis.

The raw Excel file is not included in this repository. Processed datasets and analysis outputs are provided for demonstration and reproducibility.

## Methods

### Data Preparation
- Cleaning, variable standardization, and log transformation of total Cd (`log_T_Cd`)
- Processed datasets exported to `data/processed_data/`

### Exploratory Data Analysis
- Distribution plots, Cd source-type boxplots, correlation matrices
- Kruskal-Wallis tests for BA_GP differences across `Method` and `Type` groups

### Feature Group Ablation Study (`scripts/03`)
Systematic evaluation of five feature group configurations across four models (Ridge Regression, Random Forest, Gradient Boosting, XGBoost) using RepeatedKFold(n_splits=5, n_repeats=10):

| Feature group | Description |
|---|---|
| Total-Cd-only | log_T_Cd only — baseline |
| Soil-properties-only | pH, SOM, Clay, Silt, Sand, Fe |
| Cd + Soil | log_T_Cd + soil properties |
| Cd + Soil + Method | + in vitro digestion method |
| Full-feature | + soil source type |

### Hyperparameter Tuning
- Random Forest tuned via RandomizedSearchCV (20 iterations, `scripts/04`)
- XGBoost tuned via RandomizedSearchCV (30 iterations, `scripts/06`)
- Tuned models evaluated under the same RepeatedKFold(5, 10) protocol (`scripts/05`, `scripts/07`)

### Final Model Interpretation (`scripts/08`)
- Permutation importance (30 repeats) on 80/20 hold-out test set
- Partial dependence plots for `log_T_Cd`, `Fe`, `SOM`, `pH`

### Bioaccessibility-Informed Risk Extension (`notebooks/10`)
Bioaccessible Cd computed as:

```
Bioaccessible Cd_GP = T_Cd × BA_GP / 100
```

Samples grouped into Low / Medium / High relative risk categories and summarized by soil source type.

### Unified Performance Comparison (`scripts/generate_unified_comparison.py`)
All ablation and tuned-model results share the same RepeatedKFold(5, 10) CV protocol (50 evaluation runs each). A unified comparison table (`results/unified_performance_comparison.csv`) and structured summary (`PERFORMANCE_SUMMARY.md`) were generated to explicitly answer RQ4 and verify H3 robustness.

## Key Results

### H3: Full-feature models outperform the Total-Cd-only baseline

Under the unified 10-repeat CV benchmark:

| Model | Feature group | MAE | R² |
|---|---|---|---|
| XGBoost (best baseline) | Total-Cd-only | 21.06 ± 2.86 | 0.047 ± 0.201 |
| Random Forest (best full) | Full-feature | 16.15 ± 2.23 | 0.432 ± 0.124 |

MAE reduced by 4.92 (23.3%); R² increased by +0.385. **H3 is supported.**

### Feature group ablation (Random Forest)

Each step of adding features improved prediction:

| Feature group | MAE | R² | Δ MAE vs baseline |
|---|---|---|---|
| Total-Cd-only | 21.09 | 0.055 | — |
| Soil-properties-only | 17.76 | 0.331 | −3.34 |
| Cd + Soil | 17.12 | 0.375 | −3.98 |
| Cd + Soil + Method | 16.68 | 0.402 | −4.42 |
| Full-feature | 16.15 | 0.432 | −4.95 |
| Full-feature (Tuned RF) | **15.81** | **0.462** | −5.28 |

### RQ4: Hyperparameter tuning provides a modest but consistent improvement

| Model | MAE | R² |
|---|---|---|
| Random Forest (default) | 16.15 ± 2.23 | 0.432 ± 0.124 |
| Tuned Random Forest | 15.81 ± 2.13 | 0.462 ± 0.113 |
| Tuned XGBoost | 16.08 ± 2.29 | 0.433 ± 0.142 |

Tuning gain for RF: ΔMAE = −0.34 (−2.1%), ΔR² = +0.030 (+6.9%).  
The tuned Random Forest is the best-performing model overall.

### H2: Method and Type significantly affect BA_GP variability

Kruskal-Wallis tests confirmed significant differences in BA_GP across both `Method` and `Type` groups. Including these variables progressively improved model performance (see ablation table above).

### Feature importance (Tuned RF, permutation importance on hold-out set)

Top predictors: **Fe > SOM > log_T_Cd > Clay > Method > Sand > Type > Silt > pH**  
Fe and SOM together account for the largest share of permutation importance.  
pH showed near-zero importance in the tuned RF, suggesting its effect may be mediated through correlated soil properties.

### Risk extension

Bioaccessible Cd (gastric phase) varies substantially across soil source types, highlighting that total Cd concentration alone can misrepresent actual exposure-related risk.

## Limitations

- No geographic coordinates; not a spatial prediction model.
- Literature-derived data are heterogeneous in digestion protocols and sampling regions.
- Model interpretation does not imply causality.
- Risk groups are relative and for demonstration only; they do not correspond to regulatory thresholds.
- Permutation importance was computed on a single 80/20 split, not under repeated CV; rankings are indicative.
- R² ≈ 0.46 for the best model indicates moderate predictive power; substantial unexplained variance remains, likely reflecting measurement heterogeneity.
- `notebooks/09` uses single KFold(5) rather than RepeatedKFold; its saved result files are not directly comparable to the script-path results in `unified_performance_comparison.csv`.

## Future Work

- Add spatial covariates if geographic coordinates become available.
- Expand dataset with more soil properties and contamination history records.
- Add SHAP interpretation for more granular feature attribution.
- Extend from sample-level prediction to regional-scale risk modeling.
- Align `notebooks/09` CV protocol with the script pipeline to ensure full cross-path comparability.
- Conduct statistical significance testing (e.g., Wilcoxon signed-rank) on the 50 CV fold scores to formally confirm the tuning improvement.
