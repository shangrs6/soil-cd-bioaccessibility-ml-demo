# Performance Summary

> **CV benchmark:** RepeatedKFold(n_splits=5, n_repeats=10, random_state=42) — 50 evaluation runs per model  
> **Dataset:** n = 190 samples  
> **Target:** BA_GP (gastric-phase Cd bioaccessibility, %)  
> All rows in this document share the same CV protocol, making metrics directly comparable.

---

## H3 Verification: Full-feature Models vs Total-Cd-only Baseline

**H3 states:** Machine learning models integrating soil properties and methodological variables outperform total-Cd-only baseline models.

| Comparison | Best model | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| Total-Cd-only baseline | XGBoost | 21.06 ± 2.86 | 0.047 ± 0.201 |
| Full-feature (default params) | Random Forest | 16.15 ± 2.23 | 0.432 ± 0.124 |

**Performance gap (Full-feature best vs Total-Cd-only best):**

- MAE improvement: 4.92 (23.3% reduction)
- R² improvement: +0.385 (+814.4% relative gain)

**Conclusion:** H3 is supported under the unified 10-repeat CV benchmark.  
The best full-feature model (Random Forest) reduces MAE by 4.92 percentage points and raises R² by 0.385 compared to the best total-Cd-only baseline (XGBoost).  
Both models were evaluated under identical RepeatedKFold(5, 10) protocol, confirming the result is not an artifact of CV variance.

---

## RQ4: Does Hyperparameter Tuning Improve Performance?

**RQ4 asks:** Can hyperparameter-tuned Random Forest and XGBoost outperform default-parameter models?

| Model | Params | MAE (mean ± std) | RMSE (mean ± std) | R² (mean ± std) |
|---|---|---|---|---|
| Random Forest (default) | n_estimators=500, min_samples_leaf=2 | 16.15 ± 2.23 | 20.50 ± 2.65 | 0.432 ± 0.124 |
| Tuned Random Forest | n_estimators=300, max_depth=20, min_samples_leaf=1, max_features=sqrt | 15.81 ± 2.13 | 19.97 ± 2.49 | 0.462 ± 0.113 |
| Tuned XGBoost | n_estimators=100, max_depth=3, lr=0.1 | 16.08 ± 2.29 | 20.45 ± 2.79 | 0.433 ± 0.142 |

**Tuning gain (Tuned RF vs default RF, same Full-feature input, same CV protocol):**

- MAE: 16.15 → 15.81  (0.34 reduction, 2.1%)
- R²: 0.432 → 0.462  (+0.030, +6.9%)

**Conclusion:** Hyperparameter tuning provides a modest but consistent improvement for Random Forest.  
The tuned RF achieves the best overall MAE (15.81) and R² (0.462) among all evaluated models.  
The tuned XGBoost (MAE = 16.08, R² = 0.433) slightly underperforms the tuned RF but outperforms the default XGBoost (MAE = 16.24).

---

## Full Comparison Table (all models, unified CV benchmark)

Values reported as mean ± std across 50 CV folds (5-fold × 10 repeats).  
Rows marked **Tuned** were evaluated by scripts/05 and scripts/07; all others by scripts/03.

| Feature_group | Model | MAE | R² |
| --- | --- | --- | --- |
| Total-Cd-only | Gradient Boosting | 21.12 ± 2.73 | 0.036 ± 0.209 |
| Total-Cd-only | Random Forest | 21.09 ± 3.01 | 0.055 ± 0.217 |
| Total-Cd-only | Ridge Regression | 22.03 ± 1.90 | 0.090 ± 0.089 |
| Total-Cd-only | XGBoost | 21.06 ± 2.86 | 0.047 ± 0.201 |
| Soil-properties-only | Gradient Boosting | 18.29 ± 2.21 | 0.281 ± 0.138 |
| Soil-properties-only | Random Forest | 17.76 ± 2.15 | 0.331 ± 0.130 |
| Soil-properties-only | Ridge Regression | 21.87 ± 2.28 | 0.098 ± 0.125 |
| Soil-properties-only | XGBoost | 18.31 ± 2.14 | 0.291 ± 0.132 |
| Cd + Soil | Gradient Boosting | 17.81 ± 2.33 | 0.300 ± 0.160 |
| Cd + Soil | Random Forest | 17.12 ± 2.23 | 0.375 ± 0.129 |
| Cd + Soil | Ridge Regression | 20.81 ± 2.38 | 0.127 ± 0.150 |
| Cd + Soil | XGBoost | 17.68 ± 2.27 | 0.328 ± 0.147 |
| Cd + Soil + Method | Gradient Boosting | 17.41 ± 2.31 | 0.332 ± 0.153 |
| Cd + Soil + Method | Random Forest | 16.68 ± 2.29 | 0.402 ± 0.129 |
| Cd + Soil + Method | Ridge Regression | 20.26 ± 2.45 | 0.146 ± 0.179 |
| Cd + Soil + Method | XGBoost | 16.90 ± 2.22 | 0.380 ± 0.137 |
| Full-feature | Gradient Boosting | 16.63 ± 2.22 | 0.374 ± 0.147 |
| Full-feature | Random Forest | 16.15 ± 2.23 | 0.432 ± 0.124 |
| Full-feature | Ridge Regression | 19.85 ± 2.62 | 0.182 ± 0.171 |
| Full-feature | XGBoost | 16.24 ± 2.23 | 0.413 ± 0.139 |
| Full-feature | Tuned Random Forest | 15.81 ± 2.13 | 0.462 ± 0.113 |
| Full-feature | Tuned XGBoost | 16.08 ± 2.29 | 0.433 ± 0.142 |

---

## Feature Group Ablation Summary (Random Forest only)

Illustrates the progressive effect of adding feature groups on RF performance.

| Feature group | MAE (mean ± std) | R² (mean ± std) | Δ MAE vs Total-Cd-only | Δ R² vs Total-Cd-only |
|---|---|---|---|---|
| Total-Cd-only | 21.09 ± 3.01 | 0.055 ± 0.217 | +0.00 | −0.000 |
| Soil-properties-only | 17.76 ± 2.15 | 0.331 ± 0.130 | −3.34 | +0.276 |
| Cd + Soil | 17.12 ± 2.23 | 0.375 ± 0.129 | −3.98 | +0.320 |
| Cd + Soil + Method | 16.68 ± 2.29 | 0.402 ± 0.129 | −4.42 | +0.347 |
| Full-feature | 16.15 ± 2.23 | 0.432 ± 0.124 | −4.95 | +0.378 |
| Full-feature **(Tuned RF)** | **15.81 ± 2.13** | **0.462 ± 0.113** | −5.28 | +0.407 |

---

## Design Notes

- **CV protocol consistency:** All rows in this document use `RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)` (50 evaluation runs each). Metrics are directly comparable without statistical correction.
- **scripts/03 vs scripts/04–08:** `scripts/03` uses default model parameters defined inline; `scripts/04/06` run `RandomizedSearchCV` independently. The default RF in `scripts/03` uses `n_estimators=500, min_samples_leaf=2`, while the tuned RF found `n_estimators=300, max_depth=20, min_samples_leaf=1, max_features="sqrt"`.
- **notebook 09 comparability:** `notebooks/09_model_ba_gp.ipynb` uses a single `KFold(n_splits=5)` (not repeated), producing higher-variance estimates. Results from that notebook (`model_performance_ba_gp.csv`, `model_comparison_baseline_vs_full.csv`) are **not directly comparable** to the values in this document.
- **Risk of overfitting interpretation:** The dataset contains 190 samples. R² ≈ 0.46 for the best model suggests moderate predictive power. The remaining unexplained variance likely reflects measurement heterogeneity from multiple in vitro digestion protocols and literature sources.
- **Feature importance note:** Permutation importance from `scripts/08` was computed on a single 80/20 hold-out split, not under repeated CV. Importance rankings should be interpreted as indicative, not definitive.
