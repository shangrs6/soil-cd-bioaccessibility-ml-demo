from pathlib import Path
import json
import warnings
import yaml
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_validate

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config.yaml"
RESULTS_DIR = BASE_DIR / "results"

BEST_PARAMS_PATH = RESULTS_DIR / "best_xgb_params.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(selected_features, numeric_features, categorical_features):
    selected_numeric = [f for f in selected_features if f in numeric_features]
    selected_categorical = [f for f in selected_features if f in categorical_features]

    transformers = []

    if selected_numeric:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )
        transformers.append(("num", numeric_transformer, selected_numeric))

    if selected_categorical:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot_encoder())
            ]
        )
        transformers.append(("cat", categorical_transformer, selected_categorical))

    return ColumnTransformer(transformers=transformers)


def clean_best_params(best_params):
    cleaned = {}
    for key, value in best_params.items():
        if key.startswith("model__"):
            cleaned[key.replace("model__", "")] = value
    return cleaned


def main():
    config = load_config()

    data_path = BASE_DIR / config["data"]["processed_file"]
    df = pd.read_csv(data_path)

    target = config["target"]
    selected_features = config["feature_groups"]["Full-feature"]

    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    df = df.dropna(subset=[target]).copy()

    X = df[selected_features].copy()
    y = df[target].copy()

    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        best_params_raw = json.load(f)

    best_params = clean_best_params(best_params_raw)

    print("Using tuned XGBoost parameters:")
    print(best_params)

    preprocessor = build_preprocessor(
        selected_features=selected_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    tuned_xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_params
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", tuned_xgb)
        ]
    )

    cv = RepeatedKFold(
        n_splits=config["cv"]["n_splits"],
        n_repeats=config["cv"]["n_repeats"],
        random_state=config["cv"]["random_state"]
    )

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    mae_scores = -scores["test_MAE"]
    rmse_scores = -scores["test_RMSE"]
    r2_scores = scores["test_R2"]

    result = pd.DataFrame([{
        "Feature_group": "Full-feature",
        "Model": "Tuned XGBoost",
        "MAE_mean": np.mean(mae_scores),
        "MAE_std": np.std(mae_scores),
        "RMSE_mean": np.mean(rmse_scores),
        "RMSE_std": np.std(rmse_scores),
        "R2_mean": np.mean(r2_scores),
        "R2_std": np.std(r2_scores),
        "n_samples": len(df),
        "n_cv_runs": len(mae_scores)
    }])

    output_path = RESULTS_DIR / "tuned_xgb_evaluation.csv"
    result.to_csv(output_path, index=False, lineterminator="\n")

    print("\nTuned XGBoost evaluation:")
    print(result)
    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()