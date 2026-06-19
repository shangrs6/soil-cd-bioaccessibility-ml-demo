from pathlib import Path
import warnings
import yaml
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, cross_validate
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config.yaml"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_onehot_encoder():
    """
    Compatible with different scikit-learn versions.
    """
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


def build_models(random_state=42):
    models = {
        "Ridge Regression": Ridge(alpha=1.0),

        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            min_samples_leaf=2
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=random_state
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models


def evaluate_feature_groups(df, config):
    target = config["target"]
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]
    feature_groups = config["feature_groups"]

    cv_config = config["cv"]
    cv = RepeatedKFold(
        n_splits=cv_config["n_splits"],
        n_repeats=cv_config["n_repeats"],
        random_state=cv_config["random_state"]
    )

    models = build_models(random_state=cv_config["random_state"])

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }

    records = []

    df = df.dropna(subset=[target]).copy()

    for group_name, selected_features in feature_groups.items():
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            print(f"[Warning] Skip {group_name}, missing features: {missing_features}")
            continue

        X = df[selected_features].copy()
        y = df[target].copy()

        for model_name, model in models.items():
            print(f"Running: {group_name} | {model_name}")

            preprocessor = build_preprocessor(
                selected_features=selected_features,
                numeric_features=numeric_features,
                categorical_features=categorical_features
            )

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model)
                ]
            )

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

            records.append({
                "Feature_group": group_name,
                "Model": model_name,
                "MAE_mean": np.mean(mae_scores),
                "MAE_std": np.std(mae_scores),
                "RMSE_mean": np.mean(rmse_scores),
                "RMSE_std": np.std(rmse_scores),
                "R2_mean": np.mean(r2_scores),
                "R2_std": np.std(r2_scores),
                "n_samples": len(df),
                "n_cv_runs": len(mae_scores)
            })

    return pd.DataFrame(records)


def main():
    config = load_config()
    data_path = BASE_DIR / config["data"]["processed_file"]

    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    df = pd.read_csv(data_path)

    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())

    results = evaluate_feature_groups(df, config)

    output_path = RESULTS_DIR / "feature_group_ablation_results.csv"
    results.to_csv(output_path, index=False, lineterminator="\n")

    print("\nSaved results to:", output_path)
    print(results)


if __name__ == "__main__":
    main()