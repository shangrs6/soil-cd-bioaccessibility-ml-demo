from pathlib import Path
import json
import warnings
import yaml
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV


warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config.yaml"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


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


def main():
    config = load_config()

    data_path = BASE_DIR / config["data"]["processed_file"]
    df = pd.read_csv(data_path)

    target = config["target"]
    selected_features = config["feature_groups"]["Full-feature"]

    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    df = df.dropna(subset=[target]).copy()

    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    X = df[selected_features].copy()
    y = df[target].copy()

    preprocessor = build_preprocessor(
        selected_features=selected_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42))
        ]
    )

    param_distributions = {
        "model__n_estimators": [200, 300, 500, 800],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 3, 5],
        "model__max_features": ["sqrt", 0.7, 1.0]
    }

    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    search.fit(X, y)

    tuning_results = pd.DataFrame(search.cv_results_)
    tuning_results_path = RESULTS_DIR / "rf_tuning_results.csv"
    tuning_results.to_csv(tuning_results_path, index=False, lineterminator="\n")

    best_params_path = RESULTS_DIR / "best_rf_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(search.best_params_, f, indent=4)

    summary = {
        "best_score_neg_mae": search.best_score_,
        "best_mae": -search.best_score_,
        "best_params": search.best_params_
    }

    summary_path = RESULTS_DIR / "rf_tuning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("Best MAE:", -search.best_score_)
    print("Best parameters:", search.best_params_)
    print("Saved tuning results to:", tuning_results_path)
    print("Saved best params to:", best_params_path)


if __name__ == "__main__":
    main()