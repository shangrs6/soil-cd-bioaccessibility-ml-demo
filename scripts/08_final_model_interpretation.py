from pathlib import Path
import json
import warnings
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config.yaml"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
BEST_PARAMS_PATH = RESULTS_DIR / "best_rf_params.json"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


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

    preprocessor = build_preprocessor(
        selected_features=selected_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    tuned_rf = RandomForestRegressor(
        random_state=42,
        **best_params
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", tuned_rf)
        ]
    )

    # Use a hold-out set for permutation importance
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Permutation importance on original feature columns
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance_mean": perm.importances_mean,
        "Importance_std": perm.importances_std
    }).sort_values("Importance_mean", ascending=False)

    importance_path = RESULTS_DIR / "final_tuned_rf_permutation_importance.csv"
    importance_df.to_csv(importance_path, index=False, lineterminator="\n")

    print("Permutation importance:")
    print(importance_df)

    # Plot permutation importance
    plt.figure(figsize=(8, 5))
    plot_df = importance_df.sort_values("Importance_mean", ascending=True)
    plt.barh(plot_df["Feature"], plot_df["Importance_mean"], xerr=plot_df["Importance_std"])
    plt.xlabel("Permutation importance based on MAE decrease")
    plt.ylabel("Feature")
    plt.title("Permutation Importance of Tuned Random Forest")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_tuned_rf_permutation_importance.png", dpi=300)
    plt.close()

    # PDP for key numeric variables
    pdp_features = ["log_T_Cd", "Fe", "SOM", "pH"]

    available_pdp_features = [f for f in pdp_features if f in selected_features]

    for feature in available_pdp_features:
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(
            model,
            X_train,
            features=[feature],
            ax=ax
        )
        ax.set_title(f"Partial Dependence of BA_GP on {feature}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"pdp_tuned_rf_{feature}.png", dpi=300)
        plt.close()

    print("Saved interpretation outputs:")
    print(f"- {importance_path}")
    print("- figures/final_tuned_rf_permutation_importance.png")
    for feature in available_pdp_features:
        print(f"- figures/pdp_tuned_rf_{feature}.png")


if __name__ == "__main__":
    main()