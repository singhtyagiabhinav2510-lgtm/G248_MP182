import hashlib
import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import yaml

from src.bakery_sales.preprocess import clean_data, get_preprocessor

def compute_sha256(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    # 1. Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    raw_train_path = params["data"]["raw_train_path"]
    num_features = params["preprocessing"]["features"]["numerical"]
    cat_features = params["preprocessing"]["features"]["categorical"]
    target_col = params["preprocessing"]["target"]
    random_state = params["train"]["random_state"]
    test_size = params["train"]["test_size"]
    experiment_name = params["tracking"]["experiment_name"]
    model_name = params["tracking"]["registered_model_name"]
    model_output_path = params["tracking"]["model_output_path"]

    # Compute data sha256
    data_sha = compute_sha256(raw_train_path)

    # 2. Load and clean data
    df_raw = pd.read_csv(raw_train_path)
    df_cleaned = clean_data(df_raw, is_training=True)

    feature_cols = num_features + cat_features
    X = df_cleaned[feature_cols]
    y = df_cleaned[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Configure MLflow (SQLite tracking required by Handbook)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    # 4. Define 3 candidate models to fulfill the 3+ runs requirement
    candidate_models = [
        (
            "Ridge_Regression",
            Ridge(alpha=1.0, random_state=random_state),
            {"model_type": "Ridge", "alpha": 1.0},
        ),
        (
            "Random_Forest",
            RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state),
            {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 8},
        ),
        (
            "Gradient_Boosting",
            GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=random_state),
            {"model_type": "GradientBoosting", "n_estimators": 120, "max_depth": 4, "learning_rate": 0.08},
        ),
    ]

    best_mae = float("inf")
    best_pipeline = None
    best_run_id = None
    best_run_name = None

    for run_name, regressor, hyperparams in candidate_models:
        with mlflow.start_run(run_name=run_name) as run:
            # Build end-to-end pipeline
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", get_preprocessor(num_features, cat_features)),
                    ("regressor", regressor),
                ]
            )

            # Train
            pipeline.fit(X_train, y_train)

            # Predict and evaluate
            preds = pipeline.predict(X_val)
            val_mae = mean_absolute_error(y_val, preds)

            # Mandatory logging
            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_params(hyperparams)
            mlflow.log_metric("mae", val_mae)
            mlflow.log_metric("mae_val", val_mae)

            # Log model artifact with cloudpickle format (safe for MLflow 3/2)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            print(f"Run '{run_name}' | Run ID: {run.info.run_id} | Validation MAE: {val_mae:.3f}")

            if val_mae < best_mae:
                best_mae = val_mae
                best_pipeline = pipeline
                best_run_id = run.info.run_id
                best_run_name = run_name

    print("\n" + "=" * 60)
    print(f"Best Model: {best_run_name} (Run ID: {best_run_id}) with MAE: {best_mae:.3f}")
    print("=" * 60)

    # 5. Register best model and assign champion alias
    model_uri = f"runs:/{best_run_id}/model"
    reg_model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=reg_model_version.version,
    )
    print(f"Registered model '{model_name}' Version {reg_model_version.version} with alias 'champion'.")

    # 6. Save model to models/model.joblib for Stage 3 API
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(best_pipeline, model_output_path)
    print(f"Saved champion pipeline to '{model_output_path}'.")

    # 7. Print summary for findings.yaml
    print("\n--- UPDATE findings.yaml WITH THESE VALUES ---")
    print(f"tracking:")
    print(f"  experiment_name: \"{experiment_name}\"")
    print(f"  best_run_id: \"{best_run_id}\"")
    print(f"  registered_model_name: \"{model_name}\"")
    print(f"  champion_version: {reg_model_version.version}")

if __name__ == "__main__":
    main()