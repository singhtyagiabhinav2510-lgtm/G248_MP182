from __future__ import annotations

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from src.bakery_sales.preprocess import clean_data, get_preprocessor


NUMERIC_FEATURES = [
    "price_rs",
    "temp_c",
    "footfall_index",
    "competitor_shops",
    "shelf_hours",
    "orders_previous_day",
]
OUTLETS = ["College Gate", "Mall Road", "Station Road"]
V1_RUN_ID = "87b82141cc374817a7766f68c71f9592"
REGISTERED_MODEL_NAME = "MP182_Bread_Sales_Model"


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute PSI using bins derived from the reference distribution."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if reference.empty or current.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference.quantile(quantiles).to_numpy())
    if len(edges) == 1:
        edges = np.array([-np.inf, np.inf])
    else:
        edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))

    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)

    reference_pct = reference_counts / reference_counts.sum()
    current_pct = current_counts / current_counts.sum()

    epsilon = 1e-6
    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


def load_v1_model(mlruns_path: Path) -> object:
    artifact_path = mlruns_path / "1" / V1_RUN_ID / "artifacts" / "model"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Existing v1 MLflow artifact not found: {artifact_path}")
    return mlflow.sklearn.load_model(str(artifact_path))


def build_pipeline(num_features: list[str], cat_features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", get_preprocessor(num_features, cat_features)),
            ("regressor", Ridge(alpha=1.0, random_state=42)),
        ]
    )


def update_monitoring_findings(path: Path, monitoring: dict) -> None:
    text = path.read_text(encoding="utf-8")
    monitoring_yaml = yaml.safe_dump({"monitoring": monitoring}, sort_keys=False).rstrip()

    marker = "\nmonitoring:\n"
    if marker not in text:
        raise ValueError("findings.yaml does not contain a monitoring section")

    prefix = text.split(marker, 1)[0].rstrip()
    path.write_text(prefix + "\n\n" + monitoring_yaml + "\n", encoding="utf-8")


def main() -> dict:
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)

    with (root / "params.yaml").open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    num_features = params["preprocessing"]["features"]["numerical"]
    cat_features = params["preprocessing"]["features"]["categorical"]
    target = params["preprocessing"]["target"]

    train_raw = pd.read_csv(root / params["data"]["raw_train_path"])
    later_raw = pd.read_csv(root / params["data"]["raw_later_path"])
    holdout = pd.read_csv(root / "data/raw/holdout_features.csv")

    train_clean = clean_data(train_raw, is_training=True)
    later_clean = clean_data(later_raw, is_training=False)

    psi = {
        column: compute_psi(train_clean[column], later_clean[column], bins=10)
        for column in NUMERIC_FEATURES
    }
    most_drifted = max(psi, key=psi.get)

    v1 = load_v1_model(root / "mlruns")
    later_features = num_features + cat_features
    X_later = later_clean[later_features]
    y_later = later_clean[target]

    v1_later_predictions = v1.predict(X_later)
    mae_by_group = {}
    for outlet in OUTLETS:
        mask = later_clean["outlet"] == outlet
        mae_by_group[outlet] = float(
            mean_absolute_error(y_later.loc[mask], v1_later_predictions[mask.to_numpy()])
        )
    worst_group = max(mae_by_group, key=mae_by_group.get)

    later_sorted = later_clean.copy()
    later_sorted["date"] = pd.to_datetime(later_sorted["date"])
    later_sorted = later_sorted.sort_values("date", kind="stable").reset_index(drop=True)

    first_70_count = int(len(later_sorted) * 0.70)
    first_70 = later_sorted.iloc[:first_70_count]
    final_30 = later_sorted.iloc[first_70_count:]

    retrain_frame = pd.concat([train_clean, first_70], ignore_index=True)
    X_retrain = retrain_frame[later_features]
    y_retrain = retrain_frame[target]

    v2 = build_pipeline(num_features, cat_features)
    v2.fit(X_retrain, y_retrain)

    X_check = final_30[later_features]
    y_check = final_30[target]
    v1_check_predictions = v1.predict(X_check)
    v2_check_predictions = v2.predict(X_check)

    mae_v1_check = float(mean_absolute_error(y_check, v1_check_predictions))
    mae_v2_check = float(mean_absolute_error(y_check, v2_check_predictions))
    promote = mae_v2_check <= 0.85 * mae_v1_check

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(params["tracking"]["experiment_name"])

    with mlflow.start_run(run_name="Ridge_v2_Stage4") as run:
        mlflow.log_param("data_sha256", __import__("src.bakery_sales.train", fromlist=["compute_sha256"]).compute_sha256(params["data"]["raw_train_path"]))
        mlflow.log_params({"model_type": "Ridge", "alpha": 1.0, "stage": 4})
        mlflow.log_metric("mae", mae_v2_check)
        mlflow.log_metric("mae_v2_check", mae_v2_check)
        mlflow.log_metric("mae_v1_check", mae_v1_check)
        mlflow.log_metric("promotion_threshold", 0.85 * mae_v1_check)
        mlflow.sklearn.log_model(
            sk_model=v2,
            artifact_path="model",
            serialization_format="cloudpickle",
        )
        v2_run_id = run.info.run_id

    registered = mlflow.register_model(
        model_uri=f"runs:/{v2_run_id}/model",
        name=REGISTERED_MODEL_NAME,
    )

    champion = v1
    champion_version = 1
    if promote:
        MlflowClient().set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=registered.version,
        )
        champion = v2
        champion_version = int(registered.version)

    model_output = root / params["tracking"]["model_output_path"]
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, model_output)

    holdout_features = holdout[num_features + cat_features]
    holdout_predictions = champion.predict(holdout_features)
    predictions_dir = root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "record_id": holdout["record_id"],
            "prediction": holdout_predictions,
        }
    ).to_csv(predictions_dir / "holdout_predictions.csv", index=False)

    monitoring = {
        "psi": psi,
        "most_drifted_feature": most_drifted,
        "mae_by_group": {outlet: mae_by_group[outlet] for outlet in OUTLETS},
        "worst_served_group": worst_group,
        "mae_v1_on_check": mae_v1_check,
        "mae_v2_on_check": mae_v2_check,
        "decision": "promote" if promote else "keep",
    }
    update_monitoring_findings(root / "findings.yaml", monitoring)

    return {
        **monitoring,
        "v2_run_id": v2_run_id,
        "v2_registered_version": int(registered.version),
        "champion_version": champion_version,
        "first_70_count": first_70_count,
        "final_30_count": len(final_30),
    }


if __name__ == "__main__":
    print(main())
