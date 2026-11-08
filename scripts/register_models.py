"""Register both trained Keras models in MLflow and assign the best one champion."""
import os
from pathlib import Path

import mlflow
import tensorflow as tf
from mlflow import MlflowClient


MODEL_NAME = "VisionGuard"


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("Set MLFLOW_TRACKING_URI before running this script.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("VisionGuard")
    client = MlflowClient(tracking_uri=tracking_uri)

    models = [
        ("baseline", Path("models/baseline.keras"), 0.6530),
        ("transfer", Path("models/transfer.keras"), 0.5608),
    ]

    for model_type, model_path, auc in models:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = tf.keras.models.load_model(model_path)
        with mlflow.start_run(run_name=f"registry-{model_type}") as run:
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("model_path", str(model_path))
            mlflow.log_param("seed", 42)
            mlflow.log_param("dataset_version", "data/dataset.dvc")
            mlflow.log_metric("roc_auc", auc)
            mlflow.set_tag("project", "VisionGuard")
            mlflow.set_tag("registered_candidate", "true")
            mlflow.tensorflow.log_model(
                model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )
            print(f"Registered {model_type}: run_id={run.info.run_id}")

    versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
    if len(versions) < 2:
        raise RuntimeError(f"Expected at least two versions of {MODEL_NAME}; found {len(versions)}.")

    baseline_versions = [
        version for version in versions
        if version.tags.get("model_type") == "baseline"
    ]
    if baseline_versions:
        champion = max(baseline_versions, key=lambda version: int(version.version))
    else:
        champion = max(versions, key=lambda version: int(version.version))

    client.set_registered_model_alias(MODEL_NAME, "champion", champion.version)
    print(f"Registered model: {MODEL_NAME}")
    print(f"Champion version: {champion.version}")
    print(f"Model URI: models:/{MODEL_NAME}@champion")


if __name__ == "__main__":
    main()
