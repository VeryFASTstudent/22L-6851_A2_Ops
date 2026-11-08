"""Log comparable local model evaluations to MLflow."""
import argparse
import json
import os
from pathlib import Path

import mlflow
import tensorflow as tf
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", default="artifacts/mlflow")
    args = parser.parse_args()
    paths = [str(path) for name in ("RealArt", "AiArtData") for path in sorted(Path(args.data, name).glob("*"))
             if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}]
    labels = [0 if Path(path).parent.name == "RealArt" else 1 for path in paths]
    _, paths, _, labels = train_test_split(paths, labels, test_size=.2, random_state=42, stratify=labels)
    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(image, (224, 224)), label
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels)).map(load).batch(32)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "VisionGuard"))
    for model_path in args.models:
        with mlflow.start_run(run_name=Path(model_path).stem) as run:
            model = tf.keras.models.load_model(model_path)
            metrics = model.evaluate(dataset, verbose=0, return_dict=True)
            mlflow.log_params({"model_path": model_path, "image_size": 224, "seed": 42,
                               "dataset_version": "data/dataset.dvc"})
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
            metric_file = output / f"{Path(model_path).stem}.json"
            metric_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(metric_file))
            print(run.info.run_id, model_path, metrics)


if __name__ == "__main__":
    main()
