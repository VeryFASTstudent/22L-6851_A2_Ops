"""Run a training command under MLflow; credentials are supplied by environment variables."""
import os, subprocess, sys
import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "VisionGuard"))
with mlflow.start_run() as run:
    mlflow.set_tags({"seed": "42", "project": "VisionGuard"})
    mlflow.log_params({"command": " ".join(sys.argv[1:])})
    result = subprocess.run(sys.argv[1:], check=False)
    mlflow.log_metric("process_exit_code", result.returncode)
    if result.returncode: raise SystemExit(result.returncode)
    print(run.info.run_id)

