# VisionGuard

VisionGuard is an end-to-end AI-generated-image classifier: TensorFlow/Keras training, MLflow experiment tracking, Flask inference, SQLite/PostgreSQL-ready request history, Docker Compose, and GitHub Actions.

## Run locally

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m flask --app app run --port 8000
```

The UI is at `http://localhost:8000`. Without `models/champion.keras`, `/health` is intentionally `503` and predictions are disabled. Train one with `python scripts/train.py --data path\to\dataset`.

## Dataset and tracking

`python scripts/audit_dataset.py path\to\dataset --output data\audit.csv` records dimensions, mode, format, size, SHA-256, corrupt files, and duplicate hashes. Run training through `python scripts/track_run.py python scripts/train.py ...`; set `MLFLOW_TRACKING_URI` to the DagsHub MLflow endpoint and provide credentials through environment variables, never source code.

## Docker

Create `.env` with `POSTGRES_PASSWORD`, then run `docker compose up --build`. PostgreSQL is internal-only; the web service is published on port 8000. The model is baked into the image from `models/champion.keras` for deterministic deployment. Do not commit credentials or model artifacts to Git.

## Deployment

Deploy the Compose stack on an Ubuntu VM or managed container host, open only TCP 80/443 (reverse proxy to 8000), set environment secrets in the provider secret manager, and verify `/health` from a second network. A live URL and DagsHub/GitHub links are account-specific deliverables and must be added after provisioning.
