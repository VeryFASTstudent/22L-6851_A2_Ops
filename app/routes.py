from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from .db import add_prediction, list_predictions

bp = Blueprint("main", __name__)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif"}


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.get("/")
def index():
    return render_template("index.html")


@bp.post("/predict")
def predict():
    uploaded = request.files.get("image")
    if not uploaded or not uploaded.filename:
        flash("Choose an image to upload.", "error")
        return redirect(url_for("main.index"))
    filename = secure_filename(uploaded.filename)
    if not allowed(filename):
        flash("Unsupported file type. Use JPG, PNG, WEBP, or GIF.", "error")
        return redirect(url_for("main.index"))
    inference = current_app.extensions.get("inference")
    if inference is None:
        flash(current_app.extensions.get("inference_error", "Model is unavailable."), "error")
        return redirect(url_for("main.index"))
    try:
        label, confidence = inference.predict(uploaded.stream)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.index"))
    add_prediction(
        current_app.config["DATABASE_URL"], filename, label, confidence, inference.model_name, inference.model_version
    )
    return render_template(
        "result.html", label=label, confidence=confidence, filename=filename, model_name=inference.model_name
    )


@bp.get("/history")
def history():
    return render_template("history.html", predictions=list_predictions(current_app.config["DATABASE_URL"]))


@bp.get("/health")
def health():
    inference = current_app.extensions.get("inference")
    status = "ok" if inference else "degraded"
    return {"status": status, "model_loaded": inference is not None}, (200 if inference else 503)
