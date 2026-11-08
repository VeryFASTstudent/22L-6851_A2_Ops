import os
from pathlib import Path

from flask import Flask

from .db import init_db
from .inference import ModelInference, ModelLoadError
from .routes import bp


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True,
                template_folder="../templates", static_folder="../static")
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "change-me-in-production"),
        MAX_CONTENT_LENGTH=int(os.getenv("MAX_CONTENT_LENGTH", str(10 * 1024 * 1024))),
        DATABASE_URL=os.getenv("DATABASE_URL", "sqlite:///data/visionguard.db"),
    )
    if test_config:
        app.config.update(test_config)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    init_db(app.config["DATABASE_URL"])
    try:
        app.extensions["inference"] = ModelInference.from_environment()
    except ModelLoadError as exc:
        app.extensions["inference_error"] = str(exc)
        app.extensions["inference"] = None
    app.register_blueprint(bp)
    return app
