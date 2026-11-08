import os
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


class ModelLoadError(RuntimeError):
    pass


class ModelInference:
    def __init__(self, model, threshold, model_name, model_version, image_size=(224, 224)):
        self.model = model
        self.threshold = threshold
        self.model_name = model_name
        self.model_version = model_version
        self.image_size = image_size

    @classmethod
    def from_environment(cls, configured_path=None):
        model_path = Path(configured_path or os.getenv("MODEL_PATH", "models/champion.keras"))
        if not model_path.exists():
            raise ModelLoadError(f"Champion model not found at {model_path}. Train a model and set MODEL_PATH.")
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(model_path)
        except Exception as exc:
            raise ModelLoadError(f"Could not load champion model: {exc}") from exc
        return cls(
            model,
            float(os.getenv("MODEL_THRESHOLD", "0.5")),
            os.getenv("MODEL_NAME", "visionguard"),
            os.getenv("MODEL_VERSION", "local"),
        )

    def predict(self, image_stream):
        try:
            image = Image.open(image_stream).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The uploaded file is not a readable image.") from exc
        array = np.asarray(image.resize(self.image_size), dtype=np.float32) / 255.0
        probability = float(np.asarray(self.model.predict(array[None, ...], verbose=0)).reshape(-1)[0])
        label = "AI-generated" if probability >= self.threshold else "Real"
        confidence = probability if label == "AI-generated" else 1.0 - probability
        return label, confidence
