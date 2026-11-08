"""Load a registered champion or local fallback and classify one image."""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model-uri", default=None)
    parser.add_argument("--local-model", default="models/champion.keras")
    args = parser.parse_args()
    uri = args.model_uri or os.getenv("MODEL_URI")
    model = tf.keras.models.load_model(uri or args.local_model)
    with Image.open(args.image) as image:
        array = np.asarray(image.convert("RGB").resize((224, 224)), dtype="float32") / 255
    probability = float(model.predict(array[None, ...], verbose=0).reshape(-1)[0])
    print({"label": "AI-generated" if probability >= .5 else "Real",
           "confidence": probability if probability >= .5 else 1 - probability,
           "model": uri or args.local_model})


if __name__ == "__main__":
    main()
