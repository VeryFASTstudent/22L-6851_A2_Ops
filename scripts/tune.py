"""Run toy and VisionGuard Keras Tuner searches.

The VisionGuard search is intentionally bounded; increase --trials on a GPU for the
assignment's full eight-trial run.
"""

import argparse
from pathlib import Path

import tensorflow as tf


def main():
    try:
        import keras_tuner as kt
    except ImportError as exc:
        raise SystemExit("Install keras-tuner before running this script.") from exc
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default="artifacts/tuner")
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.data is None:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train, x_test = x_train[:10000].astype("float32") / 255, x_test[:2000].astype("float32") / 255
        y_train, y_test = y_train[:10000], y_test[:2000]

        def build(hp):
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                    tf.keras.layers.Dense(hp.Int("units", 32, 128, step=32), activation="relu"),
                    tf.keras.layers.Dense(10, activation="softmax"),
                ]
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model

        tuner = kt.RandomSearch(
            build, objective="val_accuracy", max_trials=args.trials, directory=output, project_name="fashion_mnist"
        )
        tuner.search(x_train, y_train, validation_split=0.2, epochs=3, verbose=0)
    else:
        from train import make_model

        dataset = tf.keras.utils.image_dataset_from_directory(
            args.data,
            validation_split=0.2,
            subset="training",
            seed=42,
            class_names=["RealArt", "AiArtData"],
            image_size=(224, 224),
            batch_size=32,
        )
        validation = tf.keras.utils.image_dataset_from_directory(
            args.data,
            validation_split=0.2,
            subset="validation",
            seed=42,
            class_names=["RealArt", "AiArtData"],
            image_size=(224, 224),
            batch_size=32,
        )

        def build(hp):
            model = make_model("baseline", dropout=hp.Float("dropout", 0.1, 0.5, step=0.2))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")),
                loss="binary_crossentropy",
                metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
            )
            return model

        tuner = kt.RandomSearch(
            build, objective="val_auc", max_trials=args.trials, directory=output, project_name="visionguard"
        )
        tuner.search(dataset, validation_data=validation, epochs=2, verbose=0)
    best = tuner.get_best_hyperparameters(1)[0]
    (output / "best_hyperparameters.txt").write_text(str(best.values), encoding="utf-8")
    print(best.values)


if __name__ == "__main__":
    main()
