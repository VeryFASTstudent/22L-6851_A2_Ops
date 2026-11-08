"""Train baseline and transfer-learning models with reproducible callbacks."""

import argparse
import os
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42


def seed_everything():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def make_model(kind="baseline", image_size=224, dropout=0.3, unfrozen=0):
    inputs = tf.keras.Input((image_size, image_size, 3))
    augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.08)])
    x = augmentation(inputs)
    if kind == "transfer":
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
        )
        backbone.trainable = unfrozen > 0
        if unfrozen:
            for layer in backbone.layers[:-unfrozen]:
                layer.trainable = False
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
        x = backbone(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        for filters in (32, 64, 128):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(1, activation="sigmoid")(x))


def make_dataset(paths, labels, training):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image = tf.image.resize(image, (224, 224))
        return tf.cast(image, tf.float32), tf.cast(label, tf.float32)

    return dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/champion.keras")
    parser.add_argument("--kind", choices=("baseline", "transfer"), default="transfer")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    seed_everything()
    class_names = ["RealArt", "AiArtData"]
    paths = [
        str(path)
        for name in class_names
        for path in sorted(Path(args.data, name).glob("*"))
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    ]
    labels = [0 if Path(path).parent.name == "RealArt" else 1 for path in paths]
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    train = make_dataset(train_paths, train_labels, True)
    val = make_dataset(val_paths, val_labels, False)
    model = make_model(args.kind)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        train.prefetch(tf.data.AUTOTUNE),
        validation_data=val.prefetch(tf.data.AUTOTUNE),
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2),
            tf.keras.callbacks.ModelCheckpoint(args.output, save_best_only=True),
        ],
    )


if __name__ == "__main__":
    main()
