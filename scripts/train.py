"""Train baseline and transfer-learning models with reproducible callbacks."""
import argparse, os, random
from pathlib import Path
import numpy as np
import tensorflow as tf

SEED = 42
def seed_everything():
    os.environ["PYTHONHASHSEED"] = str(SEED); random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

def make_model(kind="baseline", image_size=224, dropout=.3, unfrozen=0):
    inputs = tf.keras.Input((image_size, image_size, 3))
    augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(.08)])
    x = augmentation(inputs)
    if kind == "transfer":
        backbone = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(image_size,image_size,3))
        backbone.trainable = unfrozen > 0
        if unfrozen:
            for layer in backbone.layers[:-unfrozen]: layer.trainable = False
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
        x = backbone(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        for filters in (32, 64, 128):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(1, activation="sigmoid")(x))

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data", required=True); parser.add_argument("--output", default="models/champion.keras")
    parser.add_argument("--kind", choices=("baseline","transfer"), default="transfer"); parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args(); seed_everything()
    class_names = ["RealArt", "AiArtData"]
    train = tf.keras.utils.image_dataset_from_directory(
        args.data, validation_split=.2, subset="training", seed=SEED,
        class_names=class_names, image_size=(224, 224), batch_size=32
    )
    val = tf.keras.utils.image_dataset_from_directory(
        args.data, validation_split=.2, subset="validation", seed=SEED,
        class_names=class_names, image_size=(224, 224), batch_size=32
    )
    model = make_model(args.kind); model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.fit(train.prefetch(tf.data.AUTOTUNE), validation_data=val.prefetch(tf.data.AUTOTUNE), epochs=args.epochs,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(patience=2), tf.keras.callbacks.ModelCheckpoint(args.output, save_best_only=True)])

if __name__ == "__main__": main()
