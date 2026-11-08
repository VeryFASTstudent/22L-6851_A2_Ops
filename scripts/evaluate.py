"""Evaluate a saved classifier and write metrics, curves, and error-analysis artifacts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="artifacts/evaluation")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        str(path)
        for name in ("RealArt", "AiArtData")
        for path in sorted(Path(args.data, name).glob("*"))
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    ]
    labels = np.array([0 if Path(path).parent.name == "RealArt" else 1 for path in paths])
    _, paths, _, labels = train_test_split(paths, labels, test_size=0.2, random_state=42, stratify=labels)

    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(image, (224, 224)), label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels)).map(load).batch(32)
    model = tf.keras.models.load_model(args.model)
    probabilities = model.predict(dataset, verbose=0).reshape(-1)
    labels = np.concatenate([labels.numpy() for _, labels in dataset])
    thresholds = np.linspace(0.1, 0.9, 81)
    threshold = float(max(thresholds, key=lambda value: f1_score(labels, probabilities >= value)))
    predictions = probabilities >= threshold
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savetxt(
        output / "predictions.csv",
        np.column_stack([labels, probabilities, predictions.astype(int)]),
        delimiter=",",
        header="label,probability,prediction",
        comments="",
    )
    matrix = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Real", "AI"])
    plt.yticks([0, 1], ["Real", "AI"])
    for (row, column), value in np.ndenumerate(matrix):
        plt.text(column, row, value, ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close()
    false_indices = np.where(labels != predictions)[0]
    with (output / "misclassified.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "probability", "prediction"])
        writer.writerows(
            (int(index), int(labels[index]), float(probabilities[index]), int(predictions[index]))
            for index in false_indices
        )
    fpr, tpr, _ = roc_curve(labels, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "roc_curve.png", dpi=150)
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
