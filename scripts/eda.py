"""Create reproducible visual EDA artifacts for the image dataset."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError


def collect(root):
    rows = []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
            continue
        try:
            with Image.open(path) as image:
                rows.append((path.parent.name, path, image.width, image.height, path.stat().st_size))
        except (UnidentifiedImageError, OSError):
            continue
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output", default="artifacts/eda")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = collect(args.dataset)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "count", "width_min", "width_median", "width_max",
                         "height_min", "height_median", "height_max", "size_min",
                         "size_median", "size_max"])
        for label in sorted({row[0] for row in rows}):
            selected = [row for row in rows if row[0] == label]
            widths = np.array([row[2] for row in selected])
            heights = np.array([row[3] for row in selected])
            sizes = np.array([row[4] for row in selected])
            writer.writerow([label, len(selected), widths.min(), np.median(widths), widths.max(),
                             heights.min(), np.median(heights), heights.max(),
                             sizes.min(), np.median(sizes), sizes.max()])
    for value, title, filename, xlabel in (
        (2, "Image widths by class", "widths.png", "Width (pixels)"),
        (3, "Image heights by class", "heights.png", "Height (pixels)"),
        (4, "File sizes by class", "file_sizes.png", "File size (bytes)"),
    ):
        plt.figure(figsize=(8, 5))
        for label in sorted({row[0] for row in rows}):
            values = [row[value] for row in rows if row[0] == label]
            plt.hist(values, bins=30, alpha=.55, label=label)
        plt.title(title); plt.xlabel(xlabel); plt.ylabel("Images"); plt.legend()
        plt.tight_layout(); plt.savefig(output / filename, dpi=150); plt.close()
    sample_rows = []
    for label in sorted({row[0] for row in rows}):
        sample_rows.extend([row for row in rows if row[0] == label][:8])
    figure, axes = plt.subplots(2, 8, figsize=(18, 5))
    for axis, row in zip(axes.ravel(), sample_rows):
        with Image.open(row[1]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(row[0]); axis.axis("off")
    figure.tight_layout(); figure.savefig(output / "sample_grid.png", dpi=150); plt.close(figure)
    print(f"EDA artifacts written to {output} ({len(rows)} images)")


if __name__ == "__main__":
    main()
