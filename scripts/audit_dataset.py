"""Audit an image folder and write a reproducible CSV plus summary."""
import argparse, csv, hashlib
from collections import Counter
from pathlib import Path
from PIL import Image, UnidentifiedImageError

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

def audit(root):
    rows, hashes, corrupt = [], Counter(), []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in EXTENSIONS: continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            with Image.open(path) as image:
                row = {"class": path.parent.name, "file": str(path), "size_bytes": path.stat().st_size,
                       "width": image.width, "height": image.height, "mode": image.mode,
                       "format": image.format, "sha256": digest, "error": ""}
        except (UnidentifiedImageError, OSError) as exc:
            corrupt.append(str(path)); row = {"class": path.parent.name, "file": str(path),
                "size_bytes": path.stat().st_size, "width": "", "height": "", "mode": "",
                "format": "", "sha256": digest, "error": str(exc)}
        rows.append(row); hashes[digest] += 1
    return rows, corrupt, sum(count > 1 for count in hashes.values())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("dataset"); parser.add_argument("--output", default="data/audit.csv")
    args = parser.parse_args(); rows, corrupt, duplicates = audit(args.dataset)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else ["class"]); writer.writeheader(); writer.writerows(rows)
    print({"images": len(rows), "corrupt": len(corrupt), "duplicate_hashes": duplicates})

