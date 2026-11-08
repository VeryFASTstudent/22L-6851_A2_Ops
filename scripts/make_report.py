"""Build the assignment PDF from the generated local artifacts."""
import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)


ROOT = Path(__file__).resolve().parents[1]


def image(path, width=6.4 * inch):
    item = Image(str(ROOT / path))
    item.drawWidth = width
    item.drawHeight = width * item.imageHeight / item.imageWidth
    return item


def build():
    output = ROOT / "22L-6851_VisionGuard.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Cover", parent=styles["Title"], alignment=TA_CENTER, fontSize=24, leading=30))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=45, leftMargin=45, topMargin=42, bottomMargin=42)
    story = [
        Paragraph("VisionGuard", styles["Cover"]),
        Spacer(1, 18),
        Paragraph("AI-Generated Image Classification and Deployment", styles["Heading2"]),
        Paragraph("Student: Hassan Musa<br/>Roll number: 22L-6851<br/>GitHub: https://github.com/VeryFASTstudent/22L-6851_A2_Ops<br/>DagsHub: https://dagshub.com/VeryFASTstudent/22L-6851_A2_Ops<br/>Live URL: To be added after cloud deployment", styles["BodyText"]),
        Spacer(1, 18),
        Paragraph("This report documents the reproducible local implementation, dataset audit, model training, evaluation artifacts, experiment tooling, Flask service, database trail, and container configuration. Cloud credentials, public deployment, and account screenshots remain operator-owned evidence.", styles["BodyText"]),
        PageBreak(),
        Paragraph("1. Environment and dataset", styles["Heading1"]),
        Paragraph("The project uses Python 3.11, TensorFlow 2.18.0, Flask 3.1.0, MLflow 2.19.0, and CPU execution. The dataset was copied from the supplied archive into data/dataset and tracked with DVC through data/dataset.dvc.", styles["BodyText"]),
    ]
    with (ROOT / "artifacts/eda/summary.csv").open(encoding="utf-8") as file:
        rows = list(csv.reader(file))
    story += [Spacer(1, 10), Table(rows, repeatRows=1, style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#315efb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7), ("ALIGN", (0, 0), (-1, -1), "CENTER")
    ])), PageBreak()]
    story += [
        Paragraph("2. Visual EDA", styles["Heading1"]),
        Paragraph("The supplied data contains 539 AI-art images and 435 readable real-art images in the generated EDA summary. Width, height, and file-size distributions are broad, so the input pipeline resizes images to 224x224 while preserving three color channels. Resizing standardizes the tensor shape but can distort aspect ratio; this is a documented limitation.", styles["BodyText"]),
        image("artifacts/eda/sample_grid.png"), Paragraph("Figure 1. Representative images from both classes.", styles["Small"]),
        image("artifacts/eda/widths.png"), Paragraph("Figure 2. Width distribution by class.", styles["Small"]),
        PageBreak(), image("artifacts/eda/heights.png"), Paragraph("Figure 3. Height distribution by class.", styles["Small"]),
        image("artifacts/eda/file_sizes.png"), Paragraph("Figure 4. File-size distribution by class.", styles["Small"]),
        PageBreak(),
        Paragraph("3. Models and evaluation", styles["Heading1"]),
        Paragraph("The baseline model is a custom CNN with convolution, batch normalization, pooling, global average pooling, dropout, and a sigmoid output. A MobileNetV2 transfer-learning model was also trained. Training uses a fixed seed, augmentation, early stopping, learning-rate reduction, and checkpointing. A later correction replaced directory-tail validation with a stratified file-level split so both classes are represented.", styles["BodyText"]),
    ]
    metrics = json.loads((ROOT / "artifacts/evaluation/metrics.json").read_text(encoding="utf-8"))
    metric_rows = [["Metric", "Champion result"]] + [[key, f"{value:.4f}" if isinstance(value, float) else str(value)] for key, value in metrics.items()]
    story += [Spacer(1, 10), Table(metric_rows, style=TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#315efb")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .25, colors.grey)])), image("artifacts/evaluation/confusion_matrix.png", 3.2 * inch), image("artifacts/evaluation/roc_curve.png", 5 * inch), PageBreak()]
    story += [
        Paragraph("4. Tuning and experiment tracking", styles["Heading1"]),
        Paragraph("The Keras Tuner toy search completed eight Fashion-MNIST trials and recorded the best units and learning rate in artifacts/tuner/best_hyperparameters.txt. The MLflow comparison script logs model paths, image size, seed, dataset pointer, evaluation metrics, and JSON artifacts. Set MLFLOW_TRACKING_URI to the DagsHub MLflow endpoint to repeat the same runs remotely.", styles["BodyText"]),
        Paragraph("Best toy tuner settings: " + (ROOT / "artifacts/tuner/best_hyperparameters.txt").read_text(encoding="utf-8"), styles["BodyText"]),
        Paragraph("5. Serving and operations", styles["Heading1"]),
        Paragraph("The Flask service loads the champion model once at startup, validates extensions and decodability, does not persist uploaded images, records every successful prediction, and exposes /health and /history. Docker Compose provides a web service and an internal PostgreSQL service with a named volume and health-gated startup. GitHub Actions runs tests, linting, and a Docker build.", styles["BodyText"]),
    ]
    document.build(story)
    print(output)


if __name__ == "__main__":
    build()
