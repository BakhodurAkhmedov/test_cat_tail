"""Run inference on images/video using trained tail segmentation model."""

import argparse
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path(__file__).resolve().parent / "weights/best.pt"


def main():
    parser = argparse.ArgumentParser(description="Cat tail segmentation inference")
    parser.add_argument("--source", required=True, help="Image, directory, or video path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS), help="Model weights path")
    args = parser.parse_args()

    model = YOLO(args.weights)

    save_dir = "runs/predict/result"

    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
        project="runs/predict",
        name="result",
        exist_ok=True,
    )

    print(f"\nProcessed {len(results)} image(s)")
    print(f"Results saved to {save_dir}/")


if __name__ == "__main__":
    main()
