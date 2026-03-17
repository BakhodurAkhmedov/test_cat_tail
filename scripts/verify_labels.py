"""Visualize YOLO segment labels overlaid on images for QA."""

from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VERIFY_DIR = DATA_DIR / "verify"
COLOR = (0, 255, 0)  # Green
ALPHA = 0.4


def draw_label(img_path: Path, txt_path: Path, out_path: Path):
    """Draw polygon masks on image and save."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Cannot read {img_path.name}")
        return False

    h, w = img.shape[:2]
    overlay = img.copy()

    with open(txt_path) as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        parts = line.split()
        coords = list(map(float, parts[1:]))

        points = []
        for i in range(0, len(coords), 2):
            px = int(coords[i] * w)
            py = int(coords[i + 1] * h)
            points.append([px, py])
        polygon = np.array(points, dtype=np.int32)

        # Fill polygon
        cv2.fillPoly(overlay, [polygon], COLOR)
        # Draw contour
        cv2.polylines(img, [polygon], True, COLOR, 2)

    # Blend
    result = cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0)

    # Add label info
    cv2.putText(result, f"{len(lines)} tail(s)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2)

    cv2.imwrite(str(out_path), result)
    return True


def main():
    VERIFY_DIR.mkdir(exist_ok=True)

    total = 0
    verified = 0

    for split in ["train", "val"]:
        img_dir = DATA_DIR / "images" / split
        lbl_dir = DATA_DIR / "labels" / split

        for txt_path in sorted(lbl_dir.glob("*.txt")):
            # Find matching image
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = img_dir / (txt_path.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                print(f"  No image for {txt_path.name}")
                continue

            total += 1
            out_name = f"{split}_{img_path.name}"
            if draw_label(img_path, txt_path, VERIFY_DIR / out_name):
                verified += 1

    print(f"Verified {verified}/{total} images -> data/verify/")


if __name__ == "__main__":
    main()
