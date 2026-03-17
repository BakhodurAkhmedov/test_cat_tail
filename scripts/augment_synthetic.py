"""Generate synthetic training data by pasting tail cutouts onto random backgrounds."""

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_IMAGES = DATA_DIR / "images" / "train"
TRAIN_LABELS = DATA_DIR / "labels" / "train"
BG_DIR = DATA_DIR / "backgrounds"
NUM_SYNTHETIC = 50
SEED = 42
IMG_SIZE = 640


def load_tail_cutout(img_path: Path, txt_path: Path):
    """Extract tail region from image using YOLO polygon label."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None

    h, w = img.shape[:2]

    with open(txt_path) as f:
        line = f.readline().strip()

    parts = line.split()
    coords = list(map(float, parts[1:]))

    # Convert normalized coords to pixel coords
    points = []
    for i in range(0, len(coords), 2):
        px = int(coords[i] * w)
        py = int(coords[i + 1] * h)
        points.append([px, py])
    polygon = np.array(points, dtype=np.int32)

    # Create mask from polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # Crop to bounding box of the tail
    x, y, bw, bh = cv2.boundingRect(polygon)
    # Add small padding
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)

    tail_crop = img[y : y + bh, x : x + bw]
    mask_crop = mask[y : y + bh, x : x + bw]

    # Shift polygon to crop coordinates
    polygon_crop = polygon - np.array([x, y])

    return tail_crop, mask_crop, polygon_crop


def get_backgrounds():
    """Load background images. If no backgrounds dir, generate solid color/noise backgrounds."""
    backgrounds = []

    if BG_DIR.exists():
        for p in sorted(BG_DIR.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                bg = cv2.imread(str(p))
                if bg is not None:
                    backgrounds.append(bg)

    # Always add some generated backgrounds as fallback
    for _ in range(10):
        # Random solid color
        color = [random.randint(50, 230) for _ in range(3)]
        bg = np.full((IMG_SIZE, IMG_SIZE, 3), color, dtype=np.uint8)
        # Add gaussian noise for texture
        noise = np.random.normal(0, 15, bg.shape).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        backgrounds.append(bg)

    return backgrounds


# Augmentation for tail cutouts
tail_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.RandomScale(scale_limit=(-0.3, 0.3), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    ]
)


def paste_tail_on_background(tail_crop, mask_crop, background):
    """Paste augmented tail onto resized background, return image and YOLO label."""
    # Resize background
    bg = cv2.resize(background, (IMG_SIZE, IMG_SIZE))

    # Augment tail
    augmented = tail_transform(image=tail_crop, mask=mask_crop)
    aug_tail = augmented["image"]
    aug_mask = augmented["mask"]

    th, tw = aug_tail.shape[:2]

    # Skip if tail is too large for background
    if th >= IMG_SIZE or tw >= IMG_SIZE:
        # Scale down
        scale = min((IMG_SIZE - 20) / th, (IMG_SIZE - 20) / tw)
        aug_tail = cv2.resize(aug_tail, (int(tw * scale), int(th * scale)))
        aug_mask = cv2.resize(aug_mask, (int(tw * scale), int(th * scale)))
        th, tw = aug_tail.shape[:2]

    # Random position
    max_x = IMG_SIZE - tw
    max_y = IMG_SIZE - th
    if max_x <= 0 or max_y <= 0:
        return None, None
    off_x = random.randint(0, max_x)
    off_y = random.randint(0, max_y)

    # Paste using mask
    roi = bg[off_y : off_y + th, off_x : off_x + tw]
    mask_3ch = cv2.merge([aug_mask, aug_mask, aug_mask])
    mask_norm = mask_3ch.astype(np.float32) / 255.0
    blended = (aug_tail * mask_norm + roi * (1 - mask_norm)).astype(np.uint8)
    bg[off_y : off_y + th, off_x : off_x + tw] = blended

    # Generate YOLO label from pasted mask
    contours, _ = cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Take largest contour
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None, None

    # Simplify contour a bit
    epsilon = 0.01 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    if len(contour) < 3:
        return None, None

    # Convert to normalized YOLO coords
    coords = []
    for pt in contour.reshape(-1, 2):
        nx = (pt[0] + off_x) / IMG_SIZE
        ny = (pt[1] + off_y) / IMG_SIZE
        coords.append(f"{nx:.6f}")
        coords.append(f"{ny:.6f}")

    label = "0 " + " ".join(coords)
    return bg, label


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Collect all train image-label pairs
    pairs = []
    for img_path in sorted(TRAIN_IMAGES.iterdir()):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            txt_path = TRAIN_LABELS / img_path.with_suffix(".txt").name
            if txt_path.exists():
                pairs.append((img_path, txt_path))

    if not pairs:
        print("No training pairs found!")
        return

    backgrounds = get_backgrounds()
    print(f"Source images: {len(pairs)}, Backgrounds: {len(backgrounds)}")

    generated = 0
    attempts = 0
    max_attempts = NUM_SYNTHETIC * 5

    while generated < NUM_SYNTHETIC and attempts < max_attempts:
        attempts += 1

        img_path, txt_path = random.choice(pairs)
        result = load_tail_cutout(img_path, txt_path)
        if result[0] is None:
            continue
        tail_crop, mask_crop, _ = result

        bg = random.choice(backgrounds)
        synth_img, synth_label = paste_tail_on_background(tail_crop, mask_crop, bg)
        if synth_img is None:
            continue

        # Save
        name = f"synthetic_{generated:04d}"
        cv2.imwrite(str(TRAIN_IMAGES / f"{name}.jpg"), synth_img)
        (TRAIN_LABELS / f"{name}.txt").write_text(synth_label)
        generated += 1

    print(f"Generated {generated} synthetic images ({attempts} attempts)")


if __name__ == "__main__":
    main()
