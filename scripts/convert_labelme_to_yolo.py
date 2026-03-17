"""Convert LabelMe JSON annotations to YOLO segment format."""

import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CLASS_MAP = {"tail": 0}


def convert_one(json_path: Path) -> str | None:
    """Convert a single LabelMe JSON to YOLO segment line(s)."""
    with open(json_path) as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in CLASS_MAP:
            print(f"  Skipping unknown label '{label}' in {json_path.name}")
            continue

        if shape["shape_type"] != "polygon":
            print(f"  Skipping non-polygon shape in {json_path.name}")
            continue

        class_id = CLASS_MAP[label]
        coords = []
        for x, y in shape["points"]:
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")

        lines.append(f"{class_id} " + " ".join(coords))

    return "\n".join(lines) if lines else None


def main():
    json_files = sorted(RAW_DIR.glob("*.json"))
    print(f"Found {len(json_files)} annotation files in {RAW_DIR}")

    converted = 0
    for json_path in json_files:
        result = convert_one(json_path)
        if result is None:
            print(f"  No valid shapes in {json_path.name}, skipping")
            continue

        txt_path = json_path.with_suffix(".txt")
        txt_path.write_text(result)
        converted += 1

    print(f"Converted {converted}/{len(json_files)} files")


if __name__ == "__main__":
    main()
