"""Draw YOLO boxes on sample images for visual dataset QA."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_names(data_yaml: Path) -> list[str]:
    """Read class names from a simple YOLO data.yaml file."""
    import yaml

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def yolo_to_xyxy(parts: list[str], width: int, height: int) -> tuple[float, float, float, float]:
    """Convert normalized YOLO xywh values to absolute xyxy."""
    _, x_center, y_center, box_width, box_height = [float(part) for part in parts[:5]]
    abs_width = box_width * width
    abs_height = box_height * height
    center_x = x_center * width
    center_y = y_center * height
    return (
        center_x - abs_width / 2,
        center_y - abs_height / 2,
        center_x + abs_width / 2,
        center_y + abs_height / 2,
    )


def draw_sample(image_path: Path, label_path: Path, names: list[str], output_path: Path) -> None:
    """Draw all label boxes for one image."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        label = names[class_id] if class_id < len(names) else str(class_id)
        xyxy = yolo_to_xyxy(parts, *image.size)
        draw.rectangle(xyxy, outline=(220, 38, 38), width=3)
        draw.text((xyxy[0], max(0, xyxy[1] - 12)), label, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def find_images_for_class(
    dataset_dir: Path, class_id: int, split: str, limit: int
) -> list[tuple[Path, Path]]:
    """Find image/label pairs containing a target class ID."""
    label_dir = dataset_dir / split / "labels"
    image_dir = dataset_dir / split / "images"
    pairs: list[tuple[Path, Path]] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        ids = {
            int(float(line.split()[0]))
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.split()
        }
        if class_id not in ids:
            continue
        for suffix in (".jpg", ".jpeg", ".png"):
            image_path = image_dir / f"{label_path.stem}{suffix}"
            if image_path.exists():
                pairs.append((image_path, label_path))
                break
        if len(pairs) >= limit:
            break
    return pairs


def main() -> None:
    """Run the visual QA sampler."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/GENERAL_DIR"))
    parser.add_argument("--data-yaml", type=Path, default=Path("data/GENERAL_DIR/data.yaml"))
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/inspection"))
    args = parser.parse_args()

    names = read_names(args.data_yaml)
    pairs = find_images_for_class(args.dataset_dir, args.class_id, args.split, args.limit)
    for index, (image_path, label_path) in enumerate(pairs, start=1):
        output_path = (
            args.output_dir / f"{args.dataset_dir.name}_class_{args.class_id}_{index:02d}.jpg"
        )
        draw_sample(image_path, label_path, names, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
