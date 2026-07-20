"""Merge Roboflow PPE datasets into the canonical 8-class YOLO dataset."""

from __future__ import annotations

import shutil
from pathlib import Path

LABCOAT_DIR = Path("data/LABCOAT_DIR")
GENERAL_DIR = Path("data/GENERAL_DIR")
OUTPUT_DIR = Path("data/processed")

CANONICAL_NAMES = [
    "person",
    "helmet",
    "safety_vest",
    "gloves",
    "safety_boots",
    "safety_goggles",
    "face_mask",
    "lab_coat",
]

LABCOAT_REMAP = {
    2: 3,
    3: 5,
    4: 5,
    5: 3,
    7: 7,
    8: 7,
    9: 6,
}

GENERAL_REMAP = {
    0: 3,
    1: 1,
    2: 0,
    3: 4,
    4: 2,
    5: 4,
    6: 5,
    7: 3,
    8: 1,
    9: 1,
    15: 2,
}

SPLITS = ["train", "valid", "test"]


def remap_label_file(src_label_path: Path, remap: dict[int, int]) -> list[str]:
    """Read a YOLO label file and return remapped label lines."""
    if not src_label_path.exists():
        return []
    new_lines: list[str] = []
    for line in src_label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        old_id = int(float(parts[0]))
        if old_id in remap:
            new_lines.append(" ".join([str(remap[old_id]), *parts[1:]]))
    return new_lines


def merge_source(
    src_dir: Path, remap: dict[int, int], prefix: str, out_dir: Path
) -> tuple[int, int]:
    """Copy and remap one source dataset into the merged output directory."""
    copied = 0
    empty = 0
    for split in SPLITS:
        img_dir = src_dir / split / "images"
        lbl_dir = src_dir / split / "labels"
        if not img_dir.exists():
            continue

        out_img_dir = out_dir / split / "images"
        out_lbl_dir = out_dir / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            new_stem = f"{prefix}_{img_path.stem}"
            new_lines = remap_label_file(lbl_dir / f"{img_path.stem}.txt", remap)
            if not new_lines:
                empty += 1

            shutil.copy2(img_path, out_img_dir / f"{new_stem}{img_path.suffix}")
            (out_lbl_dir / f"{new_stem}.txt").write_text("\n".join(new_lines), encoding="utf-8")
            copied += 1
    return copied, empty


def write_data_yaml(out_dir: Path) -> None:
    """Write the unified YOLO data.yaml."""
    yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(CANONICAL_NAMES)}
names: {CANONICAL_NAMES}
"""
    (out_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")


def main() -> None:
    """Merge all configured sources."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lab_count, lab_empty = merge_source(LABCOAT_DIR, LABCOAT_REMAP, "labcoat", OUTPUT_DIR)
    gen_count, gen_empty = merge_source(GENERAL_DIR, GENERAL_REMAP, "general", OUTPUT_DIR)
    write_data_yaml(OUTPUT_DIR)
    print(f"labcoat: copied {lab_count} images ({lab_empty} zero-box labels)")
    print(f"general: copied {gen_count} images ({gen_empty} zero-box labels)")
    print(f"wrote {OUTPUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
