"""
Merge the 'labcoat' and 'general' Roboflow PPE datasets into one unified
8-class dataset for YOLO fine-tuning.

Usage:
    python merge_datasets.py

Expects the two source datasets already downloaded/extracted with their
standard Roboflow YOLO export structure:

    <SRC>/train/images/*.jpg
    <SRC>/train/labels/*.txt
    <SRC>/valid/images/*.jpg
    <SRC>/valid/labels/*.txt
    <SRC>/test/images/*.jpg
    <SRC>/test/labels/*.txt

Edit LABCOAT_DIR and GENERAL_DIR below to point at your two extracted
dataset folders, and OUTPUT_DIR to where you want the merged dataset written.
"""

import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. EDIT THESE PATHS
# ---------------------------------------------------------------------------
LABCOAT_DIR = Path("data/LABCOAT_DIR")
GENERAL_DIR = Path("data/GENERAL_DIR")
OUTPUT_DIR = Path("data/processed")     # merged output goes here

# ---------------------------------------------------------------------------
# 2. CANONICAL SCHEMA (do not change order — indices matter)
# ---------------------------------------------------------------------------
CANONICAL_NAMES = [
    "person",          # 0
    "helmet",          # 1
    "safety_vest",     # 2
    "gloves",          # 3
    "safety_boots",    # 4
    "safety_goggles",  # 5
    "face_mask",       # 6
    "lab_coat",        # 7
]

# ---------------------------------------------------------------------------
# 3. REMAP TABLES: old_class_id -> new_class_id
#    Any old_class_id NOT present in these dicts is DROPPED (its boxes are
#    removed from the label file, e.g. all the "no_*" negative classes).
# ---------------------------------------------------------------------------
LABCOAT_REMAP = {
    2: 3,   # gloves       -> gloves
    3: 5,   # goggle       -> safety_goggles
    4: 5,   # goggles      -> safety_goggles
    5: 3,   # hand gloves  -> gloves
    7: 7,   # lab coat     -> lab_coat
    8: 7,   # labcoat      -> lab_coat
    9: 6,   # mask         -> face_mask
    # dropped: 0 Headcap, 1 full cover, 6 head cap, 10-18 all "no_*"/shoe cover
}

GENERAL_REMAP = {
    0: 3,   # Gloves       -> gloves
    1: 1,   # Helmet       -> helmet
    2: 0,   # Human        -> person
    3: 4,   # Safety Boot  -> safety_boots
    4: 2,   # Safety Vest  -> safety_vest
    5: 4,   # boots        -> safety_boots
    6: 5,   # glasses      -> safety_goggles  (verify visually — see notes)
    7: 3,   # gloves       -> gloves
    8: 1,   # hat          -> helmet          (verify visually — see notes)
    9: 1,   # helmet       -> helmet
    15: 2,  # vest         -> safety_vest
    # dropped: 10-14 all "no_*"
}

SPLITS = ["train", "valid", "test"]


def remap_label_file(src_label_path: Path, remap: dict) -> list[str]:
    """Read a YOLO label file and return remapped lines (dropped classes removed)."""
    if not src_label_path.exists():
        return []
    new_lines = []
    for line in src_label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_id = int(parts[0])
        if old_id in remap:
            new_id = remap[old_id]
            new_lines.append(" ".join([str(new_id)] + parts[1:]))
        # else: class not in remap -> silently dropped
    return new_lines


def merge_source(src_dir: Path, remap: dict, prefix: str, out_dir: Path):
    """Copy + remap one source dataset into the merged output, prefixing filenames
    to avoid collisions between the two sources."""
    copied, empty = 0, 0
    for split in SPLITS:
        img_dir = src_dir / split / "images"
        lbl_dir = src_dir / split / "labels"
        if not img_dir.exists():
            print(f"  [skip] {img_dir} not found")
            continue

        out_img_dir = out_dir / split / "images"
        out_lbl_dir = out_dir / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = img_path.stem
            new_stem = f"{prefix}_{stem}"

            label_path = lbl_dir / f"{stem}.txt"
            new_lines = remap_label_file(label_path, remap)

            if not new_lines:
                empty += 1
                # still copy the image only if you want background/negative
                # images in training; comment out the next line to skip them
                # continue

            shutil.copy(img_path, out_img_dir / f"{new_stem}{img_path.suffix}")
            (out_lbl_dir / f"{new_stem}.txt").write_text("\n".join(new_lines))
            copied += 1

    print(f"  {prefix}: copied {copied} images ({empty} ended up with zero boxes after remap)")


def write_data_yaml(out_dir: Path):
    yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(CANONICAL_NAMES)}
names: {CANONICAL_NAMES}
"""
    (out_dir / "data.yaml").write_text(yaml_content)


if __name__ == "__main__":
    print("Merging lab coat dataset...")
    merge_source(LABCOAT_DIR, LABCOAT_REMAP, prefix="labcoat", out_dir=OUTPUT_DIR)

    print("Merging general dataset...")
    merge_source(GENERAL_DIR, GENERAL_REMAP, prefix="general", out_dir=OUTPUT_DIR)

    write_data_yaml(OUTPUT_DIR)
    print(f"\nDone. Merged dataset written to: {OUTPUT_DIR.resolve()}")
    print("Wrote data.yaml with unified 8-class schema.")