"""Visualization helpers for annotated PPE detections."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from src.ppe_verification.overlap_rules import Assignment
from src.utils.bbox import BBox, Detection

COLOR_OK = (24, 128, 74)
COLOR_MISSING = (190, 48, 48)
COLOR_OPTIONAL = (105, 111, 122)
COLOR_PERSON = (42, 91, 215)


def _draw_label(
    draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: tuple[int, int, int]
) -> None:
    font = ImageFont.load_default()
    left, top = xy
    text_bbox = draw.textbbox((left, top), text, font=font)
    padding = 3
    draw.rectangle(
        (
            text_bbox[0] - padding,
            text_bbox[1] - padding,
            text_bbox[2] + padding,
            text_bbox[3] + padding,
        ),
        fill=color,
    )
    draw.text((left, top), text, fill=(255, 255, 255), font=font)


def draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: BBox,
    label: str,
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    """Draw a labeled bounding box."""
    draw.rectangle(bbox.to_xyxy(), outline=color, width=width)
    _draw_label(draw, (bbox.x1, max(0, bbox.y1 - 13)), label, color)


def annotate_image(
    image: Image.Image,
    detections: list[Detection],
    assignments: list[Assignment],
    required_items: set[str],
    ignored_items: set[str],
) -> Image.Image:
    """Draw detections with compliance-aware colors."""
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    assignment_by_detection = {id(assignment.detection): assignment for assignment in assignments}

    for detection in detections:
        if detection.is_person:
            draw_box(draw, detection.bbox, "person", COLOR_PERSON, width=2)
            continue

        assignment = assignment_by_detection.get(id(detection))
        worn = assignment.worn if assignment else False
        if detection.class_name in ignored_items:
            color = COLOR_OPTIONAL
            status = "ignored"
        elif detection.class_name in required_items and worn:
            color = COLOR_OK
            status = "worn"
        elif detection.class_name in required_items:
            color = COLOR_MISSING
            status = "not worn"
        else:
            color = COLOR_OPTIONAL
            status = "optional"
        draw_box(draw, detection.bbox, f"{detection.class_name} ({status})", color)

    return output
