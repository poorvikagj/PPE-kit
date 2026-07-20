"""Bounding-box and detection primitives used across the pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Absolute pixel bounding box in xyxy format."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Invalid BBox coordinates: {self}")

    @property
    def width(self) -> float:
        """Return box width."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Return box height."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Return box area."""
        return self.width * self.height

    def intersection(self, other: BBox) -> float:
        """Return intersection area with another box."""
        left = max(self.x1, other.x1)
        top = max(self.y1, other.y1)
        right = min(self.x2, other.x2)
        bottom = min(self.y2, other.y2)
        if right <= left or bottom <= top:
            return 0.0
        return (right - left) * (bottom - top)

    def iou(self, other: BBox) -> float:
        """Return intersection-over-union with another box."""
        union = self.area + other.area - self.intersection(other)
        if union <= 0:
            return 0.0
        return self.intersection(other) / union

    def overlap_ratio(self, region: BBox) -> float:
        """Return the fraction of this box covered by a target region."""
        if self.area <= 0:
            return 0.0
        return self.intersection(region) / self.area

    def subregion(
        self,
        x_start: float,
        x_end: float,
        y_start: float,
        y_end: float,
    ) -> BBox:
        """Return a fractional subregion of this box."""
        return BBox(
            x1=self.x1 + self.width * x_start,
            y1=self.y1 + self.height * y_start,
            x2=self.x1 + self.width * x_end,
            y2=self.y1 + self.height * y_end,
        )

    def clamp(self, width: int, height: int) -> BBox:
        """Clamp this box to image dimensions."""
        return BBox(
            x1=max(0.0, min(float(width), self.x1)),
            y1=max(0.0, min(float(height), self.y1)),
            x2=max(0.0, min(float(width), self.x2)),
            y2=max(0.0, min(float(height), self.y2)),
        )

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return coordinates as an xyxy tuple."""
        return self.x1, self.y1, self.x2, self.y2

    @classmethod
    def from_yolo(
        cls,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
    ) -> BBox:
        """Create an absolute xyxy box from normalized YOLO xywh values."""
        box_width = width * image_width
        box_height = height * image_height
        center_x = x_center * image_width
        center_y = y_center * image_height
        return cls(
            x1=center_x - box_width / 2,
            y1=center_y - box_height / 2,
            x2=center_x + box_width / 2,
            y2=center_y + box_height / 2,
        ).clamp(image_width, image_height)


@dataclass(frozen=True)
class Detection:
    """A model or label-derived object detection."""

    class_id: int
    class_name: str
    bbox: BBox
    confidence: float = 1.0

    @property
    def is_person(self) -> bool:
        """Return whether this detection is the canonical person class."""
        return self.class_name == "person"


def class_names_from_iterable(names: Iterable[str]) -> dict[int, str]:
    """Convert an ordered class-name iterable to an id-to-name mapping."""
    return {index: name for index, name in enumerate(names)}
