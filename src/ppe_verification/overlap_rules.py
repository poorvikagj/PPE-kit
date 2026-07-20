"""Region-overlap rules for deciding whether PPE is worn by a person."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.utils.bbox import BBox, Detection

LOGGER = logging.getLogger(__name__)

PPE_REGION_FRACTIONS: dict[str, list[tuple[float, float, float, float]]] = {
    "helmet": [(0.0, 1.0, 0.0, 0.25)],
    "safety_goggles": [(0.0, 1.0, 0.10, 0.30)],
    "face_mask": [(0.0, 1.0, 0.20, 0.40)],
    "safety_vest": [(0.0, 1.0, 0.25, 0.65)],
    "lab_coat": [(0.0, 1.0, 0.25, 0.75)],
    "gloves": [(0.0, 0.20, 0.50, 0.90), (0.80, 1.0, 0.50, 0.90)],
    "safety_boots": [(0.0, 1.0, 0.80, 1.0)],
}

PPE_VERIFY_THRESHOLDS: dict[str, float] = {
    "helmet": 0.18,
    "safety_goggles": 0.18,
    "face_mask": 0.18,
    "safety_vest": 0.15,
    "lab_coat": 0.15,
    "gloves": 0.12,
    "safety_boots": 0.15,
}


@dataclass(frozen=True)
class Assignment:
    """Best person assignment for a single PPE detection."""

    person_index: int | None
    item_type: str
    detection: Detection
    overlap: float
    worn: bool


@dataclass(frozen=True)
class VerificationResult:
    """Per-frame worn PPE assignment result."""

    worn_by_person: dict[int, set[str]] = field(default_factory=dict)
    assignments: list[Assignment] = field(default_factory=list)
    unassigned: list[Detection] = field(default_factory=list)

    @property
    def all_worn_items(self) -> set[str]:
        """Return the union of worn PPE across all detected people."""
        worn: set[str] = set()
        for items in self.worn_by_person.values():
            worn.update(items)
        return worn


def relevant_regions(person_bbox: BBox, item_type: str) -> list[BBox]:
    """Return person-relative verification regions for a PPE item type."""
    if item_type not in PPE_REGION_FRACTIONS:
        return []
    return [
        person_bbox.subregion(x_start, x_end, y_start, y_end)
        for x_start, x_end, y_start, y_end in PPE_REGION_FRACTIONS[item_type]
    ]


def best_overlap(person_bbox: BBox, item_bbox: BBox, item_type: str) -> float:
    """Return the highest item-area overlap with valid regions on a person."""
    regions = relevant_regions(person_bbox, item_type)
    if not regions:
        return 0.0
    return max(item_bbox.overlap_ratio(region) for region in regions)


def item_center_in_region(person_bbox: BBox, item_bbox: BBox, item_type: str) -> bool:
    """Return whether the detected item center falls inside a valid PPE region."""
    regions = relevant_regions(person_bbox, item_type)
    if not regions:
        return False

    center_x = (item_bbox.x1 + item_bbox.x2) / 2
    center_y = (item_bbox.y1 + item_bbox.y2) / 2
    return any(
        region.x1 <= center_x <= region.x2 and region.y1 <= center_y <= region.y2
        for region in regions
    )


def verify_worn(
    person_bbox: BBox,
    item_bbox: BBox,
    item_type: str,
    threshold: float = 0.18,
) -> bool:
    """Return whether a PPE item is worn by a person using region overlap."""
    threshold = PPE_VERIFY_THRESHOLDS.get(item_type, threshold)
    return best_overlap(person_bbox, item_bbox, item_type) >= threshold or item_center_in_region(
        person_bbox, item_bbox, item_type
    )


def assign_ppe_to_people(
    person_bboxes: list[BBox],
    detections: list[Detection],
    threshold: float = 0.18,
) -> VerificationResult:
    """Assign PPE detections to the best matching person and mark worn state."""
    worn_by_person: dict[int, set[str]] = {index: set() for index in range(len(person_bboxes))}
    assignments: list[Assignment] = []
    unassigned: list[Detection] = []

    for detection in detections:
        if detection.is_person:
            continue
        if detection.class_name not in PPE_REGION_FRACTIONS:
            unassigned.append(detection)
            continue

        scored_people = [
            (person_index, best_overlap(person_bbox, detection.bbox, detection.class_name))
            for person_index, person_bbox in enumerate(person_bboxes)
        ]
        best_person_index: int | None = None
        best_score = 0.0
        if scored_people:
            best_person_index, best_score = max(scored_people, key=lambda item: item[1])

        worn = best_person_index is not None and verify_worn(
            person_bboxes[best_person_index], detection.bbox, detection.class_name, threshold
        )
        if worn and best_person_index is not None:
            worn_by_person[best_person_index].add(detection.class_name)

        assignments.append(
            Assignment(
                person_index=best_person_index,
                item_type=detection.class_name,
                detection=detection,
                overlap=best_score,
                worn=worn,
            )
        )

    LOGGER.info(
        "verified_ppe",
        extra={
            "person_count": len(person_bboxes),
            "ppe_count": len([d for d in detections if not d.is_person]),
            "worn_by_person": {key: sorted(value) for key, value in worn_by_person.items()},
        },
    )
    return VerificationResult(
        worn_by_person=worn_by_person,
        assignments=assignments,
        unassigned=unassigned,
    )


def get_worn_ppe(
    person_bboxes: list[BBox],
    detections: list[Detection],
    threshold: float = 0.4,
) -> dict[int, set[str]]:
    """Return a mapping of person index to worn PPE item names."""
    return assign_ppe_to_people(person_bboxes, detections, threshold).worn_by_person
