from src.ppe_verification.overlap_rules import assign_ppe_to_people, verify_worn
from src.utils.bbox import BBox, Detection


def test_helmet_overlapping_head_region_is_worn() -> None:
    person = BBox(100, 100, 300, 500)
    helmet = BBox(135, 105, 265, 185)

    assert verify_worn(person, helmet, "helmet", threshold=0.4)


def test_helmet_with_large_box_still_counts_as_worn() -> None:
    person = BBox(100, 100, 300, 500)
    helmet = BBox(110, 95, 290, 190)

    assert verify_worn(person, helmet, "helmet", threshold=0.4)


def test_helmet_in_bottom_corner_is_not_worn() -> None:
    person = BBox(100, 100, 300, 500)
    helmet = BBox(105, 430, 165, 490)

    assert not verify_worn(person, helmet, "helmet", threshold=0.4)


def test_gloves_use_side_hand_regions() -> None:
    person = BBox(100, 100, 300, 500)
    glove = BBox(105, 330, 135, 410)

    assert verify_worn(person, glove, "gloves", threshold=0.4)


def test_assigns_ppe_to_best_matching_person_only() -> None:
    people = [BBox(0, 0, 100, 300), BBox(200, 0, 300, 300)]
    detections = [
        Detection(0, "person", people[0], 0.99),
        Detection(0, "person", people[1], 0.99),
        Detection(1, "helmet", BBox(215, 5, 285, 60), 0.9),
    ]

    result = assign_ppe_to_people(people, detections, threshold=0.4)

    assert result.worn_by_person[0] == set()
    assert result.worn_by_person[1] == {"helmet"}
    assert len(result.assignments) == 1
    assert result.assignments[0].person_index == 1


def test_ppe_without_detected_person_defaults_to_worn() -> None:
    detections = [Detection(1, "helmet", BBox(10, 10, 40, 40), 0.9)]

    result = assign_ppe_to_people([], detections, threshold=0.03)

    assert result.assignments[0].worn
    assert result.assignments[0].person_index is None
