"""End-to-end PPE compliance pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from src.ppe_verification.overlap_rules import VerificationResult, assign_ppe_to_people
from src.risk_scoring.risk_calculator import RiskResult, calculate_risk
from src.rule_engine.rule_engine import ComplianceResult, evaluate_compliance
from src.utils.bbox import BBox, Detection


@dataclass(frozen=True)
class FrameResult:
    """Complete result for one processed image or frame."""

    detections: list[Detection]
    person_bboxes: list[BBox]
    verification: VerificationResult
    compliance: ComplianceResult
    risk: RiskResult


def analyze_detections(
    detections: list[Detection],
    scenario: dict,
    weights: dict[str, float],
    overlap_threshold: float = 0.4,
) -> FrameResult:
    """Run verification, compliance evaluation, and risk scoring for detections."""
    person_bboxes = [detection.bbox for detection in detections if detection.is_person]
    verification = assign_ppe_to_people(person_bboxes, detections, threshold=overlap_threshold)
    worn_items = verification.all_worn_items
    compliance = evaluate_compliance(worn_items=worn_items, scenario=scenario)
    risk = calculate_risk(
        required_missing=compliance.required_missing,
        weights=weights,
        required_items=set(scenario.get("required", [])),
        site_display_name=str(scenario.get("display_name", "the site")),
        action_verb=str(scenario.get("action_verb", "entering")),
    )
    return FrameResult(
        detections=detections,
        person_bboxes=person_bboxes,
        verification=verification,
        compliance=compliance,
        risk=risk,
    )


def analyze_image(
    image: Image.Image,
    detector,
    scenario: dict,
    weights: dict[str, float],
    overlap_threshold: float = 0.4,
) -> FrameResult:
    """Detect and analyze one PIL image."""
    detections = detector.predict(image)
    return analyze_detections(detections, scenario, weights, overlap_threshold)
