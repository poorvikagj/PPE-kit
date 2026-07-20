"""Weighted, normalized risk score calculation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = Path(__file__).with_name("weights.json")


@dataclass(frozen=True)
class RiskResult:
    """Risk score, band, and recommendation for a compliance result."""

    raw_score: float
    normalized_score: float
    severity: str
    recommendation: str


def load_weights(path: str | Path = DEFAULT_WEIGHTS_PATH) -> dict[str, float]:
    """Load PPE risk weights from JSON."""
    with Path(path).open("r", encoding="utf-8") as file:
        weights = json.load(file)
    return {str(key): float(value) for key, value in weights.items()}


def severity_for_score(score: float) -> str:
    """Map a normalized 0-100 score to a severity band."""
    if score <= 20:
        return "Safe"
    if score <= 40:
        return "Low Risk"
    if score <= 60:
        return "Moderate Risk"
    if score <= 80:
        return "High Risk"
    return "Critical Risk"


def humanize_item(item: str) -> str:
    """Return a display-friendly PPE item label."""
    return item.replace("_", " ")


def join_items(items: list[str]) -> str:
    """Join item names into readable English."""
    labels = [humanize_item(item) for item in items]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


def calculate_risk(
    required_missing: set[str] | list[str],
    weights: dict[str, float],
    required_items: set[str] | list[str],
    site_display_name: str = "the site",
    action_verb: str = "entering",
) -> RiskResult:
    """Calculate weighted normalized risk from required missing PPE."""
    missing = set(required_missing)
    required = set(required_items)
    raw_score = sum(weights.get(item, 0.0) for item in missing)
    denominator = sum(weights.get(item, 0.0) for item in required)
    normalized_score = (raw_score / denominator * 100.0) if denominator else 0.0
    normalized_score = max(0.0, min(100.0, normalized_score))
    severity = severity_for_score(normalized_score)

    if missing:
        recommendation = (
            f"Wear {join_items(sorted(missing))} before {action_verb} {site_display_name}."
        )
    else:
        recommendation = f"All required PPE is worn for {site_display_name}."

    LOGGER.info(
        "calculated_risk",
        extra={
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "severity": severity,
        },
    )
    return RiskResult(
        raw_score=raw_score,
        normalized_score=normalized_score,
        severity=severity,
        recommendation=recommendation,
    )
