"""Config-driven compliance rule evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_SCENARIOS_PATH = Path(__file__).with_name("scenarios.json")


@dataclass(frozen=True)
class ComplianceResult:
    """Compliance outcome for one person or aggregated frame."""

    required_met: set[str]
    required_missing: set[str]
    optional_present: set[str]
    optional_missing: set[str]
    ignored: set[str]
    detected_relevant: set[str]

    @property
    def is_compliant(self) -> bool:
        """Return whether all required PPE is present and worn."""
        return not self.required_missing


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_scenarios(path: str | Path = DEFAULT_SCENARIOS_PATH) -> dict[str, dict[str, Any]]:
    """Load scenario rules from a JSON file."""
    scenarios = _read_json(path)
    for key, scenario in scenarios.items():
        for field_name in ("display_name", "required", "optional", "forbidden"):
            if field_name not in scenario:
                raise ValueError(f"Scenario '{key}' is missing '{field_name}'")
    return scenarios


def save_scenarios(
    scenarios: dict[str, dict[str, Any]],
    path: str | Path = DEFAULT_SCENARIOS_PATH,
) -> None:
    """Persist scenario rules to a JSON file."""
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(scenarios, file, indent=2)
        file.write("\n")


def evaluate_compliance(worn_items: set[str], scenario: dict[str, Any]) -> ComplianceResult:
    """Evaluate worn PPE against a site scenario."""
    required = set(scenario.get("required", []))
    optional = set(scenario.get("optional", []))
    forbidden = set(scenario.get("forbidden", []))
    applicable = required | optional

    result = ComplianceResult(
        required_met=required & worn_items,
        required_missing=required - worn_items,
        optional_present=optional & worn_items,
        optional_missing=optional - worn_items,
        ignored=(worn_items & forbidden) | (worn_items - applicable),
        detected_relevant=worn_items & applicable,
    )
    LOGGER.info(
        "evaluated_compliance",
        extra={
            "required_missing": sorted(result.required_missing),
            "is_compliant": result.is_compliant,
        },
    )
    return result
