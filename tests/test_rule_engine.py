from src.rule_engine.rule_engine import evaluate_compliance, load_scenarios


def test_loads_configured_scenarios() -> None:
    scenarios = load_scenarios()

    assert "construction" in scenarios
    assert scenarios["construction"]["required"] == ["helmet", "safety_vest", "safety_boots"]


def test_evaluates_required_optional_and_ignored_items() -> None:
    scenario = {
        "required": ["helmet", "safety_vest"],
        "optional": ["gloves"],
        "forbidden": ["lab_coat"],
    }

    result = evaluate_compliance({"helmet", "gloves", "lab_coat"}, scenario)

    assert result.required_met == {"helmet"}
    assert result.required_missing == {"safety_vest"}
    assert result.optional_present == {"gloves"}
    assert result.ignored == {"lab_coat"}
    assert not result.is_compliant
