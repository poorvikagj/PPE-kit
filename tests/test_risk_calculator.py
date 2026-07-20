from src.risk_scoring.risk_calculator import calculate_risk, severity_for_score


def test_calculates_normalized_required_missing_risk() -> None:
    weights = {"helmet": 30, "safety_boots": 20, "safety_vest": 15}

    result = calculate_risk(
        required_missing={"helmet"},
        weights=weights,
        required_items={"helmet", "safety_boots", "safety_vest"},
        site_display_name="Construction Site",
    )

    assert result.raw_score == 30
    assert round(result.normalized_score, 1) == 46.2
    assert result.severity == "Moderate Risk"
    assert "helmet" in result.recommendation


def test_no_required_missing_is_safe() -> None:
    result = calculate_risk(set(), {"helmet": 30}, {"helmet"}, "Construction Site")

    assert result.raw_score == 0
    assert result.normalized_score == 0
    assert result.severity == "Safe"


def test_severity_boundaries() -> None:
    assert severity_for_score(20) == "Safe"
    assert severity_for_score(21) == "Low Risk"
    assert severity_for_score(40) == "Low Risk"
    assert severity_for_score(41) == "Moderate Risk"
    assert severity_for_score(61) == "High Risk"
    assert severity_for_score(81) == "Critical Risk"
