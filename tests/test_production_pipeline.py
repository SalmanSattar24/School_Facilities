from school_facilities.production_pipeline import _correct, _risk_stratum


def test_solar_area_uses_frozen_absolute_and_relative_tolerance() -> None:
    assert _correct("solar_area_m2", 120, 100)
    assert not _correct("solar_area_m2", 140, 100)
    assert _correct("solar_area_m2", 20, 0)
    assert not _correct("solar_area_m2", 30, 0)


def test_risk_stratum_prioritizes_unknown() -> None:
    assert _risk_stratum("unknown", False) == "abstained_unknown"
    assert _risk_stratum("yes", True) == "flagged_known"
    assert _risk_stratum(0, False) == "auto_accept_known"
