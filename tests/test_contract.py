from sc_novelty_reliability import alignment_contract, expected_calibration_error, risk_coverage_curve

def test_metrics_are_deterministic():
    y, p = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    assert abs(expected_calibration_error(y, p, bins=2) - 0.15) < 1e-12
    assert risk_coverage_curve(y, p, (0.5, 1.0))[0]["requested_coverage"] == 0.5

def test_alignment_is_fail_closed():
    primary = {"type_n_total": {"A": 2, "B": 1}, "gse_donor_n": {"D1": 2, "D2": 1}, "held_gse_donor": "D1", "cal_gse_donor": "D2", "train_gse_donors": []}
    assert alignment_contract(["A", "A", "B"], ["D1", "D1", "D2"], primary)["status"] == "COMPARABLE"
    assert alignment_contract(["A", "A", "DRIFT"], ["D1", "D1", "D2"], primary)["status"] == "NOT_COMPARABLE"
