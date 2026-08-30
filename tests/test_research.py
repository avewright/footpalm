import numpy as np

from footpalm.research import apply, _metrics


def test_temperature_below_one_sharpens():
    p = np.array([0.2, 0.4, 0.6, 0.8])
    q = apply("temperature", p, p, {"T": 0.7})
    assert q[0] < p[0]
    assert q[-1] > p[-1]


def test_sigma_rebuilds_from_margin():
    margin = np.array([0.0, 14.5])
    p = apply("sigma", np.array([0.9, 0.9]), margin, {"s": 14.5})
    assert abs(p[0] - 0.5) < 1e-6
    assert abs(p[1] - 1 / (1 + np.exp(-1.0))) < 1e-6


def test_promotion_rule_rejects_tiny_gains():
    y = np.array([1.0, 0.0, 1.0, 1.0])
    raw = np.array([0.6, 0.4, 0.7, 0.8])
    tiny = raw * 0.999 + 0.001 * y
    raw_m = _metrics(y, raw)
    tiny_m = _metrics(y, tiny)
    assert tiny_m["brier"] < raw_m["brier"]
    assert raw_m["brier"] - tiny_m["brier"] < 0.002


def test_platt_identity_is_noop():
    p = np.array([0.2, 0.5, 0.8])
    q = apply("platt", p, p, {"a": 0.0, "b": 1.0})
    assert np.allclose(p, q)


def test_apply_promoted_is_idempotent():
    from footpalm.research import apply_promoted

    games = [{"home_win_prob": 0.8, "pred_margin": 10.0, "home_won": 1}]
    report = {"promoted": "temperature", "experiments": [{"id": "temperature", "params": {"T": 0.7}}]}
    once = apply_promoted(games, report)
    twice = apply_promoted(once, report)
    assert once[0]["home_win_prob"] == twice[0]["home_win_prob"]
    assert once[0]["home_win_prob_raw"] == 0.8
