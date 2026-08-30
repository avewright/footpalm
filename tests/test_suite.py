import numpy as np

from footpalm.suite import _row, _temperature


def test_temperature_sharpens_confident_probs():
    y = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([0.7, 0.65, 0.35, 0.3])
    q, t = _temperature(y, p, p)
    assert t < 1.0
    assert q[0] > p[0]
    assert q[-1] < p[-1]


def test_row_metrics_round_trip():
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([0.9, 0.1, 0.8])
    row = _row("demo", y, p, y, p)
    assert row["holdout"]["brier"] < 0.05
    assert row["id"] == "demo"
