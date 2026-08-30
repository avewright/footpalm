import numpy as np

from footpalm.trees import _gain_rows, _metrics


def test_gain_rows_sort_and_share():
    rows = _gain_rows(["a", "b", "c"], np.array([1.0, 3.0, 0.0]))
    assert rows[0]["feature"] == "b"
    assert rows[0]["share"] == 0.75


def test_metrics_perfect_is_zero_brier():
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 1.0])
    m = _metrics(y, p)
    assert m["brier"] == 0.0
    assert m["accuracy"] == 1.0
