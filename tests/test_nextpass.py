import numpy as np

from footpalm.nextpass import fit_blend_weights, thin_column
from footpalm.predict import FEATURE_NAMES


def test_thin_column_reads_home_and_away_games():
    X = np.zeros((3, len(FEATURE_NAMES)))
    home = FEATURE_NAMES.index("home_games")
    away = FEATURE_NAMES.index("away_games")
    X[0, home], X[0, away] = 0, 0
    X[1, home], X[1, away] = 5, 5
    X[2, home], X[2, away] = 2, 10
    assert np.allclose(thin_column(X), [1.0, 0.0, 1.0])


def test_blend_weights_sum_to_one_and_stay_nonneg():
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    parts = np.column_stack(
        [
            np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.4]),
            np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4]),
            np.array([0.7, 0.3, 0.7, 0.3, 0.7, 0.3]),
        ]
    )
    w = fit_blend_weights(y, parts)
    assert w.min() >= 0
    assert abs(w.sum() - 1.0) < 1e-9


def test_blend_weights_ignore_holdout_labels():
    y_train = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    good = y_train.copy()
    bad = 1.0 - y_train
    mid = np.full(6, 0.5)
    w = fit_blend_weights(y_train, np.column_stack([good, bad, mid]))
    y_hold = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w_again = fit_blend_weights(y_train, np.column_stack([good, bad, mid]))
    assert np.allclose(w, w_again)
    assert w[0] > 0.8
    assert y_hold.sum() == 0
