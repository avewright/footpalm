import numpy as np

from footpalm.walkpass import expanding_masks, promote_row, season_stability, slice_metrics


def test_expanding_masks_never_train_on_the_hold_year_or_the_future():
    season = np.array([2014, 2014, 2015, 2015, 2016, 2016])
    fbs = np.array([True, False, True, True, True, True])
    folds = expanding_masks(season, fbs)
    years = [year for year, _train, _hold in folds]
    assert years == [2015, 2016]
    for year, train, hold in folds:
        assert set(season[hold]) == {year}
        assert np.all(season[train] < year)
        assert not np.any(season[train] > year)


def test_season_stability_counts_years_that_actually_improve():
    extras = [{"season": y, "brier": 0.20} for y in range(2015, 2026)]
    full = [{"season": y, "brier": 0.19 if y < 2023 else 0.21} for y in range(2015, 2026)]
    stab = season_stability(extras, full)
    assert stab["years"] == 11
    assert stab["years_better"] == 8
    assert stab["median_season_delta"] < 0


def test_promote_rejects_a_one_year_trick():
    extras_seasons = [{"season": y, "brier": 0.184} for y in range(2015, 2026)]
    full_seasons = [{"season": y, "brier": 0.185} for y in range(2015, 2026)]
    full_seasons[-1]["brier"] = 0.150
    extras = {
        "engine": "logistic",
        "pooled": {"brier": 0.1840, "logloss": 0.54},
        "seasons": extras_seasons,
    }
    full = {
        "engine": "logistic",
        "pooled": {"brier": 0.1810, "logloss": 0.53},
        "seasons": full_seasons,
    }
    row = promote_row(extras, full)
    assert row["delta_brier"] == -0.003
    assert row["years_better"] == 1
    assert row["pass"] is False


def test_promote_accepts_a_spread_out_drop():
    extras_seasons = [{"season": y, "brier": 0.184} for y in range(2015, 2026)]
    full_seasons = [{"season": y, "brier": 0.181} for y in range(2015, 2026)]
    extras = {
        "engine": "logistic",
        "pooled": {"brier": 0.1840, "logloss": 0.54},
        "seasons": extras_seasons,
    }
    full = {
        "engine": "logistic",
        "pooled": {"brier": 0.1810, "logloss": 0.53},
        "seasons": full_seasons,
    }
    assert promote_row(extras, full)["pass"] is True


def test_slice_metrics_split_early_mid_late():
    y = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([0.9, 0.1, 0.1, 0.9])
    week = np.array([1, 5, 6, 12])
    slices = slice_metrics(y, p, week)
    assert slices["early"]["n"] == 1
    assert slices["mid"]["n"] == 2
    assert slices["late"]["n"] == 1
    assert slices["early"]["brier"] < slices["late"]["brier"]
