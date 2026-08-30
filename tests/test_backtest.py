import numpy as np
import pandas as pd

from footpalm.predict import FEATURE_NAMES, game_features, logistic_from_features
from footpalm.rate import RatingBook, TeamRating, publish_table, win_prob


def _book() -> RatingBook:
    teams = {
        "Alpha": TeamRating("Alpha", 0.1, -0.1, 0.0, 130, 12.0, 8.0, -4.0, 6),
        "Beta": TeamRating("Beta", -0.05, 0.05, 0.0, 130, -6.0, -4.0, 2.0, 6),
    }
    return RatingBook(2025, 4, teams, 0.02, 0.0, 65.0, 27.0, {"Alpha", "Beta"}, {})


def test_win_prob_is_half_on_a_pick():
    assert abs(float(win_prob([0.0])[0]) - 0.5) < 1e-9


def test_publish_table_keeps_elo():
    book = _book()
    sides = pd.DataFrame(
        [
            {
                "team": "Alpha",
                "opponent": "Beta",
                "home_team": "Alpha",
                "away_team": "Beta",
                "game_id": 1,
                "won": True,
                "lost": False,
                "points": 31,
                "opp_points": 17,
                "is_home": True,
                "neutral_site": False,
            },
            {
                "team": "Beta",
                "opponent": "Alpha",
                "home_team": "Alpha",
                "away_team": "Beta",
                "game_id": 1,
                "won": False,
                "lost": True,
                "points": 17,
                "opp_points": 31,
                "is_home": False,
                "neutral_site": False,
            },
        ]
    )
    table = publish_table(book, sides, min_games=0, elo={"Alpha": 1682.4, "Beta": 1410.1})
    by_team = {row["team"]: row for row in table["teams"]}
    assert by_team["Alpha"]["elo"] == 1682
    assert by_team["Beta"]["elo"] == 1410


def test_favorite_is_above_a_coin_flip():
    book = _book()
    x = game_features(book, "Alpha", "Beta", neutral=False)
    pred = logistic_from_features(x, 27.0)
    assert pred["home_win_prob"] > 0.7
    assert pred["pred_margin"] > 0


def test_feature_vector_is_locked_and_finite():
    book = _book()
    x = game_features(book, "Alpha", "Beta", neutral=True)
    assert len(x) == len(FEATURE_NAMES)
    assert x[7] == 0.0
    assert all(v == v for v in x)


def test_walk_forward_does_not_use_future_slates():
    import pandas as pd

    from footpalm.backtest import walk_forward

    rows = []
    for slate, home, away, hf, af in [
        (1, "Alpha", "Beta", 31, 17),
        (2, "Beta", "Alpha", 10, 24),
        (3, "Alpha", "Beta", 28, 14),
    ]:
        for team, opp, pts, opp_pts, is_home in [
            (home, away, hf, af, True),
            (away, home, af, hf, False),
        ]:
            rows.append(
                {
                    "game_id": slate,
                    "slate": slate,
                    "week": slate,
                    "season_type": "regular",
                    "team": team,
                    "opponent": opp,
                    "home_team": home,
                    "away_team": away,
                    "points": pts,
                    "opp_points": opp_pts,
                    "won": pts > opp_pts,
                    "lost": pts < opp_pts,
                    "is_home": is_home,
                    "neutral_site": False,
                    "off_epa": 0.1 if team == "Alpha" else -0.05,
                    "def_epa": -0.05 if team == "Alpha" else 0.1,
                    "st_epa": 0.0,
                    "game_plays": 120,
                    "spread": -7.0 if is_home else 7.0,
                }
            )
    sides = pd.DataFrame(rows)
    result = walk_forward(
        sides,
        {"Alpha", "Beta"},
        {"Alpha": "SEC", "Beta": "SEC"},
        2025,
        use_tabpfn=False,
    )
    assert result["games"]
    first = [g for g in result["games"] if g["slate"] == 1][0]
    assert first["engine"] == "logistic"
    assert first["pred_margin"] == first["pred_margin"]
    assert result["features"][-1] == "quality_win_diff"
    hist = result["_history"]
    assert hist is not None
    assert hist[0].shape[1] == 20
    assert np.allclose(hist[0][0, 10:], 0.0)
    assert hist[0][1, 10] != 0.0


def test_replay_then_extras_matrix_is_week0_elo_only():
    from footpalm.form import extras_matrix, replay_form
    from footpalm.predict import FEATURE_NAMES

    games = [
        {
            "season": 2025,
            "slate": 1,
            "home": "Alpha",
            "away": "Beta",
            "neutral": False,
            "actual_margin": 21.0,
            "home_won": 1,
            "actual_home": 31.0,
            "actual_away": 10.0,
            "home_conf": "SEC",
            "away_conf": "SEC",
        }
    ]
    locked = np.array([[18.0, 12.0, -6.0, 0, 0, 0, 0, 1, 1, 1]], dtype=float)
    form = replay_form(games, locked)
    form.new_season()
    live = extras_matrix(_book(), form, [{"home": "Alpha", "away": "Beta", "slate": 1, "neutral": False}])
    assert live.shape == (1, 20)
    assert live[0, FEATURE_NAMES.index("pom_diff")] == 18.0
    assert live[0, 10] != 0.0
    assert np.allclose(live[0, 11:], 0.0)
