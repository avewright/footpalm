import pandas as pd

from footpalm.pace import PACE_NAMES, PaceBook, PaceSnap, snaps_from_pbp


def test_first_game_is_zero_then_walk_forward():
    book = PaceBook()
    assert len(book.vector("Alpha", "Beta")) == len(PACE_NAMES)
    assert (book.vector("Alpha", "Beta") == 0).all()
    book.apply(
        {
            "Alpha": PaceSnap(giveaways=0, takeaways=2, rush_yds=80, rush_n=20, wall_sec=300, wall_n=10, clock_sec=200, clock_n=20),
            "Beta": PaceSnap(giveaways=2, takeaways=0, rush_yds=40, rush_n=20, wall_sec=400, wall_n=10, clock_sec=300, clock_n=20),
        },
        "Alpha",
        "Beta",
    )
    later = book.vector("Alpha", "Beta")
    assert later[PACE_NAMES.index("to_margin_diff")] > 0
    assert later[PACE_NAMES.index("ypc_diff")] > 0
    assert later[PACE_NAMES.index("play_speed_diff")] < 0
    assert later[PACE_NAMES.index("sec_per_play_diff")] < 0


def test_same_slate_does_not_see_apply():
    book = PaceBook()
    before = book.vector("Alpha", "Gamma")
    book.apply({"Alpha": PaceSnap(takeaways=3, rush_yds=100, rush_n=20)}, "Alpha", "Beta")
    assert (before == 0).all()


def test_snaps_from_tiny_pbp():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "pos_team": ["A", "A", "B", "B"],
            "def_pos_team": ["B", "B", "A", "A"],
            "play_type": ["Rush", "Rush", "Pass Reception", "Pass Incompletion"],
            "pos_unit": ["Offense"] * 4,
            "rush": [1, 1, 0, 0],
            "sack": [0, 0, 0, 0],
            "yds_rushed": [5.0, 7.0, 0.0, 0.0],
            "turnover": [0, 0, 1, 0],
            "TimeSecsRem": [900.0, 880.0, 860.0, 840.0],
            "wallclock": [
                "2025-09-01T18:00:00Z",
                "2025-09-01T18:00:40Z",
                "2025-09-01T18:01:20Z",
                "2025-09-01T18:02:00Z",
            ],
            "game_play_number": [1, 2, 3, 4],
            "period": [1, 1, 1, 1],
            "kickoff_play": [0, 0, 0, 0],
            "punt": [0, 0, 0, 0],
            "fg_inds": [0, 0, 0, 0],
            "penalty_no_play": [0, 0, 0, 0],
            "pos_team_score": [0, 0, 0, 0],
            "def_pos_team_score": [0, 0, 0, 0],
        }
    )
    snaps = snaps_from_pbp(df)
    a = snaps.loc[snaps["team"].eq("A")].iloc[0]
    b = snaps.loc[snaps["team"].eq("B")].iloc[0]
    assert a["rush_n"] == 2
    assert a["rush_yds"] == 12
    assert b["giveaways"] == 1
    assert a["takeaways"] == 1
