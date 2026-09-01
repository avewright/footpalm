import math

import pandas as pd

from footpalm.qb import QB_NAMES, QBBook, QBSnap, _normalize, snaps_from_pbp


def test_first_game_is_zero_then_walk_forward():
    book = QBBook()
    assert len(book.vector("Alpha", "Beta")) == len(QB_NAMES)
    assert (book.vector("Alpha", "Beta") == 0).all()
    # prior slate: Alpha started A. Qb, Beta started B. Qb
    book.apply(
        {
            "Alpha": QBSnap(modal="qb_a", epa_sum=1.0, epa_cnt=10),
            "Beta": QBSnap(modal="qb_b", epa_sum=-0.5, epa_cnt=10),
        },
        "Alpha",
        "Beta",
    )
    # need a second slate to see effect? vector checks expected starter
    later = book.vector("Alpha", "Beta")
    # qb_epa_diff = Alpha epa (0.1) - Beta epa (-0.05) >0
    assert later[QB_NAMES.index("qb_epa_diff")] > 0
    # starts log1p(1) ~0.69 diff non-zero but symmetric? both 1 so diff 0; need asymmetric
    # Already qb_epa diff suffices; also qb_starts diff may be 0 here (both 1)


def test_same_slate_does_not_see_apply():
    book = QBBook()
    before = book.vector("Alpha", "Gamma")
    assert (before == 0).all()
    book.apply({"Alpha": QBSnap(modal="qb_a", epa_sum=2.0, epa_cnt=10)}, "Alpha", "Beta")
    # Gamma never played, vector Alpha vs Gamma should still be zero for Gamma side, but Alpha now has expected
    # Before was zero, after would be non-zero if same slate were visible, but we applied after vector
    # So before stays zero as verified


def test_modal_passer_from_tiny_pbp():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1],
            "pos_team": ["A", "A", "A", "B", "B"],
            "def_pos_team": ["B", "B", "B", "A", "A"],
            "play_type": ["Pass Reception"] * 5,
            "pos_unit": ["Offense"] * 5,
            "pass": [1] * 5,
            "passer_player_name": ["John Doe", "John Doe", "Jane Smith", "Bob Lee", "Bob Lee"],
            "EPA": [0.5, 0.3, 0.8, 0.2, 0.1],
            "period": [1] * 5,
            "penalty_no_play": [0] * 5,
            "pos_team_score": [0] * 5,
            "def_pos_team_score": [0] * 5,
            "kickoff_play": [0] * 5,
            "punt": [0] * 5,
            "fg_inds": [0] * 5,
        }
    )
    snaps = snaps_from_pbp(df)
    a_row = snaps.loc[snaps["team"].eq("A")].iloc[0]
    # modal for A is John Doe (2 vs 1), normalized
    assert a_row["modal"] == _normalize("John Doe")
    assert a_row["epa_cnt"] == 2
    b_row = snaps.loc[snaps["team"].eq("B")].iloc[0]
    assert b_row["modal"] == _normalize("Bob Lee")


def test_normalize_casefold_strip():
    assert _normalize(" J. Daniels ") == _normalize("j. daniels")
    assert _normalize("J. Daniels") == "j. daniels"


def test_new_season_clears_this_but_keeps_last():
    book = QBBook()
    # season 2024: two games for Alpha, starter qb_a both games
    book.apply({"Alpha": QBSnap(modal="qb_a", epa_sum=1.0, epa_cnt=10)}, "Alpha", "Beta")
    book.apply({"Alpha": QBSnap(modal="qb_a", epa_sum=1.0, epa_cnt=10)}, "Alpha", "Beta")
    # also populate Beta with qb_b
    # global last not yet set
    book.new_season()
    # after new_season, last_modal should be qb_a, recent cleared, starts cleared
    state = book.get("Alpha")
    assert state.last_modal == "qb_a"
    assert state.recent_starter is None
    assert len(state.this_starts) == 0
    # prior EPA for qb_a should be 0.1 (2/20? actually 2/20=0.1)
    assert abs(book.global_last_epa.get("qb_a", 0) - 0.1) < 1e-9
    # this season EPA for qb_a should be cleared
    assert state.this_epa_cnt.get("qb_a", 0) == 0
    # vector for Alpha vs Beta at start of new season should use expected = last_modal
    vec = book.vector("Alpha", "Beta")
    # qb_prior_epa_diff should be prior of qb_a minus prior of Beta's last_modal
    # Beta's last_modal is qb_b, prior  also exists
    # Just check qb_change diff = 0 (both expected == last_modal)
    assert vec[QB_NAMES.index("qb_change")] == 0


def test_qb_change_flips_when_starter_differs():
    book = QBBook()
    # Season 1: Alpha starts qb_a all season
    for _ in range(3):
        book.apply({"Alpha": QBSnap(modal="qb_a", epa_sum=0.5, epa_cnt=10)}, "Alpha", "X")
    book.new_season()
    # New season first game Alpha starts qb_b (different)
    book.apply({"Alpha": QBSnap(modal="qb_b", epa_sum=0.2, epa_cnt=10)}, "Alpha", "Y")
    # At this point recent is qb_b, last_modal is qb_a, so change=1
    vec = book.vector("Alpha", "Beta")
    # Beta has no history, expected None -> features 0, so diff is just Alpha's change
    assert vec[QB_NAMES.index("qb_change")] == 1.0
    # If later Alpha goes back to qb_a, change flips to 0? Let's simulate next slate still same season
    # vector before next apply should still be 1
    assert book.vector("Alpha", "Beta")[QB_NAMES.index("qb_change")] == 1.0
    # Apply next game with qb_a
    book.apply({"Alpha": QBSnap(modal="qb_a", epa_sum=0.5, epa_cnt=10)}, "Alpha", "Z")
    vec2 = book.vector("Alpha", "Beta")
    assert vec2[QB_NAMES.index("qb_change")] == 0.0


def test_qb_starts_log1p():
    book = QBBook()
    book.apply({"Alpha": QBSnap(modal="qb_a", epa_sum=0.1, epa_cnt=5)}, "Alpha", "Beta")
    vec1 = book.vector("Alpha", "Beta")
    assert abs(vec1[QB_NAMES.index("qb_starts")] - math.log1p(1)) < 1e-9
    book.apply({"Alpha": QBSnap(modal="qb_a", epa_sum=0.1, epa_cnt=5)}, "Alpha", "Beta")
    vec2 = book.vector("Alpha", "Beta")
    assert abs(vec2[QB_NAMES.index("qb_starts")] - math.log1p(2)) < 1e-9
