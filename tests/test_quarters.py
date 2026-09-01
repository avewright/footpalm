import numpy as np
import pandas as pd
import pytest

from footpalm.predict import MAX_TABPFN_ROWS
from footpalm.quarters import (
    ALL_SLICES,
    HALVES,
    HALVES_THREE,
    PAIRS,
    SINGLES,
    THREE_CONSEC,
    expand_rows,
    quarter_points_from_pbp,
    scale_quarters,
    synth_outcomes,
    take_block,
)


def test_combo_counts_are_locked():
    assert len(SINGLES) == 4
    assert len(PAIRS) == 6
    assert len(ALL_SLICES) == 10
    assert HALVES == ((1, 2), (3, 4))
    assert THREE_CONSEC == ((1, 2, 3), (2, 3, 4))
    assert len(HALVES_THREE) == 4


def test_user_examples():
    q_home = np.array([7.0, 13.0, 3.0, 0.0])
    q_away = np.array([0.0, 10.0, 7.0, 0.0])
    assert scale_quarters(q_home, q_away, (1,)) == (28.0, 0.0)
    # 20-10 at half is Q1+Q2
    half_h = np.array([12.0, 8.0, 0.0, 0.0])
    half_a = np.array([3.0, 7.0, 0.0, 0.0])
    assert scale_quarters(half_h, half_a, (1, 2)) == (40.0, 20.0)
    assert scale_quarters(q_home, q_away, (1, 3)) == (20.0, 14.0)
    # 7-0, 13-10, 3-7 in Q1–Q3 → 23-17 × 4/3
    three = scale_quarters(q_home, q_away, (1, 2, 3))
    assert three is not None
    assert three[0] == pytest.approx(23 * 4 / 3)
    assert three[1] == pytest.approx(17 * 4 / 3)


def test_synth_drops_ties_and_keeps_a_decisive_slice():
    q_home = np.array([7.0, 0.0, 7.0, 0.0])
    q_away = np.array([7.0, 0.0, 0.0, 7.0])
    rows = synth_outcomes(q_home, q_away)
    combos = {combo for combo, _h, _a in rows}
    assert (1,) not in combos
    assert (1, 2) not in combos
    assert (1, 3) in combos


def test_pbp_end_of_period_minus_previous():
    df = pd.DataFrame(
        {
            "game_id": [9] * 8,
            "home": ["A"] * 8,
            "away": ["B"] * 8,
            "pos_team": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "pos_team_score": [7, 0, 14, 10, 17, 17, 17, 24],
            "def_pos_team_score": [0, 7, 3, 14, 10, 17, 24, 17],
            "period": [1, 1, 2, 2, 3, 3, 4, 4],
            "game_play_number": list(range(1, 9)),
        }
    )
    # period ends: Q1 7-0, Q2 14-10, Q3 17-17, Q4 17-24
    out = quarter_points_from_pbp(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["q1_home"] == 7 and row["q1_away"] == 0
    assert row["q2_home"] == 7 and row["q2_away"] == 10
    assert row["q3_home"] == 3 and row["q3_away"] == 7
    assert row["q4_home"] == 0 and row["q4_away"] == 7


def test_expand_keeps_x_and_marks_real():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 0.0])
    m = np.array([7.0, -3.0])
    ids = np.array([1.0, 2.0])
    quarters = {1: (np.array([7.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0]))}
    Xe, ye, me, real = expand_rows(X, y, m, ids, quarters, slices=SINGLES)
    assert real[0]
    assert Xe[0].tolist() == [1.0, 2.0]
    assert not real[1]
    assert ye[1] == 1.0
    assert me[1] == 28.0
    assert real.tolist().count(True) == 2


def test_block_keeps_whole_bundles_under_cap():
    # 3 games, each real + 2 synth = 9 rows. Cap 8 keeps the last two bundles (6).
    X = np.arange(9).reshape(9, 1).astype(float)
    y = np.ones(9)
    m = np.arange(9, dtype=float)
    real = np.array([True, False, False, True, False, False, True, False, False])
    xb, yb, mb, rb = take_block(X, y, m, real, cap=8)
    assert len(xb) == 6
    assert xb[0, 0] == 3.0
    assert rb.tolist() == [True, False, False, True, False, False]
    assert take_block(X, y, m, real, cap=MAX_TABPFN_ROWS)[0].shape[0] == 9
