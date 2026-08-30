import numpy as np

from footpalm.featurize import extras_from_games
from footpalm.form import (
    ALL_NAMES,
    CONF_NAMES,
    CRAFT_NAMES,
    ELO_MEAN,
    ELO_TANH,
    EXTRA_NAMES,
    FORM_SOS_SCALE,
    LOSO_NAMES,
    PROD_SCALE,
    SIGNAL_NAMES,
    TIME_NAMES,
    is_power,
    FormBook,
    elo_snapshots_from_games,
    game_features_full,
    revert_elo,
    same_conf,
    signed_log,
    thin_feature,
    tier_win_points,
    time_features,
    trim_mean,
)
from footpalm.predict import FEATURE_NAMES, game_features, logistic_from_features
from footpalm.rate import RatingBook, TeamRating


def _book() -> RatingBook:
    teams = {
        "Alpha": TeamRating("Alpha", 0.1, -0.1, 0.0, 130, 12.0, 8.0, -4.0, 6),
        "Beta": TeamRating("Beta", -0.05, 0.05, 0.0, 130, -6.0, -4.0, 2.0, 6),
    }
    return RatingBook(2025, 4, teams, 0.02, 0.0, 65.0, 27.0, {"Alpha", "Beta"}, {})


def test_fresh_extras_are_zero():
    form = FormBook()
    extra = form.extras("Alpha", "Beta", 1)
    assert len(extra) == len(EXTRA_NAMES)
    assert np.allclose(extra, 0.0)


def test_full_vector_is_locked_plus_extras():
    x = game_features_full(_book(), FormBook(), "Alpha", "Beta", slate=1, neutral=True)
    assert len(x) == len(ALL_NAMES)
    assert x[FEATURE_NAMES.index("is_home")] == 0.0


def test_form_does_not_see_the_current_slate():
    form = FormBook()
    before = form.extras("Alpha", "Beta", 1)
    form.apply_game(
        "Alpha",
        "Beta",
        home_won=True,
        margin=21.0,
        home_pom=12.0,
        away_pom=-6.0,
        slate=1,
        neutral=False,
    )
    after = form.extras("Alpha", "Beta", 2)
    assert np.allclose(before, 0.0)
    assert after[EXTRA_NAMES.index("elo_diff")] > 0
    assert after[EXTRA_NAMES.index("form4_win_diff")] > 0
    assert after[EXTRA_NAMES.index("avg_margin_diff")] > 0


def test_elo_favorite_beats_a_coin_flip():
    form = FormBook()
    form.apply_game(
        "Alpha",
        "Beta",
        home_won=True,
        margin=28.0,
        home_pom=12.0,
        away_pom=-6.0,
        slate=1,
        neutral=False,
    )
    assert form.get("Alpha").elo > ELO_MEAN
    assert form.get("Beta").elo < ELO_MEAN


def test_new_season_clears_form_and_reverts_elo():
    form = FormBook()
    form.apply_game(
        "Alpha",
        "Beta",
        home_won=True,
        margin=21.0,
        home_pom=12.0,
        away_pom=-6.0,
        slate=1,
        neutral=False,
    )
    elo_before = form.get("Alpha").elo
    form.new_season()
    assert form.get("Alpha").results == []
    assert form.get("Alpha").last_slate is None
    assert ELO_MEAN < form.get("Alpha").elo < elo_before
    assert np.allclose(form.extras("Alpha", "Beta", 1)[1:], 0.0)


def test_logistic_uses_named_home_flag_if_extras_appended():
    book = _book()
    x = game_features(book, "Alpha", "Beta", neutral=True)
    wide = np.concatenate([x, np.ones(len(EXTRA_NAMES))])
    pred = logistic_from_features(wide, 27.0)
    assert pred["pred_margin"] == x[0]


def test_extras_from_games_are_walk_forward():
    games = [
        {
            "season": 2014,
            "slate": 1,
            "home": "Alpha",
            "away": "Beta",
            "neutral": False,
            "fbs_fbs": True,
            "actual_margin": 21.0,
            "home_won": 1,
        },
        {
            "season": 2014,
            "slate": 1,
            "home": "Gamma",
            "away": "Delta",
            "neutral": False,
            "fbs_fbs": True,
            "actual_margin": 7.0,
            "home_won": 1,
        },
        {
            "season": 2014,
            "slate": 2,
            "home": "Alpha",
            "away": "Beta",
            "neutral": False,
            "fbs_fbs": True,
            "actual_margin": 14.0,
            "home_won": 1,
        },
    ]
    X = np.array(
        [
            [18.0, 12.0, -6.0, 0, 0, 0, 0, 1, 0, 0],
            [4.0, 2.0, -2.0, 0, 0, 0, 0, 1, 0, 0],
            [18.0, 12.0, -6.0, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=float,
    )
    extras = extras_from_games(games, X)
    assert extras.shape == (3, len(EXTRA_NAMES))
    assert np.allclose(extras[0], 0.0)
    assert np.allclose(extras[1], 0.0)
    assert extras[2, EXTRA_NAMES.index("elo_diff")] > 0
    assert extras[2, EXTRA_NAMES.index("form4_win_diff")] > 0
    assert extras[2, EXTRA_NAMES.index("rest_diff")] == 0.0


def test_signal_is_walk_forward_and_uses_scores():
    form = FormBook()
    before = form.signal("Alpha", "Beta", 1)
    assert len(before) == len(SIGNAL_NAMES)
    assert np.allclose(before, 0.0)
    form.apply_game(
        "Alpha",
        "Beta",
        home_won=True,
        margin=21.0,
        home_pom=12.0,
        away_pom=-6.0,
        slate=1,
        neutral=False,
        home_points=38.0,
        away_points=17.0,
    )
    same_slate = form.signal("Alpha", "Gamma", 1)
    later = form.signal("Alpha", "Beta", 2)
    assert later[SIGNAL_NAMES.index("avg_margin_diff")] > 0
    assert later[SIGNAL_NAMES.index("pf_diff")] > 0
    assert later[SIGNAL_NAMES.index("pa_diff")] < 0
    assert later[SIGNAL_NAMES.index("elo_momentum_diff")] > 0
    assert later[SIGNAL_NAMES.index("residual_margin_diff")] > 0
    assert later[SIGNAL_NAMES.index("h2h_margin")] == 21.0
    # H2H is written at apply time. A later same-slate read would see it, which
    # is why extras/signal are computed for the whole slate before any apply.
    assert before[SIGNAL_NAMES.index("h2h_margin")] == 0.0
    assert same_slate[SIGNAL_NAMES.index("avg_margin_diff")] > 0


def test_h2h_survives_new_season_but_form_does_not():
    form = FormBook()
    form.apply_game(
        "Alpha",
        "Beta",
        home_won=True,
        margin=14.0,
        home_pom=8.0,
        away_pom=-4.0,
        slate=12,
        neutral=False,
        home_points=31.0,
        away_points=17.0,
    )
    form.new_season()
    nxt = form.signal("Alpha", "Beta", 1)
    assert np.allclose(nxt[SIGNAL_NAMES.index("avg_margin_diff")], 0.0)
    assert nxt[SIGNAL_NAMES.index("h2h_margin")] == 14.0


def test_signal_does_not_use_current_game_points():
    form = FormBook()
    form.apply_game(
        "Alpha",
        "Beta",
        home_won=True,
        margin=7.0,
        home_pom=4.0,
        away_pom=-2.0,
        slate=1,
        neutral=False,
        home_points=24.0,
        away_points=17.0,
    )
    pre = form.signal("Alpha", "Gamma", 2)
    assert pre[SIGNAL_NAMES.index("form2_margin_diff")] == 7.0
    assert pre[SIGNAL_NAMES.index("pf_diff")] == 24.0
    form.apply_game(
        "Alpha",
        "Gamma",
        home_won=True,
        margin=42.0,
        home_pom=4.0,
        away_pom=-20.0,
        slate=2,
        neutral=False,
        home_points=56.0,
        away_points=14.0,
    )
    after = form.signal("Alpha", "Delta", 3)
    assert after[SIGNAL_NAMES.index("form2_margin_diff")] == 24.5
    assert after[SIGNAL_NAMES.index("pf_diff")] == 40.0


def test_elo_snapshots_are_end_of_season():
    games = [
        {
            "season": 2014,
            "slate": 1,
            "home": "Alpha",
            "away": "Beta",
            "neutral": False,
            "actual_margin": 21.0,
            "home_won": 1,
        },
        {
            "season": 2014,
            "slate": 2,
            "home": "Alpha",
            "away": "Beta",
            "neutral": False,
            "actual_margin": 14.0,
            "home_won": 1,
        },
    ]
    snaps, form = elo_snapshots_from_games(games)
    assert set(snaps) == {2014}
    assert snaps[2014]["Alpha"] > ELO_MEAN
    assert snaps[2014]["Beta"] < ELO_MEAN
    before = snaps[2014]["Alpha"]
    form.new_season()
    assert ELO_MEAN < form.get("Alpha").elo < before
    assert abs(form.get("Alpha").elo - revert_elo(before)) < 1e-9


def test_time_features_are_year_zero_and_week_over_52():
    assert TIME_NAMES == ["year_idx", "week52"]
    assert np.allclose(time_features(2014, 0), [0.0, 0.0])
    assert np.allclose(time_features(2016, 13), [2.0, 13 / 52])
    assert time_features(2026, 1)[0] == 12.0


def test_thin_is_one_when_either_side_is_under_three_games():
    assert thin_feature(0, 0) == 1.0
    assert thin_feature(2, 10) == 1.0
    assert thin_feature(5, 5) == 0.0
    assert thin_feature(3, 3) == 0.0


def test_same_conf_needs_both_names():
    assert same_conf("SEC", "SEC")
    assert not same_conf("SEC", "ACC")
    assert not same_conf("", "")
    assert not same_conf("SEC", "")


def test_signed_log_compresses_and_keeps_sign():
    assert signed_log(0.0) == 0.0
    assert signed_log(14.0) == np.log1p(14.0)
    assert signed_log(-14.0) == -np.log1p(14.0)
    assert abs(signed_log(40.0)) < 40.0


def _play(form, home, away, margin, *, home_pom=8.0, away_pom=-4.0, slate=1, home_conf="SEC", away_conf="ACC"):
    form.apply_game(
        home,
        away,
        home_won=margin > 0,
        margin=margin,
        home_pom=home_pom,
        away_pom=away_pom,
        slate=slate,
        neutral=False,
        home_points=max(margin, 0) + 17,
        away_points=17 if margin > 0 else 17 - margin,
        home_conf=home_conf,
        away_conf=away_conf,
    )


def test_ncsos_skips_conference_games():
    form = FormBook()
    _play(form, "Alpha", "Beta", 21.0, away_pom=-6.0, home_conf="SEC", away_conf="ACC")
    _play(form, "Alpha", "Gamma", 7.0, away_pom=10.0, slate=2, home_conf="SEC", away_conf="SEC")
    assert form.get("Alpha").ncsos() == -6.0
    assert form.get("Alpha").sos() == 2.0


def test_srs_and_colley_rank_the_winner():
    form = FormBook()
    _play(form, "Alpha", "Beta", 21.0)
    assert form.srs_of("Alpha") > form.srs_of("Beta")
    assert form.colley_of("Alpha") > form.colley_of("Beta")
    assert form.srs_of("Nobody") == 0.0
    assert form.colley_of("Nobody") == 0.5


def test_margin_std_needs_two_games():
    form = FormBook()
    _play(form, "Alpha", "Beta", 7.0)
    assert form.get("Alpha").margin_std() == 0.0
    _play(form, "Alpha", "Gamma", 21.0, slate=2)
    assert form.get("Alpha").margin_std() > 0.0


def test_craft_is_walk_forward_and_matches_formulas():
    form = FormBook()
    locked = np.array([18.0, 12.0, -6.0, 4.0, -2.0, 0.0, 3.0, 1.0, 1.0, 1.0], dtype=float)
    before = form.craft("Alpha", "Beta", 1, locked)
    assert len(before) == len(CRAFT_NAMES)
    assert before[CRAFT_NAMES.index("pom_sum")] == 6.0
    assert before[CRAFT_NAMES.index("pom_abs")] == 18.0
    assert before[CRAFT_NAMES.index("tempo_abs")] == 3.0
    assert before[CRAFT_NAMES.index("log_pom_diff")] == signed_log(18.0)
    _play(form, "Alpha", "Beta", 21.0, home_pom=12.0, away_pom=-6.0)
    same_slate = form.craft("Alpha", "Gamma", 1, locked)
    later = form.craft("Alpha", "Beta", 2, locked)
    assert later[CRAFT_NAMES.index("srs_diff")] > 0
    assert later[CRAFT_NAMES.index("colley_diff")] > 0
    assert later[CRAFT_NAMES.index("ncsos_diff")] == form.get("Alpha").ncsos() - form.get("Beta").ncsos()
    elo = form.get("Alpha").elo - form.get("Beta").elo
    assert later[CRAFT_NAMES.index("tanh_elo_diff")] == np.tanh(elo / ELO_TANH)
    assert later[CRAFT_NAMES.index("pom_elo_prod")] == 18.0 * elo / PROD_SCALE
    form4 = 21.0 - (-21.0)
    sos = form.get("Alpha").sos() - form.get("Beta").sos()
    assert later[CRAFT_NAMES.index("form_sos_prod")] == form4 * sos / FORM_SOS_SCALE
    # Same-slate read after apply would leak. Features are computed before apply.
    assert same_slate[CRAFT_NAMES.index("srs_diff")] > 0
    assert before[CRAFT_NAMES.index("srs_diff")] == 0.0


def test_craft_products_use_prior_slate_only():
    form = FormBook()
    locked = np.zeros(len(FEATURE_NAMES), dtype=float)
    locked[0] = 10.0
    locked[1] = 8.0
    locked[2] = -2.0
    _play(form, "Alpha", "Beta", 14.0, home_pom=8.0, away_pom=-2.0)
    pre = form.craft("Alpha", "Gamma", 2, locked)
    _play(form, "Alpha", "Gamma", 42.0, home_pom=8.0, away_pom=-20.0, slate=2, away_conf="MAC")
    after = form.craft("Alpha", "Delta", 3, locked)
    assert after[CRAFT_NAMES.index("ncsos_diff")] != pre[CRAFT_NAMES.index("ncsos_diff")]


def test_tier_and_trim_helpers():
    assert tier_win_points(12.0) == 6.0
    assert tier_win_points(1.0) == 4.0
    assert tier_win_points(-8.0) == 2.0
    assert tier_win_points(-20.0) == 0.25
    assert trim_mean([1.0, 2.0, 3.0]) == 2.0
    assert trim_mean([0.0, 1.0, 2.0, 3.0, 100.0]) < 20.0


def test_loso_is_walk_forward_and_uses_prior_year():
    form = FormBook()
    before = form.loso("Alpha", "Beta", 1)
    assert len(before) == len(LOSO_NAMES)
    assert np.allclose(before, 0.0)
    _play(form, "Alpha", "Beta", 21.0, home_pom=12.0, away_pom=-6.0, home_conf="SEC", away_conf="ACC")
    later = form.loso("Alpha", "Beta", 2)
    assert later[LOSO_NAMES.index("glm_quality_diff")] > 0
    assert later[LOSO_NAMES.index("tier_win_diff")] > 0
    assert later[LOSO_NAMES.index("form10_win_diff")] > 0
    assert later[LOSO_NAMES.index("log_margin_diff")] > 0
    assert later[LOSO_NAMES.index("conf_pom_diff")] != 0
    assert later[LOSO_NAMES.index("elo_ratio")] > 0
    assert later[LOSO_NAMES.index("yoy_margin_diff")] == 0.0
    form.new_season()
    _play(form, "Alpha", "Gamma", 7.0, home_pom=12.0, away_pom=-2.0, home_conf="SEC", away_conf="MAC")
    yoy = form.loso("Alpha", "Beta", 1)
    assert yoy[LOSO_NAMES.index("yoy_margin_diff")] < 0


def test_late_win_ignores_early_slates():
    form = FormBook()
    _play(form, "Alpha", "Beta", 21.0, slate=1)
    _play(form, "Alpha", "Gamma", -14.0, slate=8, away_conf="MAC")
    late = form.loso("Alpha", "Delta", 9)
    assert late[LOSO_NAMES.index("late_win_diff")] < 0
    assert late[LOSO_NAMES.index("form10_win_diff")] == 0.0


def test_conference_axis_is_walk_forward():
    form = FormBook()
    form.touch("Alpha", 12.0, "SEC")
    form.touch("Beta", -6.0, "ACC")
    before = form.conference("Alpha", "Beta", "SEC", "ACC")
    assert len(before) == len(CONF_NAMES)
    assert before[CONF_NAMES.index("p4_diff")] == 0.0
    assert before[CONF_NAMES.index("same_conf")] == 0.0
    assert before[CONF_NAMES.index("conf_pom_diff")] > 0
    _play(form, "Alpha", "Gamma", 21.0, home_pom=12.0, away_pom=-4.0, home_conf="SEC", away_conf="SEC")
    later = form.conference("Alpha", "Beta", "SEC", "ACC")
    assert later[CONF_NAMES.index("conf_win_diff")] > 0
    assert later[CONF_NAMES.index("conf_margin_diff")] > 0
    assert later[CONF_NAMES.index("ooc_win_diff")] == 0.0
    form.touch("Delta", -20.0, "MAC")
    g5 = form.conference("Alpha", "Delta", "SEC", "MAC")
    assert g5[CONF_NAMES.index("p4_diff")] == 1.0
    assert is_power("SEC") == 1.0
    assert is_power("MAC") == 0.0
