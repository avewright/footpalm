from footpalm.specials import SPECIAL_NAMES, SpecialSnap, SpecialsBook


def test_first_game_zero_then_streak_and_momentum():
    book = SpecialsBook()
    assert len(book.vector("Alpha", "Beta")) == len(SPECIAL_NAMES)
    assert (book.vector("Alpha", "Beta") == 0).all()
    book.apply(None, "Alpha", "Beta", home_won=True, margin=21.0)
    after_one = book.vector("Alpha", "Beta")
    assert after_one[SPECIAL_NAMES.index("win_streak_diff")] > 0
    for _ in range(3):
        book.apply(None, "Alpha", "Gamma", home_won=True, margin=14.0)
    book.apply(None, "Alpha", "Delta", home_won=False, margin=-21.0)
    later = book.vector("Alpha", "Omega")
    assert later[SPECIAL_NAMES.index("win_streak_diff")] < 0
    assert later[SPECIAL_NAMES.index("margin_momentum_diff")] < 0


def test_kicker_and_punt_walk_forward():
    book = SpecialsBook()
    book.apply(
        {
            "Alpha": SpecialSnap(
                fg_make_yds=80, fg_make_n=2, fg_made=2, fg_exp=1.4, fg_att=2,
                punt_n=3, punt_yds=120, plays=60,
            ),
            "Beta": SpecialSnap(
                fg_make_yds=30, fg_make_n=1, fg_made=1, fg_exp=1.2, fg_att=2,
                punt_n=6, punt_yds=180, plays=50,
            ),
        },
        "Alpha",
        "Beta",
        home_won=True,
        margin=7.0,
    )
    later = book.vector("Alpha", "Beta")
    assert later[SPECIAL_NAMES.index("fg_avg_make_diff")] > 0
    assert later[SPECIAL_NAMES.index("fg_make_adj_diff")] > 0
    assert later[SPECIAL_NAMES.index("punt_rate_diff")] < 0
    assert later[SPECIAL_NAMES.index("plays_pg_diff")] > 0
