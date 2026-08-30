from footpalm.project import resolve_name
from footpalm.rate import RatingBook, TeamRating, dump_book, load_book


def test_resolve_prefers_existing_palm_name():
    known = {"App State", "Ohio State"}
    assert resolve_name("App State", known) == "App State"
    assert resolve_name("Ohio State", known) == "Ohio State"


def test_predict_slate_batches_without_tabpfn():
    from footpalm.predict import predict_slate

    book = RatingBook(
        2025,
        None,
        {
            "Alpha": TeamRating("Alpha", 0.1, -0.1, 0.0, 130, 12.0, 8.0, -4.0, 12),
            "Beta": TeamRating("Beta", -0.05, 0.05, 0.0, 130, -6.0, -4.0, 2.0, 12),
        },
        0.02,
        0.0,
        65.0,
        27.0,
        {"Alpha", "Beta"},
        {},
    )
    preds = predict_slate(book, [{"home": "Alpha", "away": "Beta", "neutral": False}])
    assert len(preds) == 1
    assert preds[0]["home_win_prob"] > 0.7


def test_week0_copies_last_season():
    from footpalm.rate import publish_preseason

    book = RatingBook(
        2026,
        None,
        {
            "Alpha": TeamRating("Alpha", 0.1, -0.1, 0.0, 130, 12.0, 8.0, -4.0, 0),
            "Beta": TeamRating("Beta", -0.05, 0.05, 0.0, 130, -6.0, -4.0, 2.0, 0),
        },
        0.02,
        0.0,
        65.0,
        27.0,
        {"Alpha", "Beta"},
        {"Alpha": "SEC", "Beta": "B1G"},
    )
    table = publish_preseason(
        book,
        prior_season=2025,
        prior_table={
            "teams": [
                {
                    "team": "Alpha",
                    "wins": 11,
                    "losses": 2,
                    "games": 13,
                    "pom": 18.4,
                    "adjo": 10.1,
                    "adjd": -7.8,
                    "adjst": 0.5,
                    "tempo": 66.2,
                    "sos": 4.1,
                    "luck": 0.02,
                    "elo": 1760,
                    "nil_roster": 40_000_000,
                    "nil_quality": "published",
                }
            ]
        },
    )
    alpha = next(row for row in table["teams"] if row["team"] == "Alpha")
    assert table["week"] == 0
    assert "Week 0" in table["method"]
    assert alpha["wins"] == 11
    assert alpha["losses"] == 2
    assert alpha["pom"] == 18.4
    assert alpha["elo"] == 1760
    assert alpha["nil_roster"] == 40_000_000
    assert table["home_adv_epa"] == 0.02


def test_book_roundtrip():
    book = RatingBook(
        2025,
        None,
        {"Alpha": TeamRating("Alpha", 0.1, -0.1, 0.0, 130, 12.0, 8.0, -4.0, 12)},
        0.02,
        0.0,
        65.0,
        27.0,
        {"Alpha"},
        {"Alpha": "SEC"},
    )
    again = load_book(dump_book(book))
    assert again.pom("Alpha") == 12.0
    assert again.games_played("Alpha") == 12
