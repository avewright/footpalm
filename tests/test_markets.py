from footpalm.markets import american, attach, bind_book, match_team, parse_event


def test_american_odds():
    assert american(0.5) == -100
    assert american(0.75) == -300
    assert american(0.25) == 300


def test_match_team_strips_mascot():
    known = {"UNLV", "North Texas", "Indiana", "Hawai'i", "Stanford", "NC State"}
    assert match_team("UNLV Runnin'", known) == "UNLV"
    assert match_team("Hawaii", known) == "Hawai'i"
    assert match_team("North Carolina State", known) == "NC State"
    assert match_team("North Texas", known) == "North Texas"
    assert match_team("Louisiana-Monroe", {"UL Monroe", "Mississippi State"}) == "UL Monroe"


def test_parse_and_bind_home_favorite():
    event = parse_event(
        {
            "title": "North Texas vs. Indiana",
            "slug": "cfb-ntx-ind-2026-09-05",
            "markets": [
                {
                    "sportsMarketType": "moneyline",
                    "outcomes": '["North Texas", "Indiana"]',
                    "outcomePrices": '["0.09", "0.91"]',
                    "gameStartTime": "2026-09-05 16:00:00+00",
                },
                {
                    "sportsMarketType": "spreads",
                    "line": -40.5,
                    "outcomes": '["Indiana", "North Texas"]',
                    "outcomePrices": '["0.48", "0.52"]',
                },
            ],
        }
    )
    assert event is not None
    book = bind_book(event, home="Indiana", away="North Texas")
    assert book is not None
    assert book["spread"] == -40.5
    assert book["ml_home"] == 0.91
    assert book["ml_away"] == 0.09
    assert book["ml_home_american"] == -1011
    assert book["spread_p_home"] == 0.48


def test_attach_requires_same_day():
    events = [
        parse_event(
            {
                "title": "San Jose State vs. USC",
                "slug": "cfb-sjst-usc-2026-08-29",
                "markets": [
                    {
                        "sportsMarketType": "moneyline",
                        "outcomes": '["San Jose State", "USC"]',
                        "outcomePrices": '["0.02", "0.98"]',
                        "gameStartTime": "2026-08-29 19:00:00+00",
                    }
                ],
            }
        )
    ]
    games = [
        {"home": "USC", "away": "San José State", "start": "2026-08-29T19:00:00.000Z"},
        {"home": "Eastern Michigan", "away": "San José State", "start": "2026-09-04T23:00:00.000Z"},
    ]
    n = attach(games, events)
    assert n == 1
    assert games[0]["books"]["polymarket"]["ml_home"] == 0.98
    assert "books" not in games[1]
