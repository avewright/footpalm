from datetime import datetime, timezone

from footpalm.markets import (
    american,
    attach,
    bind_book,
    empty_log,
    live_ml,
    match_team,
    parse_event,
    parse_kalshi_game,
    parse_kalshi_spread,
    upsert_log,
)


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


def test_parse_skips_settled_moneyline():
    event = parse_event(
        {
            "title": "Hawai'i vs. Stanford",
            "slug": "cfb-haw-stan-2026-08-30",
            "markets": [
                {
                    "sportsMarketType": "moneyline",
                    "outcomes": '["Hawai\'i", "Stanford"]',
                    "outcomePrices": '["0.0", "1.0"]',
                    "gameStartTime": "2026-08-30 02:30:00+00",
                },
                {
                    "sportsMarketType": "spreads",
                    "line": -4.0,
                    "outcomes": '["Stanford", "Hawai\'i"]',
                    "outcomePrices": '["0.52", "0.48"]',
                },
            ],
        }
    )
    assert event is not None
    assert "ml" not in event
    book = bind_book(event, home="Stanford", away="Hawai'i")
    assert book is not None
    assert "ml_home" not in book
    assert book["spread"] == -4.0


def test_attach_keeps_existing_books_when_event_is_gone():
    games = [
        {
            "home": "Stanford",
            "away": "Hawai'i",
            "start": "2026-08-30T02:30:00.000Z",
            "books": {"polymarket": {"ml_home": 0.68, "spread": -4.0}},
        }
    ]
    log = empty_log(2026)
    assert attach(games, [], log) == 1
    assert games[0]["books"]["polymarket"]["ml_home"] == 0.68
    assert live_ml(list(log["games"].values())[0]["polymarket"]["ml_home"])


def test_log_locks_moneyline_after_kickoff():
    now = datetime(2026, 8, 30, 12, tzinfo=timezone.utc)
    game = {
        "season": 2026,
        "week": 1,
        "home": "Stanford",
        "away": "Hawai'i",
        "start": "2026-08-30T02:30:00.000Z",
        "completed": True,
    }
    log = empty_log(2026)
    upsert_log(log, game, {"ml_home": 0.68, "ml_away": 0.32, "spread": -4.0}, now)
    upsert_log(log, game, {"ml_home": 0.91, "ml_away": 0.09, "spread": -5.5}, now)
    book = list(log["games"].values())[0]
    assert book["locked"] is True
    assert book["polymarket"]["ml_home"] == 0.68
    assert book["polymarket"]["spread"] == -5.5


def test_parse_kalshi_game_and_spread_main_line():
    game = parse_kalshi_game(
        {
            "title": "UMass vs Rutgers",
            "event_ticker": "KXNCAAFGAME-26SEP03MASSRUTG",
            "markets": [
                {
                    "yes_sub_title": "Rutgers",
                    "yes_bid_dollars": "0.9700",
                    "yes_ask_dollars": "0.9800",
                    "occurrence_datetime": "2026-09-04T01:00:00Z",
                },
                {
                    "yes_sub_title": "UMass",
                    "yes_bid_dollars": "0.0200",
                    "yes_ask_dollars": "0.0300",
                    "occurrence_datetime": "2026-09-04T01:00:00Z",
                },
            ],
        }
    )
    assert game is not None
    assert game["source"] == "kalshi"
    assert game["home_raw"] == "Rutgers"
    assert {name for name, _p in game["ml"]} == {"Rutgers", "UMass"}

    spread = parse_kalshi_spread(
        {
            "title": "UMass vs Rutgers: Spread",
            "event_ticker": "KXNCAAFSPREAD-26SEP03MASSRUTG",
            "markets": [
                {
                    "yes_sub_title": "Rutgers wins by over 51.5 points",
                    "floor_strike": 51.5,
                    "yes_bid_dollars": "0.0700",
                    "yes_ask_dollars": "0.1500",
                },
                {
                    "yes_sub_title": "Rutgers wins by over 27.5 points",
                    "floor_strike": 27.5,
                    "yes_bid_dollars": "0.5200",
                    "yes_ask_dollars": "0.5400",
                },
                {
                    "yes_sub_title": "Rutgers wins by over 21.5 points",
                    "floor_strike": 21.5,
                    "yes_bid_dollars": "0.6500",
                    "yes_ask_dollars": "0.7000",
                },
            ],
        }
    )
    assert spread is not None
    assert spread["spread_line"] == -27.5
    book = bind_book({**game, **spread}, home="Rutgers", away="Massachusetts")
    assert book is not None
    assert book["source"] == "kalshi"
    assert book["spread"] == -27.5
    assert book["ml_home"] == 0.975


def test_kalshi_upsert_does_not_wipe_polymarket():
    now = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    game = {"home": "Rutgers", "away": "Massachusetts", "start": "2026-09-04T01:00:00Z"}
    log = empty_log(2026)
    upsert_log(log, game, {"source": "polymarket", "spread": -40.5, "ml_home": 0.91}, now)
    upsert_log(log, game, {"source": "kalshi", "spread": -27.5, "ml_home": 0.97}, now)
    row = list(log["games"].values())[0]
    assert row["polymarket"]["spread"] == -40.5
    assert row["kalshi"]["spread"] == -27.5
