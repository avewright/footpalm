from datetime import datetime, timezone

from footpalm.espn import (
    bind_event,
    empty_log,
    kickoff_passed,
    parse_event,
    parse_fpi,
    parse_odds,
    parse_qbs,
    parse_summary,
    parse_weather,
    snapshot,
    stamp_log,
    upsert_log,
)


NOW = datetime(2026, 8, 31, 17, 0, tzinfo=timezone.utc)
KICK = "2026-09-05T16:00:00Z"


def _game(**extra):
    row = {
        "game_id": 401858425,
        "home": "Indiana",
        "away": "North Texas",
        "start": KICK,
        "completed": False,
    }
    row.update(extra)
    return row


def test_parse_weather_and_fpi_from_summary():
    summary = {
        "header": {
            "id": "401858425",
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"id": "84", "location": "Indiana"}},
                        {"homeAway": "away", "team": {"id": "249", "location": "North Texas"}},
                    ]
                }
            ],
        },
        "gameInfo": {
            "venue": {
                "fullName": "Memorial Stadium (Bloomington, IN)",
                "address": {"city": "Bloomington", "state": "IN"},
                "grass": False,
            },
            "weather": {"temperature": 82, "gust": 2, "precipitation": 5, "conditionId": "4"},
        },
        "predictor": {
            "homeTeam": {"gameProjection": "98.2"},
            "awayTeam": {"gameProjection": "1.8"},
        },
        "pickcenter": [{"provider": {"name": "DraftKings"}, "spread": -40.5, "overUnder": 55.5}],
    }
    row = parse_summary(summary)
    assert row["weather"]["temperature"] == 82
    assert row["weather"]["city"] == "Bloomington"
    assert row["weather"]["grass"] is False
    assert row["fpi"]["home_win"] == 0.982
    assert row["odds"]["spread"] == -40.5
    assert row["odds"]["source"] == "draftkings"


def test_parse_qbs_from_roster_groups():
    roster = {
        "athletes": [
            {
                "position": "offense",
                "items": [
                    {
                        "displayName": "Josh Hoover",
                        "jersey": "10",
                        "position": {"abbreviation": "QB"},
                        "experience": {"displayValue": "Senior"},
                    },
                    {"displayName": "Omar Cooper", "position": {"abbreviation": "WR"}},
                ],
            }
        ]
    }
    qbs = parse_qbs(roster)
    assert qbs == [{"name": "Josh Hoover", "jersey": "10", "year": "Senior", "group": "offense"}]


def test_bind_event_uses_location():
    event = parse_event(
        {
            "id": "401858425",
            "date": KICK,
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"id": "84", "location": "Indiana", "displayName": "Indiana Hoosiers"}},
                        {"homeAway": "away", "team": {"id": "249", "location": "North Texas"}},
                    ]
                }
            ],
        }
    )
    assert event is not None
    assert bind_event(event, "Indiana", "North Texas") is not None


def test_kickoff_lock_freezes_weather():
    log = empty_log(2026)
    game = _game()
    upsert_log(log, game, {"weather": {"temperature": 80}, "espn_id": "1"}, NOW)
    later = datetime(2026, 9, 5, 17, 0, tzinfo=timezone.utc)
    upsert_log(log, game, {"weather": {"temperature": 50}, "espn_id": "1"}, later)
    entry = log["games"]["id:401858425"]
    assert entry["locked"] is True
    assert entry["weather"]["temperature"] == 80
    assert len(entry["snaps"]) == 1


def test_snaps_append_when_forecast_moves():
    log = empty_log(2026)
    game = _game()
    upsert_log(log, game, {"weather": {"temperature": 80}}, NOW)
    upsert_log(log, game, {"weather": {"temperature": 76}}, NOW.replace(hour=20))
    entry = log["games"]["id:401858425"]
    assert entry["locked"] is False
    assert entry["weather"]["temperature"] == 76
    assert [s["weather"]["temperature"] for s in entry["snaps"]] == [80, 76]


def test_snapshot_matches_and_stamps(tmp_path):
    games = [_game()]
    event = {
        "espn_id": "401858425",
        "start": KICK,
        "home_raw": "Indiana",
        "away_raw": "North Texas",
        "home_id": "84",
        "away_id": "249",
        "odds": {"source": "draftkings", "spread": -40.5, "total": 55.5},
    }

    def board(_date):
        return [event]

    def summary(_eid):
        return {
            "weather": parse_weather(
                {
                    "venue": {"fullName": "Memorial Stadium", "address": {"city": "Bloomington", "state": "IN"}},
                    "weather": {"temperature": 82, "gust": 2, "precipitation": 5},
                }
            ),
            "fpi": parse_fpi({"homeTeam": {"gameProjection": "98.2"}}),
            "odds": parse_odds([{"provider": {"name": "DraftKings"}, "spread": -40.5, "overUnder": 55.5}]),
        }

    def roster(tid):
        return [{"name": "Josh Hoover", "jersey": "10", "year": "Senior"}] if tid == "84" else [{"name": "Drew Mestemaker"}]

    n = snapshot(
        tmp_path,
        2026,
        games,
        now=NOW,
        fetch_board=board,
        fetch_sum=summary,
        fetch_ros=roster,
    )
    assert n == 1
    espn = games[0]["espn"]
    assert espn["weather"]["temperature"] == 82
    assert espn["qbs"]["home"][0]["name"] == "Josh Hoover"
    assert espn["qbs"]["away"][0]["name"] == "Drew Mestemaker"
    assert espn["locked"] is False
    assert espn["snaps"] == 1
    assert kickoff_passed(games[0], NOW) is False


def test_stamp_skips_games_without_a_row():
    games = [_game(), _game(game_id=2, home="Alabama", away="East Carolina")]
    log = empty_log(2026)
    upsert_log(log, games[0], {"weather": {"temperature": 70}}, NOW)
    assert stamp_log(games, log) == 1
    assert "espn" in games[0]
    assert "espn" not in games[1]
