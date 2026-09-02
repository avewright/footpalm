import json
from pathlib import Path

from footpalm.ask import Session, Warehouse


def _warehouse(tmp_path: Path) -> Warehouse:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "ratings-2025.json").write_text(
        json.dumps(
            {
                "season": 2025,
                "teams": [
                    {
                        "rank": 1,
                        "team": "Ohio State",
                        "conf": "B1G",
                        "wins": 12,
                        "losses": 2,
                        "pom": 31.2,
                        "elo": 1889,
                        "adjo": 15.6,
                        "adjd": -16.2,
                        "adjst": -0.6,
                        "tempo": 61.1,
                        "sos": 6.5,
                        "luck": 0.02,
                        "nil_roster": 40420000,
                    },
                    {
                        "rank": 20,
                        "team": "Michigan",
                        "conf": "B1G",
                        "wins": 9,
                        "losses": 4,
                        "pom": 12.0,
                        "elo": 1700,
                        "adjo": 8.0,
                        "adjd": -4.0,
                        "adjst": 0.0,
                        "tempo": 64.0,
                        "sos": 5.0,
                        "luck": -0.01,
                        "nil_roster": 22100000,
                    },
                ],
            }
        )
        + "\n"
    )
    (processed / "predictions-2025.json").write_text(
        json.dumps(
            {
                "season": 2025,
                "games": [
                    {
                        "season": 2025,
                        "week": 10,
                        "home": "Ohio State",
                        "away": "Michigan",
                        "pred_margin": 14.0,
                        "home_win_prob": 0.82,
                        "actual_home": 21,
                        "actual_away": 24,
                        "actual_margin": -3,
                        "home_won": 0,
                    },
                    {
                        "season": 2025,
                        "week": 1,
                        "home": "Ohio State",
                        "away": "Akron",
                        "pred_margin": 40.0,
                        "home_win_prob": 0.99,
                        "actual_home": 52,
                        "actual_away": 6,
                        "actual_margin": 46,
                        "home_won": 1,
                    },
                    {
                        "season": 2025,
                        "week": 2,
                        "home": "Indiana",
                        "away": "North Texas",
                        "pred_margin": 21.4,
                        "home_win_prob": 0.86,
                        "spread": -40.5,
                        "start": "2026-09-05T16:00:00.000Z",
                        "completed": False,
                        "books": {"polymarket": {"spread": -40.5}},
                    },
                    {
                        "season": 2025,
                        "week": 2,
                        "home": "Nebraska",
                        "away": "Ohio",
                        "pred_margin": 14.9,
                        "home_win_prob": 0.78,
                        "spread": -23.5,
                        "start": "2026-09-05T19:30:00.000Z",
                        "completed": False,
                        "books": {"polymarket": {"spread": -23.5}},
                    },
                ],
            }
        )
        + "\n"
    )
    (processed / "research.json").write_text(
        json.dumps({"promoted": "temperature", "conclusion": "Promoted temperature.", "holdout_season": 2025})
        + "\n"
    )
    (processed / "people-2025.json").write_text(
        json.dumps(
            {
                "season": 2025,
                "people": [
                    {
                        "name": "Eli Stowers",
                        "team": "Vanderbilt",
                        "pos": "TE",
                        "class": "SR",
                        "recruit": {"stars": 3, "year": 2021},
                        "portal": [{"season": 2023, "origin": "LSU", "destination": "Vanderbilt"}],
                        "draft": {"year": 2026, "round": 2, "overall": 54, "nfl": "Philadelphia"},
                    },
                    {
                        "name": "Wyatt Young",
                        "team": "North Texas",
                        "pos": "WR",
                        "class": "JR",
                    },
                ],
            }
        )
        + "\n"
    )
    (processed / "draft.json").write_text(
        json.dumps(
            {
                "years": [2025, 2026],
                "picks": [
                    {
                        "name": "Cam Ward",
                        "college": "Miami",
                        "year": 2025,
                        "round": 1,
                        "pick": 1,
                        "overall": 1,
                        "nfl": "Tennessee",
                        "position": "Quarterback",
                    },
                    {
                        "name": "Eli Stowers",
                        "college": "Vanderbilt",
                        "year": 2026,
                        "round": 2,
                        "pick": 22,
                        "overall": 54,
                        "nfl": "Philadelphia",
                        "position": "Tight End",
                    },
                    {
                        "name": "Kendrick Law",
                        "college": "Kentucky",
                        "year": 2026,
                        "round": 5,
                        "pick": 4,
                        "overall": 168,
                        "nfl": "Detroit",
                        "position": "Wide Receiver",
                    },
                ],
            }
        )
        + "\n"
    )
    (processed / "leaders-2025.json").write_text(
        json.dumps(
            {
                "season": 2025,
                "rushing": [
                    {"rank": 1, "player": "Cam Cook", "team": "Jacksonville State", "att": 283, "yds": 1586, "td": 11, "ypc": 5.6}
                ],
                "yac": [
                    {"rank": 1, "player": "Wyatt Young", "team": "North Texas", "rec": 67, "yds": 1103, "td": 7, "yac": 460, "yac_plays": 39}
                ],
                "receiving": [],
                "passing": [],
            }
        )
        + "\n"
    )
    return Warehouse(tmp_path)


def test_resolve_aliases_and_fragments(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2025)
    assert session.find_teams("osu")["matches"] == ["Ohio State"]
    assert session.find_teams("ohio")["matches"] == ["Ohio State"]
    assert session.get_team("ohio state")["team"]["pom"] == 31.2


def test_leaderboard_and_compare(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2025)
    board = session.leaderboard(metric="pom", n=2)
    assert [t["team"] for t in board["teams"]] == ["Ohio State", "Michigan"]
    cmp = session.compare_teams(["Ohio State", "Michigan"])
    assert len(cmp["teams"]) == 2
    assert cmp["missing"] == []


def test_games_and_upsets(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2025)
    one = session.get_game("Ohio State", "Michigan")
    assert one["games"][0]["home_won"] == 0
    upsets = session.list_games(team="Ohio State", upsets=True)
    assert upsets["n"] == 1
    assert upsets["games"][0]["away"] == "Michigan"


def test_show_collects_cards(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2025)
    session.show("stats", "Ohio State", items=[{"label": "Pom", "value": 31.2}])
    session.show(
        "bars",
        "Pom",
        items=[{"label": "Ohio State", "value": 31.2}, {"label": "Michigan", "value": 12.0}],
    )
    session.show("table", "Board", columns=["Team", "Pom"], rows=[["Ohio State", 31.2]])
    session.show("line", "Pom", items=[{"label": "W1", "value": 20}, {"label": "W2", "value": 31}])
    session.show(
        "graph",
        "OSU",
        nodes=[{"id": "Ohio State"}, {"id": "Michigan"}],
        edges=[{"source": "Ohio State", "target": "Michigan", "label": "+3"}],
    )
    session.show(
        "scatter",
        "NIL vs Pom",
        x_label="Roster $M",
        y_label="Pom",
        points=[{"label": "Ohio State", "x": 40.4, "y": 31.2}],
    )
    assert [c["kind"] for c in session.cards] == ["stats", "bars", "table", "line", "graph", "scatter"]


def test_catalog_and_research(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2025)
    cat = session.catalog()
    assert cat["seasons"][0]["season"] == 2025
    assert cat["seasons"][0]["teams"] == 2
    assert session.research()["promoted"] == "temperature"


def test_rushing_and_yac_leaders_draw_cards(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2026)
    rush = session.leaders("rushing", season=2025, n=5)
    assert rush["leaders"][0]["player"] == "Cam Cook"
    yac = session.leaders("yac", n=5)
    assert yac["season"] == 2025
    assert yac["leaders"][0]["yac"] == 460
    avg = session.leaders("yac_avg", season=2025, n=5)
    assert avg["stat"] == "yac_avg"
    assert avg["leaders"][0]["player"] == "Wyatt Young"
    assert avg["leaders"][0]["yac_avg"] == 11.8
    kinds = [c["kind"] for c in session.cards]
    assert kinds.count("table") == 3
    assert kinds.count("bars") == 0


def test_drafted_matches_names_and_jr(tmp_path: Path):
    from footpalm.draft import colleges_match, person_key

    assert person_key("Marcus Sanders Jr.") == person_key("Marcus Sanders")
    assert colleges_match("Kennesaw St", "Kennesaw State")
    session = Session(_warehouse(tmp_path), 2025)
    out = session.drafted(
        players=["Eli Stowers", "Wyatt Young", "Kendrick Law"],
        colleges=["Vanderbilt", "North Texas", "Kentucky"],
    )
    assert out["asked"] == 3
    assert out["drafted"] == 2
    assert out["undrafted"] == 1
    by_name = {r["player"]: r for r in out["players"]}
    assert by_name["Eli Stowers"]["overall"] == 54
    assert by_name["Wyatt Young"]["drafted"] is False
    assert any(c["kind"] == "table" and c["title"] == "NFL draft" for c in session.cards)


def test_drafted_lists_latest_class(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2026)
    out = session.drafted()
    assert out["year"] == 2026
    names = [p["name"] for p in out["picks"]]
    assert "Eli Stowers" in names
    assert "Cam Ward" not in names
    assert any(c["kind"] == "table" and "2026" in (c.get("title") or "") for c in session.cards)


def test_drafted_year_filter(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2026)
    out = session.drafted(year=2025)
    assert [p["name"] for p in out["picks"]] == ["Cam Ward"]


def test_resolve_ask_season_keeps_draft_on_current():
    from footpalm.ask import resolve_ask_season

    season, last = resolve_ask_season("do you know who got drafted last year", 2026)
    assert season == 2026
    assert last == 2025
    season, last = resolve_ask_season("rushing leaders last year", 2026)
    assert season == 2025
    assert last == 2025
    season, last = resolve_ask_season("highest ev bets for next saturday", 2025, live=2026)
    assert season == 2026
    assert last == 2025


def test_clean_answer_drops_markdown_table():
    from footpalm.ask import _clean_answer

    text = _clean_answer(
        "Cam Cook led.\n\n| Player | Yds |\n|---|---|\n| Cam Cook | 1586 |\n",
        [{"kind": "table", "title": "Rushing"}],
    )
    assert "Cam Cook led." in text
    assert "|" not in text


def test_clean_answer_drops_numbered_dump():
    from footpalm.ask import _clean_answer, _prune_cards

    blob = (
        "The 2026 NFL draft's first round (top 32): "
        "1. Fernando Mendoza, QB (Indiana) -> Las Vegas "
        "2. David Bailey, EDGE (Texas Tech) -> New York "
        "3. Jeremiyah Love, RB (Notre Dame) -> Arizona "
        "Ohio State leads with four first-rounders; Miami has three."
    )
    text = _clean_answer(blob, [{"kind": "table", "title": "NFL draft · 2026"}])
    assert "Ohio State leads" in text
    assert "Fernando Mendoza" not in text
    assert "1." not in text

    bold = (
        "The 2026 NFL draft's first round (top 32): "
        "**1. Fernando Mendoza, QB** (Indiana) -> Las Vegas "
        "**2. David Bailey, EDGE** (Texas Tech) -> New York "
        "**3. Jeremiyah Love, RB** (Notre Dame) -> Arizona"
    )
    text = _clean_answer(bold, [{"kind": "table", "title": "NFL draft · 2026"}])
    assert "Fernando Mendoza" not in text
    assert text == ""

    caption = _clean_answer(
        "32 first-round picks. Fernando Mendoza (Indiana QB) went No. 1 to Las Vegas.",
        [{"kind": "table", "title": "NFL draft · 2026"}],
    )
    assert caption == "32 first-round picks. Fernando Mendoza (Indiana QB) went No. 1 to Las Vegas."

    cards = _prune_cards(
        [
            {"kind": "table", "title": "NFL draft · 2026", "rows": []},
            {"kind": "stats", "title": "NFL draft · 2026", "items": []},
        ]
    )
    assert [c["kind"] for c in cards] == ["table"]

    junk = _prune_cards(
        [
            {"kind": "table", "title": "Games · 2025", "columns": ["Wk"], "rows": []},
            {"kind": "table", "title": "Highest EV · Saturday · 2026", "columns": ["EV"], "rows": []},
            {"kind": "stats", "title": "Research", "items": []},
            {"kind": "stats", "title": "Backtest · 2025", "items": []},
            {"kind": "table", "title": "North Texas at Indiana", "rows": []},
        ],
        "highest ev bets for next saturday",
    )
    assert [c["title"] for c in junk] == ["Highest EV · Saturday · 2026"]


def test_search_open_list(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2025)
    found = session.search("stowers", kind="player")
    assert found["n"] == 1
    assert found["hits"][0]["name"] == "Eli Stowers"
    team = session.search("osu", kind="team")
    assert team["hits"][0]["name"] == "Ohio State"
    pick = session.search("cam ward", kind="draft")
    assert pick["hits"][0]["year"] == 2025
    opened = session.open_record(kind="team", name="Ohio State")
    assert opened["team"]["pom"] == 31.2
    assert any(c["kind"] == "stats" and c["title"] == "Ohio State" for c in session.cards)
    board = session.list_board("ratings", n=2)
    assert [r["team"] for r in board["teams"]] == ["Ohio State", "Michigan"]
    assert any(c["kind"] == "table" and c["title"] == "Ratings · 2025" for c in session.cards)
    draft = session.list_board("draft", year=2026)
    assert draft["year"] == 2026
    rush = session.list_board("leaders", stat="rushing", season=2025)
    assert rush["leaders"][0]["player"] == "Cam Cook"


def test_player_dossier_and_board(tmp_path: Path):
    session = Session(_warehouse(tmp_path), 2025)
    one = session.player(name="Eli Stowers", team="Vanderbilt")
    assert one["found"] == 1
    assert one["people"][0]["draft"]["overall"] == 54
    assert one["people"][0]["portal"][0]["origin"] == "LSU"
    board = session.player(names=["Eli Stowers", "Wyatt Young"], colleges=["Vanderbilt", "North Texas"])
    assert board["drafted"] == 1
    assert board["portal"] == 1
    assert board["missing"] == []


def test_saturday_ev_board(tmp_path: Path):
    from datetime import datetime, timezone

    from footpalm.ask import _next_saturday

    assert _next_saturday(datetime(2026, 8, 30, 18, tzinfo=timezone.utc)).isoformat() == "2026-09-05"
    session = Session(
        _warehouse(tmp_path),
        2025,
        question="What are my projections and highest ev bets for next saturday",
        now=datetime(2026, 8, 30, 18, tzinfo=timezone.utc),
    )
    out = session.list_board("games")
    assert out["when"] == "saturday"
    assert out["sort"] == "ev"
    assert out["n"] == 2
    assert [g["away"] for g in out["games"]] == ["North Texas", "Ohio"]
    card = next(c for c in session.cards if c["kind"] == "table")
    assert card["title"] == "Highest EV · Saturday · 2025"
    assert card["columns"] == ["When", "Away", "Home", "Us", "Mkt", "Pick", "EV"]
    session.list_board("games", season=2025, completed=True)
    assert len([c for c in session.cards if c["kind"] == "table"]) == 1
    assert session.cards[-1]["title"] == "Highest EV · Saturday · 2025"


def test_money_scatter(tmp_path: Path):
    from footpalm.ask import _want_scatter

    assert _want_scatter("visualize NIL on the teams vs Pom 2025")
    assert not _want_scatter("who is ranked first")
    session = Session(_warehouse(tmp_path), 2025)
    out = session.list_board("money")
    assert out["n"] == 2
    card = session.cards[-1]
    assert card["kind"] == "scatter"
    assert card["title"] == "NIL vs Pom · 2025"
    assert card["points"][0]["label"] == "Ohio State"
    assert card["points"][0]["x"] == 40.4


def test_merge_delta_joins_stream_chunks():
    from footpalm.ask import _merge_delta

    msg: dict = {"role": "assistant", "content": ""}
    _merge_delta(msg, {"content": "Ohio "})
    _merge_delta(msg, {"content": "State"})
    _merge_delta(
        msg,
        {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "li", "arguments": ""}}]},
    )
    _merge_delta(msg, {"tool_calls": [{"index": 0, "function": {"name": "st", "arguments": "{\"q\""}}]})
    assert msg["content"] == "Ohio State"
    assert msg["tool_calls"][0]["id"] == "c1"
    assert msg["tool_calls"][0]["function"]["name"] == "list"
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"q"'


