import json
from pathlib import Path

from footpalm.cfbd import dataset_path
from footpalm.people import build_season, find_people, write_all


def _dump(root: Path, name: str, year: int, rows: list) -> None:
    dest = dataset_path(root, name, year)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows) + "\n")


def test_join_roster_portal_recruit_draft_usage(tmp_path: Path):
    _dump(
        tmp_path,
        "roster",
        2025,
        [
            {
                "firstName": "Eli",
                "lastName": "Stowers",
                "team": "Vanderbilt",
                "position": "TE",
                "year": 4,
                "height": 77,
                "weight": 240,
                "jersey": 9,
                "homeCity": "Brentwood",
                "homeState": "TN",
            },
            {
                "firstName": "Wyatt",
                "lastName": "Young",
                "team": "North Texas",
                "position": "WR",
                "year": 3,
                "height": 72,
                "weight": 185,
                "jersey": 1,
                "homeCity": "Denton",
                "homeState": "TX",
            },
        ],
    )
    _dump(
        tmp_path,
        "usage",
        2025,
        [{"name": "Eli Stowers", "team": "Vanderbilt", "usage": {"overall": 0.12, "pass": 0.2, "rush": 0}}],
    )
    _dump(
        tmp_path,
        "portal",
        2023,
        [
            {
                "season": 2023,
                "firstName": "Eli",
                "lastName": "Stowers",
                "origin": "LSU",
                "destination": "Vanderbilt",
                "position": "TE",
            }
        ],
    )
    _dump(
        tmp_path,
        "recruits",
        2021,
        [{"name": "Eli Stowers", "committedTo": "LSU", "stars": 3, "ranking": 900, "year": 2021, "position": "TE"}],
    )
    _dump(
        tmp_path,
        "stats_receiving",
        2025,
        [
            {"player": "Eli Stowers", "team": "Vanderbilt", "category": "receiving", "statType": "REC", "stat": "58"},
            {"player": "Eli Stowers", "team": "Vanderbilt", "category": "receiving", "statType": "YDS", "stat": "700"},
        ],
    )
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "draft.json").write_text(
        json.dumps(
            {
                "picks": [
                    {
                        "name": "Eli Stowers",
                        "college": "Vanderbilt",
                        "year": 2026,
                        "round": 2,
                        "overall": 54,
                        "nfl": "Philadelphia",
                    }
                ]
            }
        )
        + "\n"
    )
    write_all(tmp_path, [2025])
    payload = json.loads((processed / "people-2025.json").read_text())
    eli = find_people(payload, "Eli Stowers", "Vanderbilt")[0]
    assert eli["pos"] == "TE"
    assert eli["class"] == "SR"
    assert eli["usage"]["overall"] == 0.12
    assert eli["stats"]["receiving"]["rec"] == 58
    assert eli["portal"][0]["origin"] == "LSU"
    assert eli["draft"]["overall"] == 54
    assert eli["recruit"]["stars"] == 3
    wyatt = find_people(payload, "Wyatt Young")[0]
    assert "draft" not in wyatt


def test_build_season_skips_empty_roster(tmp_path: Path):
    payload = build_season(tmp_path, 2024, draft_picks=[], recruits={}, portals={})
    assert payload["n"] == 0
    assert payload["people"] == []
