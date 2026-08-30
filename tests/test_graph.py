import pandas as pd

from footpalm.graph import build_graph


def _sides(games: list[tuple]) -> pd.DataFrame:
    rows = []
    for slate, home, away, home_pts, away_pts in games:
        for team, opp, pts, opp_pts in (
            (home, away, home_pts, away_pts),
            (away, home, away_pts, home_pts),
        ):
            rows.append(
                {
                    "game_id": slate,
                    "slate": slate,
                    "week": slate,
                    "team": team,
                    "opponent": opp,
                    "home_team": home,
                    "away_team": away,
                    "points": pts,
                    "opp_points": opp_pts,
                    "neutral_site": False,
                }
            )
    return pd.DataFrame(rows)


def test_better_teams_are_bigger_than_their_neighbors():
    sides = _sides(
        [
            (1, "Alpha", "Beta", 35, 14),
            (2, "Beta", "Gamma", 28, 10),
            (3, "Alpha", "Gamma", 42, 7),
        ]
    )
    ratings = {
        "season": 2025,
        "teams": [
            {"team": "Alpha", "conf": "SEC", "pom": 20.0, "wins": 2, "losses": 0},
            {"team": "Beta", "conf": "SEC", "pom": 4.0, "wins": 1, "losses": 1},
            {"team": "Gamma", "conf": "SEC", "pom": -12.0, "wins": 0, "losses": 2},
        ],
    }
    graph = build_graph(sides, {"Alpha", "Beta", "Gamma"}, {"Alpha": "SEC", "Beta": "SEC", "Gamma": "SEC"}, ratings, {})
    by_team = {n["team"]: n for n in graph["nodes"]}

    assert by_team["Alpha"]["size"] > by_team["Beta"]["size"] > by_team["Gamma"]["size"]
    assert by_team["Alpha"]["winningness"] == 1.0
    assert by_team["Beta"]["winningness"] == 0.5
    assert by_team["Gamma"]["winningness"] == 0.0
    assert by_team["Alpha"]["vs_neighbors"] > 0
    assert by_team["Gamma"]["vs_neighbors"] < 0
    assert by_team["Alpha"]["pagerank"] > by_team["Gamma"]["pagerank"]
    assert by_team["Alpha"]["margin_vs"] > 0
    assert {e["source"] for e in graph["edges"]} == {"Alpha", "Beta"}
    assert all(e["source"] != "Gamma" for e in graph["edges"])
    net = graph["network"]
    assert net["n"] == 3
    assert net["undirected_edges"] == 3
    assert net["average_path"] == 1
    assert net["bound_path"] <= net["average_path"] + 1e-9
    assert net["distances"] == {"1": 3}
    assert graph["nodes"][0]["degree"] == 2


def test_directed_cycle_is_a_circuit():
    sides = _sides(
        [
            (1, "Alpha", "Beta", 21, 14),
            (2, "Beta", "Gamma", 17, 10),
            (3, "Gamma", "Alpha", 24, 21),
        ]
    )
    ratings = {
        "season": 2025,
        "teams": [
            {"team": "Alpha", "conf": "SEC", "pom": 10.0, "wins": 1, "losses": 1},
            {"team": "Beta", "conf": "SEC", "pom": 4.0, "wins": 1, "losses": 1},
            {"team": "Gamma", "conf": "SEC", "pom": -2.0, "wins": 1, "losses": 1},
        ],
    }
    graph = build_graph(sides, {"Alpha", "Beta", "Gamma"}, {"Alpha": "SEC", "Beta": "SEC", "Gamma": "SEC"}, ratings, {})
    cycles = graph["network"]["cycles"]
    assert len(cycles) == 1
    assert set(cycles[0]["teams"]) == {"Alpha", "Beta", "Gamma"}
    assert graph["network"]["mst"]
    assert "sx" in graph["nodes"][0]
    assert "wx" in graph["nodes"][0]
    assert graph["nodes"][0]["fiedler"] is not None
    assert graph["network"]["algebraic_connectivity"] > 0
