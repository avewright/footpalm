from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx

from footpalm.plays import listed_games

SIZE_MIN = 8.0
SIZE_MAX = 26.0


def _finite(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _scale(values: dict[str, float], lo: float = SIZE_MIN, hi: float = SIZE_MAX) -> dict[str, float]:
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    if vmax <= vmin:
        mid = round((lo + hi) / 2, 2)
        return {key: mid for key in values}
    span = vmax - vmin
    return {key: round(lo + (hi - lo) * (value - vmin) / span, 2) for key, value in values.items()}


def _week(row) -> int | None:
    week = getattr(row, "week", None)
    if week is None or week != week:
        return None
    return int(week)


def _undirected(nodes, edges) -> nx.Graph:
    graph = nx.Graph()
    for row in nodes:
        graph.add_node(row["id"], team=row.get("team"), conf=row.get("conf"), pom=_finite(row.get("pom", row.get("palm"))))
    for edge in edges:
        if not edge.get("fbs_fbs"):
            continue
        source, target = edge["source"], edge["target"]
        if source not in graph or target not in graph or source == target:
            continue
        margin = float(edge.get("margin") or 0)
        if graph.has_edge(source, target):
            graph[source][target]["margin"] = max(graph[source][target]["margin"], margin)
        else:
            graph.add_edge(source, target, margin=margin)
    return graph


def _directed(nodes, edges) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in nodes:
        graph.add_node(row["id"], pom=_finite(row.get("pom", row.get("palm"))))
    for edge in edges:
        if not edge.get("fbs_fbs"):
            continue
        source, target = edge["source"], edge["target"]
        if source not in graph or target not in graph or source == target:
            continue
        graph.add_edge(source, target, margin=float(edge.get("margin") or 0))
    return graph


def _distance_counts(graph: nx.Graph) -> dict[str, int]:
    counts: dict[int, int] = {}
    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        for target, dist in lengths.items():
            if source < target:
                counts[dist] = counts.get(dist, 0) + 1
    return {str(key): counts[key] for key in sorted(counts)}


def _xy(layout: dict) -> dict[str, list[float]]:
    return {team: [round(float(x), 4), round(float(y), 4)] for team, (x, y) in layout.items()}


def _fiedler(graph: nx.Graph) -> dict[str, float]:
    if graph.number_of_nodes() < 3:
        return {}
    values = nx.fiedler_vector(graph, weight=None)
    return {team: round(float(value), 5) for team, value in zip(graph.nodes(), values)}


def _directed_triangles(directed: nx.DiGraph, limit: int = 12) -> list[dict]:
    cycles = []
    seen: set[tuple[str, str, str]] = set()
    for winner, loser in directed.edges():
        for third in directed.successors(loser):
            if third == winner or not directed.has_edge(third, winner):
                continue
            key = tuple(sorted((winner, loser, third)))
            if key in seen:
                continue
            seen.add(key)
            path = ((winner, loser), (loser, third), (third, winner))
            margins = [directed[a][b]["margin"] for a, b in path]
            tension = 0.0
            for a, b in path:
                tension += max(0.0, directed.nodes[b]["pom"] - directed.nodes[a]["pom"])
            cycles.append(
                {
                    "teams": [winner, loser, third],
                    "margins": [round(v, 1) for v in margins],
                    "tension": round(tension, 2),
                }
            )
    cycles.sort(key=lambda row: (-row["tension"], row["teams"]))
    return cycles[:limit]


def analyze_network(nodes, edges) -> dict:
    """Undirected who-played-whom stats. Bound is Moore-style: diameter 2 on a k-regular graph."""
    undirected = _undirected(nodes, edges)
    directed = _directed(nodes, edges)
    n = undirected.number_of_nodes()
    m = undirected.number_of_edges()
    empty = {
        "n": n,
        "undirected_edges": m,
        "degree_mean": 0.0,
        "degree_min": 0,
        "degree_max": 0,
        "components": nx.number_connected_components(undirected) if n else 0,
        "diameter": 0,
        "radius": 0,
        "average_path": 0.0,
        "bound_path": 0.0,
        "distances": {},
        "triangles": 0,
        "bridges": 0,
        "mst": [],
        "cycles": [],
        "algebraic_connectivity": 0.0,
        "layout": {},
        "tree_layout": {},
        "spectral_layout": {},
        "weight_layout": {},
        "fiedler": {},
        "betweenness": {},
        "eccentricity": {},
        "degree": {},
    }
    if n < 2 or m == 0:
        return empty

    giant = undirected.subgraph(max(nx.connected_components(undirected), key=len)).copy()
    degrees = dict(undirected.degree())
    mean_deg = sum(degrees.values()) / n
    bound = 2 - mean_deg / (n - 1)
    distances = _distance_counts(giant)
    betweenness = nx.betweenness_centrality(undirected)
    eccentricity = nx.eccentricity(giant)
    layout = nx.spring_layout(undirected, seed=7, iterations=120)
    weighted = nx.spring_layout(undirected, weight="margin", seed=11, iterations=160)
    spectral = nx.spectral_layout(giant, dim=2) if giant.number_of_nodes() >= 3 else layout
    tree = nx.maximum_spanning_tree(undirected, weight="margin")
    tree_layout = nx.spring_layout(tree, seed=3, iterations=160)
    mst = [
        {"source": a, "target": b, "margin": round(float(data.get("margin") or 0), 1)}
        for a, b, data in tree.edges(data=True)
    ]
    mst.sort(key=lambda row: (-row["margin"], row["source"], row["target"]))
    connectivity = float(nx.algebraic_connectivity(giant)) if giant.number_of_nodes() >= 3 else 0.0

    return {
        "n": n,
        "undirected_edges": m,
        "degree_mean": round(mean_deg, 2),
        "degree_min": min(degrees.values()),
        "degree_max": max(degrees.values()),
        "components": nx.number_connected_components(undirected),
        "diameter": int(nx.diameter(giant)),
        "radius": int(nx.radius(giant)),
        "average_path": round(nx.average_shortest_path_length(giant), 4),
        "bound_path": round(bound, 4),
        "algebraic_connectivity": round(connectivity, 4),
        "distances": distances,
        "triangles": int(sum(nx.triangles(undirected).values()) // 3),
        "bridges": int(sum(1 for _ in nx.bridges(undirected))),
        "mst": mst,
        "cycles": _directed_triangles(directed),
        "layout": _xy(layout),
        "tree_layout": _xy(tree_layout),
        "spectral_layout": _xy(spectral),
        "weight_layout": _xy(weighted),
        "fiedler": _fiedler(giant),
        "betweenness": {team: round(value, 5) for team, value in betweenness.items()},
        "eccentricity": {team: int(value) for team, value in eccentricity.items()},
        "degree": degrees,
    }


def _put_xy(row: dict, point: list[float] | None, x_key: str, y_key: str) -> None:
    if not point:
        return
    row[x_key] = point[0]
    row[y_key] = point[1]


def attach_network(graph: dict) -> dict:
    network = analyze_network(graph.get("nodes") or [], graph.get("edges") or [])
    layout = network.pop("layout")
    tree_layout = network.pop("tree_layout")
    spectral_layout = network.pop("spectral_layout")
    weight_layout = network.pop("weight_layout")
    fiedler = network.pop("fiedler")
    betweenness = network.pop("betweenness")
    eccentricity = network.pop("eccentricity")
    degree = network.pop("degree")
    for row in graph.get("nodes") or []:
        team = row["id"]
        row["degree"] = degree.get(team, 0)
        row["betweenness"] = betweenness.get(team, 0.0)
        row["eccentricity"] = eccentricity.get(team)
        row["fiedler"] = fiedler.get(team)
        _put_xy(row, layout.get(team), "nx", "ny")
        _put_xy(row, tree_layout.get(team), "tx", "ty")
        _put_xy(row, spectral_layout.get(team), "sx", "sy")
        _put_xy(row, weight_layout.get(team), "wx", "wy")
    graph["network"] = network
    graph["note"] = (
        "NetworkX game graph. Edges point winner → loser. "
        "The undirected schedule is nearly 12-regular. "
        "bound_path is the Moore-style minimum average distance if every leftover pair sat at distance 2. "
        "sx/sy are the Laplacian spectral embedding; wx/wy are a margin-weighted spring."
    )
    return graph


def refresh_graphs(root: Path | None = None) -> list[Path]:
    """Recompute NetworkX layouts on published graph JSON without rerating."""
    root = root or Path(__file__).resolve().parents[1]
    written: list[Path] = []
    for src in sorted((root / "data" / "processed").glob("graph-*.json")):
        graph = json.loads(src.read_text())
        attach_network(graph)
        text = json.dumps(graph, indent=2) + "\n"
        src.write_text(text)
        dest = root / "web" / "public" / "data" / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        written.append(src)
        print(f"refreshed {src.name}")
    return written


if __name__ == "__main__":
    refresh_graphs()


def build_graph(sides, fbs: set[str], conferences: dict[str, str], ratings: dict, money: dict[str, float]) -> dict:
    """Directed winner→loser game graph. Node size is winningness; gravity is margin-weighted."""
    games = listed_games(sides)
    team_rows = {t["team"]: t for t in ratings.get("teams", [])}

    played = nx.MultiDiGraph()
    credit = nx.DiGraph()
    for team in fbs:
        row = team_rows.get(team, {})
        played.add_node(
            team,
            team=team,
            conf=conferences.get(team, row.get("conf", "")),
            pom=row.get("pom", row.get("palm")),
            wins=row.get("wins", 0),
            losses=row.get("losses", 0),
            nil_roster=money.get(team),
        )
        credit.add_node(team)

    edges = []
    for row in games.itertuples(index=False):
        home, away = row.home_team, row.away_team
        if home not in fbs and away not in fbs:
            continue
        margin = float(row.points - row.opp_points)
        if margin == 0:
            continue
        winner, loser = (home, away) if margin > 0 else (away, home)
        abs_margin = abs(margin)
        fbs_fbs = home in fbs and away in fbs
        edge = {
            "id": f"{home}-{away}-{int(row.slate)}",
            "source": winner,
            "target": loser,
            "winner": winner,
            "loser": loser,
            "home": home,
            "away": away,
            "margin": round(abs_margin, 1),
            "week": _week(row),
            "slate": int(row.slate),
            "neutral": bool(row.neutral_site),
            "fbs_fbs": fbs_fbs,
        }
        edges.append(edge)
        played.add_edge(winner, loser, **edge)
        if fbs_fbs:
            weight = credit[loser][winner]["weight"] + abs_margin if credit.has_edge(loser, winner) else abs_margin
            credit.add_edge(loser, winner, weight=weight)

    pagerank = nx.pagerank(credit, weight="weight") if credit.number_of_nodes() else {}
    neighbors = played.subgraph(fbs).to_undirected()
    poms = {team: _finite(played.nodes[team].get("pom")) for team in fbs}
    winningness = {}
    for team in fbs:
        row = team_rows.get(team, {})
        wins = int(row.get("wins", 0) or 0)
        losses = int(row.get("losses", 0) or 0)
        played_n = wins + losses
        winningness[team] = wins / played_n if played_n else 0.0
    sizes = _scale(winningness)

    nodes = []
    for team in sorted(fbs):
        data = played.nodes[team]
        neigh = [n for n in neighbors.neighbors(team)] if team in neighbors else []
        neighbor_pom = sum(poms[n] for n in neigh) / len(neigh) if neigh else 0.0
        signed = []
        if team in played:
            for _, _, payload in played.out_edges(team, data=True):
                if payload.get("fbs_fbs"):
                    signed.append(float(payload["margin"]))
            for _, _, payload in played.in_edges(team, data=True):
                if payload.get("fbs_fbs"):
                    signed.append(-float(payload["margin"]))
        pom = poms[team]
        nodes.append(
            {
                "id": team,
                "team": team,
                "conf": data.get("conf", ""),
                "pom": data.get("pom"),
                "wins": data.get("wins", 0),
                "losses": data.get("losses", 0),
                "nil_roster": data.get("nil_roster"),
                "pagerank": round(pagerank.get(team, 0.0), 5),
                "neighbor_pom": round(neighbor_pom, 2),
                "vs_neighbors": round(pom - neighbor_pom, 2),
                "margin_vs": round(sum(signed) / len(signed), 1) if signed else 0.0,
                "winningness": round(winningness.get(team, 0.0), 3),
                "size": sizes.get(team, SIZE_MIN),
            }
        )

    return attach_network(
        {
            "season": ratings.get("season"),
            "directed": "winner -> loser",
            "nodes": nodes,
            "edges": edges,
        }
    )
