"""In-app DeepSeek agent over published FootPalm files. The key never leaves the server."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from footpalm.accounts import Accounts
from footpalm.cfbd import load_dotenv
from footpalm.draft import colleges_match, list_picks, lookup as draft_lookup, person_key
from footpalm.fetch import repo_root
from footpalm.names import canon, key as name_key
from footpalm.people import find_people

CHAT_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"
FALLBACK_MODEL = "deepseek-chat"
MAX_ROUNDS = 10
MAX_GAMES = 24
MAX_EV = 16
HOST = "127.0.0.1"
PORT = 8766
ET = ZoneInfo("America/New_York")
JUICE = 110

SYSTEM = """You work like a coding agent over FootPalm files. Search, then open, then list. No greeting. No capability list.

Pom = points vs an average FBS team on a neutral field. Elo is not the board. Spreads and NIL are context, not model inputs.

1. search — find a name, team, pick, or game. Read the hits. Do not guess.
2. open — one record. Use the kind and name from search.
3. list — a board: ratings, games, leaders, draft, people, backtest, money. One table or scatter.
4. show — a chart only if they asked for one. scatter for x vs y. list/open already draw the card.

The card is the answer. One or two sentences. Never restate card rows, numbered picks, or a markdown table.

Default season is {season}. Last year for teams and player stats is {last_year}. Draft last year is the {draft_year} NFL draft. Do not pass year={last_year} for draft unless they said the {last_year} draft.
Player stats: rushing, receiving, yac, yac_avg, passing. yac_avg is YAC per catch-point play. 2026 has no play-by-play; boards stop at {last_year}.
A draft miss is "not in the file", not "went undrafted". Many stayed in school. Not a ratings input.

Picks, EV, Saturday, weekend, upcoming: the slate card is already drawn. Do not catalog. Do not list again. Do not open research or backtest. Do not list a finished week.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "catalog",
            "description": "What seasons and files are on disk.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Find teams, players, draft picks, or games by name. Returns hits. Then open one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "kind": {"type": "string", "enum": ["team", "player", "draft", "game"]},
                    "season": {"type": "integer"},
                    "team": {"type": "string"},
                    "n": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open",
            "description": "Read one record and draw its card. kind: team, player, draft, game, research. Omit kind to open the top search hit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["team", "player", "draft", "game", "research"]},
                    "name": {"type": "string"},
                    "team": {"type": "string"},
                    "season": {"type": "integer"},
                    "year": {"type": "integer"},
                    "home": {"type": "string"},
                    "away": {"type": "string"},
                    "week": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list",
            "description": "Draw one board. source: ratings, games, leaders, draft, people, backtest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["ratings", "games", "leaders", "draft", "people", "backtest", "money"],
                    },
                    "season": {"type": "integer"},
                    "n": {"type": "integer"},
                    "team": {"type": "string"},
                    "teams": {"type": "array", "items": {"type": "string"}},
                    "metric": {
                        "type": "string",
                        "enum": ["pom", "elo", "adjo", "adjd", "adjst", "tempo", "sos", "luck", "wins"],
                    },
                    "bottom": {"type": "boolean"},
                    "conf": {"type": "string"},
                    "stat": {"type": "string", "enum": ["rushing", "receiving", "yac", "yac_avg", "passing"]},
                    "year": {"type": "integer"},
                    "round": {"type": "integer"},
                    "position": {"type": "string"},
                    "college": {"type": "string"},
                    "nfl": {"type": "string"},
                    "week": {"type": "integer"},
                    "upsets": {"type": "boolean"},
                    "completed": {"type": "boolean"},
                    "when": {"type": "string", "enum": ["upcoming", "saturday", "weekend"]},
                    "sort": {"type": "string", "enum": ["date", "ev", "week"]},
                    "names": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show",
            "description": "A chart the list did not draw. Only if they asked for one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["stats", "table", "bars", "line", "graph", "scatter"]},
                    "title": {"type": "string"},
                    "x_label": {"type": "string"},
                    "y_label": {"type": "string"},
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "hint": {"type": "string"},
                            },
                            "required": ["label", "x", "y"],
                        },
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "value": {},
                                "tone": {"type": "string", "enum": ["good", "bad", ""]},
                            },
                            "required": ["label", "value"],
                        },
                    },
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "rows": {"type": "array", "items": {"type": "array"}},
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                                "value": {},
                            },
                            "required": ["id"],
                        },
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "label": {"type": "string"},
                            },
                            "required": ["source", "target"],
                        },
                    },
                },
                "required": ["kind", "title"],
            },
        },
    },
]


def _read(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _slim_team(row: dict) -> dict:
    keep = (
        "rank",
        "team",
        "conf",
        "wins",
        "losses",
        "games",
        "pom",
        "elo",
        "adjo",
        "adjd",
        "adjst",
        "tempo",
        "sos",
        "luck",
        "nil_roster",
        "nil_quality",
    )
    return {k: row.get(k) for k in keep if k in row}


def _slim_game(row: dict) -> dict:
    keep = (
        "season",
        "week",
        "season_type",
        "home",
        "away",
        "neutral",
        "fbs_fbs",
        "pred_margin",
        "home_win_prob",
        "pred_home",
        "pred_away",
        "actual_home",
        "actual_away",
        "actual_margin",
        "home_won",
        "spread",
        "engine",
        "completed",
        "start",
    )
    out = {k: row.get(k) for k in keep if k in row}
    if out.get("completed") is None:
        out["completed"] = row.get("actual_home") is not None
    if row.get("start"):
        out["when"] = _fmt_when(row)
    side = _ats_side(row)
    mkt = _market_spread(row)
    if mkt is not None:
        out["mkt"] = mkt
    if side:
        out["pick"] = _bet_label(side)
        out["ev"] = round(side["ev"], 3)
    return out


def _portal_cell(row: dict) -> str:
    moves = row.get("portal") or []
    if not moves:
        return ""
    last = moves[-1]
    origin = last.get("origin") or "?"
    dest = last.get("destination") or "?"
    return f"{origin} → {dest}"


def _draft_cell(row: dict) -> str:
    draft = row.get("draft") or {}
    if not draft:
        return ""
    return f"{draft.get('year')} R{draft.get('round')}-{draft.get('overall')} {draft.get('nfl')}"


def _name_score(name: str, query: str) -> int | None:
    pk = person_key(name)
    q = name_key(query)
    if not pk or not q:
        return None
    if pk == q:
        return 0
    if q in pk:
        return 1
    last = pk.split()[-1]
    if q == last:
        return 2
    return None


def _game_score_line(game: dict) -> str:
    away = game.get("actual_away")
    home = game.get("actual_home")
    if away is None or home is None:
        return ""
    return f"{away}-{home}"


def _signed(n: float) -> str:
    return f"{n:+.1f}"


def _ev_pct(ev: float) -> str:
    return f"{ev * 100:+.0f}%"


def _clip(p: float) -> float:
    return min(1 - 1e-6, max(1e-6, p))


def _market_spread(game: dict) -> float | None:
    books = game.get("books") or {}
    for src in ("kalshi", "polymarket"):
        raw = (books.get(src) or {}).get("spread")
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    raw = game.get("spread")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _ats_side(game: dict) -> dict | None:
    spread = _market_spread(game)
    if spread is None:
        return None
    pred = float(game.get("pred_margin") or 0)
    p_home_win = float(game.get("home_win_prob") or 0.5)
    z = math.log(_clip(p_home_win) / (1 - _clip(p_home_win)))
    if abs(z) < 1e-9 or abs(pred) < 1e-6 or (pred > 0) != (z > 0):
        sigma = 14.5
    else:
        sigma = min(22.0, max(10.0, pred / z))
    p_home = 1 / (1 + math.exp(-(pred + spread) / sigma))
    take_home = p_home >= 0.5
    p_cover = p_home if take_home else 1 - p_home
    return {
        "who": game.get("home") if take_home else game.get("away"),
        "line": spread if take_home else -spread,
        "ev": p_cover * (100 / JUICE) - (1 - p_cover),
        "p_cover": p_cover,
    }


def _bet_label(side: dict) -> str:
    return f"{side['who']} {_signed(side['line'])}"


def _parse_start(game: dict) -> datetime | None:
    raw = game.get("start")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _et_date(game: dict) -> date | None:
    start = _parse_start(game)
    if start is None:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return start.astimezone(ET).date()


def _next_saturday(now: datetime | None = None) -> date:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local = now.astimezone(ET)
    days = (5 - local.weekday()) % 7
    if days == 0 and local.weekday() != 5:
        days = 7
    if local.weekday() == 6:
        days = 6
    return (local + timedelta(days=days)).date()


def _when_dates(when: str | None, now: datetime | None = None) -> set[date] | None:
    if not when:
        return None
    w = when.strip().lower()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", w):
        return {date.fromisoformat(w)}
    if w in {"saturday", "sat"}:
        return {_next_saturday(now)}
    if w == "weekend":
        sat = _next_saturday(now)
        return {sat - timedelta(days=1), sat, sat + timedelta(days=1)}
    return None


def _fmt_when(game: dict) -> str:
    day = _et_date(game)
    if day is None:
        return ""
    return f"{day.month}/{day.day}"


def _unplayed(game: dict) -> bool:
    return game.get("actual_home") is None and not game.get("completed")


def _want_scatter(question: str) -> bool:
    q = question or ""
    if not re.search(r"\b(nil|payroll|roster)\b", q, re.I):
        return False
    return bool(re.search(r"\b(vs|versus|against|scatter|plot|graph|chart|visuali[sz]e)\b", q, re.I))


def _live_slate(question: str) -> bool:
    q = question or ""
    if re.search(r"\blast (year|season)\b", q, re.I):
        return False
    return bool(
        re.search(
            r"\b(saturday|sat|weekend|upcoming|slate|projections?|highest ev|\bev\b|bets?|picks?)\b",
            q,
            re.I,
        )
    )


def _live_season(warehouse: Warehouse, fallback: int) -> int:
    years = warehouse.seasons()
    for year in reversed(years):
        if any(_unplayed(g) for g in warehouse.games(year)):
            return year
    return years[-1] if years else fallback


def _slate_title(title: str) -> bool:
    return bool(re.match(r"^(Games|Saturday|Weekend|Upcoming|Highest EV)\b", title or ""))


def _matchup_title(title: str) -> bool:
    return " at " in (title or "") and "·" not in (title or "")


def _card_rank(card: dict) -> int:
    title = card.get("title") or ""
    cols = card.get("columns") or []
    if "EV" in cols or "Highest EV" in title:
        return 3
    if _slate_title(title):
        return 2
    if _matchup_title(title):
        return 1
    return 0


_MD_TABLE = re.compile(r"(?m)(?:^\s*\|.*\|\s*$\n?)+")
_NFL_CITY = (
    "Las Vegas|Los Angeles|New England|New Orleans|New York|Green Bay|"
    "Kansas City|San Francisco|Tampa Bay|Arizona|Atlanta|Baltimore|Buffalo|"
    "Carolina|Chicago|Cincinnati|Cleveland|Dallas|Denver|Detroit|Houston|"
    "Indianapolis|Jacksonville|Miami|Minnesota|Philadelphia|Pittsburgh|"
    "Seattle|Tennessee|Washington"
)
_PICK_ARROW = re.compile(rf"\d+\.\s+.+?\s*->\s+(?:{_NFL_CITY})")
_PICK_RUN = re.compile(r"(?:^|\s)\d+\.\s+\S.{8,160}?(?=\s+\d+\.\s+|$)")
_LINE_NUMBERED = re.compile(r"(?m)(?:^\s*\d+\.\s+.+\n?){3,}")


def _strip_numbered_dump(text: str) -> str:
    items = list(_PICK_ARROW.finditer(text))
    if len(items) < 3:
        items = list(_PICK_RUN.finditer(text))
    if len(items) < 3:
        return text
    return (text[: items[0].start()] + " " + text[items[-1].end() :]).strip()


_SENTENCE = re.compile(r"(?<!\bNo\.)(?<!\bRd\.)(?<!\bvs\.)(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENTENCE.split(text) if p.strip()]


def _looks_like_dump(text: str) -> bool:
    return text.count("->") >= 2 or len(re.findall(r"\d+\.", text)) >= 3


def _clean_answer(text: str, cards: list) -> str:
    out = (text or "").strip()
    if not any(c.get("kind") == "table" for c in cards):
        return out
    out = re.sub(r"[*_`]", "", out)
    out = _MD_TABLE.sub("", out)
    out = _LINE_NUMBERED.sub("", out)
    out = _strip_numbered_dump(out)
    out = re.sub(r"^[^.\n]{0,120}:\s*", "", out)
    keep = [s for s in _sentences(out) if not _looks_like_dump(s)]
    out = " ".join(keep[:2])
    return re.sub(r"\s+", " ", out).strip()


def _prune_cards(cards: list, question: str = "") -> list:
    q = question or ""
    want_research = bool(re.search(r"\b(research|promoted|temperature|holdout)\b", q, re.I))
    want_backtest = bool(re.search(r"\b(backtest|brier)\b", q, re.I))
    tables = {c.get("title") for c in cards if c.get("kind") == "table"}
    kept: list[dict] = []
    slates: list[dict] = []
    for card in cards:
        title = card.get("title") or ""
        if card.get("kind") == "stats" and title in tables:
            continue
        if title == "Research" and not want_research:
            continue
        if title.startswith("Backtest") and not want_backtest:
            continue
        if card.get("kind") == "table" and (_slate_title(title) or _matchup_title(title)):
            slates.append(card)
            continue
        kept.append(card)
    if slates:
        best = max(_card_rank(c) for c in slates)
        ranked = [c for c in slates if _card_rank(c) == best]
        kept.append(ranked[-1])
    return kept[:3]


def _with_yac_avg(row: dict) -> dict:
    item = dict(row)
    plays = float(item.get("yac_plays") or 0)
    yac = float(item.get("yac") or 0)
    item["yac_avg"] = round(yac / plays, 1) if plays else 0.0
    return item


def _num(row: dict, metric: str) -> float:
    if metric == "wins":
        return float(row.get("wins") or 0)
    raw = row.get(metric, row.get("palm"))
    return float(raw) if raw is not None else 0.0


class Warehouse:
    def __init__(self, root: Path):
        self.root = root
        self.processed = root / "data" / "processed"
        self._cache: dict[str, Any] = {}

    def _load(self, name: str) -> Any:
        if name not in self._cache:
            self._cache[name] = _read(self.processed / name)
        return self._cache[name]

    def seasons(self) -> list[int]:
        years = []
        for path in self.processed.glob("ratings-*.json"):
            try:
                years.append(int(path.stem.split("-")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(years)

    def leader_seasons(self) -> list[int]:
        years = []
        for path in self.processed.glob("leaders-*.json"):
            try:
                years.append(int(path.stem.split("-")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(years)

    def leaders(self, season: int) -> dict | None:
        return self._load(f"leaders-{season}.json")

    def draft(self) -> dict | None:
        return self._load("draft.json")

    def people_seasons(self) -> list[int]:
        years = []
        for path in self.processed.glob("people-*.json"):
            try:
                years.append(int(path.stem.split("-")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(years)

    def people(self, season: int) -> dict | None:
        return self._load(f"people-{season}.json")

    def ratings(self, season: int) -> dict | None:
        return self._load(f"ratings-{season}.json")

    def games(self, season: int) -> list[dict]:
        payload = self._load(f"predictions-{season}.json")
        if not payload:
            return []
        return payload.get("games") or []

    def graph(self, season: int) -> dict | None:
        return self._load(f"graph-{season}.json")

    def roster(self, season: int) -> list[str]:
        data = self.ratings(season)
        if not data:
            return []
        return [str(row["team"]) for row in data.get("teams") or [] if row.get("team")]

    def resolve(self, query: str, season: int) -> list[str]:
        names = self.roster(season)
        if not names:
            return []
        wanted = canon(query)
        wanted_key = name_key(wanted)
        by_key = {name_key(n): n for n in names}
        if wanted_key in by_key:
            return [by_key[wanted_key]]
        hits = [n for n in names if wanted_key in name_key(n) or name_key(n) in wanted_key]
        if len(hits) == 1:
            return hits
        if hits:
            return hits[:8]
        return [n for n in names if wanted_key[:4] and wanted_key[:4] in name_key(n)][:8]

    def team_row(self, team: str, season: int) -> dict | None:
        data = self.ratings(season)
        if not data:
            return None
        hits = self.resolve(team, season)
        if not hits:
            return None
        name = hits[0]
        for row in data.get("teams") or []:
            if row.get("team") == name:
                return row
        return None


class Session:
    def __init__(self, warehouse: Warehouse, season: int, question: str = "", now: datetime | None = None):
        self.wh = warehouse
        self.season = season
        self.question = question or ""
        self.now = now
        self.cards: list[dict] = []

    def _year(self, season: int | None) -> int:
        return int(season) if season else self.season

    def _pbp_year(self, season: int | None) -> int:
        year = self._year(season)
        if self.wh.leaders(year):
            return year
        have = self.wh.leader_seasons()
        return have[-1] if have else year

    def catalog(self) -> dict:
        years = self.wh.seasons()
        files = []
        for year in years:
            files.append(
                {
                    "season": year,
                    "ratings": self.wh.ratings(year) is not None,
                    "games": bool(self.wh.games(year)),
                    "graph": self.wh.graph(year) is not None,
                    "teams": len(self.wh.roster(year)),
                }
            )
        research = self.wh._load("research.json") or {}
        return {
            "seasons": files,
            "default_season": self.season,
            "player_boards": self.wh.leader_seasons(),
            "draft_years": (self.wh.draft() or {}).get("years") or [],
            "people": self.wh.people_seasons(),
            "promoted": research.get("promoted"),
            "conclusion": research.get("conclusion"),
        }

    def find_teams(self, query: str, season: int | None = None) -> dict:
        year = self._year(season)
        return {"season": year, "matches": self.wh.resolve(query, year)}

    def get_team(self, team: str, season: int | None = None) -> dict:
        year = self._year(season)
        row = self.wh.team_row(team, year)
        if not row:
            return {"error": f"no team matching {team!r} in {year}", "tried": self.wh.resolve(team, year)}
        name = row["team"]
        money = self.wh._load("money.json") or {}
        cash = next((m for m in money.get("teams") or [] if m.get("team") == name), None)
        node = None
        graph = self.wh.graph(year)
        if graph:
            node = next((n for n in graph.get("nodes") or [] if n.get("team") == name), None)
        games = [_slim_game(g) for g in self.wh.games(year) if g.get("home") == name or g.get("away") == name]
        return {
            "season": year,
            "team": _slim_team(row),
            "money": (
                {
                    "nil_roster": (cash or {}).get("nil_roster"),
                    "nil_quality": (cash or {}).get("nil_quality"),
                    "athletic_spend": (cash or {}).get("athletic_spend"),
                }
                if cash
                else None
            ),
            "graph": (
                {
                    "degree": (node or {}).get("degree"),
                    "winningness": (node or {}).get("winningness"),
                    "neighbor_pom": (node or {}).get("neighbor_pom"),
                    "vs_neighbors": (node or {}).get("vs_neighbors"),
                    "pagerank": (node or {}).get("pagerank"),
                }
                if node
                else None
            ),
            "games": games[:MAX_GAMES],
        }

    def compare_teams(self, teams: list[str], season: int | None = None) -> dict:
        year = self._year(season)
        rows = []
        missing = []
        for team in teams:
            row = self.wh.team_row(team, year)
            if row:
                rows.append(_slim_team(row))
            else:
                missing.append(team)
        return {"season": year, "teams": rows, "missing": missing}

    def leaderboard(
        self,
        season: int | None = None,
        metric: str = "pom",
        n: int = 10,
        bottom: bool = False,
        conf: str | None = None,
    ) -> dict:
        year = self._year(season)
        data = self.wh.ratings(year)
        if not data:
            return {"error": f"no ratings for {year}"}
        rows = list(data.get("teams") or [])
        if conf:
            want = conf.lower()
            rows = [r for r in rows if str(r.get("conf") or "").lower() == want]
        invert = metric == "adjd"
        rows.sort(key=lambda r: _num(r, metric), reverse=not bottom and not invert)
        take = max(1, min(int(n or 10), 25))
        return {
            "season": year,
            "metric": metric,
            "conf": conf,
            "teams": [_slim_team(r) for r in rows[:take]],
        }

    def list_games(
        self,
        season: int | None = None,
        team: str | None = None,
        week: int | None = None,
        upsets: bool = False,
        completed: bool | None = None,
        when: str | None = None,
        sort: str | None = None,
    ) -> dict:
        live = _live_slate(self.question)
        if live and not re.search(r"\b20\d{2}\b", self.question):
            year = self.season
        else:
            year = self._year(season)
        if live and team is None and week is None:
            if completed is None:
                completed = False
            if not when:
                q = self.question.lower()
                if re.search(r"\bsat(urday)?\b", q):
                    when = "saturday"
                elif "weekend" in q:
                    when = "weekend"
                else:
                    when = "upcoming"
            if not sort:
                sort = "ev" if re.search(r"\b(ev|bet|bets|pick|picks)\b", self.question, re.I) else "date"
        games = self.wh.games(year)
        name = None
        if team:
            hits = self.wh.resolve(team, year)
            name = hits[0] if hits else team
            games = [g for g in games if g.get("home") == name or g.get("away") == name]
        if week is not None:
            games = [g for g in games if g.get("week") == week]
        if completed is True:
            games = [g for g in games if not _unplayed(g)]
        if completed is False:
            games = [g for g in games if _unplayed(g)]
        days = _when_dates(when, self.now)
        if days:
            games = [g for g in games if _et_date(g) in days]
        if upsets:
            picked = []
            for game in games:
                p = game.get("home_win_prob")
                won = game.get("home_won")
                if p is None or won is None:
                    continue
                if (p < 0.4 and won == 1) or (p > 0.6 and won == 0):
                    picked.append(game)
            games = picked
        keyed = []
        for game in games:
            start = _parse_start(game)
            side = _ats_side(game)
            keyed.append((game, start.timestamp() if start else 0, side["ev"] if side else -99, game.get("week") or 0))
        key = (sort or "date").lower()
        if key == "ev":
            keyed.sort(key=lambda row: (-row[2], row[1]))
        elif key == "week":
            keyed.sort(key=lambda row: (row[3], row[1]))
        else:
            keyed.sort(key=lambda row: (row[1], row[3]))
        take = MAX_EV if key == "ev" else MAX_GAMES
        slim = [_slim_game(g) for g, *_ in keyed[:take]]
        return {
            "season": year,
            "team": name,
            "when": when,
            "sort": key,
            "n": len(keyed),
            "shown": len(slim),
            "games": slim,
        }

    def get_game(self, home: str, away: str, season: int | None = None, week: int | None = None) -> dict:
        year = self._year(season)
        homes = self.wh.resolve(home, year) or [home]
        aways = self.wh.resolve(away, year) or [away]
        home_name, away_name = homes[0], aways[0]
        found = []
        for game in self.wh.games(year):
            sides = {game.get("home"), game.get("away")}
            if home_name in sides and away_name in sides:
                if week is None or game.get("week") == week:
                    found.append(_slim_game(game))
        if not found:
            return {"error": f"no {home_name} vs {away_name} in {year}", "home": home_name, "away": away_name}
        return {
            "season": year,
            "games": found,
            "ratings": {
                home_name: _slim_team(self.wh.team_row(home_name, year) or {}),
                away_name: _slim_team(self.wh.team_row(away_name, year) or {}),
            },
        }

    def backtest(self, season: int | None = None) -> dict:
        year = self._year(season)
        row = self.wh._load(f"backtest-{year}.json")
        summary = self.wh._load("backtest-summary.json") or {}
        year_row = next((s for s in summary.get("seasons") or [] if s.get("season") == year), None)
        if not row and not year_row:
            return {"error": f"no backtest for {year}"}
        out: dict[str, Any] = {"season": year}
        if row:
            out["all_fbs"] = row.get("all_fbs")
            out["tabpfn"] = row.get("tabpfn")
            out["logistic"] = row.get("logistic")
            out["engine_counts"] = row.get("engine_counts")
        if year_row:
            out["summary"] = year_row
        return out

    def leaders(self, stat: str, season: int | None = None, n: int = 10, team: str | None = None) -> dict:
        year = self._pbp_year(season)
        data = self.wh.leaders(year)
        if not data:
            return {"error": f"no player boards for {year}"}
        key = {
            "rushing": "rushing",
            "receiving": "receiving",
            "yac": "yac",
            "yac_avg": "yac_avg",
            "avg": "yac_avg",
            "average": "yac_avg",
            "avg_yac": "yac_avg",
            "passing": "passing",
        }.get((stat or "").lower())
        if not key:
            return {"error": f"unknown stat {stat}", "have": ["rushing", "receiving", "yac", "yac_avg", "passing"]}
        note = None
        if key == "yac_avg":
            rows = [_with_yac_avg(r) for r in (data.get("yac") or data.get("receiving") or [])]
            rows = [r for r in rows if (r.get("yac_plays") or 0) >= 10]
            rows.sort(key=lambda r: r.get("yac_avg") or 0, reverse=True)
            for i, row in enumerate(rows, start=1):
                row["rank"] = i
            if not rows:
                note = f"{year} has no catch-point YAC."
        else:
            rows = [_with_yac_avg(r) for r in (data.get(key) or [])]
            if key == "yac" and not rows:
                rows = [_with_yac_avg(r) for r in (data.get("receiving") or [])]
                note = f"{year} play-by-play has no catch-point YAC. Showing receiving yards."
                key = "receiving"
        if team:
            hits = self.wh.resolve(team, year) or [team]
            name = hits[0]
            rows = [r for r in rows if r.get("team") == name]
        take = max(1, min(int(n or 10), 25))
        rows = rows[:take]
        title = {
            "rushing": "Rushing yards",
            "receiving": "Receiving yards",
            "yac": "Yards after catch",
            "yac_avg": "YAC per catch",
            "passing": "Passing yards",
        }[key]
        if key == "rushing":
            columns = ["#", "Player", "Team", "Att", "Yds", "TD", "YPC"]
            table = [[r["rank"], r["player"], r["team"], r["att"], r["yds"], r["td"], r["ypc"]] for r in rows]
        elif key == "passing":
            columns = ["#", "Player", "Team", "Cmp", "Att", "Yds", "TD"]
            table = [[r["rank"], r["player"], r["team"], r["cmp"], r["att"], r["yds"], r["td"]] for r in rows]
        elif key == "yac_avg":
            columns = ["#", "Player", "Team", "Rec", "YAC", "Plays", "Avg"]
            table = [[r["rank"], r["player"], r["team"], r["rec"], r.get("yac", 0), r.get("yac_plays", 0), r["yac_avg"]] for r in rows]
        else:
            columns = ["#", "Player", "Team", "Rec", "Yds", "YAC", "Avg", "TD"]
            table = [[r["rank"], r["player"], r["team"], r["rec"], r["yds"], r.get("yac", 0), r.get("yac_avg", 0), r["td"]] for r in rows]
        self.show("table", f"{title} · {year}", columns=columns, rows=table)
        return {
            "season": year,
            "stat": key,
            "note": note,
            "used": year if year == self._year(season) else f"{year} (latest PBP)",
            "n": len(rows),
            "leaders": rows,
        }

    def drafted(
        self,
        players: list | None = None,
        colleges: list | None = None,
        year: int | None = None,
        round: int | None = None,
        position: str | None = None,
        college: str | None = None,
        nfl: str | None = None,
        n: int | None = None,
    ) -> dict:
        data = self.wh.draft()
        if not data:
            return {"error": "no NFL draft file"}
        picks = list(data.get("picks") or [])
        years = list(data.get("years") or [])
        names = list(players or [])
        if names:
            if year is not None:
                wanted = {int(year), int(year) + 1}
                picks = [p for p in picks if p.get("year") in wanted]
                years = sorted(set(wanted) & set(years))
            rows = draft_lookup(names, picks, list(colleges or []))
            drafted_n = sum(1 for r in rows if r["drafted"])
            self.show(
                "table",
                "NFL draft",
                columns=["Player", "College", "Drafted", "Year", "Rd", "Overall", "NFL"],
                rows=[
                    [
                        r["player"],
                        r["college"],
                        "Yes" if r["drafted"] else "Not in file",
                        r["year"] or "",
                        r["round"] or "",
                        r["overall"] or "",
                        r["nfl"],
                    ]
                    for r in rows
                ],
            )
            return {
                "years": years,
                "asked": len(rows),
                "drafted": drafted_n,
                "undrafted": len(rows) - drafted_n,
                "note": "Not in file means they are not in the CFBD pick list. Many stayed in school.",
                "players": rows,
            }
        if not years:
            return {"error": "no NFL draft years"}
        draft_year = int(year) if year is not None else max(years)
        rows = list_picks(picks, draft_year, rnd=round, position=position, college=college, nfl=nfl, n=n)
        title = f"NFL draft · {draft_year}"
        if round is not None:
            title += f" · Rd {round}"
        elif nfl:
            title += f" · {nfl}"
        elif college:
            title += f" · {college}"
        elif position:
            title += f" · {position}"
        self.show(
            "table",
            title,
            columns=["Rd", "Pick", "Player", "College", "Pos", "NFL"],
            rows=[
                [
                    r.get("round") or "",
                    r.get("overall") or "",
                    r.get("name") or "",
                    r.get("college") or "",
                    r.get("position") or "",
                    r.get("nfl") or "",
                ]
                for r in rows
            ],
        )
        return {"year": draft_year, "years": years, "n": len(rows), "picks": rows}

    def _people_year(self, season: int | None) -> int:
        year = self._year(season)
        if self.wh.people(year):
            return year
        have = self.wh.people_seasons()
        return have[-1] if have else year

    def player(
        self,
        name: str | None = None,
        names: list | None = None,
        team: str | None = None,
        colleges: list | None = None,
        season: int | None = None,
        players: list | None = None,
    ) -> dict:
        wanted = [n for n in [name] if n] + list(names or []) + list(players or [])
        if not wanted:
            return {"error": "pass a player name"}
        year = self._people_year(season)
        data = self.wh.people(year)
        if not data:
            return {"error": f"no player file for {year}"}
        colleges = list(colleges or [])
        rows = []
        missing = []
        for i, raw in enumerate(wanted):
            college = team if team else (colleges[i] if i < len(colleges) else None)
            hits = find_people(data, raw, college)
            hit_year = year
            if not hits and season is None:
                for other in reversed(self.wh.people_seasons()):
                    if other == year:
                        continue
                    hits = find_people(self.wh.people(other) or {}, raw, college)
                    if hits:
                        hit_year = other
                        break
            if not hits:
                missing.append(raw)
                rows.append({"name": raw, "team": college or "", "found": False})
                continue
            row = dict(hits[0])
            row["found"] = True
            row["season"] = hit_year
            rows.append(row)
        drafted_n = sum(1 for r in rows if r.get("draft"))
        portal_n = sum(1 for r in rows if r.get("portal"))
        if len(rows) == 1 and rows[0].get("found"):
            row = rows[0]
            draft = row.get("draft") or {}
            recruit = row.get("recruit") or {}
            usage = row.get("usage") or {}
            items = [
                {"label": "Team", "value": row.get("team") or ""},
                {"label": "Pos", "value": row.get("pos") or ""},
                {"label": "Class", "value": row.get("class") or ""},
                {"label": "Stars", "value": recruit.get("stars") or "—"},
            ]
            if usage.get("overall") is not None:
                items.append({"label": "Usage", "value": usage.get("overall")})
            if draft:
                items.append({"label": "Draft", "value": f"{draft.get('year')} R{draft.get('round')} {draft.get('overall')} {draft.get('nfl')}"})
            self.show("stats", row.get("name") or "Player", items=items)
        else:
            self.show(
                "table",
                f"Players · {year}",
                columns=["Player", "Team", "Pos", "Yr", "Stars", "Portal", "Draft"],
                rows=[
                    [
                        r.get("name") or "",
                        r.get("team") or "",
                        r.get("pos") or "",
                        r.get("class") or "",
                        (r.get("recruit") or {}).get("stars") or "",
                        _portal_cell(r),
                        _draft_cell(r),
                    ]
                    for r in rows
                ],
            )
        return {
            "season": year,
            "asked": len(rows),
            "found": sum(1 for r in rows if r.get("found")),
            "missing": missing,
            "drafted": drafted_n,
            "portal": portal_n,
            "people": rows,
        }

    def search(
        self,
        query: str,
        kind: str | None = None,
        season: int | None = None,
        team: str | None = None,
        n: int | None = None,
    ) -> dict:
        q = (query or "").strip()
        if not q:
            return {"error": "empty query"}
        kinds = [kind] if kind else ["team", "player", "draft", "game"]
        year = self._year(season)
        take = max(1, min(int(n or 12), 24))
        hits: list[dict] = []

        if "team" in kinds:
            names = self.wh.resolve(q, year)
            if not names:
                qq = name_key(q)
                names = [name for name in self.wh.roster(year) if qq and qq in name_key(name)]
            for i, name in enumerate(names):
                row = self.wh.team_row(name, year) or {}
                hits.append(
                    {
                        "kind": "team",
                        "name": name,
                        "season": year,
                        "hint": f"#{row.get('rank')} · Pom {row.get('pom')}" if row.get("rank") is not None else "",
                        "score": i,
                    }
                )

        if "player" in kinds:
            years = [self._people_year(season)]
            last = years[0] - 1
            if last in self.wh.people_seasons() and last not in years:
                years.append(last)
            seen: set[tuple] = set()
            for y in years:
                data = self.wh.people(y) or {}
                for row in data.get("people") or []:
                    score = _name_score(row.get("name") or "", q)
                    if score is None:
                        continue
                    if team and not colleges_match(team, row.get("team") or ""):
                        continue
                    key = (person_key(row.get("name") or ""), name_key(row.get("team") or ""), y)
                    if key in seen:
                        continue
                    seen.add(key)
                    hits.append(
                        {
                            "kind": "player",
                            "name": row.get("name"),
                            "team": row.get("team"),
                            "season": y,
                            "hint": " · ".join(p for p in (row.get("pos"), row.get("class"), _draft_cell(row)) if p),
                            "score": score,
                        }
                    )

        if "draft" in kinds:
            picks = (self.wh.draft() or {}).get("picks") or []
            qq = name_key(q)
            for pick in picks:
                score = _name_score(pick.get("name") or "", q)
                blob = name_key(f"{pick.get('college')} {pick.get('nfl')} {pick.get('position')}")
                if score is None:
                    if qq and qq in blob:
                        score = 4
                    else:
                        continue
                if team and not colleges_match(team, pick.get("college") or ""):
                    continue
                hits.append(
                    {
                        "kind": "draft",
                        "name": pick.get("name"),
                        "team": pick.get("college"),
                        "year": pick.get("year"),
                        "hint": f"{pick.get('year')} R{pick.get('round')}-{pick.get('overall')} {pick.get('nfl')}",
                        "score": score,
                    }
                )

        if "game" in kinds:
            qq = name_key(q)
            for game in self.wh.games(year):
                blob = name_key(f"{game.get('home')} {game.get('away')}")
                if not qq or qq not in blob:
                    continue
                hits.append(
                    {
                        "kind": "game",
                        "name": f"{game.get('away')} at {game.get('home')}",
                        "home": game.get("home"),
                        "away": game.get("away"),
                        "season": year,
                        "week": game.get("week"),
                        "hint": f"W{game.get('week')} · {_game_score_line(game) or 'pred ' + str(game.get('pred_margin'))}",
                        "score": 3,
                    }
                )

        hits.sort(key=lambda h: (h.get("score") or 9, str(h.get("kind")), str(h.get("name"))))
        hits = hits[:take]
        for hit in hits:
            hit.pop("score", None)
        return {"query": q, "n": len(hits), "hits": hits}

    def open_record(
        self,
        kind: str | None = None,
        name: str | None = None,
        team: str | None = None,
        season: int | None = None,
        year: int | None = None,
        home: str | None = None,
        away: str | None = None,
        week: int | None = None,
    ) -> dict:
        if not kind and name:
            found = self.search(name, season=season, team=team, n=1)
            hits = found.get("hits") or []
            if not hits:
                return {"error": f"no match for {name!r}"}
            hit = hits[0]
            kind = hit.get("kind")
            name = hit.get("name")
            team = team or hit.get("team")
            year = year or hit.get("year")
            home = home or hit.get("home")
            away = away or hit.get("away")
            week = week if week is not None else hit.get("week")
            if hit.get("season") is not None and season is None:
                season = hit.get("season")
        k = (kind or "").lower()
        if k == "research":
            out = self.research()
            if out.get("error"):
                return out
            self.show(
                "stats",
                "Research",
                items=[
                    {"label": "Promoted", "value": out.get("promoted") or "—"},
                    {"label": "Holdout", "value": out.get("holdout_season") or ""},
                    {"label": "Brier", "value": out.get("baseline_holdout_brier") or ""},
                ],
            )
            return out
        if k == "team":
            out = self.get_team(name or team or "", season)
            row = out.get("team")
            if row:
                rec = f"{row.get('wins')}-{row.get('losses')}"
                self.show(
                    "stats",
                    row.get("team") or name,
                    items=[
                        {"label": "Rank", "value": row.get("rank")},
                        {"label": "Rec", "value": rec},
                        {"label": "Pom", "value": row.get("pom")},
                        {"label": "Elo", "value": row.get("elo")},
                        {"label": "Conf", "value": row.get("conf") or ""},
                    ],
                )
            return out
        if k == "player":
            return self.player(name=name, team=team, season=season)
        if k == "draft":
            return self.drafted(players=[name] if name else None, colleges=[team] if team else None, year=year)
        if k == "game":
            if not home or not away:
                return {"error": "pass home and away"}
            out = self.get_game(home, away, season=season, week=week)
            games = out.get("games") or []
            if games:
                self.show(
                    "table",
                    f"{away} at {home}",
                    columns=["Wk", "Away", "Home", "Pred", "Actual"],
                    rows=[
                        [
                            g.get("week"),
                            g.get("away"),
                            g.get("home"),
                            g.get("pred_margin"),
                            _game_score_line(g),
                        ]
                        for g in games
                    ],
                )
            return out
        return {"error": f"unknown kind {kind!r}", "have": ["team", "player", "draft", "game", "research"]}

    def list_board(
        self,
        source: str,
        season: int | None = None,
        n: int | None = None,
        team: str | None = None,
        teams: list | None = None,
        metric: str = "pom",
        bottom: bool = False,
        conf: str | None = None,
        stat: str | None = None,
        year: int | None = None,
        round: int | None = None,
        position: str | None = None,
        college: str | None = None,
        nfl: str | None = None,
        week: int | None = None,
        upsets: bool = False,
        completed: bool | None = None,
        when: str | None = None,
        sort: str | None = None,
        names: list | None = None,
    ) -> dict:
        src = (source or "").lower()
        if src in {"ratings", "leaderboard", "teams"}:
            if teams:
                out = self.compare_teams(list(teams), season=season)
                rows = out.get("teams") or []
            else:
                out = self.leaderboard(season=season, metric=metric, n=n or 10, bottom=bottom, conf=conf)
                rows = out.get("teams") or []
            if rows:
                year_n = out.get("season") or self._year(season)
                self.show(
                    "table",
                    f"Ratings · {year_n}",
                    columns=["#", "Team", "Rec", "Pom", "Elo", "Conf"],
                    rows=[
                        [
                            r.get("rank"),
                            r.get("team"),
                            f"{r.get('wins')}-{r.get('losses')}",
                            r.get("pom"),
                            r.get("elo"),
                            r.get("conf") or "",
                        ]
                        for r in rows
                    ],
                )
            return out
        if src in {"games", "slate"}:
            out = self.list_games(
                season=season,
                team=team,
                week=week,
                upsets=upsets,
                completed=completed,
                when=when,
                sort=sort,
            )
            games = out.get("games") or []
            if games:
                ev_board = bool(out.get("when") or out.get("sort") == "ev")
                if ev_board:
                    title = {
                        "saturday": "Saturday",
                        "weekend": "Weekend",
                        "upcoming": "Upcoming",
                    }.get(out.get("when") or "", "Games")
                    if out.get("sort") == "ev":
                        title = f"Highest EV · {title} · {out.get('season')}"
                    else:
                        title = f"{title} · {out.get('season')}"
                    self.show(
                        "table",
                        title,
                        columns=["When", "Away", "Home", "Us", "Mkt", "Pick", "EV"],
                        rows=[
                            [
                                g.get("when") or "",
                                g.get("away"),
                                g.get("home"),
                                _signed(-float(g["pred_margin"])) if g.get("pred_margin") is not None else "",
                                _signed(g["mkt"]) if g.get("mkt") is not None else "",
                                g.get("pick") or "",
                                _ev_pct(g["ev"]) if g.get("ev") is not None else "",
                            ]
                            for g in games
                        ],
                    )
                else:
                    self.show(
                        "table",
                        f"Games · {out.get('season')}",
                        columns=["Wk", "Away", "Home", "Pred", "Actual"],
                        rows=[
                            [
                                g.get("week"),
                                g.get("away"),
                                g.get("home"),
                                g.get("pred_margin"),
                                _game_score_line(g),
                            ]
                            for g in games[:32]
                        ],
                    )
            return out
        if src in {"leaders", "stats"}:
            return self.leaders(stat or "rushing", season=season, n=n or 10, team=team)
        if src in {"draft", "drafted"}:
            return self.drafted(
                players=list(names or []) or None,
                year=year,
                round=round,
                position=position,
                college=college or team,
                nfl=nfl,
                n=n,
            )
        if src in {"people", "players", "roster"}:
            if names:
                return self.player(names=list(names), team=team, season=season)
            year_n = self._people_year(season)
            data = self.wh.people(year_n) or {}
            rows = list(data.get("people") or [])
            if team:
                hits = self.wh.resolve(team, year_n) or [team]
                want = hits[0]
                rows = [r for r in rows if colleges_match(want, r.get("team") or "")]
            rows = rows[: max(1, min(int(n or 20), 32))]
            if not rows:
                return {"error": f"no people for {team or year_n}"}
            self.show(
                "table",
                f"Players · {year_n}",
                columns=["Player", "Team", "Pos", "Yr", "Stars", "Portal", "Draft"],
                rows=[
                    [
                        r.get("name") or "",
                        r.get("team") or "",
                        r.get("pos") or "",
                        r.get("class") or "",
                        (r.get("recruit") or {}).get("stars") or "",
                        _portal_cell(r),
                        _draft_cell(r),
                    ]
                    for r in rows
                ],
            )
            return {"season": year_n, "n": len(rows), "people": rows}
        if src == "backtest":
            out = self.backtest(season=season)
            if out.get("error"):
                return out
            summary = out.get("summary") or {}
            tab = summary.get("tabpfn") or out.get("tabpfn") or {}
            fbs = summary.get("fbs") or out.get("all_fbs") or {}
            self.show(
                "stats",
                f"Backtest · {out.get('season')}",
                items=[
                    {"label": "Brier", "value": tab.get("tabpfn_brier") or fbs.get("fbs_brier") or "—"},
                    {"label": "n", "value": tab.get("tabpfn_n") or fbs.get("fbs_n") or ""},
                ],
            )
            return out
        if src in {"money", "nil"}:
            return self.money_plot(season=season)
        if src == "research":
            return self.open_record(kind="research")
        return {
            "error": f"unknown source {source!r}",
            "have": ["ratings", "games", "leaders", "draft", "people", "backtest", "money"],
        }

    def money_plot(self, season: int | None = None) -> dict:
        year = self._year(season)
        data = self.wh.ratings(year)
        if not data:
            return {"error": f"no ratings for {year}"}
        points = []
        for row in data.get("teams") or []:
            nil = row.get("nil_roster")
            pom = row.get("pom", row.get("palm"))
            name = row.get("team")
            if nil is None or pom is None or not name:
                continue
            points.append(
                {
                    "label": name,
                    "x": round(float(nil) / 1_000_000, 1),
                    "y": round(float(pom), 1),
                    "hint": row.get("conf") or "",
                }
            )
        if not points:
            return {"error": f"no NIL on {year} ratings"}
        self.show("scatter", f"NIL vs Pom · {year}", x_label="Roster $M", y_label="Pom", points=points)
        return {"season": year, "n": len(points), "x": "nil_roster", "y": "pom"}

    def research(self) -> dict:
        data = self.wh._load("research.json")
        if not data:
            return {"error": "no research.json"}
        return {
            "promoted": data.get("promoted"),
            "conclusion": data.get("conclusion"),
            "holdout_season": data.get("holdout_season"),
            "baseline_holdout_brier": data.get("baseline_holdout_brier"),
            "promote_if": data.get("promote_if"),
            "note": data.get("note"),
        }

    def show(
        self,
        kind: str,
        title: str,
        items: list | None = None,
        columns: list | None = None,
        rows: list | None = None,
        nodes: list | None = None,
        edges: list | None = None,
        points: list | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> dict:
        card = {"kind": kind, "title": title}
        if kind in {"stats", "bars", "line"}:
            card["items"] = (items or [])[:24]
        if kind == "table":
            card["columns"] = columns or []
            card["rows"] = (rows or [])[:32]
        if kind == "graph":
            card["nodes"] = (nodes or [])[:20]
            card["edges"] = (edges or [])[:36]
        if kind == "scatter":
            card["points"] = (points or [])[:160]
            card["x_label"] = x_label or ""
            card["y_label"] = y_label or ""
        if kind == "table" and _slate_title(title):
            existing = [c for c in self.cards if _slate_title(c.get("title") or "")]
            incoming = _card_rank({"title": title, "columns": columns or []})
            if existing and incoming < max(_card_rank(c) for c in existing):
                return {"ok": True, "n": len(self.cards)}
            self.cards = [
                c for c in self.cards if not _slate_title(c.get("title") or "") and not _matchup_title(c.get("title") or "")
            ]
        if any(c.get("kind") == kind and c.get("title") == title for c in self.cards):
            return {"ok": True, "n": len(self.cards)}
        if kind == "stats" and any(c.get("kind") == "table" and c.get("title") == title for c in self.cards):
            return {"ok": True, "n": len(self.cards)}
        self.cards.append(card)
        return {"ok": True, "n": len(self.cards)}


HANDLERS = {
    "catalog": lambda s, **kw: s.catalog(),
    "search": lambda s, **kw: s.search(**kw),
    "open": lambda s, **kw: s.open_record(**kw),
    "list": lambda s, **kw: s.list_board(**kw),
    "show": lambda s, **kw: s.show(**kw),
    "find_teams": lambda s, **kw: s.find_teams(**kw),
    "get_team": lambda s, **kw: s.get_team(**kw),
    "compare_teams": lambda s, **kw: s.compare_teams(**kw),
    "leaderboard": lambda s, **kw: s.leaderboard(**kw),
    "list_games": lambda s, **kw: s.list_games(**kw),
    "get_game": lambda s, **kw: s.get_game(**kw),
    "backtest": lambda s, **kw: s.backtest(**kw),
    "leaders": lambda s, **kw: s.leaders(**kw),
    "drafted": lambda s, **kw: s.drafted(**kw),
    "player": lambda s, **kw: s.player(**kw),
    "research": lambda s, **kw: s.research(),
}


def api_key() -> str:
    return (os.environ.get("DEEPSEEK") or os.environ.get("DEEPSEEK_API_KEY") or "").strip()


def model_name() -> str:
    return (os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL).strip()


def _complete(messages: list[dict], model: str, key: str) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "temperature": 0.2,
        }
    ).encode()
    req = Request(
        CHAT_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"DeepSeek {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"DeepSeek network error: {exc}") from exc


def _run_tool(session: Session, name: str, raw_args: str) -> str:
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        return json.dumps({"error": "bad tool arguments"})
    if not isinstance(args, dict):
        return json.dumps({"error": "tool arguments must be an object"})
    fn = HANDLERS.get(name)
    if not fn:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        result = fn(session, **args)
    except TypeError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(result, default=str)


def _last_user(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return ""


def resolve_ask_season(question: str, season: int, live: int | None = None) -> tuple[int, int]:
    last_year = season - 1 if season > 2014 else season
    draft_only = bool(
        re.search(r"\b(draft|drafted)\b", question or "", re.I)
        and not re.search(r"\b(rush|receiv|pass|yac|leader|stat|yard)\b", question or "", re.I)
    )
    if question and re.search(r"\blast (year|season)\b", question, re.I) and not draft_only:
        return last_year, last_year
    if _live_slate(question or "") and live and not re.search(r"\b20\d{2}\b", question or ""):
        year = live
        return year, year - 1 if year > 2014 else year
    return season, last_year


def ask(warehouse: Warehouse, messages: list[dict], season: int) -> dict:
    last = _last_user(messages)
    live = _live_season(warehouse, season)
    season, last_year = resolve_ask_season(last, season, live=live)
    session = Session(warehouse, season, question=last)
    if _live_slate(last):
        session.list_board("games")
    if _want_scatter(last):
        session.list_board("money")
        if session.cards:
            return {"text": "", "cards": session.cards, "tools": ["list"], "model": "local"}
    key = api_key()
    if not key:
        raise RuntimeError("DEEPSEEK is missing. Put it in .env.")
    draft_years = (warehouse.draft() or {}).get("years") or []
    draft_year = max(draft_years) if draft_years else season
    history = [{"role": "system", "content": SYSTEM.format(season=season, last_year=last_year, draft_year=draft_year)}]
    for msg in messages:
        role = msg.get("role")
        text = (msg.get("content") or "").strip()
        if role in {"user", "assistant"} and text:
            history.append({"role": role, "content": text[:2000]})
    if len(history) < 2:
        raise RuntimeError("empty question")

    used: list[str] = []
    model = model_name()
    for _ in range(MAX_ROUNDS):
        try:
            payload = _complete(history, model, key)
        except RuntimeError as exc:
            if model != FALLBACK_MODEL and "model" in str(exc).lower():
                model = FALLBACK_MODEL
                payload = _complete(history, model, key)
            else:
                raise
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        history.append({k: message[k] for k in message if k in {"role", "content", "tool_calls", "reasoning_content"}})
        calls = message.get("tool_calls") or []
        if not calls:
            cards = _prune_cards(session.cards, last)
            text = _clean_answer(message.get("content") or "", cards)
            return {"text": text, "cards": cards, "tools": used, "model": model}
        for call in calls:
            fn = (call.get("function") or {}) if isinstance(call, dict) else {}
            name = fn.get("name") or ""
            used.append(name)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": _run_tool(session, name, fn.get("arguments") or "{}"),
                }
            )
    return {"text": "Stopped after too many tool calls.", "cards": session.cards, "tools": used, "model": model}


def _cors(handler: BaseHTTPRequestHandler) -> None:
    origin = handler.headers.get("Origin")
    if origin:
        handler.send_header("Access-Control-Allow-Origin", origin)
        handler.send_header("Access-Control-Allow-Credentials", "true")
    else:
        handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "content-type, authorization")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")


class Handler(BaseHTTPRequestHandler):
    warehouse: Warehouse
    accounts: Accounts

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"ask {self.address_string()} {fmt % args}")

    def _send(self, code: int, payload: dict, cookie: str | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        _cors(self)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self, limit: int) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > limit:
            self._send(413, {"error": "payload too large"})
            raise ValueError("payload too large")
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "bad json"})
            raise ValueError("bad json")

    def _accounts(self, method: str, body: dict | None = None) -> bool:
        result = self.accounts.dispatch(self, method, body)
        if result is None:
            return False
        self._send(result["code"], result["payload"], result.get("cookie"))
        return True

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def do_GET(self) -> None:
        if self._accounts("GET"):
            return
        if urlparse(self.path).path.rstrip("/") == "/api/health":
            years = self.warehouse.seasons()
            self._send(
                200,
                {
                    "ok": True,
                    "key": bool(api_key()),
                    "seasons": years,
                    "model": model_name(),
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/")
        try:
            body = self._read_body(2_000_000 if path.startswith("/api/models") else 80_000)
        except ValueError:
            return
        if self._accounts("POST", body):
            return
        if path != "/api/ask":
            self._send(404, {"error": "not found"})
            return
        if not api_key():
            self._send(502, {"error": "DEEPSEEK is missing. Put it in .env."})
            return
        messages = body.get("messages") or []
        if not isinstance(messages, list) or len(messages) > 24:
            self._send(400, {"error": "messages must be a short list"})
            return
        years = self.warehouse.seasons()
        season = int(body.get("season") or (years[-1] if years else 2026))
        try:
            self._send(200, ask(self.warehouse, messages, season))
        except RuntimeError as exc:
            self._send(502, {"error": str(exc)})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def do_PATCH(self) -> None:
        try:
            body = self._read_body(80_000)
        except ValueError:
            return
        if not self._accounts("PATCH", body):
            self._send(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        if not self._accounts("DELETE", {}):
            self._send(404, {"error": "not found"})


def serve(root: Path | None = None, host: str = HOST, port: int = PORT) -> None:
    root = root or repo_root()
    load_dotenv(root)
    Handler.warehouse = Warehouse(root)
    Handler.accounts = Accounts(root)
    httpd = ThreadingHTTPServer((host, port), Handler)
    key = "ask on" if api_key() else "ask off — set DEEPSEEK to enable"
    print(f"ask http://{host}:{port}  seasons={Handler.warehouse.seasons()}  {key}", flush=True)
    httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek agent over published FootPalm files")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
