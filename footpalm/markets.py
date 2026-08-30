from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from footpalm.names import canon, key
from footpalm.project import _write_json
from footpalm.trees import _repo_root

POLY_SERIES = 12756
POLY_EVENTS = "https://gamma-api.polymarket.com/events"
POLY_EVENT_URL = "https://polymarket.com/event/{slug}"


def american(p: float) -> int:
    p = min(1 - 1e-6, max(1e-6, p))
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def _get(url: str) -> object:
    req = Request(url, headers={"User-Agent": "footpalm/1.0", "Accept": "application/json"})
    with urlopen(req, timeout=45) as response:
        return json.loads(response.read().decode())


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def _split_vs(title: str) -> tuple[str, str] | None:
    parts = re.split(r"\s+vs\.?\s+", title.strip(), maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def match_team(raw: str, known: set[str]) -> str | None:
    parts = key(raw).split()
    candidates = [" ".join(parts[:i]) for i in range(len(parts), 0, -1)]
    by_key = {key(team): team for team in known}
    for cand in candidates:
        mapped = canon(cand)
        if mapped in known:
            return mapped
        if key(mapped) in by_key:
            return by_key[key(mapped)]
        if key(cand) in by_key:
            return by_key[key(cand)]
    hits = [team for team in known if key(team).startswith(key(raw)) or key(raw).startswith(key(team))]
    if len(hits) == 1:
        return hits[0]
    return None


def _prices(market: dict) -> list[tuple[str, float]]:
    names = [str(x) for x in _as_list(market.get("outcomes"))]
    prices = [float(x) for x in _as_list(market.get("outcomePrices"))]
    return list(zip(names, prices, strict=False))


def parse_event(event: dict) -> dict | None:
    split = _split_vs(event.get("title") or "")
    if not split:
        return None
    away_raw, home_raw = split
    markets = event.get("markets") or []
    moneyline = next((m for m in markets if m.get("sportsMarketType") == "moneyline"), None)
    spreads = [m for m in markets if m.get("sportsMarketType") == "spreads" and m.get("line") is not None]
    row = {
        "title": event.get("title"),
        "slug": event.get("slug"),
        "url": POLY_EVENT_URL.format(slug=event.get("slug") or ""),
        "away_raw": away_raw,
        "home_raw": home_raw,
        "start": None,
    }
    if moneyline:
        row["ml"] = _prices(moneyline)
        row["start"] = moneyline.get("gameStartTime") or moneyline.get("endDate")
    if spreads:
        main = spreads[0]
        row["spread_line"] = float(main["line"])
        row["spread_outcomes"] = _prices(main)
        row["start"] = row.get("start") or main.get("gameStartTime") or main.get("endDate")
    if "ml" not in row and "spread_line" not in row:
        return None
    return row


def fetch_polymarket() -> list[dict]:
    events: list[dict] = []
    offset = 0
    while True:
        page = _get(f"{POLY_EVENTS}?series_id={POLY_SERIES}&closed=false&limit=100&offset={offset}")
        if not isinstance(page, list) or not page:
            break
        events.extend(page)
        if len(page) < 100:
            break
        offset += 100
        if offset > 800:
            break
    parsed = []
    for event in events:
        row = parse_event(event)
        if row:
            parsed.append(row)
    return parsed


def _start_day(value: str | None) -> str | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00").replace(" ", "T", 1)
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc).date().isoformat()
    except ValueError:
        return value[:10] if len(value) >= 10 else None


def bind_book(event: dict, home: str, away: str) -> dict | None:
    known = {home, away}
    ev_home = match_team(event["home_raw"], known)
    ev_away = match_team(event["away_raw"], known)
    if {ev_home, ev_away} != known:
        return None
    book = {
        "source": "polymarket",
        "slug": event.get("slug"),
        "url": event.get("url"),
        "title": event.get("title"),
    }
    if event.get("ml"):
        p_by = {}
        for name, price in event["ml"]:
            team = match_team(name, known)
            if team:
                p_by[team] = price
        if home in p_by and away in p_by:
            book["ml_home"] = round(p_by[home], 4)
            book["ml_away"] = round(p_by[away], 4)
            book["ml_home_american"] = american(p_by[home])
            book["ml_away_american"] = american(p_by[away])
    if event.get("spread_line") is not None and event.get("spread_outcomes"):
        listed, _p = event["spread_outcomes"][0]
        listed_team = match_team(listed, known)
        if listed_team:
            line = float(event["spread_line"])
            book["spread"] = round(line if listed_team == home else -line, 1)
            p_cover = {match_team(name, known): price for name, price in event["spread_outcomes"]}
            if home in p_cover:
                book["spread_p_home"] = round(p_cover[home], 4)
    if "ml_home" not in book and "spread" not in book:
        return None
    return book


def attach(games: list[dict], events: list[dict]) -> int:
    n = 0
    for game in games:
        home, away = game["home"], game["away"]
        game_day = _start_day(game.get("start"))
        hits = []
        for event in events:
            book = bind_book(event, home, away)
            if not book:
                continue
            ev_day = _start_day(event.get("start"))
            if game_day and ev_day and game_day != ev_day:
                continue
            hits.append(book)
        if not hits:
            game.pop("books", None)
            continue
        game["books"] = {"polymarket": hits[0]}
        n += 1
    return n


def run(root: Path | None = None, season: int = 2026) -> dict:
    root = root or _repo_root()
    events = fetch_polymarket()
    matched = 0
    for dest in (root / "data" / "processed", root / "web" / "public" / "data"):
        path = dest / f"predictions-{season}.json"
        if not path.exists():
            continue
        body = json.loads(path.read_text())
        matched = attach(body["games"], events)
        body["books"] = {
            "source": "polymarket",
            "series_id": POLY_SERIES,
            "pulled_at": datetime.now(timezone.utc).isoformat(),
            "events": len(events),
            "matched": matched,
            "note": "Display only. Do not train on Polymarket or any market line.",
        }
        _write_json(path, body)
    print(f"polymarket events={len(events)} matched={matched} season={season}")
    return {"events": len(events), "matched": matched}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Polymarket CFB moneyline/spread onto predictions")
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()
    run(season=args.season)


if __name__ == "__main__":
    main()
