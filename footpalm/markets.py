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


def _settled(prices: list) -> bool:
    return bool(prices) and all(p <= 0.001 or p >= 0.999 for _name, p in prices)


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
        prices = _prices(moneyline)
        if prices and not _settled(prices):
            row["ml"] = prices
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
    seen: set[str] = set()
    for closed, cap in ((False, 800), (True, 200)):
        offset = 0
        while True:
            page = _get(
                f"{POLY_EVENTS}?series_id={POLY_SERIES}&closed={str(closed).lower()}&limit=100&offset={offset}"
            )
            if not isinstance(page, list) or not page:
                break
            for event in page:
                slug = str(event.get("slug") or "")
                if slug and slug in seen:
                    continue
                if slug:
                    seen.add(slug)
                events.append(event)
            if len(page) < 100:
                break
            offset += 100
            if offset > cap:
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


def log_path(root: Path, season: int) -> Path:
    return root / "data" / "processed" / f"markets-{season}.json"


def empty_log(season: int) -> dict:
    return {
        "source": "polymarket",
        "season": season,
        "note": "Pre-game snapshots. Locked at kickoff. Do not train TabPFN on this.",
        "games": {},
    }


def load_log(root: Path, season: int) -> dict:
    path = log_path(root, season)
    if not path.exists():
        return empty_log(season)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("games"), dict):
        return empty_log(season)
    raw.setdefault("season", season)
    return raw


def save_log(root: Path, season: int, log: dict) -> Path:
    path = log_path(root, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    log["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(log, indent=2) + "\n")
    return path


def game_key(game: dict) -> str:
    gid = game.get("game_id")
    if gid is not None:
        return f"id:{int(gid)}"
    return "|".join(
        str(x)
        for x in (
            game.get("season") or "",
            game.get("week") or "",
            game.get("away"),
            game.get("home"),
            _start_day(game.get("start")) or "",
        )
    )


def live_ml(value) -> bool:
    try:
        return 0.02 < float(value) < 0.98
    except (TypeError, ValueError):
        return False


def _kickoff_passed(game: dict, now: datetime) -> bool:
    if game.get("completed") or game.get("actual_home") is not None:
        return True
    start = game.get("start")
    if not start:
        return False
    try:
        when = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when <= now


def upsert_log(log: dict, game: dict, book: dict, now: datetime) -> None:
    key = game_key(game)
    prev = (log.setdefault("games", {})).get(key) or {}
    merged = dict(prev.get("polymarket") or {})
    incoming = dict(book)
    locked = bool(prev.get("locked")) or _kickoff_passed(game, now)
    if locked and live_ml(merged.get("ml_home")):
        for field in ("ml_home", "ml_away", "ml_home_american", "ml_away_american"):
            incoming.pop(field, None)
            if field in merged:
                incoming[field] = merged[field]
    elif not live_ml(incoming.get("ml_home")) and live_ml(merged.get("ml_home")):
        for field in ("ml_home", "ml_away", "ml_home_american", "ml_away_american"):
            if field in merged:
                incoming[field] = merged[field]
    if incoming.get("spread") is None and merged.get("spread") is not None:
        incoming["spread"] = merged["spread"]
        if "spread_p_home" in merged:
            incoming["spread_p_home"] = merged["spread_p_home"]
    if "ml_home" not in incoming and "spread" not in incoming:
        return
    has_ml = live_ml(incoming.get("ml_home"))
    log["games"][key] = {
        "home": game["home"],
        "away": game["away"],
        "start": game.get("start"),
        "logged_at": prev.get("logged_at") if locked and has_ml else now.isoformat(),
        "locked": bool(locked and has_ml),
        "polymarket": incoming,
    }


def stamp_log(games: list[dict], log: dict) -> int:
    n = 0
    for game in games:
        entry = (log.get("games") or {}).get(game_key(game))
        if not entry:
            continue
        game["books"] = {"polymarket": entry["polymarket"]}
        n += 1
    return n


def attach(games: list[dict], events: list[dict], log: dict | None = None) -> int:
    log = log if log is not None else empty_log(0)
    now = datetime.now(timezone.utc)
    for game in games:
        existing = ((game.get("books") or {}).get("polymarket")) or {}
        if live_ml(existing.get("ml_home")) or existing.get("spread") is not None:
            upsert_log(log, game, existing, now)
        home, away = game["home"], game["away"]
        game_day = _start_day(game.get("start"))
        for event in events:
            book = bind_book(event, home, away)
            if not book:
                continue
            ev_day = _start_day(event.get("start"))
            if game_day and ev_day and game_day != ev_day:
                continue
            upsert_log(log, game, book, now)
    return stamp_log(games, log)


def apply_log(root: Path, season: int, games: list[dict]) -> int:
    return stamp_log(games, load_log(root, season))


def snapshot(root: Path, season: int, games: list[dict], events: list[dict] | None = None) -> int:
    log = load_log(root, season)
    if events is None:
        events = fetch_polymarket()
    n = attach(games, events, log)
    save_log(root, season, log)
    return n


def run(root: Path | None = None, season: int = 2026) -> dict:
    root = root or _repo_root()
    events = fetch_polymarket()
    log = load_log(root, season)
    matched = 0
    for dest in (root / "data" / "processed", root / "web" / "public" / "data"):
        path = dest / f"predictions-{season}.json"
        if not path.exists():
            continue
        body = json.loads(path.read_text())
        matched = attach(body["games"], events, log)
        body["books"] = {
            "source": "polymarket",
            "series_id": POLY_SERIES,
            "pulled_at": datetime.now(timezone.utc).isoformat(),
            "events": len(events),
            "matched": matched,
            "logged": len(log.get("games") or {}),
            "note": "Display only. Do not train on Polymarket or any market line.",
        }
        _write_json(path, body)
    save_log(root, season, log)
    print(f"polymarket events={len(events)} matched={matched} logged={len(log.get('games') or {})} season={season}")
    return {"events": len(events), "matched": matched, "logged": len(log.get("games") or {})}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Polymarket CFB moneyline/spread onto predictions")
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()
    run(season=args.season)


if __name__ == "__main__":
    main()
