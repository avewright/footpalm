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

KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_GAME = "KXNCAAFGAME"
KALSHI_SPREAD = "KXNCAAFSPREAD"
KALSHI_EVENT_URL = "https://kalshi.com/markets/{series}/{ticker}"
KALSHI_WINS = re.compile(r"^(.+?) wins by over ([\d.]+) points$", re.I)
BOOK_SOURCES = ("kalshi", "polymarket")


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
    text = re.sub(r":\s*spread\s*$", "", (title or "").strip(), flags=re.I)
    parts = re.split(r"\s+vs\.?\s+", text, maxsplit=1, flags=re.I)
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


def _dollar(market: dict, *keys: str) -> float | None:
    for key_name in keys:
        raw = market.get(key_name)
        if raw is None or raw == "":
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def mid_price(market: dict) -> float | None:
    bid = _dollar(market, "yes_bid_dollars")
    ask = _dollar(market, "yes_ask_dollars")
    last = _dollar(market, "last_price_dollars")
    if bid is not None and ask is not None and ask > 0:
        return (bid + ask) / 2
    return last if last is not None else bid if bid is not None else ask


def _kalshi_url(series: str, ticker: str) -> str:
    return KALSHI_EVENT_URL.format(series=series.lower(), ticker=ticker.lower())


def fetch_kalshi_series(series: str) -> list[dict]:
    events: list[dict] = []
    cursor = None
    while True:
        url = f"{KALSHI_API}/events?series_ticker={series}&status=open&limit=200&with_nested_markets=true"
        if cursor:
            url += f"&cursor={cursor}"
        page = _get(url)
        if not isinstance(page, dict):
            break
        batch = page.get("events") or []
        events.extend(batch)
        cursor = page.get("cursor")
        if not cursor or not batch or len(events) > 800:
            break
    return events


def parse_kalshi_game(event: dict) -> dict | None:
    split = _split_vs(event.get("title") or "")
    if not split:
        return None
    away_raw, home_raw = split
    ticker = str(event.get("event_ticker") or "")
    row = {
        "source": "kalshi",
        "title": event.get("title"),
        "slug": ticker,
        "url": _kalshi_url(KALSHI_GAME, ticker),
        "away_raw": away_raw,
        "home_raw": home_raw,
        "start": None,
    }
    ml = []
    for market in event.get("markets") or []:
        name = market.get("yes_sub_title") or ""
        price = mid_price(market)
        row["start"] = row["start"] or market.get("occurrence_datetime") or market.get("close_time")
        if name and price is not None and not (price <= 0.001 or price >= 0.999):
            ml.append((name, price))
    if ml:
        row["ml"] = ml
    return row if ml else None


def parse_kalshi_spread(event: dict) -> dict | None:
    split = _split_vs(event.get("title") or "")
    if not split:
        return None
    away_raw, home_raw = split
    ticker = str(event.get("event_ticker") or "")
    best = None
    start = None
    for market in event.get("markets") or []:
        start = start or market.get("occurrence_datetime") or market.get("close_time")
        hit = KALSHI_WINS.match(str(market.get("yes_sub_title") or market.get("title") or ""))
        price = mid_price(market)
        if not hit or price is None:
            continue
        line = _dollar(market, "floor_strike")
        if line is None:
            try:
                line = float(hit.group(2))
            except ValueError:
                continue
        bid = _dollar(market, "yes_bid_dollars")
        ask = _dollar(market, "yes_ask_dollars")
        width = abs(ask - bid) if bid is not None and ask is not None else 1.0
        score = (abs(price - 0.5), width)
        cand = (score, hit.group(1).strip(), float(line), price)
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        return None
    _, team, line, price = best
    return {
        "source": "kalshi",
        "title": event.get("title"),
        "slug": ticker,
        "url": _kalshi_url(KALSHI_SPREAD, ticker),
        "away_raw": away_raw,
        "home_raw": home_raw,
        "start": start,
        "spread_line": -line,
        "spread_outcomes": [(team, price)],
    }


def _kalshi_match_key(row: dict) -> tuple[str, str, str | None]:
    return (key(row["home_raw"]), key(row["away_raw"]), _start_day(row.get("start")))


def fetch_kalshi() -> list[dict]:
    by: dict[tuple[str, str, str | None], dict] = {}
    for raw in fetch_kalshi_series(KALSHI_GAME):
        row = parse_kalshi_game(raw)
        if row:
            by[_kalshi_match_key(row)] = row
    for raw in fetch_kalshi_series(KALSHI_SPREAD):
        row = parse_kalshi_spread(raw)
        if not row:
            continue
        k = _kalshi_match_key(row)
        if k in by:
            by[k]["spread_line"] = row["spread_line"]
            by[k]["spread_outcomes"] = row["spread_outcomes"]
        else:
            by[k] = row
    return list(by.values())


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
        "source": event.get("source") or "polymarket",
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
        "source": "kalshi+polymarket",
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
    source = book.get("source") if book.get("source") in BOOK_SOURCES else "polymarket"
    gkey = game_key(game)
    prev = (log.setdefault("games", {})).get(gkey) or {}
    merged = dict(prev.get(source) or {})
    incoming = dict(book)
    incoming["source"] = source
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
    row = {
        "home": game["home"],
        "away": game["away"],
        "start": game.get("start"),
        "logged_at": prev.get("logged_at") if locked and has_ml else now.isoformat(),
        "locked": bool(locked and has_ml),
    }
    for src in BOOK_SOURCES:
        if src != source and prev.get(src):
            row[src] = prev[src]
    row[source] = incoming
    log["games"][gkey] = row


def stamp_log(games: list[dict], log: dict) -> int:
    n = 0
    for game in games:
        entry = (log.get("games") or {}).get(game_key(game))
        if not entry:
            continue
        books = {src: entry[src] for src in BOOK_SOURCES if entry.get(src)}
        if not books:
            continue
        game["books"] = books
        n += 1
    return n


def attach(games: list[dict], events: list[dict], log: dict | None = None, source: str = "polymarket") -> int:
    log = log if log is not None else empty_log(0)
    now = datetime.now(timezone.utc)
    for game in games:
        existing = ((game.get("books") or {}).get(source)) or {}
        if existing and (live_ml(existing.get("ml_home")) or existing.get("spread") is not None):
            upsert_log(log, game, {**existing, "source": source}, now)
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


def snapshot(
    root: Path,
    season: int,
    games: list[dict],
    events: list[dict] | None = None,
    *,
    sources: tuple[str, ...] | None = None,
) -> int:
    log = load_log(root, season)
    wanted = sources or (("polymarket",) if events is not None else BOOK_SOURCES)
    if "kalshi" in wanted:
        attach(games, fetch_kalshi(), log, source="kalshi")
    if "polymarket" in wanted:
        poly = events if events is not None else fetch_polymarket()
        attach(games, poly, log, source="polymarket")
    n = stamp_log(games, log)
    save_log(root, season, log)
    return n


def run(root: Path | None = None, season: int = 2026, sources: tuple[str, ...] = BOOK_SOURCES) -> dict:
    root = root or _repo_root()
    log = load_log(root, season)
    fetched: dict[str, int] = {}
    matched = 0
    events_by: dict[str, list[dict]] = {}
    if "kalshi" in sources:
        events_by["kalshi"] = fetch_kalshi()
        fetched["kalshi"] = len(events_by["kalshi"])
    if "polymarket" in sources:
        events_by["polymarket"] = fetch_polymarket()
        fetched["polymarket"] = len(events_by["polymarket"])
    for dest in (root / "data" / "processed", root / "web" / "public" / "data"):
        path = dest / f"predictions-{season}.json"
        if not path.exists():
            continue
        body = json.loads(path.read_text())
        for source, rows in events_by.items():
            matched = attach(body["games"], rows, log, source=source)
        body["books"] = {
            "source": "+".join(sources),
            "pulled_at": datetime.now(timezone.utc).isoformat(),
            "events": fetched,
            "matched": matched,
            "logged": len(log.get("games") or {}),
            "note": "Display only. Do not train on Kalshi, Polymarket, or any market line.",
        }
        _write_json(path, body)
    save_log(root, season, log)
    print(
        f"markets sources={'+'.join(sources)} events={fetched} "
        f"matched={matched} logged={len(log.get('games') or {})} season={season}"
    )
    return {"events": fetched, "matched": matched, "logged": len(log.get("games") or {})}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Kalshi/Polymarket CFB lines onto predictions")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--source",
        choices=("kalshi", "polymarket", "all"),
        default="all",
        help="kalshi is free and what the 15-minute job uses. all also hits Polymarket.",
    )
    args = parser.parse_args()
    sources = BOOK_SOURCES if args.source == "all" else (args.source,)
    run(season=args.season, sources=sources)


if __name__ == "__main__":
    main()
