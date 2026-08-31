"""ESPN site/core harvest. Weather, roster QBs, FPI, DraftKings.

Snapshots are time-controlled: append while the game is upcoming, freeze the
last pre-kickoff row at lock. Display and holdout only. Do not train TabPFN.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from footpalm.markets import game_key, match_team
from footpalm.project import _write_json
from footpalm.trees import _repo_root

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
PAUSE_S = 0.25
SNAP_CAP = 24
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _get(url: str) -> object:
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.espn.com/college-football/scoreboard",
            "Origin": "https://www.espn.com",
        },
    )
    with urlopen(req, timeout=45) as response:
        return json.loads(response.read().decode())


def _as_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_when(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        when = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when


def kickoff_passed(game: dict, now: datetime) -> bool:
    if game.get("completed") or game.get("actual_home") is not None:
        return True
    when = _parse_when(game.get("start"))
    return bool(when and when <= now)


def _start_day(value: str | None) -> str | None:
    when = _parse_when(value)
    return when.date().isoformat() if when else None


def parse_weather(game_info: dict | None) -> dict | None:
    if not game_info:
        return None
    venue = game_info.get("venue") or {}
    weather = game_info.get("weather") or {}
    address = venue.get("address") or {}
    row = {
        "venue": venue.get("fullName") or venue.get("name"),
        "city": address.get("city"),
        "state": address.get("state"),
        "grass": venue.get("grass"),
        "temperature": _as_float(weather.get("temperature")),
        "gust": _as_float(weather.get("gust") or weather.get("windSpeed")),
        "precipitation": _as_float(weather.get("precipitation")),
        "condition": weather.get("displayValue") or weather.get("conditionId"),
    }
    if all(row.get(k) is None for k in ("temperature", "gust", "precipitation", "venue")):
        return None
    return row


def parse_fpi(predictor: dict | None) -> dict | None:
    if not predictor:
        return None
    home = (predictor.get("homeTeam") or {}).get("gameProjection")
    home_p = _as_float(home)
    if home_p is None:
        return None
    if home_p > 1:
        home_p = home_p / 100.0
    row = {"home_win": round(home_p, 4), "source": "espn-fpi"}
    stats = (predictor.get("homeTeam") or {}).get("statistics") or []
    for stat in stats:
        if stat.get("name") == "teamPredPtDiff":
            margin = _as_float(stat.get("value"))
            if margin is not None:
                row["pred_margin"] = round(margin, 2)
    return row


def parse_odds(rows: list | None) -> dict | None:
    if not rows:
        return None
    main = rows[0]
    spread = _as_float(main.get("spread"))
    total = _as_float(main.get("overUnder"))
    if spread is None and total is None:
        return None
    provider = (main.get("provider") or {}).get("name") or "DraftKings"
    return {
        "source": str(provider).lower().replace(" ", ""),
        "spread": spread,
        "total": total,
        "details": main.get("details"),
        "over_odds": _as_float(main.get("overOdds")),
        "under_odds": _as_float(main.get("underOdds")),
    }


def parse_qbs(roster: dict | None) -> list[dict]:
    out = []
    seen: set[str] = set()
    for group in (roster or {}).get("athletes") or []:
        items = group.get("items") if isinstance(group, dict) else None
        if items is None and isinstance(group, dict):
            items = [group]
        for athlete in items or []:
            pos = ((athlete.get("position") or {}).get("abbreviation") or "").upper()
            if pos != "QB":
                continue
            name = athlete.get("displayName") or athlete.get("shortName")
            if not name or name in seen:
                continue
            seen.add(name)
            exp = athlete.get("experience") or {}
            out.append(
                {
                    "name": name,
                    "jersey": str(athlete.get("jersey") or "") or None,
                    "year": exp.get("displayValue") if isinstance(exp, dict) else None,
                    "group": group.get("position") or group.get("name") if isinstance(group, dict) else None,
                }
            )
    return out


def parse_event(event: dict) -> dict | None:
    comps = event.get("competitions") or []
    if not comps:
        return None
    comp = comps[0]
    home_raw = away_raw = home_id = away_id = None
    for team in comp.get("competitors") or []:
        info = team.get("team") or {}
        raw = info.get("location") or info.get("displayName") or info.get("shortDisplayName")
        tid = info.get("id")
        if team.get("homeAway") == "home":
            home_raw, home_id = raw, tid
        else:
            away_raw, away_id = raw, tid
    if not home_raw or not away_raw:
        return None
    return {
        "espn_id": str(event.get("id") or comp.get("id") or ""),
        "start": event.get("date") or comp.get("date"),
        "home_raw": home_raw,
        "away_raw": away_raw,
        "home_id": str(home_id) if home_id else None,
        "away_id": str(away_id) if away_id else None,
        "neutral": bool(comp.get("neutralSite")),
        "odds": parse_odds(comp.get("odds") or event.get("odds")),
    }


def parse_summary(summary: dict, event: dict | None = None) -> dict:
    header = summary.get("header") or {}
    parsed = parse_event({"id": header.get("id"), "date": None, "competitions": header.get("competitions") or []})
    row = dict(event or {})
    if parsed:
        row.setdefault("espn_id", parsed["espn_id"])
        row.setdefault("home_raw", parsed["home_raw"])
        row.setdefault("away_raw", parsed["away_raw"])
    weather = parse_weather(summary.get("gameInfo"))
    fpi = parse_fpi(summary.get("predictor"))
    odds = parse_odds(summary.get("pickcenter") or summary.get("odds")) or row.get("odds")
    if weather:
        row["weather"] = weather
    if fpi:
        row["fpi"] = fpi
    if odds:
        row["odds"] = odds
    return row


def bind_event(event: dict, home: str, away: str) -> dict | None:
    known = {home, away}
    ev_home = match_team(event.get("home_raw") or "", known)
    ev_away = match_team(event.get("away_raw") or "", known)
    if {ev_home, ev_away} != known:
        return None
    return event


def scoreboard_url(*, date: str | None = None) -> str:
    url = f"{SITE}/scoreboard?limit=300&groups=80"
    if date:
        url += f"&dates={date.replace('-', '')}"
    return url


def fetch_scoreboard(date: str | None = None) -> list[dict]:
    payload = _get(scoreboard_url(date=date))
    if not isinstance(payload, dict):
        return []
    out = []
    for event in payload.get("events") or []:
        parsed = parse_event(event)
        if parsed:
            out.append(parsed)
    return out


def fetch_summary(espn_id: str) -> dict:
    payload = _get(f"{SITE}/summary?event={espn_id}")
    return parse_summary(payload if isinstance(payload, dict) else {})


def fetch_roster(team_id: str) -> list[dict]:
    payload = _get(f"{SITE}/teams/{team_id}/roster")
    return parse_qbs(payload if isinstance(payload, dict) else {})


def log_path(root: Path, season: int) -> Path:
    return root / "data" / "processed" / f"espn-{season}.json"


def empty_log(season: int) -> dict:
    return {
        "source": "espn-site",
        "season": season,
        "note": "Pre-game snapshots. Locked at kickoff. Do not train TabPFN on ESPN FPI, DK, or weather.",
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


def _snap_payload(row: dict) -> dict:
    out = {}
    for key in ("weather", "fpi", "odds", "qbs"):
        if row.get(key) is not None:
            out[key] = row[key]
    return out


def _changed(prev: dict | None, incoming: dict) -> bool:
    if not prev:
        return True
    return json.dumps(_snap_payload(prev), sort_keys=True) != json.dumps(_snap_payload(incoming), sort_keys=True)


def upsert_log(log: dict, game: dict, row: dict, now: datetime) -> None:
    gkey = game_key(game)
    prev = (log.setdefault("games", {})).get(gkey) or {}
    locked = bool(prev.get("locked")) or kickoff_passed(game, now)
    incoming = dict(row)
    if locked and prev.get("logged_at"):
        for field in ("weather", "fpi", "odds", "qbs", "venue"):
            if prev.get(field) is not None:
                incoming[field] = prev[field]
        incoming["espn_id"] = prev.get("espn_id") or incoming.get("espn_id")
    snaps = list(prev.get("snaps") or [])
    if not locked and _changed(prev, incoming):
        snaps.append({"at": now.isoformat(), **_snap_payload(incoming)})
        snaps = snaps[-SNAP_CAP:]
    elif not snaps and _snap_payload(incoming):
        snaps.append({"at": now.isoformat(), **_snap_payload(incoming)})
    entry = {
        "home": game["home"],
        "away": game["away"],
        "start": game.get("start") or incoming.get("start"),
        "espn_id": incoming.get("espn_id") or prev.get("espn_id"),
        "logged_at": prev.get("logged_at") if locked and prev.get("logged_at") else now.isoformat(),
        "locked": locked,
        "snaps": snaps,
    }
    for field in ("weather", "fpi", "odds", "qbs"):
        if incoming.get(field) is not None:
            entry[field] = incoming[field]
        elif prev.get(field) is not None:
            entry[field] = prev[field]
    if incoming.get("weather"):
        entry["venue"] = {
            k: incoming["weather"].get(k) for k in ("venue", "city", "state", "grass") if incoming["weather"].get(k) is not None
        }
    elif prev.get("venue"):
        entry["venue"] = prev["venue"]
    log["games"][gkey] = entry


def public_row(entry: dict) -> dict:
    out = {
        "logged_at": entry.get("logged_at"),
        "locked": bool(entry.get("locked")),
        "snaps": len(entry.get("snaps") or []),
    }
    for key in ("weather", "venue", "qbs", "fpi", "odds"):
        if entry.get(key) is not None:
            out[key] = entry[key]
    return out


def stamp_log(games: list[dict], log: dict) -> int:
    n = 0
    for game in games:
        entry = (log.get("games") or {}).get(game_key(game))
        if not entry:
            continue
        game["espn"] = public_row(entry)
        n += 1
    return n


def apply_log(root: Path, season: int, games: list[dict]) -> int:
    return stamp_log(games, load_log(root, season))


def _dates_for(games: list[dict], now: datetime) -> list[str]:
    days = {now.date().isoformat()}
    for game in games:
        if game.get("completed"):
            continue
        day = _start_day(game.get("start"))
        if day:
            days.add(day)
    return sorted(days)[:16]


def harvest_events(games: list[dict], now: datetime, *, fetch=fetch_scoreboard) -> list[dict]:
    events: list[dict] = []
    seen: set[str] = set()
    for day in _dates_for(games, now):
        try:
            rows = fetch(day)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"espn scoreboard {day}: {exc}", flush=True)
            continue
        for row in rows:
            eid = row.get("espn_id")
            if eid and eid in seen:
                continue
            if eid:
                seen.add(eid)
            events.append(row)
        time.sleep(PAUSE_S)
    return events


def _need_summary(prev: dict, locked: bool, game: dict, now: datetime) -> bool:
    if locked:
        return False
    if prev.get("weather") is None or prev.get("fpi") is None:
        return True
    start = _parse_when(game.get("start"))
    return bool(start and (start - now).total_seconds() <= 3 * 86400)


def _need_roster(prev: dict, locked: bool) -> bool:
    if locked:
        return False
    qbs = prev.get("qbs") or {}
    return not qbs.get("home") or not qbs.get("away")


def snapshot(
    root: Path,
    season: int,
    games: list[dict],
    *,
    now: datetime | None = None,
    fetch_board=fetch_scoreboard,
    fetch_sum=fetch_summary,
    fetch_ros=fetch_roster,
) -> int:
    now = now or datetime.now(timezone.utc)
    log = load_log(root, season)
    events = harvest_events(games, now, fetch=fetch_board)
    for game in games:
        prev = (log.get("games") or {}).get(game_key(game)) or {}
        locked = bool(prev.get("locked")) or kickoff_passed(game, now)
        event = None
        game_day = _start_day(game.get("start"))
        for cand in events:
            if bind_event(cand, game["home"], game["away"]) is None:
                continue
            ev_day = _start_day(cand.get("start"))
            if game_day and ev_day and game_day != ev_day:
                continue
            event = cand
            break
        if event is None and prev.get("espn_id"):
            event = {"espn_id": prev["espn_id"], "home_raw": game["home"], "away_raw": game["away"]}
        if event is None:
            if prev:
                upsert_log(log, game, prev, now)
            continue
        row = dict(event)
        eid = row.get("espn_id")
        if eid and _need_summary(prev, locked, game, now):
            try:
                extra = fetch_sum(eid)
                if isinstance(extra, dict):
                    if extra.get("gameInfo") or extra.get("predictor") or extra.get("header"):
                        extra = parse_summary(extra, row)
                    for key, value in extra.items():
                        if value is not None:
                            row[key] = value
                time.sleep(PAUSE_S)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"espn summary {eid}: {exc}", flush=True)
        if _need_roster(prev, locked):
            qbs = dict(prev.get("qbs") or {})
            for side, tid in (
                ("home", row.get("home_id") or prev.get("home_id")),
                ("away", row.get("away_id") or prev.get("away_id")),
            ):
                if qbs.get(side) or not tid:
                    continue
                try:
                    qbs[side] = fetch_ros(str(tid))
                    time.sleep(PAUSE_S)
                except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                    print(f"espn roster {tid}: {exc}", flush=True)
            if qbs:
                row["qbs"] = qbs
        elif prev.get("qbs"):
            row["qbs"] = prev["qbs"]
        upsert_log(log, game, row, now)
    n = stamp_log(games, log)
    save_log(root, season, log)
    return n


def run(root: Path | None = None, season: int = 2026) -> dict:
    root = root or _repo_root()
    matched = 0
    logged = 0
    for dest in (root / "data" / "processed", root / "web" / "public" / "data"):
        path = dest / f"predictions-{season}.json"
        if not path.exists():
            continue
        body = json.loads(path.read_text())
        matched = snapshot(root, season, body["games"])
        logged = len(load_log(root, season).get("games") or {})
        body["espn"] = {
            "source": "espn-site",
            "pulled_at": datetime.now(timezone.utc).isoformat(),
            "matched": matched,
            "logged": logged,
            "note": "Weather, roster QBs, FPI, DraftKings. Locked at kickoff. Not a model input.",
        }
        _write_json(path, body)
    print(f"espn matched={matched} logged={logged} season={season}")
    return {"matched": matched, "logged": logged}


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest ESPN weather, QBs, FPI, DK. Lock at kickoff.")
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()
    run(season=args.season)


if __name__ == "__main__":
    main()
