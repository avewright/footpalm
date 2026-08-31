"""Stamp live scores onto existing projections. Does not refit TabPFN."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from footpalm.cfbd import load_dotenv, pull_dataset
from footpalm.fetch import LIVE_SEASON, repo_root
from footpalm.project import _publish, load_cfbd_slate

ET = ZoneInfo("America/New_York")


def et_day(start: str | None) -> str | None:
    if not start:
        return None
    return datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(ET).date().isoformat()


def grade(game: dict) -> dict | None:
    ah, aa = game.get("actual_home"), game.get("actual_away")
    if ah is None or aa is None:
        return None
    p = float(game["home_win_prob"])
    pred_m = float(game["pred_margin"])
    actual_m = float(ah) - float(aa)
    y = int(float(ah) > float(aa))
    p = min(1 - 1e-6, max(1e-6, p))
    spread = game.get("spread")
    take_home = None
    ats = None
    if spread is not None:
        take_home = (pred_m + float(spread)) > 0
        cover = actual_m + float(spread)
        if abs(cover) >= 1e-9:
            ats = take_home == (cover > 0)
    return {
        "su": (p >= 0.5) == bool(y),
        "ats": ats,
        "take_home": take_home,
        "brier": (p - y) ** 2,
        "logloss": -(y * math.log(p) + (1 - y) * math.log(1 - p)),
        "mae": abs(pred_m - actual_m),
        "day": et_day(game.get("start")),
    }


def summarize(games: list[dict]) -> dict:
    graded = [(g, grade(g)) for g in games]
    graded = [(g, row) for g, row in graded if row]
    n = len(graded)
    ats = [(g, row) for g, row in graded if row["ats"] is not None]
    su_w = sum(row["su"] for _, row in graded)
    ats_w = sum(row["ats"] for _, row in ats)
    return {
        "n": n,
        "su_w": su_w,
        "su_l": n - su_w,
        "ats_w": ats_w,
        "ats_l": len(ats) - ats_w,
        "ats_n": len(ats),
        "brier": round(sum(row["brier"] for _, row in graded) / n, 4) if n else None,
        "logloss": round(sum(row["logloss"] for _, row in graded) / n, 4) if n else None,
        "mae": round(sum(row["mae"] for _, row in graded) / n, 2) if n else None,
    }


def stamp_payload(payload: dict, slate: list[dict]) -> tuple[dict, int]:
    by_id = {int(row["game_id"]): row for row in slate if row.get("game_id") is not None}
    stamped = 0
    games = []
    for game in payload.get("games", []):
        raw = by_id.get(int(game["game_id"])) if game.get("game_id") is not None else None
        if raw is None:
            games.append(game)
            continue
        next_game = dict(game)
        if raw.get("start"):
            next_game["start"] = raw["start"]
        if raw.get("spread") is not None and next_game.get("spread") is None:
            next_game["spread"] = raw["spread"]
        if raw.get("actual_home") is not None and raw.get("actual_away") is not None:
            if next_game.get("actual_home") != raw["actual_home"] or next_game.get("actual_away") != raw["actual_away"]:
                stamped += 1
            next_game["actual_home"] = raw["actual_home"]
            next_game["actual_away"] = raw["actual_away"]
            next_game["actual_margin"] = raw["actual_margin"]
            next_game["home_won"] = raw["home_won"]
            next_game["completed"] = True
        elif raw.get("completed"):
            next_game["completed"] = True
        games.append(next_game)
    out = dict(payload)
    out["games"] = games
    out["scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out, stamped


def scorecard(games: list[dict]) -> dict:
    played = [g for g in games if g.get("actual_home") is not None]
    days = sorted({et_day(g.get("start")) for g in played if et_day(g.get("start"))})
    last = days[-1] if days else None
    last_games = [g for g in played if et_day(g.get("start")) == last] if last else []
    return {
        "last_day": last,
        "last": summarize(last_games),
        "to_date": summarize(played),
    }


def refresh_scores(root: Path, *, season: int = LIVE_SEASON, refresh: bool = True) -> dict:
    load_dotenv(root)
    if refresh:
        print(f"refreshing CFBD games/lines {season}")
        pull_dataset(root, "games", "/games", season, {"seasonType": "both"}, refresh=True)
        pull_dataset(root, "lines", "/lines", season, {"seasonType": "both"}, refresh=True)
    pred_path = root / "web" / "public" / "data" / f"predictions-{season}.json"
    if not pred_path.exists():
        raise SystemExit(f"missing {pred_path} — run footpalm.project first")
    payload = json.loads(pred_path.read_text())
    slate, _, _ = load_cfbd_slate(root, season)
    payload, stamped = stamp_payload(payload, slate)
    from footpalm.markets import apply_log, snapshot

    try:
        snapshot(root, season, payload["games"])
    except Exception as exc:
        print(f"market log: {exc}", flush=True)
        apply_log(root, season, payload["games"])
    card = scorecard(payload["games"])
    live = {
        "season": season,
        "generated_at": payload["scored_at"],
        "note": "Scores stamped onto frozen projections. TabPFN is not refit.",
        **card,
    }
    _publish(root, f"predictions-{season}.json", payload)
    _publish(root, "live.json", live)
    last = card["last"]
    print(
        f"stamped {stamped} · last {card['last_day']} "
        f"SU {last['su_w']}-{last['su_l']} ATS {last['ats_w']}-{last['ats_l']} "
        f"Brier {last['brier']}"
    )
    return live


def main() -> None:
    parser = argparse.ArgumentParser(description="Stamp CFBD scores onto live projections")
    parser.add_argument("--season", type=int, default=LIVE_SEASON)
    parser.add_argument("--no-refresh", action="store_true", help="use cached CFBD games/lines")
    parser.add_argument("--install", action="store_true", help="install Kalshi/score/project launchd jobs")
    args = parser.parse_args()
    root = repo_root()
    if args.install:
        from footpalm.cron import install

        install(root)
        return
    refresh_scores(root, season=args.season, refresh=not args.no_refresh)


if __name__ == "__main__":
    main()
