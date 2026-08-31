"""Local launchd jobs. Kalshi often, ESPN/CFBD on score, TabPFN once a week.

    uv run python -m footpalm.cron --install

Kalshi/Polymarket are free. CFBD free tier is ~1000 calls/month. Auto score
only refreshes games+lines when the cache is stale (6h weekdays, 3h Fri–Sun).
Weekly project also refreshes games+lines only — not the full satellite.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from footpalm.cfbd import cache_age_s, load_dotenv, load_usage, pull_dataset, should_refresh_cfbd
from footpalm.fetch import LIVE_SEASON, repo_root
from footpalm.markets import run as run_markets
from footpalm.score import refresh_scores

ET = ZoneInfo("America/New_York")
LINES_EVERY_S = 15 * 60
SCORE_EVERY_S = 2 * 60 * 60
PROJECT_WEEKDAY = 1  # Tuesday
PROJECT_HOUR = 10

JOBS = (
    ("com.footpalm.lines", "lines", LINES_EVERY_S, None, True),
    ("com.footpalm.score", "score", SCORE_EVERY_S, None, True),
    (
        "com.footpalm.project",
        "project",
        None,
        {"Weekday": PROJECT_WEEKDAY, "Hour": PROJECT_HOUR, "Minute": 0},
        False,
    ),
)


def _uv() -> Path:
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is not on PATH; cannot install launchd jobs")
    return Path(uv)


def plist_body(
    root: Path,
    uv: Path,
    label: str,
    command: str,
    interval_s: int | None,
    calendar: dict | None,
    run_at_load: bool,
) -> str:
    log = root / "data" / "processed" / f"{command}.log"
    when = ""
    if interval_s is not None:
        when = f"""  <key>StartInterval</key>
  <integer>{interval_s}</integer>
"""
    if calendar:
        keys = "\n".join(f"      <key>{k}</key>\n      <integer>{v}</integer>" for k, v in calendar.items())
        when += f"""  <key>StartCalendarInterval</key>
  <dict>
{keys}
  </dict>
"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{label}</string>
  <key>WorkingDirectory</key>
  <string>{root}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{uv}</string>
    <string>run</string>
    <string>python</string>
    <string>-m</string>
    <string>footpalm.cron</string>
    <string>{command}</string>
  </array>
{when}  <key>RunAtLoad</key>
  <{'true' if run_at_load else 'false'}/>
  <key>StandardOutPath</key>
  <string>{log}</string>
  <key>StandardErrorPath</key>
  <string>{log}</string>
</dict>
</plist>
"""


def _load_plist(dest: Path, label: str) -> None:
    uid = os.getuid()
    target = f"gui/{uid}/{label}"
    subprocess.run(["launchctl", "bootout", target], check=False, capture_output=True)
    loaded = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(dest)], capture_output=True, text=True)
    if loaded.returncode != 0:
        raise SystemExit(loaded.stderr.strip() or loaded.stdout.strip() or f"launchctl bootstrap failed for {label}")


def install(root: Path) -> list[Path]:
    uv = _uv()
    dests = []
    agents = Path.home() / "Library" / "LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    for label, command, interval, calendar, run_at_load in JOBS:
        dest = agents / f"{label}.plist"
        dest.write_text(plist_body(root, uv, label, command, interval, calendar, run_at_load))
        _load_plist(dest, label)
        dests.append(dest)
        print(f"installed {dest}")
    print("kalshi every 15m · score every 2h (CFBD throttled) · project Tue 10am ET")
    return dests


def uninstall() -> None:
    uid = os.getuid()
    agents = Path.home() / "Library" / "LaunchAgents"
    for label, *_rest in JOBS:
        subprocess.run(["launchctl", "bootout", f"gui/{uid}/{label}"], check=False, capture_output=True)
        dest = agents / f"{label}.plist"
        if dest.exists():
            dest.unlink()
            print(f"removed {dest}")


def run_lines(root: Path, season: int = LIVE_SEASON) -> dict:
    load_dotenv(root)
    return run_markets(root, season=season, sources=("kalshi",))


def refresh_live_slate(root: Path, season: int) -> None:
    # Two CFBD calls. Do not pull the other eight satellite datasets on a cron.
    print(f"refreshing CFBD games/lines {season} (2 calls)")
    pull_dataset(root, "games", "/games", season, {"seasonType": "both"}, refresh=True)
    pull_dataset(root, "lines", "/lines", season, {"seasonType": "both"}, refresh=True)


def project_due(root: Path, season: int, now: datetime | None = None) -> bool:
    now = now or datetime.now(ET)
    local = now.astimezone(ET) if now.tzinfo else now.replace(tzinfo=ET)
    if local.weekday() not in {1, 2}:
        return False
    if local.weekday() == 1 and local.hour < PROJECT_HOUR:
        return False
    tuesday = (local.date() - timedelta(days=local.weekday() - 1)).isoformat()
    pred = root / "web" / "public" / "data" / f"predictions-{season}.json"
    if not pred.exists():
        return True
    try:
        generated = json.loads(pred.read_text()).get("generated_at")
        day = generated.replace("Z", "+00:00")[:10] if generated else ""
    except (json.JSONDecodeError, AttributeError, TypeError):
        return True
    return day < tuesday


def run_project(root: Path, season: int = LIVE_SEASON) -> dict:
    load_dotenv(root)
    refresh_live_slate(root, season)
    from footpalm.project import project_live

    payload = project_live(root, season=season, refresh=False)
    run_markets(root, season=season, sources=("kalshi",))
    from footpalm.espn import run as run_espn

    try:
        run_espn(root, season=season)
    except Exception as exc:
        print(f"espn harvest: {exc}", flush=True)
    return payload


def run_score(root: Path, season: int = LIVE_SEASON) -> dict:
    load_dotenv(root)
    now = datetime.now(ET)
    usage = load_usage(root, now)
    age = cache_age_s(root, "games", season)
    refresh = should_refresh_cfbd(now, age, int(usage.get("calls") or 0))
    if refresh:
        print(f"cfbd refresh age={None if age is None else round(age / 3600, 1)}h calls={usage.get('calls')}/{usage.get('cap')}")
    else:
        print(f"cfbd skip (age={None if age is None else round(age / 3600, 1)}h calls={usage.get('calls')}/{usage.get('cap')})")
    live = refresh_scores(root, season=season, refresh=refresh)
    if now.weekday() == 2 and project_due(root, season, now):
        print("weekly project missed Tuesday — refitting TabPFN")
        run_project(root, season)
    return live


def main() -> None:
    parser = argparse.ArgumentParser(description="Install or run FootPalm live jobs")
    parser.add_argument("command", nargs="?", choices=("lines", "score", "project"))
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--season", type=int, default=LIVE_SEASON)
    args = parser.parse_args()
    root = repo_root()
    if args.install:
        install(root)
        return
    if args.uninstall:
        uninstall()
        return
    if args.command == "lines":
        run_lines(root, args.season)
        return
    if args.command == "score":
        run_score(root, args.season)
        return
    if args.command == "project":
        run_project(root, args.season)
        return
    raise SystemExit("usage: python -m footpalm.cron --install | lines | score | project")


if __name__ == "__main__":
    main()
