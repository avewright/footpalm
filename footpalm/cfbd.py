"""College Football Data satellite warehouse.

cfbfastR play-by-play stays the ratings engine. This module only pulls
complementary CFBD facts (schedule, books, SP+, talent) into data/raw/cfbd
and never writes over the parquet files.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

BASE = "https://api.collegefootballdata.com"
PAUSE_S = 0.4
USER_AGENT = "footpalm/0.1 (local research; cache-first)"

# One call per (dataset, season). Free tier is ~1000/month. Live jobs
# should only refresh games+lines. Do not cron the full satellite pull.
USAGE_CAP = 1000
DATASETS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("teams", "/teams/fbs", {}),
    ("calendar", "/calendar", {}),
    ("games", "/games", {"seasonType": "both"}),
    ("lines", "/lines", {"seasonType": "both"}),
    ("records", "/records", {}),
    ("sp", "/ratings/sp", {}),
    ("talent", "/talent", {}),
    ("recruiting", "/recruiting/teams", {}),
    ("rankings", "/rankings", {"seasonType": "both"}),
    ("ppa_teams", "/ppa/teams", {"excludeGarbageTime": "true"}),
)


def load_dotenv(root: Path) -> None:
    path = root / ".env"
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def api_key() -> str:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        raise SystemExit("CFBD_API_KEY is missing. Put it in .env or the environment.")
    return key


def cfbd_dir(root: Path) -> Path:
    path = root / "data" / "raw" / "cfbd"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_path(root: Path, name: str, season: int) -> Path:
    return cfbd_dir(root) / name / f"{season}.json"


def usage_path(root: Path) -> Path:
    return root / "data" / "processed" / "cfbd-usage.json"


def month_key(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


def load_usage(root: Path, now: datetime | None = None) -> dict:
    month = month_key(now)
    path = usage_path(root)
    empty = {
        "month": month,
        "calls": 0,
        "cap": USAGE_CAP,
        "note": "Free tier ~1000/month. Score/project should stay under ~350. Kalshi is free.",
    }
    if not path.exists():
        return empty
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return empty
    if not isinstance(raw, dict) or raw.get("month") != month:
        empty["prior"] = {"month": raw.get("month"), "calls": raw.get("calls")} if isinstance(raw, dict) else None
        return empty
    raw.setdefault("cap", USAGE_CAP)
    return raw


def note_calls(root: Path, n: int = 1, now: datetime | None = None) -> dict:
    row = load_usage(root, now)
    row["calls"] = int(row.get("calls") or 0) + n
    row["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dest = usage_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(row, indent=2) + "\n")
    if row["calls"] >= 700:
        print(f"cfbd usage {row['calls']}/{USAGE_CAP} this month — slow down", flush=True)
    return row


def cache_age_s(root: Path, name: str, season: int) -> float | None:
    dest = dataset_path(root, name, season)
    if not cached(dest):
        return None
    return time.time() - dest.stat().st_mtime


def should_refresh_cfbd(now: datetime, age_s: float | None, calls: int = 0) -> bool:
    """Throttle auto CFBD pulls. Manual `footpalm.score` still refreshes."""
    if age_s is None:
        return True
    if calls >= 900:
        return False
    local = now.astimezone(ET) if now.tzinfo else now.replace(tzinfo=ET)
    weekday = local.weekday()
    # Fri–Sun: scores land often. Mon–Thu: 6h is enough.
    min_s = 3 * 3600 if weekday >= 4 else 6 * 3600
    if calls >= 700:
        min_s = max(min_s, 12 * 3600)
    return age_s >= min_s


def cached(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 20


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(payload) + "\n")
    tmp.replace(path)


def get(path: str, params: dict[str, str], *, timeout: int = 90, attempts: int = 4) -> Any:
    query = urlencode({k: v for k, v in params.items() if v is not None and v != ""})
    url = f"{BASE}{path}" + (f"?{query}" if query else "")
    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    last: Exception | None = None
    for i in range(attempts):
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body) if body else []
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code in {429, 502, 503, 504} and i + 1 < attempts:
                time.sleep(PAUSE_S * (i + 2))
                last = exc
                continue
            raise SystemExit(f"CFBD {exc.code} {path}: {detail}") from exc
        except (URLError, RemoteDisconnected, TimeoutError) as exc:
            last = exc
            if i + 1 < attempts:
                time.sleep(PAUSE_S * (i + 2))
                continue
            raise SystemExit(f"CFBD network error {path}: {exc}") from exc
    raise SystemExit(f"CFBD failed {path}: {last}")


def pull_dataset(root: Path, name: str, route: str, season: int, extra: dict[str, str], *, refresh: bool) -> dict:
    dest = dataset_path(root, name, season)
    if cached(dest) and not refresh:
        rows = json.loads(dest.read_text())
        n = len(rows) if isinstance(rows, list) else 1
        return {"dataset": name, "season": season, "rows": n, "cached": True, "path": str(dest)}

    payload = get(route, {"year": str(season), **extra})
    time.sleep(PAUSE_S)
    _write(dest, payload)
    note_calls(root, 1)
    n = len(payload) if isinstance(payload, list) else 1
    print(f"  cfbd {name} {season}: {n} rows")
    return {"dataset": name, "season": season, "rows": n, "cached": False, "path": str(dest)}


def pull(root: Path, seasons: list[int], *, refresh: bool = False) -> dict:
    load_dotenv(root)
    api_key()
    pulled: list[dict] = []
    calls = 0
    for season in seasons:
        print(f"cfbd {season}")
        for name, route, extra in DATASETS:
            row = pull_dataset(root, name, route, season, extra, refresh=refresh)
            pulled.append(row)
            if not row["cached"]:
                calls += 1

    manifest = {
        "source": BASE,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "calls": calls,
        "note": (
            "Satellite only. Ratings and TabPFN still read cfbfastR parquet. "
            "Join later with footpalm.names.canon. Do not train on lines, SP+, or talent."
        ),
        "datasets": pulled,
    }
    _write(cfbd_dir(root) / "manifest.json", manifest)
    print(f"cfbd done: {calls} live calls, {len(pulled) - calls} cache hits")
    return manifest
