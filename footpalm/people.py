"""Player satellite from CFBD. Roster, portal, recruiting, usage, season stats.

One call per (dataset, year). Do not train TabPFN on this.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from footpalm.cfbd import cfbd_dir, dataset_path, load_dotenv, pull_dataset
from footpalm.draft import colleges_match, find_pick, person_key, processed_path as draft_path
from footpalm.names import key as name_key

YEARS = list(range(2014, 2027))
CLASS = {1: "FR", 2: "SO", 3: "JR", 4: "SR", 5: "GR"}

# (folder, route, extra). Draft is pulled by footpalm.draft.
DATASETS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("roster", "/roster", {"classification": "fbs"}),
    ("portal", "/player/portal", {}),
    ("recruits", "/recruiting/players", {}),
    ("usage", "/player/usage", {}),
    ("stats_passing", "/stats/player/season", {"category": "passing"}),
    ("stats_rushing", "/stats/player/season", {"category": "rushing"}),
    ("stats_receiving", "/stats/player/season", {"category": "receiving"}),
)

STAT_MAP = {
    ("passing", "ATT"): ("passing", "att"),
    ("passing", "COMPLETIONS"): ("passing", "cmp"),
    ("passing", "YDS"): ("passing", "yds"),
    ("passing", "TD"): ("passing", "td"),
    ("passing", "INT"): ("passing", "int"),
    ("rushing", "CAR"): ("rushing", "att"),
    ("rushing", "YDS"): ("rushing", "yds"),
    ("rushing", "TD"): ("rushing", "td"),
    ("receiving", "REC"): ("receiving", "rec"),
    ("receiving", "YDS"): ("receiving", "yds"),
    ("receiving", "TD"): ("receiving", "td"),
}


def _read_list(root: Path, name: str, year: int) -> list:
    path = dataset_path(root, name, year)
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, list) else []


def _full_name(row: dict) -> str:
    if row.get("name"):
        return str(row["name"]).strip()
    first = (row.get("firstName") or "").strip()
    last = (row.get("lastName") or "").strip()
    return f"{first} {last}".strip()


def _pair(name: str, team: str) -> tuple[str, str]:
    return person_key(name), name_key(team or "")


def _usage_map(rows: list[dict]) -> dict[tuple[str, str], dict]:
    out = {}
    for row in rows:
        usage = row.get("usage") or {}
        if not isinstance(usage, dict):
            continue
        slim = {k: usage.get(k) for k in ("overall", "pass", "rush") if usage.get(k) is not None}
        if slim:
            out[_pair(row.get("name") or "", row.get("team") or "")] = slim
    return out


def _stats_map(root: Path, year: int) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = defaultdict(dict)
    for name, category in (
        ("stats_passing", "passing"),
        ("stats_rushing", "rushing"),
        ("stats_receiving", "receiving"),
    ):
        for row in _read_list(root, name, year):
            key = (category, row.get("statType") or "")
            dest = STAT_MAP.get(key)
            if not dest:
                continue
            bucket, field = dest
            try:
                value = float(row.get("stat"))
                if value == int(value):
                    value = int(value)
            except (TypeError, ValueError):
                continue
            player = out[_pair(row.get("player") or "", row.get("team") or "")]
            player.setdefault(bucket, {})[field] = value
    return out


def _recruit_map(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        name = _full_name(row)
        pk = person_key(name)
        if not pk:
            continue
        out[pk].append(
            {
                "name": name,
                "team": row.get("committedTo") or "",
                "stars": row.get("stars"),
                "rank": row.get("ranking"),
                "year": row.get("year"),
                "pos": row.get("position") or "",
            }
        )
    return out


def _portal_map(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        name = _full_name(row)
        pk = person_key(name)
        if not pk:
            continue
        out[pk].append(
            {
                "season": row.get("season"),
                "origin": row.get("origin") or "",
                "destination": row.get("destination") or "",
                "pos": row.get("position") or "",
                "eligibility": row.get("eligibility") or "",
                "stars": row.get("stars"),
            }
        )
    return out


def _pick_recruit(name: str, team: str, recruits: dict[str, list[dict]]) -> dict | None:
    hits = list(recruits.get(person_key(name)) or [])
    if team:
        tight = [r for r in hits if colleges_match(team, r.get("team") or "")]
        if tight:
            hits = tight
    if not hits:
        return None
    hits.sort(key=lambda r: int(r.get("year") or 0), reverse=True)
    row = hits[0]
    return {k: row[k] for k in ("stars", "rank", "year", "pos") if row.get(k) not in (None, "")}


def _pick_portal(name: str, team: str, portals: dict[str, list[dict]]) -> list[dict]:
    hits = list(portals.get(person_key(name)) or [])
    if team:
        tight = [
            r
            for r in hits
            if colleges_match(team, r.get("origin") or "") or colleges_match(team, r.get("destination") or "")
        ]
        if tight:
            hits = tight
        elif len(hits) > 3:
            hits = []
    hits.sort(key=lambda r: int(r.get("season") or 0))
    return hits


def _class_label(value) -> str | None:
    try:
        return CLASS.get(int(value), str(int(value)))
    except (TypeError, ValueError):
        return None


def _home(row: dict) -> str:
    city = (row.get("homeCity") or "").strip()
    state = (row.get("homeState") or "").strip()
    if city and state:
        return f"{city}, {state}"
    return city or state


def build_season(root: Path, year: int, *, draft_picks: list[dict], recruits: dict, portals: dict) -> dict:
    roster = _read_list(root, "roster", year)
    usage = _usage_map(_read_list(root, "usage", year))
    stats = _stats_map(root, year)
    people = []
    for row in roster:
        name = _full_name(row)
        team = row.get("team") or ""
        if not name or not team:
            continue
        pair = _pair(name, team)
        draft = find_pick(name, team, draft_picks)
        recruit = _pick_recruit(name, team, recruits)
        portal = _pick_portal(name, team, portals)
        item = {
            "name": name,
            "team": team,
            "pos": row.get("position") or "",
            "class": _class_label(row.get("year")),
            "jersey": row.get("jersey"),
            "ht": row.get("height"),
            "wt": row.get("weight"),
            "home": _home(row),
        }
        if usage.get(pair):
            item["usage"] = usage[pair]
        if stats.get(pair):
            item["stats"] = stats[pair]
        if recruit:
            item["recruit"] = recruit
        if portal:
            item["portal"] = portal
        if draft:
            item["draft"] = {
                "year": draft.get("year"),
                "round": draft.get("round"),
                "overall": draft.get("overall"),
                "nfl": draft.get("nfl") or "",
                "position": draft.get("position") or "",
            }
        people.append(item)
    people.sort(key=lambda r: (r["team"], r["name"]))
    return {
        "season": year,
        "source": "CFBD roster, portal, recruiting, usage, season stats. Draft joined from CFBD picks.",
        "n": len(people),
        "note": "Satellite. Do not train TabPFN on this.",
        "people": people,
    }


def processed_path(root: Path, year: int) -> Path:
    return root / "data" / "processed" / f"people-{year}.json"


def write_season(root: Path, year: int, *, draft_picks: list[dict], recruits: dict, portals: dict) -> Path | None:
    if not _read_list(root, "roster", year):
        return None
    payload = build_season(root, year, draft_picks=draft_picks, recruits=recruits, portals=portals)
    dest = processed_path(root, year)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload) + "\n")
    print(f"people {year}: {payload['n']}", flush=True)
    return dest


def _available_years(root: Path, name: str) -> list[int]:
    folder = cfbd_dir(root) / name
    if not folder.exists():
        return []
    years = []
    for path in folder.glob("*.json"):
        try:
            years.append(int(path.stem))
        except ValueError:
            continue
    return sorted(years)


def _load_joins(root: Path, years: list[int]) -> tuple[list[dict], dict, dict]:
    draft = {"picks": []}
    path = draft_path(root)
    if path.exists():
        draft = json.loads(path.read_text())
    recruits: dict[str, list[dict]] = defaultdict(list)
    portals: dict[str, list[dict]] = defaultdict(list)
    for year in _available_years(root, "recruits") or years:
        for pk, rows in _recruit_map(_read_list(root, "recruits", year)).items():
            recruits[pk].extend(rows)
    for year in _available_years(root, "portal") or years:
        for pk, rows in _portal_map(_read_list(root, "portal", year)).items():
            portals[pk].extend(rows)
    return list(draft.get("picks") or []), recruits, portals


def write_all(root: Path, years: list[int] | None = None) -> list[Path]:
    years = years or YEARS
    draft_picks, recruits, portals = _load_joins(root, years)
    written = []
    for year in years:
        dest = write_season(root, year, draft_picks=draft_picks, recruits=recruits, portals=portals)
        if dest:
            written.append(dest)
    return written


def pull_and_write(root: Path | None = None, years: list[int] | None = None, *, refresh: bool = False) -> list[Path]:
    root = root or _repo_root()
    years = years or YEARS
    load_dotenv(root)
    for year in years:
        for name, route, extra in DATASETS:
            try:
                pull_dataset(root, name, route, year, extra, refresh=refresh)
            except SystemExit as exc:
                dest = dataset_path(root, name, year)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps({"empty": True, "dataset": name, "season": year, "error": str(exc)[:160]}) + "\n")
                print(f"  skip {name} {year}: {exc}", flush=True)
    return write_all(root, years)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise SystemExit("could not find repo root")


def find_people(payload: dict, name: str, team: str | None = None) -> list[dict]:
    pk = person_key(name)
    if not pk or not payload:
        return []
    rows = [r for r in payload.get("people") or [] if person_key(r.get("name") or "") == pk]
    if team:
        tight = [r for r in rows if colleges_match(team, r.get("team") or "")]
        if tight:
            return tight
    return rows


def main() -> None:
    pull_and_write()


if __name__ == "__main__":
    main()
