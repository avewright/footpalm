"""NFL draft picks from CFBD. Satellite only — do not train TabPFN on this."""

from __future__ import annotations

import json
import re
from pathlib import Path

from footpalm.cfbd import dataset_path, load_dotenv, pull_dataset
from footpalm.names import canon, key as name_key

# NFL draft year. 2025 CFB class → 2026 draft.
DRAFT_YEARS = list(range(2014, 2027))
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv)\b\.?", re.I)
SCHOOL_NOISE = re.compile(r"\b(state|st|university|univ)\b", re.I)
POS_ALIAS = {
    "qb": "quarterback",
    "rb": "running back",
    "wr": "wide receiver",
    "te": "tight end",
    "cb": "cornerback",
    "s": "safety",
    "lb": "linebacker",
    "edge": "defensive edge",
    "de": "defensive edge",
    "dt": "defensive tackle",
    "ot": "offensive tackle",
    "og": "offensive guard",
    "c": "center",
    "k": "place kicker",
    "p": "punter",
}


def person_key(name: str) -> str:
    return name_key(SUFFIX.sub("", name or ""))


def college_key(name: str) -> str:
    return name_key(canon(name or ""))


def colleges_match(left: str, right: str) -> bool:
    a, b = college_key(left), college_key(right)
    if not a or not b:
        return False
    if a == b:
        return True
    return SCHOOL_NOISE.sub("", a).strip() == SCHOOL_NOISE.sub("", b).strip()


def slim_pick(row: dict) -> dict:
    return {
        "name": row.get("name") or "",
        "college": row.get("collegeTeam") or row.get("college_team") or "",
        "year": row.get("year"),
        "round": row.get("round"),
        "pick": row.get("pick"),
        "overall": row.get("overall"),
        "nfl": row.get("nflTeam") or row.get("nfl_team") or "",
        "position": row.get("position") or "",
    }


def find_pick(name: str, college: str | None, picks: list[dict]) -> dict | None:
    pk = person_key(name)
    if not pk:
        return None
    hits = [p for p in picks if person_key(p.get("name") or "") == pk]
    if college:
        tight = [p for p in hits if colleges_match(college, p.get("college") or "")]
        if tight:
            hits = tight
        elif not hits:
            last = pk.split()[-1]
            hits = [
                p
                for p in picks
                if (person_key(p.get("name") or "").split() or [""])[-1] == last
                and colleges_match(college, p.get("college") or "")
            ]
    if not hits:
        return None
    hits.sort(key=lambda p: int(p.get("year") or 0), reverse=True)
    return hits[0]


def lookup(players: list, picks: list[dict], colleges: list[str] | None = None) -> list[dict]:
    colleges = colleges or []
    rows = []
    for i, item in enumerate(players):
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("player") or "")
            college = item.get("college") or item.get("team") or (colleges[i] if i < len(colleges) else None)
        else:
            name = str(item)
            college = colleges[i] if i < len(colleges) else None
        hit = find_pick(name, college, picks)
        rows.append(
            {
                "player": name,
                "college": college or (hit or {}).get("college") or "",
                "drafted": bool(hit),
                "year": (hit or {}).get("year"),
                "round": (hit or {}).get("round"),
                "pick": (hit or {}).get("pick"),
                "overall": (hit or {}).get("overall"),
                "nfl": (hit or {}).get("nfl") or "",
                "position": (hit or {}).get("position") or "",
            }
        )
    return rows


def list_picks(
    picks: list[dict],
    year: int,
    rnd: int | None = None,
    position: str | None = None,
    college: str | None = None,
    nfl: str | None = None,
    n: int | None = None,
) -> list[dict]:
    rows = [p for p in picks if p.get("year") == int(year)]
    if rnd is not None:
        rows = [p for p in rows if p.get("round") == int(rnd)]
    if position:
        want = POS_ALIAS.get(position.lower(), position.lower())
        rows = [p for p in rows if want in (p.get("position") or "").lower()]
    if college:
        rows = [p for p in rows if colleges_match(college, p.get("college") or "")]
    if nfl:
        want = college_key(nfl)
        rows = [p for p in rows if want and want in college_key(p.get("nfl") or "")]
    rows.sort(key=lambda p: int(p.get("overall") or 9999))
    cap = 40 if rnd is not None else 32
    take = max(1, min(int(n if n is not None else cap), 64))
    return rows[:take]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise SystemExit("could not find repo root")


def processed_path(root: Path) -> Path:
    return root / "data" / "processed" / "draft.json"


def write_index(root: Path, years: list[int] | None = None) -> Path:
    years = years or DRAFT_YEARS
    picks = []
    have = []
    for year in years:
        path = dataset_path(root, "draft", year)
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        if not isinstance(raw, list):
            continue
        have.append(year)
        picks.extend(slim_pick(row) for row in raw if row.get("name"))
    dest = processed_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "source": "CFBD /draft/picks",
                "years": have,
                "n": len(picks),
                "note": "Satellite. Do not train TabPFN on draft.",
                "picks": picks,
            }
        )
        + "\n"
    )
    print(f"draft index: {len(picks)} picks, years {have}", flush=True)
    return dest


def pull_and_write(root: Path | None = None, years: list[int] | None = None, *, refresh: bool = False) -> Path:
    root = root or _repo_root()
    years = years or DRAFT_YEARS
    load_dotenv(root)
    for year in years:
        pull_dataset(root, "draft", "/draft/picks", year, {}, refresh=refresh)
    return write_index(root, years)


def main() -> None:
    pull_and_write()


if __name__ == "__main__":
    main()
