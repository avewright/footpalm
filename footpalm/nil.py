from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from urllib.request import Request, urlopen

from footpalm.conferences import FBS_CONFERENCES
from footpalm.names import canon

NIL_URL = "https://nil-ncaa.com/"
# Typical 2026 football roster (rev share + third-party), not all-sports.
# G6 typical is ~22% of ~$13M operating revenue, most of it to football, plus a little collective.
# Top G6 coaches told USA Today the ceiling is about $10-11M.
G6_ROSTER = {
    "AAC": 4_500_000,
    "MWC": 3_500_000,
    "SBC": 3_500_000,
    "MAC": 2_500_000,
    "CUSA": 2_500_000,
    "P12": 7_000_000,
    "Ind": 4_000_000,
}
G6_HIGH = {
    "Memphis": 10_500_000,
    "Boise State": 10_000_000,
    "Tulane": 9_500_000,
    "UNLV": 9_000_000,
    "James Madison": 8_500_000,
    "South Florida": 8_500_000,
    "UTSA": 8_000_000,
    "Liberty": 8_000_000,
}
# Other football-only point estimates besides nil-ncaa.com.
# NIL Standard (Jul 2026 player-sum), College Front Office (player-sum), Sideline football split.
FOOTBALL_POINTS = {
    "Texas": [48_670_000, 47_900_000, 49_300_000],
    "Oregon": [41_390_000, 42_800_000],
    "LSU": [39_820_000, 42_800_000],
    "Ohio State": [37_550_000, 43_500_000],
    "Miami": [37_890_000, 44_000_000, 37_900_000],
    "Notre Dame": [40_400_000],
    "Texas A&M": [38_900_000, 37_600_000],
    "Alabama": [37_200_000],
    "Georgia": [34_200_000],
    "USC": [34_200_000],
    "Tennessee": [35_700_000],
    "Texas Tech": [36_300_000],
    "Michigan": [32_400_000],
    "Ole Miss": [35_200_000],
    "Oklahoma": [33_000_000],
}
# Sideline all-sports department market. Do not use as the football number.
SIDELINE_ALL_SPORTS = {
    "Texas": 73_900_000,
    "Texas A&M": 62_200_000,
    "Miami": 56_400_000,
    "Oregon": 58_900_000,
    "LSU": 57_400_000,
    "USC": 55_900_000,
    "Ohio State": 54_900_000,
    "Indiana": 53_400_000,
    "Texas Tech": 52_800_000,
    "Tennessee": 51_000_000,
    "Florida": 51_000_000,
    "Alabama": 47_000_000,
    "Michigan": 48_000_000,
    "Georgia": 46_000_000,
    "Notre Dame": 46_000_000,
}
# CBS/247 Aug 2026 industry poll, football only. Tiers, not exact dollars.
CBS_40_PLUS = frozenset({"LSU", "Miami", "Notre Dame", "Ohio State", "Oregon", "Texas"})
CBS_NEAR_40 = frozenset({"Georgia", "Michigan", "Ole Miss", "Tennessee", "USC", "Texas A&M"})
STAFF_PAYROLL = {
    "SEC": 26_931_774,
    "B1G": 24_205_387,
    "ACC": 22_082_473,
    "B12": 16_541_521,
    "P12": 8_259_208,
    "AAC": 7_392_211,
    "MWC": 5_086_308,
    "SBC": 4_613_206,
    "MAC": 3_817_671,
    "CUSA": 3_475_314,
    "Ind": 10_000_000,
}


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if any(self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _money(text: str) -> int | None:
    cleaned = text.replace(",", "").replace("$", "").replace(" ", "")
    cleaned = cleaned.replace("−", "-")
    if not re.fullmatch(r"-?\d+", cleaned):
        return None
    return int(cleaned)


def harvest_nil() -> dict:
    req = Request(NIL_URL, headers={"User-Agent": "footpalm/0.1 (research)"})
    try:
        html = urlopen(req, timeout=20).read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"NIL harvest failed ({exc}); continuing without published roster costs")
        return {"source": NIL_URL, "roster": {}, "spend": {}, "g6_roster_default": G6_ROSTER, "staff_payroll_conf": STAFF_PAYROLL}
    parser = _TableParser()
    parser.feed(html)

    roster: dict[str, dict] = {}
    spend: dict[str, dict] = {}
    for table in parser.tables:
        if not table:
            continue
        header = " ".join(table[0]).lower()
        if "roster cost" in header and "conference" in header:
            for row in table[1:]:
                if len(row) < 3:
                    continue
                value = _money(row[-1])
                if value is None:
                    continue
                team = canon(row[0], row[1])
                roster[team] = {
                    "team": team,
                    "nil_roster": value,
                    "nil_quality": "published",
                    "nil_source": "nil-ncaa.com 2026 roster cost estimate",
                }
        if "athletic department annual expenses" in header or (
            "fy 2025" in header and "spending rank" in header
        ):
            for row in table[1:]:
                if len(row) < 6:
                    continue
                subdiv = row[2] if len(row) > 2 else ""
                if subdiv.strip().upper() != "FBS":
                    continue
                value = _money(row[5])
                if value is None:
                    continue
                team = canon(row[0], row[3] if len(row) > 3 else None)
                spend[team] = {
                    "team": team,
                    "athletic_spend": value,
                    "spend_year": 2025,
                    "spend_source": "EADA via nil-ncaa.com",
                }

    return {
        "source": NIL_URL,
        "roster": roster,
        "spend": spend,
        "g6_roster_default": G6_ROSTER,
        "staff_payroll_conf": STAFF_PAYROLL,
    }


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of empty list")
    if n % 2:
        return float(ordered[n // 2])
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def estimate_football(team: str, conf: str, ncaa: int | None, athletic_spend: int | None = None) -> tuple[int | None, str]:
    """Best 2026 football roster estimate: rev share to football + third-party NIL.

    Sideline's $70M school totals are all-sports. Those stay off this number.
    """
    points = [ncaa] if ncaa else []
    points.extend(FOOTBALL_POINTS.get(team, []))
    if points:
        est = _median(points)
        quality = "blended" if len(points) > 1 else "published"
        raw = est
        if team == "LSU":
            est = max(est, 44_000_000)
        elif team in CBS_40_PLUS:
            est = max(est, 40_000_000)
        elif team in CBS_NEAR_40:
            est = max(est, 36_000_000)
        elif team == "Indiana":
            est = max(est, 29_500_000)
        elif team == "Wisconsin":
            est = max(est, 30_000_000)
        if est != raw:
            quality = "blended"
        return int(round(est, -4)), quality

    if team in G6_HIGH:
        return G6_HIGH[team], "reported"
    typical = G6_ROSTER.get(conf)
    if typical is None:
        return None, "missing"
    if athletic_spend and athletic_spend >= 70_000_000 and conf in {"AAC", "MWC", "SBC"}:
        typical = min(typical + 2_000_000, 11_000_000)
    return typical, "modeled"


def harvest_from_rows(teams: list[dict], source: str = NIL_URL) -> dict:
    roster: dict[str, dict] = {}
    spend: dict[str, dict] = {}
    for team in teams:
        name = team.get("team")
        if not name:
            continue
        if team.get("nil_roster") is not None:
            roster[name] = {
                "team": name,
                "nil_roster": team["nil_roster"],
                "nil_quality": team.get("nil_quality") or "published",
                "nil_source": source,
            }
        if team.get("athletic_spend") is not None:
            spend[name] = {
                "team": name,
                "athletic_spend": team["athletic_spend"],
                "spend_year": team.get("spend_year"),
                "spend_source": team.get("spend_source"),
            }
    return {
        "source": source,
        "final": True,
        "roster": roster,
        "spend": spend,
        "g6_roster_default": G6_ROSTER,
        "staff_payroll_conf": STAFF_PAYROLL,
    }


def attach_money(teams: list[dict], harvested: dict) -> list[dict]:
    roster = harvested["roster"]
    spend = harvested["spend"]
    frozen = bool(harvested.get("final"))
    out = []
    for team in teams:
        name = team["team"]
        row = dict(team)
        published = roster.get(name)
        spent = spend.get(name)
        athletic = spent["athletic_spend"] if spent else None
        conf = team.get("conf") or FBS_CONFERENCES.get(name, "")
        if frozen and published:
            row["nil_roster"] = published["nil_roster"]
            row["nil_quality"] = published.get("nil_quality") or "published"
        else:
            ncaa = published["nil_roster"] if published else None
            est, quality = estimate_football(name, conf, ncaa, athletic)
            row["nil_roster"] = est
            row["nil_quality"] = quality
        row["nil_all_sports"] = SIDELINE_ALL_SPORTS.get(name)
        row["athletic_spend"] = athletic
        row["staff_payroll"] = STAFF_PAYROLL.get(team.get("conf", ""), None)
        out.append(row)
    return out


def money_map(teams: list[dict]) -> dict[str, float]:
    return {t["team"]: float(t["nil_roster"]) for t in teams if t.get("nil_roster")}


NOTE = (
    "Football roster estimate: revenue share to football plus third-party NIL. "
    "P4 is a median of nil-ncaa.com, College Front Office, NIL Standard, and Sideline "
    "football splits, then floored by the Aug 2026 CBS/247 industry tiers. "
    "Sideline school totals (Texas $73.9M) are all-sports and live in that column. "
    "G6 typical is modeled; named G6 spenders use the USA Today ~$10M ceiling. "
    "None of this is a TabPFN feature."
)


def dump_money(path, harvested: dict, teams: list[dict]) -> None:
    payload = {
        "source": harvested["source"],
        "note": NOTE,
        "teams": [
            {
                "team": t["team"],
                "conf": t.get("conf"),
                "nil_roster": t.get("nil_roster"),
                "nil_quality": t.get("nil_quality"),
                "nil_all_sports": t.get("nil_all_sports"),
                "athletic_spend": t.get("athletic_spend"),
                "staff_payroll": t.get("staff_payroll"),
                "pom": t.get("pom", t.get("palm")),
            }
            for t in teams
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
