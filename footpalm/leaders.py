"""Player rushing / receiving / YAC boards from cfbfastR play-by-play."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from footpalm.fetch import DEFAULT_SEASONS, pbp_path, repo_root

CAUGHT = re.compile(r"caught at (?:the )?([A-Za-z]{2,4})(\d{1,2})", re.I)
COLS = [
    "rush_player",
    "rusher_player_name",
    "yds_rushed",
    "rush",
    "rush_td",
    "reception_player",
    "receiver_player_name",
    "yds_receiving",
    "completion",
    "pass_td",
    "passer_player_name",
    "pass",
    "play_text",
    "yards_to_goal",
    "pos_team",
    "def_pos_team",
    "home_team_division",
    "away_team_division",
    "home",
    "away",
]


def _abbrs(df: pd.DataFrame) -> dict[str, str]:
    hits: dict[str, Counter] = defaultdict(Counter)
    texts = df["play_text"].fillna("")
    teams = df["pos_team"]
    for text, team in zip(texts, teams):
        match = CAUGHT.search(text)
        if match and pd.notna(team):
            hits[str(team)][match.group(1).upper()] += 1
    return {team: counts.most_common(1)[0][0] for team, counts in hits.items() if counts}


def _yac_series(df: pd.DataFrame, abbrs: dict[str, str]) -> pd.Series:
    out = []
    for text, ytg, gain, pos, defense in zip(
        df["play_text"],
        df["yards_to_goal"],
        df["yds_receiving"],
        df["pos_team"],
        df["def_pos_team"],
    ):
        if not isinstance(text, str) or pd.isna(ytg) or pd.isna(gain):
            out.append(None)
            continue
        match = CAUGHT.search(text)
        if not match:
            out.append(None)
            continue
        token, line = match.group(1).upper(), int(match.group(2))
        if token == abbrs.get(pos):
            air_to_ez = 100 - line
        elif token == abbrs.get(defense):
            air_to_ez = line
        else:
            out.append(None)
            continue
        yac = float(gain) - (float(ytg) - air_to_ez)
        out.append(None if yac < -4 or yac > 80 else round(yac, 1))
    return pd.Series(out, index=df.index)


def _fbs_offense(df: pd.DataFrame) -> pd.Series:
    home = df["pos_team"].eq(df["home"]) & df["home_team_division"].eq("fbs")
    away = df["pos_team"].eq(df["away"]) & df["away_team_division"].eq("fbs")
    return home | away


def _rank(frame: pd.DataFrame, value: str, count: str, minimum: int, limit: int = 60) -> list[dict]:
    keep = frame.loc[frame[count] >= minimum].sort_values(value, ascending=False).head(limit)
    rows = []
    for i, row in enumerate(keep.to_dict(orient="records"), start=1):
        clean = {}
        for k, v in row.items():
            if k in {"att", "rec", "td", "yac_plays", "cmp", "yds"} and v is not None:
                clean[k] = int(round(float(v)))
            elif isinstance(v, float):
                clean[k] = round(v, 1)
            else:
                clean[k] = v
        clean["rank"] = i
        rows.append(clean)
    return rows


def build_season(root: Path, season: int) -> dict:
    path = pbp_path(root, season)
    if not path.exists():
        raise FileNotFoundError(f"no play-by-play for {season}")
    have = {f.name for f in pq.ParquetFile(path).schema_arrow}
    cols = [c for c in COLS if c in have]
    df = pd.read_parquet(path, columns=cols)
    df = df.loc[_fbs_offense(df)].copy()
    rusher = df["rush_player"] if "rush_player" in df else df["rusher_player_name"]
    if "rusher_player_name" in df:
        rusher = rusher.fillna(df["rusher_player_name"])
    receiver = df["reception_player"] if "reception_player" in df else df["receiver_player_name"]
    if "receiver_player_name" in df:
        receiver = receiver.fillna(df["receiver_player_name"])
    df["rusher"] = rusher
    df["receiver"] = receiver.where(~receiver.astype(str).str.startswith("#"))

    rushes = df.loc[df["rush"].fillna(0).eq(1) & df["rusher"].notna()]
    rush = (
        rushes.groupby(["rusher", "pos_team"], as_index=False)
        .agg(att=("rusher", "size"), yds=("yds_rushed", "sum"), td=("rush_td", "sum"))
        .rename(columns={"rusher": "player", "pos_team": "team"})
    )
    rush["ypc"] = (rush["yds"] / rush["att"]).round(2)

    recs = df.loc[df["completion"].fillna(0).eq(1) & df["receiver"].notna()].copy()
    abbrs = _abbrs(recs) if "play_text" in recs.columns else {}
    recs["yac"] = _yac_series(recs, abbrs) if abbrs else pd.NA
    rec = recs.groupby(["receiver", "pos_team"], as_index=False).agg(
        rec=("receiver", "size"),
        yds=("yds_receiving", "sum"),
        td=("pass_td", "sum"),
        yac=("yac", "sum"),
        yac_plays=("yac", "count"),
    ).rename(columns={"receiver": "player", "pos_team": "team"})
    rec["ypr"] = (rec["yds"] / rec["rec"]).round(2)

    passes = df.loc[df["pass"].fillna(0).eq(1) & df["passer_player_name"].notna()]
    passing = (
        passes.groupby(["passer_player_name", "pos_team"], as_index=False)
        .agg(
            att=("passer_player_name", "size"),
            cmp=("completion", "sum"),
            yds=("yds_receiving", "sum"),
            td=("pass_td", "sum"),
        )
        .rename(columns={"passer_player_name": "player", "pos_team": "team"})
    )

    return {
        "season": season,
        "source": "cfbfastR play-by-play. YAC from ESPN catch-point text when present.",
        "rushing": _rank(rush, "yds", "att", 40),
        "receiving": _rank(rec, "yds", "rec", 20),
        "yac": _rank(rec, "yac", "yac_plays", 10),
        "passing": _rank(passing, "yds", "att", 80),
    }


def write_season(root: Path, season: int) -> Path:
    payload = build_season(root, season)
    dest = root / "data" / "processed" / f"leaders-{season}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload) + "\n")
    print(
        f"leaders {season}: rush {len(payload['rushing'])} rec {len(payload['receiving'])} "
        f"yac {len(payload['yac'])}",
        flush=True,
    )
    return dest


def write_all(root: Path | None = None, seasons: list[int] | None = None) -> list[Path]:
    root = root or repo_root()
    years = seasons or [s for s in DEFAULT_SEASONS if pbp_path(root, s).exists()]
    return [write_season(root, year) for year in years]


def main() -> None:
    write_all()


if __name__ == "__main__":
    main()
