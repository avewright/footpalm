from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from footpalm.fetch import DEFAULT_SEASONS, pbp_path
from footpalm.form import ALL_NAMES
from footpalm.plays import DEAD_PLAYS, _garbage

# Locked before this score. Walk-forward PBP only. No spread / NIL / week dummy.
PACE_NAMES = [
    "to_margin_diff",
    "ypc_diff",
    "play_speed_diff",
    "sec_per_play_diff",
]
PACE_ALL = ALL_NAMES + PACE_NAMES
WALL_LO, WALL_HI = 3.0, 55.0
CLOCK_LO, CLOCK_HI = 1.0, 40.0
CACHE_NAME = "pace-games.parquet"

COLS = [
    "game_id",
    "pos_team",
    "def_pos_team",
    "play_type",
    "pos_unit",
    "rush",
    "sack",
    "yds_rushed",
    "turnover",
    "TimeSecsRem",
    "wallclock",
    "game_play_number",
    "period",
    "kickoff_play",
    "punt",
    "fg_inds",
    "penalty_no_play",
    "pos_team_score",
    "def_pos_team_score",
]


@dataclass
class PaceSnap:
    giveaways: float = 0.0
    takeaways: float = 0.0
    rush_yds: float = 0.0
    rush_n: float = 0.0
    wall_sec: float = 0.0
    wall_n: float = 0.0
    clock_sec: float = 0.0
    clock_n: float = 0.0


class TeamPace:
    def __init__(self) -> None:
        self.games: list[PaceSnap] = []

    def to_margin(self) -> float:
        if not self.games:
            return 0.0
        return float(np.mean([g.takeaways - g.giveaways for g in self.games]))

    def ypc(self) -> float:
        n = sum(g.rush_n for g in self.games)
        return float(sum(g.rush_yds for g in self.games) / n) if n else 0.0

    def play_speed(self) -> float:
        n = sum(g.wall_n for g in self.games)
        return float(sum(g.wall_sec for g in self.games) / n) if n else 0.0

    def sec_per_play(self) -> float:
        n = sum(g.clock_n for g in self.games)
        return float(sum(g.clock_sec for g in self.games) / n) if n else 0.0


class PaceBook:
    def __init__(self) -> None:
        self.teams: dict[str, TeamPace] = {}

    def get(self, team: str) -> TeamPace:
        if team not in self.teams:
            self.teams[team] = TeamPace()
        return self.teams[team]

    def new_season(self) -> None:
        for team in self.teams.values():
            team.games = []

    def vector(self, home: str, away: str) -> np.ndarray:
        h, a = self.get(home), self.get(away)
        return np.array(
            [
                h.to_margin() - a.to_margin(),
                h.ypc() - a.ypc(),
                h.play_speed() - a.play_speed(),
                h.sec_per_play() - a.sec_per_play(),
            ],
            dtype=float,
        )

    def apply(self, snaps: dict[str, PaceSnap] | None, home: str, away: str) -> None:
        if not snaps:
            return
        if home in snaps:
            self.get(home).games.append(snaps[home])
        if away in snaps:
            self.get(away).games.append(snaps[away])


def cache_path(root: Path) -> Path:
    return root / "data" / "processed" / CACHE_NAME


def _scrimmage(df: pd.DataFrame) -> pd.DataFrame:
    live = df[~df["play_type"].isin(DEAD_PLAYS)].copy()
    if "penalty_no_play" in live.columns:
        live = live[live["penalty_no_play"].fillna(0).astype(float).eq(0)]
    margin = pd.to_numeric(live["pos_team_score"], errors="coerce") - pd.to_numeric(
        live["def_pos_team_score"], errors="coerce"
    )
    live = live.loc[~_garbage(live["period"], margin)]
    return live[
        live["pos_unit"].eq("Offense")
        & live["kickoff_play"].fillna(0).astype(float).eq(0)
        & live["punt"].fillna(0).astype(float).eq(0)
        & live["fg_inds"].fillna(0).astype(float).eq(0)
    ]


def _empty_snaps() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "game_id",
            "team",
            "giveaways",
            "takeaways",
            "rush_yds",
            "rush_n",
            "wall_sec",
            "wall_n",
            "clock_sec",
            "clock_n",
        ]
    )


def snaps_from_pbp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_snaps()
    live = _scrimmage(df)
    if live.empty:
        return _empty_snaps()
    live = live.sort_values(["game_id", "game_play_number"])
    gid = pd.to_numeric(live["game_id"], errors="coerce")
    live = live.assign(game_id=gid).dropna(subset=["game_id"])
    live["game_id"] = live["game_id"].astype(int)

    give = (
        live.loc[live["turnover"].fillna(0).astype(float).eq(1)]
        .groupby(["game_id", "pos_team"], as_index=False)
        .size()
        .rename(columns={"pos_team": "team", "size": "giveaways"})
    )
    take = (
        live.loc[live["turnover"].fillna(0).astype(float).eq(1)]
        .groupby(["game_id", "def_pos_team"], as_index=False)
        .size()
        .rename(columns={"def_pos_team": "team", "size": "takeaways"})
    )
    rush = live.loc[live["rush"].fillna(0).astype(float).eq(1) & live["sack"].fillna(0).astype(float).eq(0)]
    ypc = (
        rush.groupby(["game_id", "pos_team"], as_index=False)
        .agg(rush_yds=("yds_rushed", "sum"), rush_n=("yds_rushed", "size"))
        .rename(columns={"pos_team": "team"})
    )

    live["wc"] = pd.to_datetime(live["wallclock"], errors="coerce")
    live["dw"] = live.groupby("game_id")["wc"].diff().dt.total_seconds()
    live["dt"] = -live.groupby("game_id")["TimeSecsRem"].diff()
    same_pos = live["pos_team"].eq(live.groupby("game_id")["pos_team"].shift())
    same_per = live["period"].eq(live.groupby("game_id")["period"].shift())
    wall = live.loc[same_pos & same_per & live["dw"].between(WALL_LO, WALL_HI)]
    clock = live.loc[same_per & live["dt"].between(CLOCK_LO, CLOCK_HI)]
    speed = (
        wall.groupby(["game_id", "pos_team"], as_index=False)
        .agg(wall_sec=("dw", "sum"), wall_n=("dw", "size"))
        .rename(columns={"pos_team": "team"})
    )
    tempo = (
        clock.groupby(["game_id", "pos_team"], as_index=False)
        .agg(clock_sec=("dt", "sum"), clock_n=("dt", "size"))
        .rename(columns={"pos_team": "team"})
    )

    teams = pd.concat(
        [give[["game_id", "team"]], take[["game_id", "team"]], ypc[["game_id", "team"]], speed[["game_id", "team"]], tempo[["game_id", "team"]]],
        ignore_index=True,
    ).drop_duplicates()
    out = teams.merge(give, on=["game_id", "team"], how="left")
    out = out.merge(take, on=["game_id", "team"], how="left")
    out = out.merge(ypc, on=["game_id", "team"], how="left")
    out = out.merge(speed, on=["game_id", "team"], how="left")
    out = out.merge(tempo, on=["game_id", "team"], how="left")
    for col in ("giveaways", "takeaways", "rush_yds", "rush_n", "wall_sec", "wall_n", "clock_sec", "clock_n"):
        out[col] = out[col].fillna(0.0)
    return out


def _season_snaps(root: Path, season: int) -> pd.DataFrame:
    path = pbp_path(root, season)
    if not path.exists():
        return _empty_snaps()
    have = {f.name for f in pq.ParquetFile(path).schema_arrow}
    cols = [c for c in COLS if c in have]
    df = pd.read_parquet(path, columns=cols)
    return snaps_from_pbp(df)


def build_snaps(root: Path) -> pd.DataFrame:
    parts = []
    for season in DEFAULT_SEASONS:
        print(f"  pace snaps {season}", flush=True)
        part = _season_snaps(root, season)
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else _empty_snaps()


def load_snaps(root: Path, *, rebuild: bool = False) -> dict[int, dict[str, PaceSnap]]:
    dest = cache_path(root)
    if rebuild or not dest.exists():
        table = build_snaps(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(dest, index=False)
        print(f"wrote {dest} n={len(table)}", flush=True)
    else:
        table = pd.read_parquet(dest)
    out: dict[int, dict[str, PaceSnap]] = {}
    for row in table.itertuples(index=False):
        snap = PaceSnap(
            giveaways=float(row.giveaways),
            takeaways=float(row.takeaways),
            rush_yds=float(row.rush_yds),
            rush_n=float(row.rush_n),
            wall_sec=float(row.wall_sec),
            wall_n=float(row.wall_n),
            clock_sec=float(row.clock_sec),
            clock_n=float(row.clock_n),
        )
        out.setdefault(int(row.game_id), {})[str(row.team)] = snap
    return out
