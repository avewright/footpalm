from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from footpalm.fetch import DEFAULT_SEASONS, pbp_path
from footpalm.form import ALL_NAMES, FORM_N
from footpalm.plays import DEAD_PLAYS, _garbage

# Locked before this score. New menu — not an add-on to pace after that peek.
SPECIAL_NAMES = [
    "margin_momentum_diff",
    "win_streak_diff",
    "fg_avg_make_diff",
    "fg_make_adj_diff",
    "punt_rate_diff",
    "punt_yds_diff",
    "plays_pg_diff",
]
SPECIAL_ALL = ALL_NAMES + SPECIAL_NAMES
CACHE_NAME = "specials-games.parquet"

COLS = [
    "game_id",
    "pos_team",
    "def_pos_team",
    "play_type",
    "pos_unit",
    "punt",
    "yds_punted",
    "fg_inds",
    "fg_made",
    "fg_make_prob",
    "yds_fg",
    "kickoff_play",
    "penalty_no_play",
    "period",
    "pos_team_score",
    "def_pos_team_score",
]


@dataclass
class SpecialSnap:
    fg_make_yds: float = 0.0
    fg_make_n: float = 0.0
    fg_made: float = 0.0
    fg_exp: float = 0.0
    fg_att: float = 0.0
    punt_n: float = 0.0
    punt_yds: float = 0.0
    plays: float = 0.0


class TeamSpecials:
    def __init__(self) -> None:
        self.snaps: list[SpecialSnap] = []
        self.margins: list[float] = []
        self.streak: int = 0

    def margin_momentum(self) -> float:
        if len(self.margins) < 2:
            return 0.0
        season = float(np.mean(self.margins))
        recent = float(np.mean(self.margins[-FORM_N:]))
        return recent - season

    def win_streak(self) -> float:
        return float(self.streak)

    def fg_avg_make(self) -> float:
        n = sum(s.fg_make_n for s in self.snaps)
        return float(sum(s.fg_make_yds for s in self.snaps) / n) if n else 0.0

    def fg_make_adj(self) -> float:
        n = sum(s.fg_att for s in self.snaps)
        if not n:
            return 0.0
        made = sum(s.fg_made for s in self.snaps)
        exp = sum(s.fg_exp for s in self.snaps)
        return float((made - exp) / n)

    def punt_rate(self) -> float:
        punts = sum(s.punt_n for s in self.snaps)
        plays = sum(s.plays for s in self.snaps)
        denom = punts + plays
        return float(punts / denom) if denom else 0.0

    def punt_yds(self) -> float:
        n = sum(s.punt_n for s in self.snaps)
        return float(sum(s.punt_yds for s in self.snaps) / n) if n else 0.0

    def plays_pg(self) -> float:
        if not self.snaps:
            return 0.0
        return float(np.mean([s.plays for s in self.snaps]))


class SpecialsBook:
    def __init__(self) -> None:
        self.teams: dict[str, TeamSpecials] = {}

    def get(self, team: str) -> TeamSpecials:
        if team not in self.teams:
            self.teams[team] = TeamSpecials()
        return self.teams[team]

    def new_season(self) -> None:
        for team in self.teams.values():
            team.snaps = []
            team.margins = []
            team.streak = 0

    def vector(self, home: str, away: str) -> np.ndarray:
        h, a = self.get(home), self.get(away)
        return np.array(
            [
                h.margin_momentum() - a.margin_momentum(),
                h.win_streak() - a.win_streak(),
                h.fg_avg_make() - a.fg_avg_make(),
                h.fg_make_adj() - a.fg_make_adj(),
                h.punt_rate() - a.punt_rate(),
                h.punt_yds() - a.punt_yds(),
                h.plays_pg() - a.plays_pg(),
            ],
            dtype=float,
        )

    def apply(
        self,
        snaps: dict[str, SpecialSnap] | None,
        home: str,
        away: str,
        *,
        home_won: bool,
        margin: float,
    ) -> None:
        if snaps:
            if home in snaps:
                self.get(home).snaps.append(snaps[home])
            if away in snaps:
                self.get(away).snaps.append(snaps[away])
        self._result(self.get(home), won=home_won, margin=margin)
        self._result(self.get(away), won=not home_won, margin=-margin)

    def _result(self, team: TeamSpecials, *, won: bool, margin: float) -> None:
        team.margins.append(float(margin))
        if won:
            team.streak = team.streak + 1 if team.streak >= 0 else 1
        else:
            team.streak = team.streak - 1 if team.streak <= 0 else -1


def cache_path(root: Path) -> Path:
    return root / "data" / "processed" / CACHE_NAME


def _live(df: pd.DataFrame) -> pd.DataFrame:
    live = df[~df["play_type"].isin(DEAD_PLAYS)].copy()
    if "penalty_no_play" in live.columns:
        live = live[live["penalty_no_play"].fillna(0).astype(float).eq(0)]
    margin = pd.to_numeric(live["pos_team_score"], errors="coerce") - pd.to_numeric(
        live["def_pos_team_score"], errors="coerce"
    )
    return live.loc[~_garbage(live["period"], margin)]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "game_id",
            "team",
            "fg_make_yds",
            "fg_make_n",
            "fg_made",
            "fg_exp",
            "fg_att",
            "punt_n",
            "punt_yds",
            "plays",
        ]
    )


def snaps_from_pbp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty()
    live = _live(df)
    if live.empty:
        return _empty()
    gid = pd.to_numeric(live["game_id"], errors="coerce")
    live = live.assign(game_id=gid).dropna(subset=["game_id"])
    live["game_id"] = live["game_id"].astype(int)

    scrim = live[
        live["pos_unit"].eq("Offense")
        & live["kickoff_play"].fillna(0).astype(float).eq(0)
        & live["punt"].fillna(0).astype(float).eq(0)
        & live["fg_inds"].fillna(0).astype(float).eq(0)
    ]
    plays = (
        scrim.groupby(["game_id", "pos_team"], as_index=False)
        .size()
        .rename(columns={"pos_team": "team", "size": "plays"})
    )
    fgs = live.loc[live["fg_inds"].fillna(0).astype(float).eq(1)].copy()
    fgs["made"] = fgs["fg_made"].fillna(0).astype(float)
    exp = pd.to_numeric(fgs["fg_make_prob"], errors="coerce")
    fgs["exp"] = np.where(exp > 1.0, exp / 100.0, exp)
    fg_att = (
        fgs.groupby(["game_id", "pos_team"], as_index=False)
        .agg(fg_made=("made", "sum"), fg_exp=("exp", "sum"), fg_att=("made", "size"))
        .rename(columns={"pos_team": "team"})
    )
    makes = fgs.loc[fgs["made"].eq(1)]
    fg_dist = (
        makes.groupby(["game_id", "pos_team"], as_index=False)
        .agg(fg_make_yds=("yds_fg", "sum"), fg_make_n=("yds_fg", "size"))
        .rename(columns={"pos_team": "team"})
    )
    punts = live.loc[live["punt"].fillna(0).astype(float).eq(1)]
    punt = (
        punts.groupby(["game_id", "pos_team"], as_index=False)
        .agg(punt_n=("punt", "size"), punt_yds=("yds_punted", "sum"))
        .rename(columns={"pos_team": "team"})
    )

    teams = pd.concat(
        [plays[["game_id", "team"]], fg_att[["game_id", "team"]], fg_dist[["game_id", "team"]], punt[["game_id", "team"]]],
        ignore_index=True,
    ).drop_duplicates()
    out = teams.merge(plays, on=["game_id", "team"], how="left")
    out = out.merge(fg_att, on=["game_id", "team"], how="left")
    out = out.merge(fg_dist, on=["game_id", "team"], how="left")
    out = out.merge(punt, on=["game_id", "team"], how="left")
    for col in ("fg_make_yds", "fg_make_n", "fg_made", "fg_exp", "fg_att", "punt_n", "punt_yds", "plays"):
        out[col] = out[col].fillna(0.0)
    return out


def _season_snaps(root: Path, season: int) -> pd.DataFrame:
    path = pbp_path(root, season)
    if not path.exists():
        return _empty()
    have = {f.name for f in pq.ParquetFile(path).schema_arrow}
    cols = [c for c in COLS if c in have]
    return snaps_from_pbp(pd.read_parquet(path, columns=cols))


def build_snaps(root: Path) -> pd.DataFrame:
    parts = []
    for season in DEFAULT_SEASONS:
        print(f"  specials snaps {season}", flush=True)
        part = _season_snaps(root, season)
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else _empty()


def load_specials(root: Path, *, rebuild: bool = False) -> dict[int, dict[str, SpecialSnap]]:
    dest = cache_path(root)
    if rebuild or not dest.exists():
        table = build_snaps(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(dest, index=False)
        print(f"wrote {dest} n={len(table)}", flush=True)
    else:
        table = pd.read_parquet(dest)
    out: dict[int, dict[str, SpecialSnap]] = {}
    for row in table.itertuples(index=False):
        snap = SpecialSnap(
            fg_make_yds=float(row.fg_make_yds),
            fg_make_n=float(row.fg_make_n),
            fg_made=float(row.fg_made),
            fg_exp=float(row.fg_exp),
            fg_att=float(row.fg_att),
            punt_n=float(row.punt_n),
            punt_yds=float(row.punt_yds),
            plays=float(row.plays),
        )
        out.setdefault(int(row.game_id), {})[str(row.team)] = snap
    return out
