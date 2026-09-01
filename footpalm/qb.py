from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from footpalm.fetch import DEFAULT_SEASONS, pbp_path
from footpalm.form import ALL_NAMES
from footpalm.plays import DEAD_PLAYS, _garbage

# Locked before this score. Walk-forward PBP only. Prior slates only.
QB_NAMES = [
    "qb_epa_diff",
    "qb_change",
    "qb_starts",
    "qb_prior_epa",
]
QB_ALL = ALL_NAMES + QB_NAMES
CACHE_NAME = "qb-games.parquet"

COLS = [
    "game_id",
    "pos_team",
    "def_pos_team",
    "play_type",
    "pos_unit",
    "pass",
    "passer_player_name",
    "EPA",
    "period",
    "penalty_no_play",
    "pos_team_score",
    "def_pos_team_score",
    "kickoff_play",
    "punt",
    "fg_inds",
]


def _normalize(name: object) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().casefold()


@dataclass
class QBSnap:
    modal: str = ""
    epa_sum: float = 0.0
    epa_cnt: float = 0.0


class TeamQB:
    def __init__(self) -> None:
        self.last_modal: str | None = None
        self.recent_starter: str | None = None
        self.this_starts: Counter[str] = Counter()
        self.this_epa_sum: dict[str, float] = {}
        self.this_epa_cnt: dict[str, int] = {}


class QBBook:
    def __init__(self) -> None:
        self.teams: dict[str, TeamQB] = {}
        self.global_current_sum: dict[str, float] = {}
        self.global_current_cnt: dict[str, int] = {}
        self.global_last_epa: dict[str, float] = {}

    def get(self, team: str) -> TeamQB:
        if team not in self.teams:
            self.teams[team] = TeamQB()
        return self.teams[team]

    def new_season(self) -> None:
        # promote current season globals to last
        for qb, s in list(self.global_current_sum.items()):
            c = self.global_current_cnt.get(qb, 0)
            if c:
                self.global_last_epa[qb] = s / c
        self.global_current_sum.clear()
        self.global_current_cnt.clear()
        for team, state in self.teams.items():
            if state.this_starts:
                modal = max(state.this_starts, key=lambda k: state.this_starts[k])
                state.last_modal = modal
            # clear this season
            state.this_starts = Counter()
            state.this_epa_sum = {}
            state.this_epa_cnt = {}
            state.recent_starter = None

    def _team_features(self, team: str) -> tuple[float, float, float, float]:
        state = self.teams.get(team)
        if state is None:
            return (0.0, 0.0, 0.0, 0.0)
        expected = state.recent_starter if state.recent_starter is not None else state.last_modal
        if expected is None:
            return (0.0, 0.0, 0.0, 0.0)
        c = state.this_epa_cnt.get(expected, 0)
        s = state.this_epa_sum.get(expected, 0.0)
        epa = float(s / c) if c else 0.0
        if state.last_modal is None:
            change = 1.0
        else:
            change = 0.0 if expected == state.last_modal else 1.0
        cnt = state.this_starts.get(expected, 0)
        starts = float(math.log1p(cnt))
        prior = float(self.global_last_epa.get(expected, 0.0))
        return (epa, change, starts, prior)

    def vector(self, home: str, away: str) -> np.ndarray:
        h = self._team_features(home)
        a = self._team_features(away)
        return np.array([h[i] - a[i] for i in range(len(QB_NAMES))], dtype=float)

    def apply(self, snaps: dict[str, QBSnap] | None, home: str, away: str) -> None:
        if not snaps:
            return
        for team in (home, away):
            snap = snaps.get(team)
            if snap is None:
                continue
            modal = snap.modal
            if not modal:
                continue
            state = self.get(team)
            state.this_starts[modal] += 1
            state.this_epa_sum[modal] = state.this_epa_sum.get(modal, 0.0) + float(snap.epa_sum)
            state.this_epa_cnt[modal] = state.this_epa_cnt.get(modal, 0) + int(snap.epa_cnt)
            state.recent_starter = modal
            self.global_current_sum[modal] = self.global_current_sum.get(modal, 0.0) + float(snap.epa_sum)
            self.global_current_cnt[modal] = self.global_current_cnt.get(modal, 0) + int(snap.epa_cnt)


def cache_path(root: Path) -> Path:
    return root / "data" / "processed" / CACHE_NAME


def _empty_snaps() -> pd.DataFrame:
    return pd.DataFrame(columns=["game_id", "team", "modal", "epa_sum", "epa_cnt"])


def snaps_from_pbp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_snaps()
    # garbage + dead/penalty/non-offense drops, same as pace
    live = df[~df["play_type"].isin(DEAD_PLAYS)].copy()
    if "penalty_no_play" in live.columns:
        live = live[live["penalty_no_play"].fillna(0).astype(float).eq(0)]
    margin = pd.to_numeric(live["pos_team_score"], errors="coerce") - pd.to_numeric(
        live["def_pos_team_score"], errors="coerce"
    )
    live = live.loc[~_garbage(live["period"], margin)]
    live = live[live["pos_unit"].eq("Offense")]
    if live.empty:
        return _empty_snaps()
    # keep only pass attempts with passer and EPA
    mask = (
        live["pass"].fillna(0).astype(float).eq(1)
        & live["passer_player_name"].notna()
        & live["EPA"].notna()
    )
    pp = live.loc[mask].copy()
    if pp.empty:
        return _empty_snaps()
    pp["norm"] = pp["passer_player_name"].apply(_normalize)
    pp = pp[pp["norm"].ne("")]
    if pp.empty:
        return _empty_snaps()
    pp["EPA"] = pd.to_numeric(pp["EPA"], errors="coerce")
    pp = pp.dropna(subset=["EPA"])
    if pp.empty:
        return _empty_snaps()
    gid = pd.to_numeric(pp["game_id"], errors="coerce")
    pp = pp.assign(game_id=gid).dropna(subset=["game_id"])
    pp["game_id"] = pp["game_id"].astype(int)
    grp = pp.groupby(["game_id", "pos_team", "norm"], as_index=False).agg(
        cnt=("norm", "size"), epa_sum=("EPA", "sum")
    )
    # modal per game_id, pos_team = max cnt
    idx = grp.groupby(["game_id", "pos_team"])["cnt"].idxmax()
    modal_df = grp.loc[idx].copy()
    modal_df = modal_df.rename(columns={"pos_team": "team", "norm": "modal", "cnt": "epa_cnt"})
    modal_df = modal_df[["game_id", "team", "modal", "epa_sum", "epa_cnt"]]
    return modal_df


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
        print(f"  qb snaps {season}", flush=True)
        part = _season_snaps(root, season)
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else _empty_snaps()


def load_snaps(root: Path, *, rebuild: bool = False) -> dict[int, dict[str, QBSnap]]:
    dest = cache_path(root)
    if rebuild or not dest.exists():
        table = build_snaps(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(dest, index=False)
        print(f"wrote {dest} n={len(table)}", flush=True)
    else:
        table = pd.read_parquet(dest)
    out: dict[int, dict[str, QBSnap]] = {}
    for row in table.itertuples(index=False):
        snap = QBSnap(modal=str(row.modal), epa_sum=float(row.epa_sum), epa_cnt=float(row.epa_cnt))
        out.setdefault(int(row.game_id), {})[str(row.team)] = snap
    return out
