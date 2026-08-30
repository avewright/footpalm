from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from footpalm.conferences import D1_DIVISIONS, FBS_CONFERENCES

KEEP_COLS = [
    "season",
    "week",
    "game_id",
    "game_play_number",
    "pos_team",
    "def_pos_team",
    "home",
    "away",
    "home_team_division",
    "away_team_division",
    "home_team_conference",
    "away_team_conference",
    "neutral_site",
    "EPA",
    "play_type",
    "pos_unit",
    "rush",
    "pass",
    "sack",
    "kickoff_play",
    "punt",
    "fg_inds",
    "pos_team_score",
    "def_pos_team_score",
    "period",
    "penalty_no_play",
    "season_type",
    "spread",
]

DEAD_PLAYS = {
    "Timeout",
    "End Period",
    "End of Game",
    "End of Half",
    "End of Regulation",
}


def load_pbp(path: Path) -> pd.DataFrame:
    available = {field.name for field in pq.ParquetFile(path).schema_arrow}
    cols = [c for c in KEEP_COLS if c in available]
    if "home_team_division" not in cols:
        raise SystemExit(f"{path.name} is missing home_team_division")
    df = pd.read_parquet(path, columns=cols)
    df = df[df["home_team_division"].isin(D1_DIVISIONS) | df["away_team_division"].isin(D1_DIVISIONS)]
    df = df[df["pos_team"].notna() & df["def_pos_team"].notna()]
    df["EPA"] = pd.to_numeric(df["EPA"], errors="coerce")
    df["neutral_site"] = df["neutral_site"].fillna(False).astype(bool)
    return df


def _garbage(period: pd.Series, margin: pd.Series) -> pd.Series:
    p = pd.to_numeric(period, errors="coerce").fillna(0)
    m = margin.abs()
    return (
        ((p <= 1) & (m >= 42))
        | ((p == 2) & (m >= 38))
        | ((p == 3) & (m >= 28))
        | ((p >= 4) & (m >= 22))
    )


def _fbs_teams(df: pd.DataFrame) -> set[str]:
    home = set(df.loc[df["home_team_division"].eq("fbs"), "home"].dropna())
    away = set(df.loc[df["away_team_division"].eq("fbs"), "away"].dropna())
    return home | away


def _conference_map(df: pd.DataFrame) -> dict[str, str]:
    rows = pd.concat(
        [
            df[["home", "home_team_conference"]].rename(
                columns={"home": "team", "home_team_conference": "conference"}
            ),
            df[["away", "away_team_conference"]].rename(
                columns={"away": "team", "away_team_conference": "conference"}
            ),
        ],
        ignore_index=True,
    ).dropna()
    mode = (
        rows.groupby("team")["conference"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    return {team: FBS_CONFERENCES.get(conf, conf) for team, conf in mode.items()}


def _final_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["home_score"] = scored["pos_team_score"].where(
        scored["pos_team"].eq(scored["home"]), scored["def_pos_team_score"]
    )
    scored["away_score"] = scored["pos_team_score"].where(
        scored["pos_team"].eq(scored["away"]), scored["def_pos_team_score"]
    )
    agg = {
        "season": ("season", "first"),
        "week": ("week", "max"),
        "home": ("home", "first"),
        "away": ("away", "first"),
        "home_score": ("home_score", "max"),
        "away_score": ("away_score", "max"),
        "neutral_site": ("neutral_site", "first"),
        "home_team_division": ("home_team_division", "first"),
        "away_team_division": ("away_team_division", "first"),
        "season_type": ("season_type", "first"),
        "max_period": ("period", "max"),
    }
    if "spread" in scored.columns:
        agg["spread"] = ("spread", "first")
    meta = scored.groupby("game_id", as_index=False).agg(**agg)
    if "spread" not in meta.columns:
        meta["spread"] = pd.NA
    return meta.loc[meta["max_period"].fillna(0).ge(4)].drop(columns="max_period")


def game_observations(df: pd.DataFrame) -> tuple[pd.DataFrame, set[str], dict[str, str]]:
    """One row per team-game with raw EPA rates, ST EPA, and the final score."""
    fbs = _fbs_teams(df)
    conferences = _conference_map(df)
    scores = _final_scores(df)

    live = df[~df["play_type"].isin(DEAD_PLAYS)].copy()
    live["margin"] = pd.to_numeric(live["pos_team_score"], errors="coerce") - pd.to_numeric(
        live["def_pos_team_score"], errors="coerce"
    )
    live["garbage"] = _garbage(live["period"], live["margin"])
    if "penalty_no_play" in live.columns:
        live = live[live["penalty_no_play"].fillna(0).astype(float).eq(0)]

    scrimmage = live[
        live["pos_unit"].eq("Offense")
        & live["EPA"].notna()
        & live["kickoff_play"].fillna(0).astype(float).eq(0)
        & live["punt"].fillna(0).astype(float).eq(0)
        & live["fg_inds"].fillna(0).astype(float).eq(0)
    ]
    st = live[live["pos_unit"].ne("Offense") & live["EPA"].notna()]

    off = (
        scrimmage.loc[~scrimmage["garbage"]]
        .groupby(["game_id", "pos_team"], as_index=False)
        .agg(off_epa=("EPA", "mean"), off_plays=("EPA", "size"))
        .rename(columns={"pos_team": "team"})
    )
    tempo = (
        scrimmage.groupby("game_id", as_index=False)
        .agg(game_plays=("EPA", "size"))
    )

    st_pos = st.groupby(["game_id", "pos_team"], as_index=False).agg(st_for=("EPA", "sum"))
    st_def = st.groupby(["game_id", "def_pos_team"], as_index=False).agg(st_against=("EPA", "sum"))

    home_side = scores.rename(
        columns={
            "home": "team",
            "away": "opponent",
            "home_score": "points",
            "away_score": "opp_points",
        }
    )
    home_side["is_home"] = ~home_side["neutral_site"]
    home_side["home_team"] = home_side["team"]
    home_side["away_team"] = home_side["opponent"]
    away_side = scores.rename(
        columns={
            "away": "team",
            "home": "opponent",
            "away_score": "points",
            "home_score": "opp_points",
        }
    )
    away_side["is_home"] = False
    away_side["home_team"] = away_side["opponent"]
    away_side["away_team"] = away_side["team"]
    sides = pd.concat([home_side, away_side], ignore_index=True)
    sides["won"] = sides["points"] > sides["opp_points"]
    sides["lost"] = sides["points"] < sides["opp_points"]

    sides = sides.merge(off, on=["game_id", "team"], how="left")
    opp_off = off.rename(columns={"team": "opponent", "off_epa": "def_epa", "off_plays": "def_plays"})
    sides = sides.merge(opp_off, on=["game_id", "opponent"], how="left")
    sides = sides.merge(tempo, on="game_id", how="left")
    sides = sides.merge(
        st_pos.rename(columns={"pos_team": "team"}),
        on=["game_id", "team"],
        how="left",
    )
    sides = sides.merge(
        st_def.rename(columns={"def_pos_team": "team"}),
        on=["game_id", "team"],
        how="left",
    )
    sides["st_epa"] = sides["st_for"].fillna(0) - sides["st_against"].fillna(0)
    sides["off_epa"] = sides["off_epa"].astype(float)
    sides["def_epa"] = sides["def_epa"].astype(float)
    sides["game_plays"] = sides["game_plays"].fillna(0)
    sides["fbs"] = sides["team"].isin(fbs)
    week = pd.to_numeric(sides["week"], errors="coerce").fillna(0)
    post = sides["season_type"].eq("postseason")
    sides["slate"] = week.where(~post, week + 20).astype(int)
    return sides, fbs, conferences


def listed_games(sides: pd.DataFrame) -> pd.DataFrame:
    """One row per game from the listed-home team's perspective."""
    return sides.loc[sides["team"].eq(sides["home_team"])].copy()
