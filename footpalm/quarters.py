from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from footpalm.fetch import DEFAULT_SEASONS, pbp_path

# Locked. C(4,1)×4 and C(4,2)×2. Not a live feature.
SINGLES: tuple[tuple[int, ...], ...] = tuple((q,) for q in range(1, 5))
PAIRS: tuple[tuple[int, ...], ...] = tuple(combinations(range(1, 5), 2))
ALL_SLICES: tuple[tuple[int, ...], ...] = SINGLES + PAIRS
# QScale2. Consecutive clock only.
HALVES: tuple[tuple[int, ...], ...] = ((1, 2), (3, 4))
THREE_CONSEC: tuple[tuple[int, ...], ...] = ((1, 2, 3), (2, 3, 4))
HALVES_THREE: tuple[tuple[int, ...], ...] = HALVES + THREE_CONSEC
CACHE_NAME = "quarter-scores.parquet"
COLS = [
    "game_id",
    "home",
    "away",
    "period",
    "pos_team",
    "pos_team_score",
    "def_pos_team_score",
    "game_play_number",
]


def cache_path(root: Path) -> Path:
    return root / "data" / "processed" / CACHE_NAME


def scale_quarters(
    q_home: np.ndarray, q_away: np.ndarray, combo: tuple[int, ...]
) -> tuple[float, float] | None:
    """Scale selected quarter points to a 4-quarter total. None if a quarter is missing."""
    idx = [q - 1 for q in combo]
    home = np.asarray(q_home, dtype=float)[idx]
    away = np.asarray(q_away, dtype=float)[idx]
    if not np.all(np.isfinite(home)) or not np.all(np.isfinite(away)):
        return None
    factor = 4 / len(combo)
    return float(home.sum() * factor), float(away.sum() * factor)


def synth_outcomes(
    q_home: np.ndarray,
    q_away: np.ndarray,
    slices: tuple[tuple[int, ...], ...] = ALL_SLICES,
) -> list[tuple[tuple[int, ...], float, float]]:
    """(combo, scaled_home, scaled_away) for finite, non-tied slices."""
    out = []
    for combo in slices:
        scaled = scale_quarters(q_home, q_away, combo)
        if scaled is None:
            continue
        home, away = scaled
        if home == away:
            continue
        out.append((combo, home, away))
    return out


def quarter_points_from_pbp(df: pd.DataFrame) -> pd.DataFrame:
    live = df.copy()
    live["game_id"] = pd.to_numeric(live["game_id"], errors="coerce")
    live["period"] = pd.to_numeric(live["period"], errors="coerce")
    live = live.dropna(subset=["game_id", "period", "home", "away", "pos_team"])
    live["game_id"] = live["game_id"].astype(int)
    live["period"] = live["period"].astype(int)
    live = live[live["period"].between(1, 4)]
    if live.empty:
        return _empty_quarters()

    pos = live["pos_team"].eq(live["home"])
    pos_pts = pd.to_numeric(live["pos_team_score"], errors="coerce")
    def_pts = pd.to_numeric(live["def_pos_team_score"], errors="coerce")
    live["home_score"] = pos_pts.where(pos, def_pts)
    live["away_score"] = pos_pts.where(~pos, def_pts)
    live = live.sort_values(["game_id", "game_play_number"])
    end = live.groupby(["game_id", "period"], as_index=False).agg(
        home_end=("home_score", "last"),
        away_end=("away_score", "last"),
    )
    wide = end.pivot(index="game_id", columns="period", values=["home_end", "away_end"])
    wide.columns = [f"{side}_{int(period)}" for side, period in wide.columns]
    wide = wide.reset_index()
    for period in range(1, 5):
        if f"home_end_{period}" not in wide.columns:
            wide[f"home_end_{period}"] = np.nan
            wide[f"away_end_{period}"] = np.nan

    prev_h = 0.0
    prev_a = 0.0
    for period in range(1, 5):
        h = pd.to_numeric(wide[f"home_end_{period}"], errors="coerce")
        a = pd.to_numeric(wide[f"away_end_{period}"], errors="coerce")
        wide[f"q{period}_home"] = h - prev_h
        wide[f"q{period}_away"] = a - prev_a
        prev_h, prev_a = h, a

    cols = ["game_id"] + [f"q{p}_{side}" for p in range(1, 5) for side in ("home", "away")]
    out = wide[cols]
    qcols = [c for c in cols if c != "game_id"]
    ok = out[qcols].notna().all(axis=1) & (out[qcols] >= 0).all(axis=1)
    return out.loc[ok].reset_index(drop=True)


def _empty_quarters() -> pd.DataFrame:
    cols = ["game_id"] + [f"q{p}_{side}" for p in range(1, 5) for side in ("home", "away")]
    return pd.DataFrame(columns=cols)


def _season_quarters(root: Path, season: int) -> pd.DataFrame:
    path = pbp_path(root, season)
    if not path.exists():
        return _empty_quarters()
    have = {f.name for f in pq.ParquetFile(path).schema_arrow}
    cols = [c for c in COLS if c in have]
    if "pos_team_score" not in cols or "home" not in cols:
        return _empty_quarters()
    return quarter_points_from_pbp(pd.read_parquet(path, columns=cols))


def build_quarters(root: Path) -> pd.DataFrame:
    parts = []
    for season in DEFAULT_SEASONS:
        print(f"  quarter scores {season}", flush=True)
        part = _season_quarters(root, season)
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else _empty_quarters()


def load_quarters(root: Path, *, rebuild: bool = False) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    dest = cache_path(root)
    if rebuild or not dest.exists():
        table = build_quarters(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(dest, index=False)
        print(f"wrote {dest} n={len(table)}", flush=True)
    else:
        table = pd.read_parquet(dest)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row in table.itertuples(index=False):
        home = np.array([getattr(row, f"q{p}_home") for p in range(1, 5)], dtype=float)
        away = np.array([getattr(row, f"q{p}_away") for p in range(1, 5)], dtype=float)
        out[int(row.game_id)] = (home, away)
    return out


def expand_rows(
    X: np.ndarray,
    y: np.ndarray,
    m: np.ndarray,
    game_ids: np.ndarray,
    quarters: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    slices: tuple[tuple[int, ...], ...] = ALL_SLICES,
    include_real: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interleave each real row with its scaled-quarter copies. real_mask marks originals."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    ms: list[float] = []
    real: list[bool] = []
    for i in range(len(X)):
        if include_real:
            xs.append(X[i])
            ys.append(float(y[i]))
            ms.append(float(m[i]))
            real.append(True)
        gid = int(game_ids[i]) if np.isfinite(game_ids[i]) else None
        qs = quarters.get(gid) if gid is not None else None
        if qs is None:
            continue
        for _combo, home, away in synth_outcomes(qs[0], qs[1], slices):
            xs.append(X[i])
            ys.append(1.0 if home > away else 0.0)
            ms.append(home - away)
            real.append(False)
    if not xs:
        empty = np.zeros((0, X.shape[1]), dtype=float)
        return empty, np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)
    return (
        np.vstack(xs),
        np.asarray(ys, dtype=float),
        np.asarray(ms, dtype=float),
        np.asarray(real, dtype=bool),
    )


def take_block(
    X: np.ndarray,
    y: np.ndarray,
    m: np.ndarray,
    real: np.ndarray,
    *,
    cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep trailing whole real+synth bundles without exceeding cap."""
    if len(X) <= cap:
        return X, y, m, real
    starts = np.flatnonzero(real)
    chosen: list[int] = []
    total = 0
    for start in reversed(starts):
        end = starts[starts > start]
        stop = int(end[0]) if len(end) else len(X)
        n = stop - start
        if total + n > cap and chosen:
            break
        if total + n > cap:
            break
        chosen.append(start)
        total += n
    if not chosen:
        return X[-cap:], y[-cap:], m[-cap:], real[-cap:]
    keep = []
    for start in reversed(chosen):
        end = starts[starts > start]
        stop = int(end[0]) if len(end) else len(X)
        keep.extend(range(start, stop))
    idx = np.asarray(keep, dtype=int)
    return X[idx], y[idx], m[idx], real[idx]
