from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from footpalm.form import (
    ALL_NAMES,
    CONF_ALL,
    CONF_NAMES,
    CRAFT_ALL,
    CRAFT_NAMES,
    EXTRA_NAMES,
    FULL_CRAFT_ALL,
    LOSO_ALL,
    LOSO_NAMES,
    SIGNAL_ALL,
    SIGNAL_NAMES,
    TIME_ALL,
    TIME_NAMES,
    FormBook,
    time_features,
)
from footpalm.predict import FEATURE_NAMES
from footpalm.project import history_path

FEATURES_NAME = "features-history.npz"
HOLDOUT_SEASON = 2025
LIVE_SEASON = 2026
HOME_POM = FEATURE_NAMES.index("home_pom")
AWAY_POM = FEATURE_NAMES.index("away_pom")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise SystemExit("could not find repo root")


def features_path(root: Path) -> Path:
    return root / "data" / "processed" / FEATURES_NAME


def season_game_counts(root: Path) -> dict[int, int]:
    counts = {}
    for path in sorted((root / "web" / "public" / "data").glob("predictions-*.json")):
        try:
            season = int(path.stem.split("-")[1])
        except (IndexError, ValueError):
            continue
        if season >= LIVE_SEASON:
            continue
        counts[season] = len(json.loads(path.read_text())["games"])
    return counts


def load_prediction_games(root: Path) -> list[dict]:
    games = []
    for season in sorted(season_game_counts(root)):
        payload = json.loads((root / "web" / "public" / "data" / f"predictions-{season}.json").read_text())
        games.extend(payload["games"])
    return games


def walk_form(
    games: list[dict], X_locked: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(games) != len(X_locked):
        raise SystemExit(f"games n={len(games)} != locked features n={len(X_locked)}")
    form = FormBook()
    extras = np.zeros((len(games), len(EXTRA_NAMES)), dtype=float)
    signal = np.zeros((len(games), len(SIGNAL_NAMES)), dtype=float)
    craft = np.zeros((len(games), len(CRAFT_NAMES)), dtype=float)
    loso = np.zeros((len(games), len(LOSO_NAMES)), dtype=float)
    conf = np.zeros((len(games), len(CONF_NAMES)), dtype=float)
    i = 0
    while i < len(games):
        season = int(games[i]["season"])
        if i == 0 or int(games[i - 1]["season"]) != season:
            form.new_season()
        slate = int(games[i]["slate"])
        start = i
        while i < len(games) and int(games[i]["season"]) == season and int(games[i]["slate"]) == slate:
            i += 1
        batch = range(start, i)
        for j in batch:
            game = games[j]
            form.touch(game["home"], float(X_locked[j, HOME_POM]), str(game.get("home_conf") or ""))
            form.touch(game["away"], float(X_locked[j, AWAY_POM]), str(game.get("away_conf") or ""))
        for j in batch:
            extras[j] = form.extras(games[j]["home"], games[j]["away"], slate)
            signal[j] = form.signal(games[j]["home"], games[j]["away"], slate)
            craft[j] = form.craft(games[j]["home"], games[j]["away"], slate, X_locked[j])
            loso[j] = form.loso(games[j]["home"], games[j]["away"], slate)
            conf[j] = form.conference(
                games[j]["home"],
                games[j]["away"],
                str(games[j].get("home_conf") or ""),
                str(games[j].get("away_conf") or ""),
            )
        for j in batch:
            game = games[j]
            if game.get("actual_margin") is None:
                continue
            form.apply_game(
                game["home"],
                game["away"],
                home_won=bool(game.get("home_won")),
                margin=float(game["actual_margin"]),
                home_pom=float(X_locked[j, HOME_POM]),
                away_pom=float(X_locked[j, AWAY_POM]),
                slate=slate,
                neutral=bool(game.get("neutral")),
                home_points=float(game.get("actual_home") or 0.0),
                away_points=float(game.get("actual_away") or 0.0),
                home_conf=str(game.get("home_conf") or ""),
                away_conf=str(game.get("away_conf") or ""),
            )
    return extras, signal, craft, loso, conf


def extras_from_games(games: list[dict], X_locked: np.ndarray) -> np.ndarray:
    extras, _signal, _craft, _loso, _conf = walk_form(games, X_locked)
    return extras


def build_matrix(root: Path | None = None) -> dict[str, np.ndarray]:
    root = root or _repo_root()
    history = np.load(history_path(root))
    X_hist = np.asarray(history["X"], dtype=float)
    X_locked = X_hist[:, : len(FEATURE_NAMES)]
    y_win = np.asarray(history["y_win"], dtype=float)
    y_margin = np.asarray(history["y_margin"], dtype=float)
    games = load_prediction_games(root)
    counts = season_game_counts(root)
    if sum(counts.values()) != len(X_locked):
        raise SystemExit(f"history n={len(X_locked)} != prediction rows {sum(counts.values())}")
    extras, signal, craft, loso, conf = walk_form(games, X_locked)
    clock = np.vstack([time_features(int(g["season"]), int(g.get("week") or 0)) for g in games])
    seasons = np.array([int(g["season"]) for g in games], dtype=int)
    fbs = np.array([bool(g.get("fbs_fbs")) for g in games])
    X_full = np.concatenate([X_locked, extras], axis=1)
    X_signal = np.concatenate([X_locked, signal], axis=1)
    X_time = np.concatenate([X_full, clock], axis=1)
    X_craft = np.concatenate([X_locked, craft], axis=1)
    X_full_craft = np.concatenate([X_full, craft], axis=1)
    X_loso = np.concatenate([X_full, loso], axis=1)
    X_conf = np.concatenate([X_full, conf], axis=1)
    return {
        "X_locked": X_locked,
        "X_full": X_full,
        "X_signal": X_signal,
        "X_time": X_time,
        "X_craft": X_craft,
        "X_full_craft": X_full_craft,
        "X_loso": X_loso,
        "X_conf": X_conf,
        "extras": extras,
        "signal": signal,
        "time": clock,
        "craft": craft,
        "loso": loso,
        "conf": conf,
        "y_win": y_win,
        "y_margin": y_margin,
        "season": seasons,
        "fbs": fbs,
    }


def save_matrix(root: Path, payload: dict[str, np.ndarray]) -> Path:
    dest = features_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        **payload,
        extra_names=np.array(EXTRA_NAMES),
        signal_names=np.array(SIGNAL_NAMES),
        time_names=np.array(TIME_NAMES),
        craft_names=np.array(CRAFT_NAMES),
        loso_names=np.array(LOSO_NAMES),
        locked_names=np.array(FEATURE_NAMES),
        full_names=np.array(ALL_NAMES),
        signal_full_names=np.array(SIGNAL_ALL),
        time_full_names=np.array(TIME_ALL),
        craft_full_names=np.array(CRAFT_ALL),
        extras_craft_names=np.array(FULL_CRAFT_ALL),
        loso_full_names=np.array(LOSO_ALL),
        conf_names=np.array(CONF_NAMES),
        conf_full_names=np.array(CONF_ALL),
    )
    print(
        f"wrote {dest} n={len(payload['y_win'])} extras={len(EXTRA_NAMES)} "
        f"signal={len(SIGNAL_NAMES)} time={len(TIME_NAMES)} craft={len(CRAFT_NAMES)} "
        f"loso={len(LOSO_NAMES)} conf={len(CONF_NAMES)}"
    )
    return dest


def run(root: Path | None = None) -> dict[str, np.ndarray]:
    root = root or _repo_root()
    payload = build_matrix(root)
    save_matrix(root, payload)
    hold = payload["season"] == HOLDOUT_SEASON
    fbs = payload["fbs"]
    print(
        f"  extras ready  train={int(((payload['season'] < HOLDOUT_SEASON)).sum())}  "
        f"2025 fbs={int((hold & fbs).sum())}"
    )
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()
