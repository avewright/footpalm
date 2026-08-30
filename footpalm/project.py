from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from footpalm.cfbd import dataset_path, load_dotenv, pull
from footpalm.conferences import FBS_CONFERENCES
from footpalm.fetch import LIVE_SEASON, repo_root
from footpalm.names import canon, key
from footpalm.form import ALL_NAMES, extras_matrix, replay_form
from footpalm.predict import FEATURE_NAMES, fit_tabpfn, predict_slate
from footpalm.nil import attach_money, harvest_from_rows
from footpalm.rate import RatingBook, dump_book, fit_ratings, load_book, publish_preseason
from footpalm.research import apply_promoted

HISTORY_NAME = "tabpfn-history.npz"


def history_path(root: Path) -> Path:
    return root / "data" / "processed" / HISTORY_NAME


def book_path(root: Path, season: int) -> Path:
    return root / "data" / "processed" / f"book-{season}.json"


def save_history(root: Path, X: np.ndarray, y_win: np.ndarray, y_margin: np.ndarray) -> None:
    dest = history_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, X=X, y_win=y_win, y_margin=y_margin)
    print(f"wrote {dest} n={len(y_win)}")


def load_history(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dest = history_path(root)
    if not dest.exists():
        raise SystemExit(f"missing {dest} — run the historical build first")
    payload = np.load(dest)
    return payload["X"], payload["y_win"], payload["y_margin"]


def save_book(root: Path, book: RatingBook) -> None:
    dest = book_path(root, book.season)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(dump_book(book), indent=2) + "\n")
    print(f"wrote {dest}")


def load_prior_book(root: Path, season: int) -> RatingBook:
    dest = book_path(root, season)
    if not dest.exists():
        raise SystemExit(f"missing {dest} — run the historical build through {season}")
    return load_book(json.loads(dest.read_text()))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    n = len(payload.get("teams") or payload.get("games") or payload.get("nodes") or [])
    print(f"wrote {path} ({n})")


def _publish(root: Path, name: str, payload: dict) -> None:
    _write_json(root / "data" / "processed" / name, payload)
    _write_json(root / "web" / "public" / "data" / name, payload)


def _conf(name: str | None) -> str:
    if not name:
        return ""
    return FBS_CONFERENCES.get(name, name)


def resolve_name(name: str, known: set[str]) -> str:
    if name in known:
        return name
    mapped = canon(name)
    if mapped in known:
        return mapped
    by_key = {key(team): team for team in known}
    if key(name) in by_key:
        return by_key[key(name)]
    if key(mapped) in by_key:
        return by_key[key(mapped)]
    return mapped


def _spread(lines_by_id: dict[int, dict], game_id: int) -> float | None:
    row = lines_by_id.get(game_id)
    if not row:
        return None
    preferred = ("DraftKings", "ESPN Bet", "Bovada", "consensus")
    entries = row.get("lines") or []
    for provider in preferred:
        for line in entries:
            if line.get("provider") == provider and line.get("spread") is not None:
                return float(line["spread"])
    for line in entries:
        if line.get("spread") is not None:
            return float(line["spread"])
    return None


def load_cfbd_slate(root: Path, season: int) -> tuple[list[dict], set[str], dict[str, str]]:
    games = json.loads(dataset_path(root, "games", season).read_text())
    teams = json.loads(dataset_path(root, "teams", season).read_text())
    lines_raw = json.loads(dataset_path(root, "lines", season).read_text()) if dataset_path(root, "lines", season).exists() else []
    lines_by_id = {int(row["id"]): row for row in lines_raw if row.get("id") is not None}

    fbs = {t["school"] for t in teams if (t.get("classification") or "").lower() == "fbs"}
    conferences = {
        t["school"]: _conf(t.get("conference"))
        for t in teams
        if (t.get("classification") or "").lower() == "fbs"
    }

    slate = []
    for row in games:
        home_raw, away_raw = row.get("homeTeam"), row.get("awayTeam")
        if not home_raw or not away_raw:
            continue
        home_class = (row.get("homeClassification") or "").lower()
        away_class = (row.get("awayClassification") or "").lower()
        if home_class != "fbs" and away_class != "fbs":
            continue
        home = resolve_name(home_raw, fbs)
        away = resolve_name(away_raw, fbs)
        week = int(row.get("week") or 0)
        season_type = row.get("seasonType") or "regular"
        slate_id = week if season_type != "postseason" else week + 20
        home_pts = row.get("homePoints")
        away_pts = row.get("awayPoints")
        played = home_pts is not None and away_pts is not None
        slate.append(
            {
                "season": season,
                "slate": slate_id,
                "week": week,
                "season_type": season_type,
                "game_id": row.get("id"),
                "start": row.get("startDate"),
                "home": home,
                "away": away,
                "neutral": bool(row.get("neutralSite")),
                "home_conf": _conf(row.get("homeConference")),
                "away_conf": _conf(row.get("awayConference")),
                "fbs_fbs": home in fbs and away in fbs,
                "actual_home": float(home_pts) if played else None,
                "actual_away": float(away_pts) if played else None,
                "actual_margin": float(home_pts) - float(away_pts) if played else None,
                "home_won": int(float(home_pts) > float(away_pts)) if played else None,
                "spread": _spread(lines_by_id, int(row["id"])) if row.get("id") is not None else None,
                "completed": bool(row.get("completed")) or played,
            }
        )
    slate.sort(key=lambda g: (g["slate"], g.get("start") or "", g["home"], g["away"]))
    return slate, fbs, conferences


def _load_calibration(root: Path) -> dict | None:
    path = root / "web" / "public" / "data" / "research.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if report.get("promoted") in {None, "identity"}:
        return None
    return report


def project_live(root: Path, *, season: int = LIVE_SEASON, refresh: bool = False, use_tabpfn: bool = True) -> dict:
    load_dotenv(root)
    if refresh or not dataset_path(root, "games", season).exists():
        print(f"refreshing CFBD {season}")
        pull(root, [season], refresh=refresh)

    prior_season = season - 1
    prior = load_prior_book(root, prior_season)
    slate, fbs, conferences = load_cfbd_slate(root, season)
    known = set(prior.teams) | fbs
    fbs = {resolve_name(team, known) for team in fbs}
    conferences = {resolve_name(team, known): conf for team, conf in conferences.items()}
    for row in slate:
        row["home"] = resolve_name(row["home"], known)
        row["away"] = resolve_name(row["away"], known)
        row["fbs_fbs"] = row["home"] in fbs and row["away"] in fbs
    book = fit_ratings(
        pd.DataFrame(columns=["off_epa", "def_epa", "team", "opponent", "game_plays", "points", "is_home"]),
        fbs | prior.fbs,
        {**prior.conferences, **conferences},
        season,
        prior=prior,
    )
    prior_ratings = root / "web" / "public" / "data" / f"ratings-{prior_season}.json"
    prior_table = json.loads(prior_ratings.read_text()) if prior_ratings.exists() else {"teams": []}
    table = publish_preseason(book, prior_season=prior_season, prior_table=prior_table)
    money_path = root / "web" / "public" / "data" / "money.json"
    if money_path.exists():
        money_file = json.loads(money_path.read_text())
        harvested = harvest_from_rows(money_file.get("teams", []), money_file.get("source", ""))
    else:
        harvested = harvest_from_rows(prior_table.get("teams", []))
    table["teams"] = attach_money(table["teams"], harvested)
    table["season"] = season

    model = None
    X_hist, y_win, y_margin = load_history(root)
    if X_hist.shape[1] != len(ALL_NAMES):
        raise SystemExit(
            f"tabpfn-history is {X_hist.shape[1]} cols, extras live needs {len(ALL_NAMES)}. "
            "Rebuild: uv run python -m footpalm.build --seasons 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025"
        )
    from footpalm.featurize import load_prediction_games

    hist_games = load_prediction_games(root)
    if len(hist_games) != len(X_hist):
        raise SystemExit(f"history n={len(X_hist)} != prediction games {len(hist_games)}")
    form = replay_form(hist_games, X_hist[:, : len(FEATURE_NAMES)])
    form.new_season()
    X_live = extras_matrix(book, form, slate)
    if use_tabpfn:
        print(f"fitting tabpfn on {len(y_win)} extras rows", flush=True)
        model = fit_tabpfn(X_hist, y_win, y_margin)
        if not model.ready:
            print(f"tabpfn: {model.error}")
            model = None

    print(f"predicting {len(slate)} games", flush=True)
    preds = predict_slate(book, slate, model, X=X_live)
    games = [{**row, **pred} for row, pred in zip(slate, preds, strict=True)]

    report = _load_calibration(root)
    if report:
        games = apply_promoted(games, report)

    payload = {
        "season": season,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "preseason",
        "prior_season": prior_season,
        "train_rows": int(model.n_train) if model else 0,
        "engine": "tabpfn-3" if model else "logistic",
        "note": (
            f"{season} projections from {prior_season} Pom plus extras (Elo, form, SOS). "
            f"TabPFN trained on 2014–{prior_season}. Refresh weekly. Do not train on the market line."
        ),
        "calibration": (
            {"id": report["promoted"], "params": next(e["params"] for e in report["experiments"] if e["id"] == report["promoted"])}
            if report
            else None
        ),
        "games": games,
    }
    _publish(root, f"ratings-{season}.json", table)
    _publish(root, f"predictions-{season}.json", payload)

    index_path = root / "web" / "public" / "data" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"seasons": []}
    seasons = [s for s in index.get("seasons", []) if s.get("season") != season]
    seasons.append({"season": season, "teams": len(table["teams"])})
    seasons.sort(key=lambda s: s["season"])
    _publish(root, "index.json", {"seasons": seasons})

    upcoming = [g for g in games if not g.get("completed")]
    print(f"{season}: {len(games)} games, {len(upcoming)} unplayed, engine={payload['engine']}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Project the live season from historical Pom + TabPFN")
    parser.add_argument("--season", type=int, default=LIVE_SEASON)
    parser.add_argument("--refresh", action="store_true", help="refetch CFBD for the live season")
    parser.add_argument("--no-tabpfn", action="store_true")
    args = parser.parse_args()
    project_live(repo_root(), season=args.season, refresh=args.refresh, use_tabpfn=not args.no_tabpfn)


if __name__ == "__main__":
    main()
