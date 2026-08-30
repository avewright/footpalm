from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from footpalm.form import ALL_NAMES, EXTRA_NAMES, FormBook
from footpalm.plays import listed_games
from footpalm.predict import TabPFNPair, fit_tabpfn, game_features, logistic_from_features
from footpalm.rate import RatingBook, fit_ratings


def _metrics(rows: list[dict], prefix: str) -> dict:
    if not rows:
        return {f"{prefix}_n": 0}
    y = np.array([r["home_won"] for r in rows], dtype=float)
    p = np.clip(np.array([r["home_win_prob"] for r in rows], dtype=float), 1e-6, 1 - 1e-6)
    margin = np.array([r["pred_margin"] for r in rows], dtype=float)
    actual = np.array([r["actual_margin"] for r in rows], dtype=float)
    picks = (p >= 0.5).astype(float)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    acc = float(np.mean(picks == y))
    mae = float(np.mean(np.abs(margin - actual)))
    out = {
        f"{prefix}_n": int(len(rows)),
        f"{prefix}_accuracy": round(acc, 4),
        f"{prefix}_brier": round(brier, 4),
        f"{prefix}_logloss": round(logloss, 4),
        f"{prefix}_mae": round(mae, 2),
    }
    ats = [r for r in rows if r.get("spread") is not None and not (isinstance(r["spread"], float) and math.isnan(r["spread"]))]
    if ats:
        # spread is home-relative. Home covers if actual + spread > 0.
        covers = []
        model_ats = []
        for r in ats:
            spread = float(r["spread"])
            actual_m = float(r["actual_margin"])
            if abs(actual_m + spread) < 1e-9:
                continue
            home_covers = actual_m + spread > 0
            covers.append(home_covers)
            model_takes_home = float(r["pred_margin"]) > -spread
            model_ats.append(model_takes_home == home_covers)
        if model_ats:
            out[f"{prefix}_ats"] = round(float(np.mean(model_ats)), 4)
            out[f"{prefix}_ats_n"] = int(len(model_ats))
    return out


def _calibration(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    buckets = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        part = [r for r in rows if lo <= r["home_win_prob"] < hi or (i == 9 and r["home_win_prob"] == 1)]
        if not part:
            continue
        buckets.append(
            {
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "n": len(part),
                "pred": round(float(np.mean([r["home_win_prob"] for r in part])), 3),
                "actual": round(float(np.mean([r["home_won"] for r in part])), 3),
            }
        )
    return buckets


def walk_forward(
    sides: pd.DataFrame,
    fbs: set[str],
    conferences: dict[str, str],
    season: int,
    prior: RatingBook | None = None,
    prior_Xy: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    use_tabpfn: bool = True,
    form: FormBook | None = None,
) -> dict:
    games = listed_games(sides)
    slates = sorted(int(s) for s in games["slate"].dropna().unique())
    history_X = [] if prior_Xy is None else [prior_Xy[0]]
    history_win = [] if prior_Xy is None else [prior_Xy[1]]
    history_margin = [] if prior_Xy is None else [prior_Xy[2]]
    predictions: list[dict] = []
    engine_counts: dict[str, int] = {}
    last_error = None
    form = form or FormBook()

    for slate in slates:
        train_sides = sides.loc[sides["slate"] < slate]
        book = fit_ratings(
            train_sides,
            fbs,
            conferences,
            season,
            through_slate=slate - 1 if train_sides.empty else int(train_sides["slate"].max()),
            prior=prior,
        )
        slate_games = games.loc[games["slate"].eq(slate)]
        model: TabPFNPair | None = None
        if use_tabpfn and history_X:
            X = np.vstack(history_X)
            y_win = np.concatenate(history_win)
            y_margin = np.concatenate(history_margin)
            model = fit_tabpfn(X, y_win, y_margin)
            if not model.ready:
                last_error = model.error
                model = None

        eligible = []
        slate_X = []
        for row in slate_games.itertuples(index=False):
            home, away = row.home_team, row.away_team
            if home not in fbs and away not in fbs:
                continue
            eligible.append(row)
            locked = game_features(book, home, away, neutral=bool(row.neutral_site))
            extra = form.extras(home, away, slate)
            slate_X.append(np.concatenate([locked, extra]))

        if model is not None and slate_X:
            preds = model.predict_many(np.vstack(slate_X), book.league_ppg)
        else:
            preds = [logistic_from_features(x, book.league_ppg) for x in slate_X]

        slate_win = []
        slate_margin = []
        for row, pred in zip(eligible, preds, strict=True):
            actual_margin = float(row.points - row.opp_points)
            home_won = bool(row.won)
            spread = None if pd.isna(getattr(row, "spread", np.nan)) else float(row.spread)
            record = {
                "season": season,
                "slate": slate,
                "week": int(row.week) if pd.notna(row.week) else slate,
                "season_type": getattr(row, "season_type", "regular"),
                "game_id": int(row.game_id) if pd.notna(row.game_id) else None,
                "home": row.home_team,
                "away": row.away_team,
                "neutral": bool(row.neutral_site),
                "home_conf": conferences.get(row.home_team, ""),
                "away_conf": conferences.get(row.away_team, ""),
                "fbs_fbs": row.home_team in fbs and row.away_team in fbs,
                "pred_margin": round(float(pred["pred_margin"]), 2),
                "home_win_prob": round(float(pred["home_win_prob"]), 4),
                "pred_home": round(float(pred["pred_home"]), 1),
                "pred_away": round(float(pred["pred_away"]), 1),
                "actual_home": float(row.points),
                "actual_away": float(row.opp_points),
                "actual_margin": actual_margin,
                "home_won": int(home_won),
                "spread": spread,
                "engine": pred["engine"],
            }
            predictions.append(record)
            engine_counts[pred["engine"]] = engine_counts.get(pred["engine"], 0) + 1
            slate_win.append(float(home_won))
            slate_margin.append(actual_margin)
            form.apply_game(
                row.home_team,
                row.away_team,
                home_won=home_won,
                margin=actual_margin,
                home_pom=book.pom(row.home_team),
                away_pom=book.pom(row.away_team),
                slate=slate,
                neutral=bool(row.neutral_site),
                home_points=float(row.points),
                away_points=float(row.opp_points),
                home_conf=conferences.get(row.home_team, ""),
                away_conf=conferences.get(row.away_team, ""),
            )

        print(f"  slate {slate}: {len(eligible)} games engine={preds[0]['engine'] if preds else 'none'}", flush=True)

        if slate_X:
            history_X.append(np.vstack(slate_X))
            history_win.append(np.asarray(slate_win, dtype=float))
            history_margin.append(np.asarray(slate_margin, dtype=float))

    fbs_rows = [r for r in predictions if r["fbs_fbs"]]
    tabpfn_rows = [r for r in fbs_rows if r["engine"] == "tabpfn-3"]
    logistic_rows = [r for r in fbs_rows if r["engine"] == "logistic"]
    by_week = []
    for slate in slates:
        part = [r for r in fbs_rows if r["slate"] == slate]
        if not part:
            continue
        week = part[0]["week"]
        by_week.append({"slate": slate, "week": week, **_metrics(part, "week")})

    stacked = None
    if history_X:
        stacked = (
            np.vstack(history_X),
            np.concatenate(history_win),
            np.concatenate(history_margin),
        )

    return {
        "season": season,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": (
            "Walk-forward: rate using only prior slates, then predict the next slate. "
            "TabPFN-3 is fit only on earlier games' pre-game features. "
            "Features are locked Pom plus extras (Elo, form, SOS). NIL and market spreads are never inputs. "
            "Logistic Pom+HFA is the fallback and uses the locked 10 only."
        ),
        "features": ALL_NAMES,
        "extra_features": EXTRA_NAMES,
        "engine_counts": engine_counts,
        "tabpfn_error": last_error,
        "all_fbs": _metrics(fbs_rows, "fbs"),
        "tabpfn": _metrics(tabpfn_rows, "tabpfn"),
        "logistic": _metrics(logistic_rows, "logistic"),
        "calibration": _calibration(fbs_rows),
        "by_week": by_week,
        "games": predictions,
        "_history": stacked,
    }


def nil_residual_check(train_games: list[dict], test_games: list[dict], money: dict[str, float]) -> dict:
    """Fit one coefficient on train residuals vs log NIL gap; apply to test. Not used in live picks."""

    def gap(row: dict) -> float:
        h = money.get(row["home"])
        a = money.get(row["away"])
        if not h or not a or h <= 0 or a <= 0:
            return np.nan
        return math.log(h) - math.log(a)

    def design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
        xs, ys = [], []
        for row in rows:
            g = gap(row)
            if np.isnan(g):
                continue
            xs.append(g)
            ys.append(row["actual_margin"] - row["pred_margin"])
        if len(xs) < 20:
            return None
        return np.asarray(xs), np.asarray(ys)

    train = design(train_games)
    test = design(test_games)
    if train is None or test is None:
        return {"used": False, "reason": "not enough NIL-matched residuals"}
    x, y = train
    coef = float(np.dot(x, y) / np.dot(x, x))
    xt, yt = test
    before = float(np.mean(yt**2))
    after = float(np.mean((yt - coef * xt) ** 2))
    return {
        "used": True,
        "note": "One coefficient fit on 2024 residuals, scored on 2025. Not in the live model.",
        "coef": round(coef, 4),
        "train_n": int(len(x)),
        "test_n": int(len(xt)),
        "test_mse_before": round(before, 3),
        "test_mse_after": round(after, 3),
        "helped": after < before,
    }
