from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from footpalm.rate import RatingBook

LOCAL_REGRESSOR = Path.home() / "Library/Caches/tabpfn/tabpfn-v3-regressor-v3_20260506_ood.ckpt"
LOCAL_CLASSIFIER = Path.home() / "Library/Caches/tabpfn/tabpfn-v3-classifier-v3_default.ckpt"

# Locked a priori. Do not grid-search these on the same games you report.
HOME_ADV_POINTS = 2.5
MARGIN_SIGMA = 14.5
FCS_POM = -8.0
MIN_TABPFN_ROWS = 24
MAX_TABPFN_ROWS = 8_000

FEATURE_NAMES = [
    "pom_diff",
    "home_pom",
    "away_pom",
    "adjo_diff",
    "adjd_diff",
    "st_diff",
    "tempo_diff",
    "is_home",
    "home_games",
    "away_games",
]


def game_features(book: RatingBook, home: str, away: str, *, neutral: bool = False) -> np.ndarray:
    home_pom = book.pom(home, FCS_POM)
    away_pom = book.pom(away, FCS_POM)
    return np.array(
        [
            home_pom - away_pom,
            home_pom,
            away_pom,
            book.adjo(home) - book.adjo(away),
            book.adjd(home) - book.adjd(away),
            book.st(home) - book.st(away),
            book.tempo_pg(home) - book.tempo_pg(away),
            0.0 if neutral else 1.0,
            float(book.games_played(home)),
            float(book.games_played(away)),
        ],
        dtype=float,
    )


def logistic_from_features(x: np.ndarray, league_ppg: float) -> dict:
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    pom_diff = float(x[idx["pom_diff"]])
    home_pom = float(x[idx["home_pom"]])
    away_pom = float(x[idx["away_pom"]])
    is_home = float(x[idx["is_home"]])
    hfa = HOME_ADV_POINTS if is_home else 0.0
    margin = float(pom_diff + hfa)
    p_home = 1 / (1 + math.exp(-margin / MARGIN_SIGMA))
    home_pts = league_ppg + (home_pom + hfa) / 2
    away_pts = league_ppg - (home_pom - away_pom - hfa) / 2
    return {
        "pred_margin": margin,
        "home_win_prob": p_home,
        "pred_home": home_pts,
        "pred_away": away_pts,
        "engine": "logistic",
    }


def predict_slate(
    book: RatingBook,
    games: list[dict],
    model: "TabPFNPair | None" = None,
    X: np.ndarray | None = None,
) -> list[dict]:
    if not games:
        return []
    if X is None:
        X = np.vstack(
            [game_features(book, g["home"], g["away"], neutral=bool(g.get("neutral"))) for g in games]
        )
    if model is not None and getattr(model, "ready", False):
        raw = model.predict_many(X, book.league_ppg)
    else:
        raw = [logistic_from_features(x, book.league_ppg) for x in X]
    out = []
    for game, pred in zip(games, raw, strict=True):
        home_pom = book.pom(game["home"], FCS_POM)
        away_pom = book.pom(game["away"], FCS_POM)
        out.append(
            {
                "pred_margin": round(float(pred["pred_margin"]), 2),
                "home_win_prob": round(float(pred["home_win_prob"]), 4),
                "pred_home": round(float(pred["pred_home"]), 1),
                "pred_away": round(float(pred["pred_away"]), 1),
                "home_pom": round(home_pom, 2),
                "away_pom": round(away_pom, 2),
                "engine": pred.get("engine", "logistic"),
            }
        )
    return out


def predict_game(
    book: RatingBook,
    home: str,
    away: str,
    *,
    neutral: bool = False,
    model: "TabPFNPair | None" = None,
) -> dict:
    x = game_features(book, home, away, neutral=neutral)
    base = logistic_from_features(x, book.league_ppg)
    if model is not None and model.ready:
        tab = model.predict_one(x, book.league_ppg)
        base.update(tab)
    home_pom = book.pom(home, FCS_POM)
    away_pom = book.pom(away, FCS_POM)
    hfa = 0.0 if neutral else HOME_ADV_POINTS
    home_pts = book.league_ppg + book.adjo(home) + book.adjd(away) + hfa / 2
    away_pts = book.league_ppg + book.adjo(away) + book.adjd(home) - hfa / 2
    if base.get("engine") == "logistic":
        base["pred_home"] = home_pts
        base["pred_away"] = away_pts
        base["pred_margin"] = home_pom - away_pom + hfa
        base["home_win_prob"] = 1 / (1 + math.exp(-base["pred_margin"] / MARGIN_SIGMA))
    return {
        "home": home,
        "away": away,
        "neutral": neutral,
        "pred_margin": round(float(base["pred_margin"]), 2),
        "home_win_prob": round(float(base["home_win_prob"]), 4),
        "pred_home": round(float(base["pred_home"]), 1),
        "pred_away": round(float(base["pred_away"]), 1),
        "home_pom": round(home_pom, 2),
        "away_pom": round(away_pom, 2),
        "engine": base.get("engine", "logistic"),
    }


def _tabpfn_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class TabPFNPair:
    ready: bool = False
    error: str | None = None
    clf: object | None = None
    reg: object | None = None
    n_train: int = 0

    def predict_many(self, X: np.ndarray, league_ppg: float) -> list[dict]:
        if self.clf is not None and self.reg is not None:
            probs = np.clip(self.clf.predict_proba(X)[:, 1], 0.01, 0.99)
            margins = np.asarray(self.reg.predict(X), dtype=float)
        elif self.clf is not None:
            probs = np.clip(self.clf.predict_proba(X)[:, 1], 0.01, 0.99)
            margins = MARGIN_SIGMA * np.log(probs / (1 - probs))
        else:
            margins = np.asarray(self.reg.predict(X), dtype=float)
            probs = 1 / (1 + np.exp(-margins / MARGIN_SIGMA))
        out = []
        for margin, p in zip(margins, probs, strict=True):
            out.append(
                {
                    "pred_margin": float(margin),
                    "home_win_prob": float(p),
                    "pred_home": league_ppg + float(margin) / 2,
                    "pred_away": league_ppg - float(margin) / 2,
                    "engine": "tabpfn-3",
                }
            )
        return out

    def predict_one(self, x: np.ndarray, league_ppg: float) -> dict:
        return self.predict_many(x.reshape(1, -1), league_ppg)[0]


def fit_tabpfn(
    X: np.ndarray,
    y_win: np.ndarray,
    y_margin: np.ndarray,
    *,
    heads: str = "auto",
) -> TabPFNPair:
    if len(X) < MIN_TABPFN_ROWS:
        return TabPFNPair(ready=False, error=f"need {MIN_TABPFN_ROWS} games, have {len(X)}", n_train=len(X))
    if len(X) > MAX_TABPFN_ROWS:
        X = X[-MAX_TABPFN_ROWS:]
        y_win = y_win[-MAX_TABPFN_ROWS:]
        y_margin = y_margin[-MAX_TABPFN_ROWS:]
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    device = _tabpfn_device()
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
    except Exception as exc:
        return TabPFNPair(ready=False, error=f"tabpfn import failed: {exc}", n_train=len(X))

    try:
        if heads == "pair":
            clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3, device=device, random_state=0)
            clf.fit(X, y_win)
            if not LOCAL_REGRESSOR.exists():
                return TabPFNPair(ready=True, clf=clf, n_train=len(X))
            reg = TabPFNRegressor(model_path=str(LOCAL_REGRESSOR), device=device, random_state=0)
            reg.fit(X, y_margin)
            return TabPFNPair(ready=True, clf=clf, reg=reg, n_train=len(X))
        if heads == "reg":
            if not LOCAL_REGRESSOR.exists():
                return TabPFNPair(ready=False, error="no tabpfn regressor checkpoint", n_train=len(X))
            reg = TabPFNRegressor(model_path=str(LOCAL_REGRESSOR), device=device, random_state=0)
            reg.fit(X, y_margin)
            return TabPFNPair(ready=True, reg=reg, n_train=len(X))
        if LOCAL_CLASSIFIER.exists() and LOCAL_CLASSIFIER.stat().st_size > 1_000_000:
            clf = TabPFNClassifier.create_default_for_version(
                ModelVersion.V3, device=device, random_state=0
            )
            clf.fit(X, y_win)
            return TabPFNPair(ready=True, clf=clf, n_train=len(X))
        if LOCAL_REGRESSOR.exists():
            reg = TabPFNRegressor(model_path=str(LOCAL_REGRESSOR), device=device, random_state=0)
            reg.fit(X, y_margin)
            return TabPFNPair(ready=True, reg=reg, n_train=len(X))
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3, device=device, random_state=0)
        clf.fit(X, y_win)
        return TabPFNPair(ready=True, clf=clf, n_train=len(X))
    except Exception as exc:
        return TabPFNPair(ready=False, error=str(exc), n_train=len(X))
