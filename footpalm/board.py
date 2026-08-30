from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from footpalm.featurize import features_path, load_prediction_games, run as build_features
from footpalm.form import extras_matrix, replay_form
from footpalm.project import book_path, load_prior_book
from footpalm.trees import HOLDOUT_SEASON, _fit_lgbm, _fit_xgb, _repo_root, _write_json

LIVE_SEASON = 2026
MODELS = ("lightgbm", "xgboost", "tabpfn", "ensemble")


def _load_features(root: Path) -> dict[str, np.ndarray]:
    dest = features_path(root)
    if not dest.exists() or "X_full" not in np.load(dest).files:
        build_features(root)
    return {key: np.load(dest)[key] for key in np.load(dest).files}


def _pack(p: np.ndarray, margin: np.ndarray) -> list[dict]:
    out = []
    for prob, m in zip(p, margin, strict=True):
        out.append({"home_win_prob": round(float(np.clip(prob, 1e-6, 1 - 1e-6)), 4), "pred_margin": round(float(m), 2)})
    return out


def _tree_predict(fit, X_tr, y_tr, m_tr, X_te) -> tuple[np.ndarray, np.ndarray]:
    clf, reg = fit(X_tr, y_tr, m_tr)
    return clf.predict_proba(X_te)[:, 1], np.asarray(reg.predict(X_te), dtype=float)


def _tabpfn_job(train_X, train_y, train_m, pred_X, dest: Path) -> None:
    from footpalm.predict import fit_tabpfn

    model = fit_tabpfn(train_X, train_y, train_m)
    if not model.ready:
        raise SystemExit(model.error or "tabpfn not ready")
    rows = model.predict_many(pred_X, 27.0)
    p = np.array([r["home_win_prob"] for r in rows], dtype=float)
    m = np.array([r["pred_margin"] for r in rows], dtype=float)
    np.savez(dest, p=p, m=m)


def _tabpfn_predict(X_tr, y_tr, m_tr, X_te) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        train_p = Path(tmp) / "train.npz"
        pred_p = Path(tmp) / "pred.npy"
        out_p = Path(tmp) / "out.npz"
        np.savez(train_p, X=X_tr, y=y_tr, m=m_tr)
        np.save(pred_p, X_te)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "footpalm.board",
                "--tabpfn-job",
                "--train",
                str(train_p),
                "--pred",
                str(pred_p),
                "--out",
                str(out_p),
            ],
            check=True,
        )
        packed = np.load(out_p)
        return packed["p"], packed["m"]


def _fit_pack(X_tr, y_tr, m_tr, X_te) -> dict[str, list[dict]]:
    print(f"  trees n_train={len(y_tr)} n_pred={len(X_te)}", flush=True)
    p_l, m_l = _tree_predict(_fit_lgbm, X_tr, y_tr, m_tr, X_te)
    p_x, m_x = _tree_predict(_fit_xgb, X_tr, y_tr, m_tr, X_te)
    print("  tabpfn", flush=True)
    p_t, m_t = _tabpfn_predict(X_tr, y_tr, m_tr, X_te)
    p_e = (p_l + p_x + p_t) / 3
    m_e = (m_l + m_x + m_t) / 3
    return {
        "lightgbm": _pack(p_l, m_l),
        "xgboost": _pack(p_x, m_x),
        "tabpfn": _pack(p_t, m_t),
        "ensemble": _pack(p_e, m_e),
    }


def _attach(games: list[dict], pack: dict[str, list[dict]]) -> None:
    for i, game in enumerate(games):
        game["models"] = {name: pack[name][i] for name in MODELS}


def run() -> None:
    root = _repo_root()
    payload = _load_features(root)
    hist_games = load_prediction_games(root)
    X_full = np.asarray(payload["X_full"], dtype=float)
    y = np.asarray(payload["y_win"], dtype=float)
    m = np.asarray(payload["y_margin"], dtype=float)
    season = np.asarray(payload["season"])
    if len(hist_games) != len(X_full):
        raise SystemExit("features-history drifted from prediction games")

    def write_year(year: int, games: list[dict], train_on: str) -> None:
        for dest in (root / "data" / "processed", root / "web" / "public" / "data"):
            path = dest / f"predictions-{year}.json"
            body = json.loads(path.read_text()) if path.exists() else {"season": year, "games": []}
            body["games"] = games
            body["models"] = {
                "features": "locked-10 + extras",
                "train": train_on,
                "ensemble": "mean of lightgbm, xgboost, tabpfn",
            }
            _write_json(path, body)
        print(f"wrote models onto {year} n={len(games)}")

    hold = season == HOLDOUT_SEASON
    games_hold = [g for g in hist_games if int(g["season"]) == HOLDOUT_SEASON]
    if len(games_hold) != int(hold.sum()):
        raise SystemExit("2025 games drifted from features-history")
    print("fit holdout pack (train < 2025 → 2025)", flush=True)
    hold_pack = _fit_pack(X_full[season < HOLDOUT_SEASON], y[season < HOLDOUT_SEASON], m[season < HOLDOUT_SEASON], X_full[hold])
    _attach(games_hold, hold_pack)
    write_year(HOLDOUT_SEASON, games_hold, f"<{HOLDOUT_SEASON}")

    live_path = root / "web" / "public" / "data" / f"predictions-{LIVE_SEASON}.json"
    if live_path.exists() and book_path(root, LIVE_SEASON - 1).exists():
        live_payload = json.loads(live_path.read_text())
        live_games = live_payload["games"]
        print(f"build {LIVE_SEASON} extras from {LIVE_SEASON - 1} book", flush=True)
        book = load_prior_book(root, LIVE_SEASON - 1)
        form = replay_form(hist_games, np.asarray(payload["X_locked"], dtype=float))
        form.new_season()
        X_live = extras_matrix(book, form, live_games)
        print("fit live pack (train < 2026 → 2026)", flush=True)
        live_pack = _fit_pack(X_full, y, m, X_live)
        _attach(live_games, live_pack)
        write_year(LIVE_SEASON, live_games, f"<{LIVE_SEASON}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabpfn-job", action="store_true")
    parser.add_argument("--train")
    parser.add_argument("--pred")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.tabpfn_job:
        train = np.load(args.train)
        _tabpfn_job(train["X"], train["y"], train["m"], np.load(args.pred), Path(args.out))
        return
    run()


if __name__ == "__main__":
    main()
