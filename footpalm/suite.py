from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from footpalm.featurize import load_prediction_games
from footpalm.form import ALL_NAMES, EXTRA_NAMES, SIGNAL_ALL, SIGNAL_NAMES
from footpalm.predict import FEATURE_NAMES
from footpalm.research import _fit_temperature, apply
from footpalm.trees import HOLDOUT_SEASON, _load_features, _metrics, _repo_root, _split, _write_json

SETS = (
    ("locked", "X_locked", FEATURE_NAMES),
    ("extras", "X_full", ALL_NAMES),
    ("signal", "X_signal", SIGNAL_ALL),
)


def _temperature(y_train: np.ndarray, p_train: np.ndarray, p_hold: np.ndarray) -> tuple[np.ndarray, float]:
    t = _fit_temperature(y_train, p_train)
    return apply("temperature", p_hold, p_hold, {"T": t}), float(t)


def _row(name: str, y_tr: np.ndarray, p_tr: np.ndarray, y_te: np.ndarray, p_te: np.ndarray, extra: dict | None = None) -> dict:
    row = {
        "id": name,
        "train": _metrics(y_tr, p_tr),
        "holdout": _metrics(y_te, p_te),
        "delta_holdout_brier": None,
    }
    if extra:
        row.update(extra)
    return row


def _tree_probs(fit, train_X, train_y, train_m, hold_X):
    clf, _reg = fit(train_X, train_y, train_m)
    return clf.predict_proba(train_X)[:, 1], clf.predict_proba(hold_X)[:, 1]


def _tabpfn_probs(model, X: np.ndarray) -> np.ndarray:
    return np.array([row["home_win_prob"] for row in model.predict_many(X, 27.0)], dtype=float)


def _job_batch(key: str, dest: Path) -> None:
    from footpalm.predict import fit_tabpfn

    payload = _load_features(_repo_root())
    train_X, train_y, train_m, hold_X, hold_y, _hold_m, hold_fbs = _split(payload, key)
    model = fit_tabpfn(train_X, train_y, train_m)
    if not model.ready:
        raise SystemExit(model.error or "tabpfn not ready")
    tail = min(400, model.n_train)
    p_tr = _tabpfn_probs(model, train_X[-tail:])
    p_te = _tabpfn_probs(model, hold_X[hold_fbs])
    np.savez(dest, p_tr=p_tr, p_te=p_te, train_tail=train_y[-tail:], hold_y=hold_y[hold_fbs])


def _job_walk(key: str, dest: Path) -> None:
    from footpalm.predict import fit_tabpfn

    root = _repo_root()
    payload = _load_features(root)
    games = load_prediction_games(root)
    X = np.asarray(payload[key], dtype=float)
    y = np.asarray(payload["y_win"], dtype=float)
    m = np.asarray(payload["y_margin"], dtype=float)
    season = np.asarray(payload["season"])
    fbs = np.asarray(payload["fbs"])
    slate = np.array([int(g["slate"]) for g in games], dtype=int)
    p = np.full(len(y), np.nan)
    slates = sorted({int(s) for s in slate[season == HOLDOUT_SEASON]})
    for current in slates:
        train = (season < HOLDOUT_SEASON) | ((season == HOLDOUT_SEASON) & (slate < current))
        test = (season == HOLDOUT_SEASON) & (slate == current) & fbs
        if not np.any(test):
            continue
        model = fit_tabpfn(X[train], y[train], m[train])
        if not model.ready:
            print(f"  tabpfn walk slate {current} failed {model.error}", flush=True)
            continue
        p[test] = _tabpfn_probs(model, X[test])
        print(f"  tabpfn walk slate {current} n_train={model.n_train} n_test={int(test.sum())}", flush=True)
    hold = (season == HOLDOUT_SEASON) & fbs & np.isfinite(p)
    np.savez(dest, p=p[hold], y=y[hold], n_scored=np.array([int(hold.sum())]))


def _spawn(job: str, key: str, dest: Path) -> None:
    cmd = [sys.executable, "-m", "footpalm.suite", "--job", job, "--key", key, "--out", str(dest)]
    print(f"  spawn {job} {key}", flush=True)
    subprocess.run(cmd, check=True)


def run(*, walk_tabpfn: bool = True) -> dict:
    from footpalm.trees import _fit_lgbm, _fit_xgb

    root = _repo_root()
    payload = _load_features(root)
    season = np.asarray(payload["season"])
    fbs_all = np.asarray(payload["fbs"])
    y = np.asarray(payload["y_win"], dtype=float)
    fbs = (season == HOLDOUT_SEASON) & fbs_all
    y_te = y[fbs]

    rows = []
    extras_hold: dict[str, np.ndarray] = {}
    extras_train: dict[str, np.ndarray] = {}
    y_tr_extras = None

    for set_name, key, names in SETS:
        train_X, train_y, train_m, hold_X, hold_y, _hold_m, hold_fbs = _split(payload, key)
        hold_X_fbs = hold_X[hold_fbs]
        hold_y_fbs = hold_y[hold_fbs]
        print(f"suite {set_name} trees", flush=True)
        for family, fit in (("lightgbm", _fit_lgbm), ("xgboost", _fit_xgb)):
            p_tr, p_te = _tree_probs(fit, train_X, train_y, train_m, hold_X_fbs)
            rows.append(_row(f"{family}/{set_name}", train_y, p_tr, hold_y_fbs, p_te, {"engine": family, "set": set_name}))
            p_cal, t = _temperature(train_y, p_tr, p_te)
            rows.append(
                _row(
                    f"{family}/{set_name}+T",
                    train_y,
                    apply("temperature", p_tr, p_tr, {"T": t}),
                    hold_y_fbs,
                    p_cal,
                    {"engine": family, "set": set_name, "T": round(t, 4)},
                )
            )
            if set_name == "extras":
                extras_train[family] = p_tr
                extras_hold[family] = p_te
                y_tr_extras = train_y

        print(f"suite {set_name} tabpfn batch", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "batch.npz"
            _spawn("batch", key, dest)
            packed = np.load(dest)
        p_tr, p_te = packed["p_tr"], packed["p_te"]
        train_tail = packed["train_tail"]
        rows.append(_row(f"tabpfn/{set_name}", train_tail, p_tr, hold_y_fbs, p_te, {"engine": "tabpfn", "set": set_name}))
        p_cal, t = _temperature(train_tail, p_tr, p_te)
        rows.append(
            _row(
                f"tabpfn/{set_name}+T",
                train_tail,
                apply("temperature", p_tr, p_tr, {"T": t}),
                hold_y_fbs,
                p_cal,
                {"engine": "tabpfn", "set": set_name, "T": round(t, 4)},
            )
        )
        if set_name == "extras":
            extras_train["tabpfn"] = p_tr
            extras_hold["tabpfn"] = p_te

    if len(extras_hold) == 3 and y_tr_extras is not None:
        p_te = np.mean([extras_hold["lightgbm"], extras_hold["xgboost"], extras_hold["tabpfn"]], axis=0)
        n = min(len(extras_train["lightgbm"]), len(extras_train["xgboost"]), len(extras_train["tabpfn"]))
        p_tr = np.mean(
            [extras_train["lightgbm"][-n:], extras_train["xgboost"][-n:], extras_train["tabpfn"][-n:]],
            axis=0,
        )
        y_tr = y_tr_extras[-n:]
        rows.append(_row("blend/extras", y_tr, p_tr, y_te, p_te, {"engine": "blend", "set": "extras"}))
        p_cal, t = _temperature(y_tr, p_tr, p_te)
        rows.append(
            _row(
                "blend/extras+T",
                y_tr,
                apply("temperature", p_tr, p_tr, {"T": t}),
                y_te,
                p_cal,
                {"engine": "blend", "set": "extras", "T": round(t, 4)},
            )
        )

    walk_rows = []
    if walk_tabpfn:
        print("suite tabpfn walk-forward extras", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "walk.npz"
            _spawn("walk", "X_full", dest)
            packed = np.load(dest)
        if len(packed["y"]):
            walk_rows.append(
                {
                    "id": "tabpfn/extras-walk",
                    "holdout": _metrics(packed["y"], packed["p"]),
                    "n_scored": int(packed["n_scored"][0]),
                    "engine": "tabpfn",
                    "set": "extras",
                    "mode": "walk-forward",
                }
            )

    locked = next((r for r in rows if r["id"] == "tabpfn/locked"), None)
    baseline = locked["holdout"]["brier"] if locked and "holdout" in locked else None
    for row in rows:
        if baseline is not None and row.get("holdout"):
            row["delta_vs_tabpfn_locked"] = round(row["holdout"]["brier"] - baseline, 4)

    scored_rows = [r for r in rows if r.get("holdout")]
    scored_rows.sort(key=lambda r: r["holdout"]["brier"])
    best = scored_rows[0]["id"] if scored_rows else None

    report = {
        "protocol": (
            "Fit on 2014–2024 walk-forward features. Score 2025 FBS–FBS once. "
            "TabPFN last 8000 train rows, isolated process. Temperature fit on train only. "
            "Blend is the mean of extras probs."
        ),
        "features": FEATURE_NAMES,
        "extra_features": EXTRA_NAMES,
        "signal_features": SIGNAL_NAMES,
        "holdout_n": int(fbs.sum()),
        "best": best,
        "rows": rows,
        "walk": walk_rows,
        "promoted": False,
        "note": "Diagnostic. Live TabPFN stays on the locked 10 until a walk-forward rebuild is shipped.",
    }
    _write_json(root / "data" / "processed" / "suite.json", report)
    _write_json(root / "web" / "public" / "data" / "suite.json", report)
    for row in rows + walk_rows:
        hold = row.get("holdout")
        if not hold:
            print(f"  {row['id']:<28} {row.get('error')}")
            continue
        print(f"  {row['id']:<28} 2025 brier {hold['brier']:.4f}  logloss {hold['logloss']:.4f}  acc {hold['accuracy']:.3f}")
    print(f"  best={best}  live_promoted=False")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("batch", "walk"))
    parser.add_argument("--key")
    parser.add_argument("--out")
    parser.add_argument("--no-walk", action="store_true")
    args = parser.parse_args()
    if args.job:
        dest = Path(args.out)
        if args.job == "batch":
            _job_batch(args.key, dest)
        else:
            _job_walk(args.key, dest)
        return
    run(walk_tabpfn=not args.no_walk)


if __name__ == "__main__":
    main()
