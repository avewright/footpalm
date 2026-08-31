from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from footpalm.featurize import features_path, run as build_features
from footpalm.form import (
    ALL_NAMES,
    CRAFT_NAMES,
    EXTRA_NAMES,
    CONF_NAMES,
    GLM4_NAMES,
    LOSO_NAMES,
    SIGNAL_ALL,
    SIGNAL_NAMES,
    TIME_ALL,
    TIME_NAMES,
)
from footpalm.pace import PACE_NAMES
from footpalm.specials import SPECIAL_NAMES
from footpalm.predict import FEATURE_NAMES
from footpalm.project import history_path
from footpalm.research import PROMOTE_BRIER

HOLDOUT_SEASON = 2025
PERM_REPEATS = 5

# Locked. Do not tune these on 2025.
LGBM = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "verbose": -1,
}
XGB = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "n_jobs": 4,
    "eval_metric": "logloss",
}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise SystemExit("could not find repo root")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _load_features(root: Path) -> dict[str, np.ndarray]:
    dest = features_path(root)
    need = not dest.exists()
    if dest.exists():
        existing = np.load(dest)
        need = (
            "X_signal" not in existing.files
            or existing["signal"].shape[1] != len(SIGNAL_NAMES)
            or "X_time" not in existing.files
            or existing["time"].shape[1] != len(TIME_NAMES)
            or "X_craft" not in existing.files
            or existing["craft"].shape[1] != len(CRAFT_NAMES)
            or "X_loso" not in existing.files
            or existing["loso"].shape[1] != len(LOSO_NAMES)
            or "X_conf" not in existing.files
            or existing["conf"].shape[1] != len(CONF_NAMES)
            or "X_glm4" not in existing.files
            or existing["glm4"].shape[1] != len(GLM4_NAMES)
            or "X_pace" not in existing.files
            or existing["pace"].shape[1] != len(PACE_NAMES)
            or "X_specials" not in existing.files
            or existing["specials"].shape[1] != len(SPECIAL_NAMES)
        )
    if need:
        print("features-history missing or stale — building extras + signal")
        build_features(root)
    payload = np.load(dest)
    history = np.load(history_path(root))
    locked = np.asarray(payload["X_locked"], dtype=float)
    hist_X = np.asarray(history["X"], dtype=float)
    hist_locked = hist_X[:, : len(FEATURE_NAMES)]
    if locked.shape != hist_locked.shape or not np.allclose(locked, hist_locked):
        raise SystemExit("features-history X_locked drifted from tabpfn-history")
    return {key: payload[key] for key in payload.files}


def _split(payload: dict[str, np.ndarray], key: str):
    season = np.asarray(payload["season"])
    fbs_all = np.asarray(payload["fbs"])
    X = np.asarray(payload[key], dtype=float)
    y = np.asarray(payload["y_win"], dtype=float)
    m = np.asarray(payload["y_margin"], dtype=float)
    train = season < HOLDOUT_SEASON
    hold = season == HOLDOUT_SEASON
    return X[train], y[train], m[train], X[hold], y[hold], m[hold], fbs_all[hold]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "accuracy": round(float(np.mean((p >= 0.5) == y)), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
    }


def _perm_brier(model, X: np.ndarray, y: np.ndarray, names: list[str], repeats: int = PERM_REPEATS) -> list[dict]:
    rng = np.random.default_rng(0)
    base_p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    base = float(np.mean((base_p - y) ** 2))
    rows = []
    for i, name in enumerate(names):
        deltas = []
        for _ in range(repeats):
            shuffled = X.copy()
            rng.shuffle(shuffled[:, i])
            p = np.clip(model.predict_proba(shuffled)[:, 1], 1e-6, 1 - 1e-6)
            deltas.append(float(np.mean((p - y) ** 2)) - base)
        rows.append({"feature": name, "brier_increase": round(float(np.mean(deltas)), 5)})
    rows.sort(key=lambda r: r["brier_increase"], reverse=True)
    return rows


def _gain_rows(names: list[str], values: np.ndarray) -> list[dict]:
    total = float(np.sum(np.abs(values))) or 1.0
    rows = [
        {"feature": name, "gain": round(float(v), 5), "share": round(float(v) / total, 4)}
        for name, v in zip(names, values, strict=True)
    ]
    rows.sort(key=lambda r: r["gain"], reverse=True)
    return rows


def _fit_lgbm(X: np.ndarray, y: np.ndarray, m: np.ndarray):
    from lightgbm import LGBMClassifier, LGBMRegressor

    clf = LGBMClassifier(**LGBM, importance_type="gain")
    clf.fit(X, y)
    reg = LGBMRegressor(**{k: v for k, v in LGBM.items() if k != "verbose"})
    reg.fit(X, m)
    return clf, reg


def _fit_xgb(X: np.ndarray, y: np.ndarray, m: np.ndarray):
    from xgboost import XGBClassifier, XGBRegressor

    clf = XGBClassifier(**XGB)
    clf.fit(X, y)
    reg = XGBRegressor(**{k: v for k, v in XGB.items() if k != "eval_metric"})
    reg.fit(X, m)
    return clf, reg


def _pack(name: str, note: str, names: list[str], clf, reg, train_X, train_y, hold_X, hold_y, hold_m, fbs, gain) -> dict:
    p_tr = clf.predict_proba(train_X)[:, 1]
    p_te = clf.predict_proba(hold_X[fbs])[:, 1]
    margin = np.asarray(reg.predict(hold_X[fbs]), dtype=float)
    return {
        "id": name,
        "note": note,
        "params": {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.05},
        "train": _metrics(train_y, p_tr),
        "holdout": {
            **_metrics(hold_y[fbs], p_te),
            "mae": round(float(np.mean(np.abs(margin - hold_m[fbs]))), 2),
        },
        "gain": _gain_rows(names, gain),
        "permutation": _perm_brier(clf, hold_X[fbs], hold_y[fbs], names),
    }


def _passes(locked: dict, full: dict) -> bool:
    brier_drop = locked["holdout"]["brier"] - full["holdout"]["brier"]
    return brier_drop >= PROMOTE_BRIER and full["holdout"]["logloss"] <= locked["holdout"]["logloss"]


def _compare_pair(locked: dict, candidate: dict, set_name: str) -> dict:
    passed = _passes(locked, candidate)
    return {
        "family": f"{locked['id']} / {set_name}",
        "set": set_name,
        "locked_brier": locked["holdout"]["brier"],
        "full_brier": candidate["holdout"]["brier"],
        "locked_logloss": locked["holdout"]["logloss"],
        "full_logloss": candidate["holdout"]["logloss"],
        "delta_brier": round(candidate["holdout"]["brier"] - locked["holdout"]["brier"], 4),
        "pass": passed,
    }


def _compare(locked_models: list[dict], extra_models: list[dict], signal_models: list[dict]) -> dict:
    rows = []
    for locked, extra, signal in zip(locked_models, extra_models, signal_models, strict=True):
        rows.append(_compare_pair(locked, extra, "extras"))
        rows.append(_compare_pair(locked, signal, "signal"))
    any_pass = any(row["pass"] for row in rows)
    signal_pass = any(row["pass"] and row["set"] == "signal" for row in rows)
    return {
        "rule": f"Promote a set if 2025 Brier drops ≥ {PROMOTE_BRIER} and log loss does not rise.",
        "extras": EXTRA_NAMES,
        "signal": SIGNAL_NAMES,
        "rows": rows,
        "would_promote": any_pass,
        "would_promote_signal": signal_pass,
        "promoted": False,
        "note": (
            "Trees are diagnostic. Live TabPFN stays on the locked 10 until a walk-forward rebuild. "
            "Do not carve the signal set after seeing 2025."
        ),
    }


def run() -> dict:
    root = _repo_root()
    payload = _load_features(root)
    locked_split = _split(payload, "X_locked")
    full_split = _split(payload, "X_full")
    signal_split = _split(payload, "X_signal")
    locked_note = "Diagnostic only. Locked 10 Pom features. Not the live model."
    full_note = "Diagnostic only. Locked 10 plus March-Madness extras. Not the live model."
    signal_note = "Diagnostic only. Locked 10 plus signal-axis expansion. Not the live model."

    models = []
    locked_models = []
    extra_models = []
    signal_models = []
    l_train_X, l_train_y, l_train_m, l_hold_X, l_hold_y, l_hold_m, l_fbs = locked_split
    f_train_X, f_train_y, f_train_m, f_hold_X, f_hold_y, f_hold_m, f_fbs = full_split
    s_train_X, s_train_y, s_train_m, s_hold_X, s_hold_y, s_hold_m, s_fbs = signal_split
    for family, fit in (("lightgbm", _fit_lgbm), ("xgboost", _fit_xgb)):
        clf, reg = fit(l_train_X, l_train_y, l_train_m)
        packed = _pack(
            family,
            locked_note,
            FEATURE_NAMES,
            clf,
            reg,
            l_train_X,
            l_train_y,
            l_hold_X,
            l_hold_y,
            l_hold_m,
            l_fbs,
            clf.feature_importances_,
        )
        models.append(packed)
        locked_models.append(packed)

        clf, reg = fit(f_train_X, f_train_y, f_train_m)
        packed = _pack(
            f"{family}-full",
            full_note,
            ALL_NAMES,
            clf,
            reg,
            f_train_X,
            f_train_y,
            f_hold_X,
            f_hold_y,
            f_hold_m,
            f_fbs,
            clf.feature_importances_,
        )
        models.append(packed)
        extra_models.append(packed)

        clf, reg = fit(s_train_X, s_train_y, s_train_m)
        packed = _pack(
            f"{family}-signal",
            signal_note,
            SIGNAL_ALL,
            clf,
            reg,
            s_train_X,
            s_train_y,
            s_hold_X,
            s_hold_y,
            s_hold_m,
            s_fbs,
            clf.feature_importances_,
        )
        models.append(packed)
        signal_models.append(packed)

    comparison = _compare(locked_models, extra_models, signal_models)
    report = {
        "protocol": (
            "Fit on 2014–2024 walk-forward features. Score 2025 FBS–FBS once. "
            "Locked shallow trees. Extras and signal menus locked before the score. Not promoted."
        ),
        "features": FEATURE_NAMES,
        "extra_features": EXTRA_NAMES,
        "signal_features": SIGNAL_NAMES,
        "train_n": int(len(locked_split[1])),
        "holdout_n": int(locked_split[6].sum()),
        "comparison": comparison,
        "models": models,
    }
    _write_json(root / "data" / "processed" / "trees.json", report)
    _write_json(root / "web" / "public" / "data" / "trees.json", report)
    for model in report["models"]:
        top = ", ".join(f"{r['feature']} {r['brier_increase']:+.4f}" for r in model["permutation"][:3])
        print(
            f"  {model['id']:<16} 2025 brier {model['holdout']['brier']:.4f}  "
            f"perm {top}"
        )
    for row in comparison["rows"]:
        print(
            f"  {row['family']:<10} locked {row['locked_brier']:.4f} → full {row['full_brier']:.4f}  "
            f"Δ {row['delta_brier']:+.4f}  pass={row['pass']}"
        )
    print(f"  would_promote={comparison['would_promote']}  live_promoted={comparison['promoted']}")
    return report


def run_time() -> dict:
    root = _repo_root()
    payload = _load_features(root)
    extra_split = _split(payload, "X_full")
    time_split = _split(payload, "X_time")
    extra_models = []
    time_models = []
    e_train_X, e_train_y, e_train_m, e_hold_X, e_hold_y, e_hold_m, e_fbs = extra_split
    t_train_X, t_train_y, t_train_m, t_hold_X, t_hold_y, t_hold_m, t_fbs = time_split
    note = "Diagnostic only. Locked 10 plus extras plus year_idx and week52."
    for family, fit in (("lightgbm", _fit_lgbm), ("xgboost", _fit_xgb)):
        clf, reg = fit(e_train_X, e_train_y, e_train_m)
        extra_models.append(
            _pack(
                f"{family}-full",
                "Diagnostic only. Locked 10 plus extras.",
                ALL_NAMES,
                clf,
                reg,
                e_train_X,
                e_train_y,
                e_hold_X,
                e_hold_y,
                e_hold_m,
                e_fbs,
                clf.feature_importances_,
            )
        )
        clf, reg = fit(t_train_X, t_train_y, t_train_m)
        time_models.append(
            _pack(
                f"{family}-time",
                note,
                TIME_ALL,
                clf,
                reg,
                t_train_X,
                t_train_y,
                t_hold_X,
                t_hold_y,
                t_hold_m,
                t_fbs,
                clf.feature_importances_,
            )
        )
    rows = []
    for extra, timed in zip(extra_models, time_models, strict=True):
        row = _compare_pair(extra, timed, "time")
        row["family"] = extra["id"].replace("-full", "")
        rows.append(row)
    report = {
        "protocol": "Score extras vs extras+year_idx+week52. 2014–2024 train, 2025 FBS once. Not live TabPFN.",
        "time_features": TIME_NAMES,
        "rows": rows,
        "would_promote": any(row["pass"] for row in rows),
        "promoted": False,
        "models": extra_models + time_models,
    }
    _write_json(root / "data" / "processed" / "trees-time.json", report)
    _write_json(root / "web" / "public" / "data" / "trees-time.json", report)
    for model in time_models:
        top = ", ".join(f"{r['feature']} {r['brier_increase']:+.4f}" for r in model["permutation"][:4])
        print(f"  {model['id']:<16} 2025 brier {model['holdout']['brier']:.4f}  perm {top}")
    for row in rows:
        print(
            f"  {row['family']:<10} extras {row['locked_brier']:.4f} → time {row['full_brier']:.4f}  "
            f"Δ {row['delta_brier']:+.4f}  pass={row['pass']}"
        )
    print(f"  would_promote={report['would_promote']}")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--time", action="store_true", help="score extras vs extras+year/week only")
    args = parser.parse_args()
    if args.time:
        run_time()
        return
    run()


if __name__ == "__main__":
    main()
