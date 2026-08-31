from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from footpalm.form import ALL_NAMES
from footpalm.pace import PACE_ALL, PACE_NAMES
from footpalm.research import PROMOTE_BRIER, _fit_temperature, apply
from footpalm.suite import _row, _spawn
from footpalm.trees import (
    _compare_pair,
    _fit_lgbm,
    _fit_xgb,
    _load_features,
    _metrics,
    _pack,
    _repo_root,
    _split,
    _write_json,
)

SETS = (
    ("extras", "X_full", ALL_NAMES),
    ("extras+pace", "X_pace", PACE_ALL),
)
EXTRAS_WALK_BRIER = 0.1865
EXTRAS_WALK_LOGLOSS = 0.5529


def _temperature(y_train: np.ndarray, p_train: np.ndarray, p_hold: np.ndarray) -> tuple[np.ndarray, float]:
    t = _fit_temperature(y_train, p_train)
    return apply("temperature", p_hold, p_hold, {"T": t}), float(t)


def _append_log(root: Path, report: dict) -> None:
    log = root / "research" / "LOG.md"
    lines = [
        "",
        "## Pace (diagnostic, not live)",
        "",
        "Menu locked before this score. Turnover margin, YPC, wallclock snap gap, "
        "game-clock seconds per play. Walk-forward PBP. 2014–2024 train, 2025 FBS–FBS once.",
        "",
        "| family | extras | extras+pace | Δ | pass |",
        "|---|---|---|---|---|",
    ]
    for row in report.get("comparisons", []):
        lines.append(
            f"| {row['family']} | {row['locked_brier']} | {row['full_brier']} | "
            f"{row['delta_brier']} | {row['pass']} |"
        )
    lines += ["", "Permutation on extras+pace:", "", "| feature | LightGBM | XGBoost |", "|---|---|---|"]
    lgbm = next((m for m in report.get("models", []) if m["id"] == "lightgbm-extras+pace"), None)
    xgb = next((m for m in report.get("models", []) if m["id"] == "xgboost-extras+pace"), None)
    if lgbm and xgb:
        left = {r["feature"]: r["brier_increase"] for r in lgbm["permutation"]}
        right = {r["feature"]: r["brier_increase"] for r in xgb["permutation"]}
        for name in PACE_NAMES:
            lines.append(f"| {name} | {left.get(name, 0):+.4f} | {right.get(name, 0):+.4f} |")
    lines += ["", "| model | 2025 Brier | logloss |", "|---|---|---|"]
    for row in report.get("rows", []) + report.get("walk", []):
        hold = row.get("holdout")
        if hold:
            lines.append(f"| {row['id']} | {hold['brier']} | {hold['logloss']} |")
    lines += [
        "",
        f"would_promote={report.get('would_promote')} live_promoted={report.get('promoted')}. "
        "Do not carve a subset after seeing 2025.",
        "",
    ]
    existing = log.read_text() if log.exists() else ""
    marker = "## Pace (diagnostic"
    if marker in existing:
        start = existing.index(marker)
        existing = existing[:start].rstrip() + "\n"
    log.write_text(existing + "\n".join(lines))


def run(*, walk_tabpfn: bool = True) -> dict:
    root = _repo_root()
    payload = _load_features(root)
    rows = []
    models = []
    packed_by_set: dict[str, dict] = {}

    for set_name, key, names in SETS:
        train_X, train_y, train_m, hold_X, hold_y, hold_m, hold_fbs = _split(payload, key)
        hold_X_fbs = hold_X[hold_fbs]
        hold_y_fbs = hold_y[hold_fbs]
        print(f"pace {set_name} trees", flush=True)
        for family, fit in (("lightgbm", _fit_lgbm), ("xgboost", _fit_xgb)):
            clf, reg = fit(train_X, train_y, train_m)
            packed = _pack(
                f"{family}-{set_name}",
                f"Diagnostic only. {set_name}. Not the live model.",
                list(names),
                clf,
                reg,
                train_X,
                train_y,
                hold_X,
                hold_y,
                hold_m,
                hold_fbs,
                clf.feature_importances_,
            )
            models.append(packed)
            packed_by_set.setdefault(set_name, {})[family] = packed
            p_tr, p_te = clf.predict_proba(train_X)[:, 1], clf.predict_proba(hold_X_fbs)[:, 1]
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
            print(f"  {packed['id']:<28} 2025 brier {packed['holdout']['brier']:.4f}", flush=True)

        print(f"pace {set_name} tabpfn batch", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "batch.npz"
            _spawn("batch", key, dest)
            tab = np.load(dest)
        p_tr, p_te = tab["p_tr"], tab["p_te"]
        train_tail = tab["train_tail"]
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
        print(
            f"  tabpfn/{set_name:<16} 2025 brier {rows[-2]['holdout']['brier']:.4f}  "
            f"+T {rows[-1]['holdout']['brier']:.4f}",
            flush=True,
        )

    comparisons = []
    for family in ("lightgbm", "xgboost", "tabpfn"):
        extras = next(r for r in rows if r["id"] == f"{family}/extras")
        full = next(r for r in rows if r["id"] == f"{family}/extras+pace")
        extras_h = {"id": extras["id"], "holdout": extras["holdout"]}
        full_h = {"id": full["id"], "holdout": full["holdout"]}
        row = _compare_pair(extras_h, full_h, "extras+pace")
        row["family"] = family
        comparisons.append(row)
        print(
            f"  {family:<10} extras {row['locked_brier']:.4f} → extras+pace {row['full_brier']:.4f}  "
            f"Δ {row['delta_brier']:+.4f}  pass={row['pass']}",
            flush=True,
        )

    walk_rows = []
    if walk_tabpfn:
        print("pace tabpfn walk-forward extras+pace", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "walk.npz"
            _spawn("walk", "X_pace", dest)
            packed = np.load(dest)
        if len(packed["y"]):
            hold = _metrics(packed["y"], packed["p"])
            walk_pass = EXTRAS_WALK_BRIER - hold["brier"] >= PROMOTE_BRIER and hold["logloss"] <= EXTRAS_WALK_LOGLOSS
            walk_rows.append(
                {
                    "id": "tabpfn/extras+pace-walk",
                    "holdout": hold,
                    "n_scored": int(packed["n_scored"][0]),
                    "engine": "tabpfn",
                    "set": "extras+pace",
                    "mode": "walk-forward",
                    "delta_vs_extras_walk": round(hold["brier"] - EXTRAS_WALK_BRIER, 4),
                    "pass": walk_pass,
                }
            )
            print(
                f"  tabpfn/extras+pace-walk 2025 brier {hold['brier']:.4f}  "
                f"Δ vs extras-walk {hold['brier'] - EXTRAS_WALK_BRIER:+.4f}  pass={walk_pass}",
                flush=True,
            )

    tree_pass = any(row["pass"] and row["family"] in {"lightgbm", "xgboost"} for row in comparisons)
    walk_pass = bool(walk_rows and walk_rows[0].get("pass"))
    report = {
        "protocol": (
            "Pace menu locked before the score. 2014–2024 train, 2025 FBS–FBS once. "
            "Trees + batch TabPFN + walk-forward TabPFN on extras+pace. Not live."
        ),
        "pace_features": PACE_NAMES,
        "comparisons": comparisons,
        "rows": rows,
        "walk": walk_rows,
        "models": [
            {k: v for k, v in m.items() if k != "note"}
            for m in models
        ],
        "would_promote": tree_pass,
        "would_promote_live": tree_pass and walk_pass,
        "promoted": False,
        "note": "Do not carve a subset after seeing 2025. Live TabPFN stays on extras.",
    }
    _write_json(root / "data" / "processed" / "pacepass.json", report)
    _write_json(root / "web" / "public" / "data" / "pacepass.json", report)
    _append_log(root, report)
    print(f"  would_promote={report['would_promote']} live_promoted={report['promoted']}")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-walk", action="store_true")
    args = parser.parse_args()
    run(walk_tabpfn=not args.no_walk)


if __name__ == "__main__":
    main()
