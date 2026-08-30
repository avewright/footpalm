from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from footpalm.form import ALL_NAMES, CRAFT_ALL, CRAFT_NAMES, FULL_CRAFT_ALL
from footpalm.predict import FEATURE_NAMES
from footpalm.research import apply, _fit_temperature
from footpalm.suite import _row, _spawn
from footpalm.research import PROMOTE_BRIER
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
    ("locked", "X_locked", FEATURE_NAMES),
    ("extras", "X_full", ALL_NAMES),
    ("craft", "X_craft", CRAFT_ALL),
    ("extras+craft", "X_full_craft", FULL_CRAFT_ALL),
)
# Published extras-walk from the model-grid pass. Do not refit.
EXTRAS_WALK_BRIER = 0.1865
EXTRAS_WALK_LOGLOSS = 0.5529


def _temperature(y_train: np.ndarray, p_train: np.ndarray, p_hold: np.ndarray) -> tuple[np.ndarray, float]:
    t = _fit_temperature(y_train, p_train)
    return apply("temperature", p_hold, p_hold, {"T": t}), float(t)


def _append_log(root: Path, report: dict) -> None:
    log = root / "research" / "LOG.md"
    lines = [
        "",
        "## Craft (diagnostic, not live)",
        "",
        "Menu locked from March Madness 2025 1st and 2026 2nd/3rd before this score. "
        "SRS, Colley, ncsos, margin std, pom/tempo sums and abs, signed-log Pom, tanh Elo, "
        "and three products. Walk-forward only. 2014–2024 train, 2025 FBS–FBS once.",
        "",
        "| family | vs | baseline | candidate | Δ | pass |",
        "|---|---|---|---|---|---|",
    ]
    for row in report.get("comparisons", []):
        lines.append(
            f"| {row['family']} | {row['set']} | {row['locked_brier']} | {row['full_brier']} | "
            f"{row['delta_brier']} | {row['pass']} |"
        )
    lines += ["", "Permutation on extras+craft (unique holdout):", "", "| feature | LightGBM | XGBoost |", "|---|---|---|"]
    lgbm = next((m for m in report.get("models", []) if m["id"] == "lightgbm-extras+craft"), None)
    xgb = next((m for m in report.get("models", []) if m["id"] == "xgboost-extras+craft"), None)
    if lgbm and xgb:
        lgbm_p = {r["feature"]: r["brier_increase"] for r in lgbm["permutation"]}
        xgb_p = {r["feature"]: r["brier_increase"] for r in xgb["permutation"]}
        for name in CRAFT_NAMES:
            lines.append(f"| {name} | {lgbm_p.get(name, ''):+} | {xgb_p.get(name, ''):+} |")
    lines += ["", "| model | 2025 Brier | logloss |", "|---|---|---|"]
    for row in report.get("rows", []) + report.get("walk", []):
        hold = row.get("holdout")
        if not hold:
            continue
        lines.append(f"| {row['id']} | {hold['brier']} | {hold['logloss']} |")
    lines += [
        "",
        f"would_promote={report.get('would_promote')} live_promoted={report.get('promoted')}. "
        "Do not carve a subset after seeing 2025.",
        "",
    ]
    existing = log.read_text() if log.exists() else ""
    if "## Craft (diagnostic" in existing:
        start = existing.index("## Craft (diagnostic")
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
        print(f"craft {set_name} trees", flush=True)
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
            top = ", ".join(f"{r['feature']} {r['brier_increase']:+.4f}" for r in packed["permutation"][:3])
            print(f"  {packed['id']:<28} 2025 brier {packed['holdout']['brier']:.4f}  perm {top}", flush=True)

        print(f"craft {set_name} tabpfn batch", flush=True)
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
    for family in ("lightgbm", "xgboost"):
        locked = packed_by_set["locked"][family]
        craft = packed_by_set["craft"][family]
        extras = packed_by_set["extras"][family]
        full = packed_by_set["extras+craft"][family]
        for baseline, candidate, label in (
            (locked, craft, "craft"),
            (extras, full, "extras+craft"),
        ):
            row = _compare_pair(baseline, candidate, label)
            row["family"] = family
            comparisons.append(row)
            print(
                f"  {family:<10} {label:<14} {row['locked_brier']:.4f} → {row['full_brier']:.4f}  "
                f"Δ {row['delta_brier']:+.4f}  pass={row['pass']}",
                flush=True,
            )

    walk_rows = []
    if walk_tabpfn:
        print("craft tabpfn walk-forward extras+craft", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "walk.npz"
            _spawn("walk", "X_full_craft", dest)
            packed = np.load(dest)
        if len(packed["y"]):
            walk_rows.append(
                {
                    "id": "tabpfn/extras+craft-walk",
                    "holdout": _metrics(packed["y"], packed["p"]),
                    "n_scored": int(packed["n_scored"][0]),
                    "engine": "tabpfn",
                    "set": "extras+craft",
                    "mode": "walk-forward",
                }
            )
            hold = walk_rows[0]["holdout"]
            print(f"  tabpfn/extras+craft-walk 2025 brier {hold['brier']:.4f}  logloss {hold['logloss']:.4f}", flush=True)

    extras_craft_pass = any(row["pass"] and row["set"] == "extras+craft" for row in comparisons)
    walk_pass = False
    if walk_rows:
        walk_brier = walk_rows[0]["holdout"]["brier"]
        walk_logloss = walk_rows[0]["holdout"]["logloss"]
        walk_pass = (
            EXTRAS_WALK_BRIER - walk_brier >= PROMOTE_BRIER and walk_logloss <= EXTRAS_WALK_LOGLOSS
        )
        walk_rows[0]["delta_vs_extras_walk"] = round(walk_brier - EXTRAS_WALK_BRIER, 4)
        walk_rows[0]["pass"] = walk_pass

    report = {
        "protocol": (
            "Craft menu locked before the score. 2014–2024 train, 2025 FBS–FBS once. "
            "Trees + batch TabPFN + walk-forward TabPFN on extras+craft. Not live."
        ),
        "craft_features": CRAFT_NAMES,
        "comparisons": comparisons,
        "rows": rows,
        "walk": walk_rows,
        "models": models,
        "would_promote": extras_craft_pass,
        "would_promote_live": extras_craft_pass and walk_pass,
        "promoted": False,
        "note": "Do not carve a subset after seeing 2025. Live TabPFN stays on the locked 10.",
    }
    _write_json(root / "data" / "processed" / "craftpass.json", report)
    _write_json(root / "web" / "public" / "data" / "craftpass.json", report)
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
