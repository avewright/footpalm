from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from footpalm.form import ALL_NAMES, LOSO_ALL, LOSO_NAMES
from footpalm.research import PROMOTE_BRIER, _fit_temperature, apply
from footpalm.trees import (
    _fit_lgbm,
    _fit_xgb,
    _load_features,
    _metrics,
    _perm_brier,
    _repo_root,
    _write_json,
)

SETS = (
    ("extras", "X_full", ALL_NAMES),
    ("extras+loso", "X_loso", LOSO_ALL),
)
TREE_FAMILIES = ("logistic", "lightgbm", "xgboost")
FAMILIES = (*TREE_FAMILIES, "tabpfn")
CLIP = (0.02, 0.98)
LR_ITERS = 40
LR_RIDGE = 1.0
TABPFN_T_TAIL = 400


def _fit_lr(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    scaled = (X - mu) / sd
    design = np.column_stack([np.ones(len(X)), scaled])
    beta = np.zeros(design.shape[1])
    ridge = np.eye(design.shape[1]) * LR_RIDGE
    ridge[0, 0] = 0.0
    for _ in range(LR_ITERS):
        z = np.clip(design @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        weight = np.clip(p * (1.0 - p), 1e-6, None)
        hessian = design.T @ (weight[:, None] * design) + ridge
        grad = design.T @ (y - p) - ridge @ beta
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    return beta, mu, sd


def _predict_lr(X: np.ndarray, beta: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    scaled = (X - mu) / sd
    design = np.column_stack([np.ones(len(X)), scaled])
    z = np.clip(design @ beta, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _tabpfn_cache(root: Path, set_name: str, year: int) -> Path:
    return root / "data" / "cache" / "loso-tabpfn" / set_name / f"{year}.npz"


def _predict_tabpfn(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    m_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    cache: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if cache is not None and cache.exists():
        packed = np.load(cache)
        if int(packed["n_train"]) == len(X_tr) and int(packed["n_hold"]) == len(X_te):
            print(f"    cache {cache.name}", flush=True)
            return packed["p_tr"], packed["p_te"]
    from footpalm.board import _tabpfn_predict

    tail = min(TABPFN_T_TAIL, len(X_tr))
    pred = np.vstack([X_tr[-tail:], X_te])
    p, _m = _tabpfn_predict(X_tr, y_tr, m_tr, pred)
    p_tr, p_te = p[:tail], p[tail:]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, p_tr=p_tr, p_te=p_te, n_train=len(X_tr), n_hold=len(X_te))
    return p_tr, p_te


def _predict_family(
    family: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    m_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    cache: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, object]:
    if family == "logistic":
        beta, mu, sd = _fit_lr(X_tr, y_tr)
        return _predict_lr(X_tr, beta, mu, sd), _predict_lr(X_te, beta, mu, sd), None
    if family == "lightgbm":
        clf, _reg = _fit_lgbm(X_tr, y_tr, y_tr)
        return clf.predict_proba(X_tr)[:, 1], clf.predict_proba(X_te)[:, 1], clf
    if family == "xgboost":
        clf, _reg = _fit_xgb(X_tr, y_tr, y_tr)
        return clf.predict_proba(X_tr)[:, 1], clf.predict_proba(X_te)[:, 1], clf
    if family == "tabpfn":
        return (*_predict_tabpfn(X_tr, y_tr, m_tr, X_te, cache=cache), None)
    raise SystemExit(f"unknown family {family}")


def _loso_masks(season: np.ndarray, fbs: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    years = sorted(int(y) for y in np.unique(season))
    folds = []
    for year in years:
        hold = season == year
        if not np.any(hold & fbs):
            continue
        train = season != year
        if not np.any(train):
            continue
        folds.append((year, train, hold & fbs))
    return folds


def _append_log(root: Path, report: dict, *, heading: str, blurb: str, candidate: str, names: list[str]) -> None:
    log = root / "research" / "LOG.md"
    lines = [
        "",
        f"## {heading}",
        "",
        blurb,
        "",
        f"| family | extras | {candidate} | Δ | pass |",
        "|---|---|---|---|---|",
    ]
    for row in report.get("comparisons", []):
        lines.append(
            f"| {row['family']} | {row['locked_brier']} | {row['full_brier']} | "
            f"{row['delta_brier']} | {row['pass']} |"
        )
    lines += ["", f"Permutation on {candidate} (2025 fold):", "", "| feature | LightGBM | XGBoost |", "|---|---|---|"]
    lgbm = report.get("perm", {}).get("lightgbm", {})
    xgb = report.get("perm", {}).get("xgboost", {})
    for name in names:
        left = lgbm.get(name)
        right = xgb.get(name)
        left_s = f"{left:+.4f}" if left is not None else ""
        right_s = f"{right:+.4f}" if right is not None else ""
        lines.append(f"| {name} | {left_s} | {right_s} |")
    lines += ["", "| model | pooled Brier | pooled logloss | mean season Brier |", "|---|---|---|---|"]
    for row in report.get("rows", []):
        lines.append(
            f"| {row['id']} | {row['pooled']['brier']} | {row['pooled']['logloss']} | {row['mean_season_brier']} |"
        )
    lines += [
        "",
        f"would_promote={report.get('would_promote')} live_promoted={report.get('promoted')}. "
        "Do not carve a subset after seeing LOSO.",
        "",
    ]
    existing = log.read_text() if log.exists() else ""
    marker = f"## {heading}"
    if marker in existing:
        start = existing.index(marker)
        existing = existing[:start].rstrip() + "\n"
    log.write_text(existing + "\n".join(lines))


def score_sets(
    sets: tuple[tuple[str, str, list[str]], ...],
    *,
    candidate: str,
    feature_names: list[str],
    heading: str,
    blurb: str,
    stem: str,
    protocol: str,
    families: tuple[str, ...] = TREE_FAMILIES,
    merge_existing: bool = False,
) -> dict:
    root = _repo_root()
    payload = _load_features(root)
    season = np.asarray(payload["season"])
    fbs = np.asarray(payload["fbs"])
    y = np.asarray(payload["y_win"], dtype=float)
    margin = np.asarray(payload["y_margin"], dtype=float)
    folds = _loso_masks(season, fbs)
    rows = []
    packed: dict[str, dict] = {}
    dests = (root / "data" / "processed" / f"{stem}.json", root / "web" / "public" / "data" / f"{stem}.json")
    existing = None
    if merge_existing:
        for dest in dests:
            if dest.exists():
                existing = json.loads(dest.read_text())
                break

    for set_name, key, names in sets:
        X = np.asarray(payload[key], dtype=float)
        for family in families:
            oof_y: list[np.ndarray] = []
            oof_p: list[np.ndarray] = []
            oof_t: list[np.ndarray] = []
            oof_c: list[np.ndarray] = []
            season_rows = []
            hold_2025 = None
            clf_2025 = None
            for year, train, hold in folds:
                X_tr, y_tr, m_tr = X[train], y[train], margin[train]
                X_te, y_te = X[hold], y[hold]
                cache = _tabpfn_cache(root, set_name, year) if family == "tabpfn" else None
                p_tr, p_te, model = _predict_family(family, X_tr, y_tr, m_tr, X_te, cache=cache)
                y_cal = y_tr[-len(p_tr) :]
                t = _fit_temperature(y_cal, p_tr)
                p_cal = apply("temperature", p_te, p_te, {"T": t})
                p_clip = np.clip(p_te, CLIP[0], CLIP[1])
                hold_metrics = _metrics(y_te, p_te)
                season_rows.append(
                    {
                        "season": year,
                        "n": hold_metrics["n"],
                        "brier": hold_metrics["brier"],
                        "logloss": hold_metrics["logloss"],
                        "brier_T": _metrics(y_te, p_cal)["brier"],
                        "brier_clip": _metrics(y_te, p_clip)["brier"],
                        "T": round(float(t), 4),
                    }
                )
                oof_y.append(y_te)
                oof_p.append(p_te)
                oof_t.append(p_cal)
                oof_c.append(p_clip)
                if year == 2025:
                    hold_2025 = (X_te, y_te, names)
                    clf_2025 = model
                print(
                    f"  {family}/{set_name} {year} n={hold_metrics['n']} "
                    f"brier {hold_metrics['brier']:.4f}",
                    flush=True,
                )
            y_all = np.concatenate(oof_y)
            p_all = np.concatenate(oof_p)
            pooled = _metrics(y_all, p_all)
            packed_row = {
                "id": f"{family}/{set_name}",
                "engine": family,
                "set": set_name,
                "pooled": pooled,
                "pooled_T": _metrics(y_all, np.concatenate(oof_t)),
                "pooled_clip": _metrics(y_all, np.concatenate(oof_c)),
                "mean_season_brier": round(float(np.mean([r["brier"] for r in season_rows])), 4),
                "seasons": season_rows,
            }
            rows.append(packed_row)
            packed[packed_row["id"]] = {**packed_row, "hold_2025": hold_2025, "clf_2025": clf_2025}
            print(
                f"  {family}/{set_name:<12} pooled {pooled['brier']:.4f}  "
                f"mean-season {packed_row['mean_season_brier']:.4f}",
                flush=True,
            )
            if family == "tabpfn":
                on_disk = existing
                if on_disk is None:
                    for dest in dests:
                        if dest.exists():
                            on_disk = json.loads(dest.read_text())
                            break
                trees = [r for r in rows if r["engine"] != "tabpfn"]
                if not trees:
                    trees = [r for r in (on_disk or {}).get("rows", []) if r.get("engine") != "tabpfn"]
                tabpfn_done = [
                    {k: v for k, v in row.items() if k not in {"hold_2025", "clf_2025"}}
                    for row in rows
                    if row["engine"] == "tabpfn"
                ]
                body = dict(on_disk or {})
                body["rows"] = trees + tabpfn_done
                body["protocol"] = protocol
                for dest in dests:
                    _write_json(dest, body)

    comparisons = []
    for family in families:
        extras = packed.get(f"{family}/extras")
        full = packed.get(f"{family}/{candidate}")
        if extras is None or full is None:
            continue
        drop = extras["pooled"]["brier"] - full["pooled"]["brier"]
        passed = drop >= PROMOTE_BRIER and full["pooled"]["logloss"] <= extras["pooled"]["logloss"]
        row = {
            "family": family,
            "locked_brier": extras["pooled"]["brier"],
            "full_brier": full["pooled"]["brier"],
            "locked_logloss": extras["pooled"]["logloss"],
            "full_logloss": full["pooled"]["logloss"],
            "delta_brier": round(full["pooled"]["brier"] - extras["pooled"]["brier"], 4),
            "pass": passed,
        }
        comparisons.append(row)
        print(
            f"  {family:<10} extras {row['locked_brier']:.4f} → {candidate} {row['full_brier']:.4f}  "
            f"Δ {row['delta_brier']:+.4f}  pass={row['pass']}",
            flush=True,
        )

    perm = {}
    for family in ("lightgbm", "xgboost"):
        item = packed.get(f"{family}/{candidate}")
        if not item:
            continue
        hold = item.get("hold_2025")
        clf = item.get("clf_2025")
        if hold is None or clf is None:
            continue
        X_te, y_te, names = hold
        perm[family] = {r["feature"]: r["brier_increase"] for r in _perm_brier(clf, X_te, y_te, names)}

    keep_rows = [r for r in (existing or {}).get("rows", []) if r.get("engine") not in families]
    keep_comp = [c for c in (existing or {}).get("comparisons", []) if c.get("family") not in families]
    if not perm and existing:
        perm = existing.get("perm") or {}
    all_comp = keep_comp + comparisons
    report = {
        "protocol": protocol,
        "features": feature_names,
        "comparisons": all_comp,
        "rows": keep_rows + [{k: v for k, v in row.items() if k not in {"hold_2025", "clf_2025"}} for row in rows],
        "perm": perm,
        "would_promote": any(row["pass"] for row in all_comp if row["family"] in TREE_FAMILIES),
        "promoted": False,
        "note": "Do not carve a subset after seeing LOSO. Live TabPFN stays on extras.",
    }
    if existing and "loso_features" in existing:
        report["loso_features"] = existing["loso_features"]
    for dest in dests:
        _write_json(dest, report)
    _append_log(root, report, heading=heading, blurb=blurb, candidate=candidate, names=feature_names)
    print(f"  would_promote={report['would_promote']} live_promoted={report['promoted']}")
    return report


def run(*, families: tuple[str, ...] = FAMILIES, merge_existing: bool = False) -> dict:
    tabpfn = "tabpfn" in families
    return score_sets(
        SETS,
        candidate="extras+loso",
        feature_names=LOSO_NAMES,
        heading="LOSO (diagnostic, not live)",
        blurb=(
            "Menu locked from ten 2025/2026 March Madness writeups before this score. "
            "Leave-one-season-out on 2014–2025 FBS–FBS. Walk-forward features only."
        ),
        stem="losopass",
        protocol=(
            "LOSO menu locked before the score. Fit on all other seasons, score each "
            "2014–2025 FBS–FBS season once. Trees + logistic"
            + (" + TabPFN-3 (LOSO, not walk-forward)." if tabpfn else ". Not live TabPFN.")
        ),
        families=families,
        merge_existing=merge_existing,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabpfn-only", action="store_true", help="score TabPFN-3 and merge into losopass.json")
    parser.add_argument("--no-tabpfn", action="store_true", help="trees + logistic only")
    args = parser.parse_args()
    if args.tabpfn_only:
        run(families=("tabpfn",), merge_existing=True)
    elif args.no_tabpfn:
        run(families=TREE_FAMILIES)
    else:
        run()


if __name__ == "__main__":
    main()
