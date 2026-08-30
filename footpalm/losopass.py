from __future__ import annotations

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
CLIP = (0.02, 0.98)
LR_ITERS = 40
LR_RIDGE = 1.0


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
) -> dict:
    root = _repo_root()
    payload = _load_features(root)
    season = np.asarray(payload["season"])
    fbs = np.asarray(payload["fbs"])
    y = np.asarray(payload["y_win"], dtype=float)
    folds = _loso_masks(season, fbs)
    rows = []
    packed: dict[str, dict] = {}

    for set_name, key, names in sets:
        X = np.asarray(payload[key], dtype=float)
        for family in ("logistic", "lightgbm", "xgboost"):
            oof_y: list[np.ndarray] = []
            oof_p: list[np.ndarray] = []
            oof_t: list[np.ndarray] = []
            oof_c: list[np.ndarray] = []
            season_rows = []
            hold_2025 = None
            clf_2025 = None
            for year, train, hold in folds:
                X_tr, y_tr = X[train], y[train]
                X_te, y_te = X[hold], y[hold]
                if family == "logistic":
                    beta, mu, sd = _fit_lr(X_tr, y_tr)
                    p_tr = _predict_lr(X_tr, beta, mu, sd)
                    p_te = _predict_lr(X_te, beta, mu, sd)
                    model = None
                elif family == "lightgbm":
                    clf, _reg = _fit_lgbm(X_tr, y_tr, y_tr)
                    p_tr = clf.predict_proba(X_tr)[:, 1]
                    p_te = clf.predict_proba(X_te)[:, 1]
                    model = clf
                else:
                    clf, _reg = _fit_xgb(X_tr, y_tr, y_tr)
                    p_tr = clf.predict_proba(X_tr)[:, 1]
                    p_te = clf.predict_proba(X_te)[:, 1]
                    model = clf
                t = _fit_temperature(y_tr, p_tr)
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

    comparisons = []
    for family in ("logistic", "lightgbm", "xgboost"):
        extras = packed[f"{family}/extras"]
        full = packed[f"{family}/{candidate}"]
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
        item = packed[f"{family}/{candidate}"]
        hold = item.get("hold_2025")
        clf = item.get("clf_2025")
        if hold is None or clf is None:
            continue
        X_te, y_te, names = hold
        perm[family] = {r["feature"]: r["brier_increase"] for r in _perm_brier(clf, X_te, y_te, names)}

    report = {
        "protocol": protocol,
        "features": feature_names,
        "comparisons": comparisons,
        "rows": [{k: v for k, v in row.items() if k not in {"hold_2025", "clf_2025"}} for row in rows],
        "perm": perm,
        "would_promote": any(row["pass"] for row in comparisons),
        "promoted": False,
        "note": "Do not carve a subset after seeing LOSO. Live TabPFN stays on extras.",
    }
    _write_json(root / "data" / "processed" / f"{stem}.json", report)
    _write_json(root / "web" / "public" / "data" / f"{stem}.json", report)
    _append_log(root, report, heading=heading, blurb=blurb, candidate=candidate, names=feature_names)
    print(f"  would_promote={report['would_promote']} live_promoted={report['promoted']}")
    return report


def run() -> dict:
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
            "2014–2025 FBS–FBS season once. Trees + logistic. Not live TabPFN."
        ),
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
