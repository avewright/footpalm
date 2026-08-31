from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from footpalm.featurize import load_prediction_games
from footpalm.form import ALL_NAMES, LOSO_ALL, LOSO_NAMES
from footpalm.losopass import CLIP, TREE_FAMILIES, _predict_family, _tabpfn_cache
from footpalm.research import PROMOTE_BRIER, _fit_temperature, apply
from footpalm.trees import _load_features, _metrics, _perm_brier, _repo_root, _write_json

SETS = (
    ("extras", "X_full", ALL_NAMES),
    ("extras+loso", "X_loso", LOSO_ALL),
)
FAMILIES = (*TREE_FAMILIES, "tabpfn")
MIN_YEARS_BETTER = 8
SLICE_EARLY = 3
SLICE_MID = 8


def expanding_masks(season: np.ndarray, fbs: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    years = sorted(int(y) for y in np.unique(season))
    folds = []
    for year in years:
        hold = (season == year) & fbs
        train = season < year
        if not np.any(hold) or not np.any(train):
            continue
        folds.append((year, train, hold))
    return folds


def slice_name(week: int) -> str:
    if week <= SLICE_EARLY:
        return "early"
    if week <= SLICE_MID:
        return "mid"
    return "late"


def slice_metrics(y: np.ndarray, p: np.ndarray, week: np.ndarray) -> dict[str, dict]:
    out = {}
    for name, mask in (
        ("early", week <= SLICE_EARLY),
        ("mid", (week > SLICE_EARLY) & (week <= SLICE_MID)),
        ("late", week > SLICE_MID),
    ):
        if np.any(mask):
            out[name] = _metrics(y[mask], p[mask])
    return out


def season_stability(extras_seasons: list[dict], full_seasons: list[dict]) -> dict:
    left = {int(s["season"]): float(s["brier"]) for s in extras_seasons}
    right = {int(s["season"]): float(s["brier"]) for s in full_seasons}
    years = sorted(set(left) & set(right))
    deltas = [right[year] - left[year] for year in years]
    if not deltas:
        return {"years": 0, "years_better": 0, "median_season_delta": 0.0}
    return {
        "years": len(years),
        "years_better": int(sum(d < -1e-9 for d in deltas)),
        "median_season_delta": round(float(np.median(deltas)), 4),
    }


def promote_row(extras: dict, full: dict) -> dict:
    drop = extras["pooled"]["brier"] - full["pooled"]["brier"]
    logloss_ok = full["pooled"]["logloss"] <= extras["pooled"]["logloss"]
    stab = season_stability(extras["seasons"], full["seasons"])
    stable = stab["median_season_delta"] < 0 or stab["years_better"] >= MIN_YEARS_BETTER
    passed = drop >= PROMOTE_BRIER and logloss_ok and stable
    return {
        "family": extras["engine"],
        "locked_brier": extras["pooled"]["brier"],
        "full_brier": full["pooled"]["brier"],
        "locked_logloss": extras["pooled"]["logloss"],
        "full_logloss": full["pooled"]["logloss"],
        "delta_brier": round(full["pooled"]["brier"] - extras["pooled"]["brier"], 4),
        "years": stab["years"],
        "years_better": stab["years_better"],
        "median_season_delta": stab["median_season_delta"],
        "pass": passed,
    }


def _weeks(root: Path, n: int) -> np.ndarray:
    games = load_prediction_games(root)
    if len(games) != n:
        raise SystemExit(f"prediction games n={len(games)} != feature rows {n}")
    return np.array([int(g.get("week") or 0) for g in games], dtype=int)


def _append_log(root: Path, report: dict, *, heading: str, blurb: str, candidate: str) -> None:
    log = root / "research" / "LOG.md"
    lines = [
        "",
        f"## {heading}",
        "",
        blurb,
        "",
        f"| family | extras | {candidate} | Δ | years better | median Δ | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report.get("comparisons", []):
        lines.append(
            f"| {row['family']} | {row['locked_brier']} | {row['full_brier']} | "
            f"{row['delta_brier']} | {row.get('years_better', '')}/{row.get('years', '')} | "
            f"{row.get('median_season_delta', '')} | {row['pass']} |"
        )
    lines += ["", "| model | pooled Brier | pooled logloss | mean season Brier |", "|---|---|---|---|"]
    for row in report.get("rows", []):
        lines.append(
            f"| {row['id']} | {row['pooled']['brier']} | {row['pooled']['logloss']} | {row['mean_season_brier']} |"
        )
    lines += ["", "Pooled slices (weeks ≤3 / 4–8 / ≥9):", "", "| model | early | mid | late |", "|---|---|---|---|"]
    for row in report.get("rows", []):
        sl = row.get("slices") or {}
        lines.append(
            f"| {row['id']} | {sl.get('early', {}).get('brier', '')} | "
            f"{sl.get('mid', {}).get('brier', '')} | {sl.get('late', {}).get('brier', '')} |"
        )
    lines += [
        "",
        f"would_promote={report.get('would_promote')} live_promoted={report.get('promoted')}. "
        "Do not carve a subset after seeing the expanding-year score.",
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
    week = _weeks(root, len(season))
    folds = expanding_masks(season, fbs)
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
            oof_w: list[np.ndarray] = []
            season_rows = []
            hold_2025 = None
            clf_2025 = None
            for year, train, hold in folds:
                X_tr, y_tr, m_tr = X[train], y[train], margin[train]
                X_te, y_te = X[hold], y[hold]
                cache = _tabpfn_cache(root, f"expanding-{set_name}", year) if family == "tabpfn" else None
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
                oof_w.append(week[hold])
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
            w_all = np.concatenate(oof_w)
            pooled = _metrics(y_all, p_all)
            packed_row = {
                "id": f"{family}/{set_name}",
                "engine": family,
                "set": set_name,
                "mode": "expanding-year",
                "pooled": pooled,
                "pooled_T": _metrics(y_all, np.concatenate(oof_t)),
                "pooled_clip": _metrics(y_all, np.concatenate(oof_c)),
                "mean_season_brier": round(float(np.mean([r["brier"] for r in season_rows])), 4),
                "seasons": season_rows,
                "slices": slice_metrics(y_all, p_all, w_all),
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
        row = promote_row(extras, full)
        comparisons.append(row)
        print(
            f"  {family:<10} extras {row['locked_brier']:.4f} → {candidate} {row['full_brier']:.4f}  "
            f"Δ {row['delta_brier']:+.4f}  {row['years_better']}/{row['years']} years  "
            f"median Δ {row['median_season_delta']:+.4f}  pass={row['pass']}",
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
        "mode": "expanding-year",
        "features": feature_names,
        "comparisons": all_comp,
        "rows": keep_rows + [{k: v for k, v in row.items() if k not in {"hold_2025", "clf_2025"}} for row in rows],
        "perm": perm,
        "would_promote": any(row["pass"] for row in all_comp if row["family"] in TREE_FAMILIES),
        "promoted": False,
        "note": (
            "Expanding-year fit: train season < hold. Peek-LOSO is an audit only. "
            "Live TabPFN stays on extras until a slate walk-forward also clears the bar."
        ),
    }
    for dest in dests:
        _write_json(dest, report)
    _append_log(root, report, heading=heading, blurb=blurb, candidate=candidate)
    print(f"  would_promote={report['would_promote']} live_promoted={report['promoted']}")
    return report


def run(*, families: tuple[str, ...] = TREE_FAMILIES, merge_existing: bool = False) -> dict:
    tabpfn = "tabpfn" in families
    return score_sets(
        SETS,
        candidate="extras+loso",
        feature_names=LOSO_NAMES,
        heading="Expanding-year (leak-free)",
        blurb=(
            "Same extras vs extras+loso menu. Fit only on earlier seasons "
            "(`season < Y`), score that year's FBS–FBS. 2014 is train-only. "
            "Not peek-LOSO. Not live TabPFN"
            + (" until a slate walk-forward confirm." if tabpfn else ".")
        ),
        stem="walkpass",
        protocol=(
            "Expanding-year screen. train = season < hold. Score 2015–2025 FBS–FBS. "
            "Walk-forward features only. Promote if pooled Brier drops ≥ 0.002, "
            "log loss does not rise, and the drop is not one year "
            f"(median season Δ < 0 or ≥ {MIN_YEARS_BETTER} years better). "
            + ("TabPFN-3 expanding-year, not slate walk-forward." if tabpfn else "Trees + logistic. Not live TabPFN.")
        ),
        families=families,
        merge_existing=merge_existing,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabpfn-only", action="store_true", help="score TabPFN-3 and merge into walkpass.json")
    parser.add_argument("--tabpfn", action="store_true", help="trees + logistic + TabPFN-3")
    args = parser.parse_args()
    if args.tabpfn_only:
        run(families=("tabpfn",), merge_existing=True)
    elif args.tabpfn:
        run(families=FAMILIES)
    else:
        run()


if __name__ == "__main__":
    main()
