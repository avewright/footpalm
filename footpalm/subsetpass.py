from __future__ import annotations

from pathlib import Path

import numpy as np

from footpalm.form import ALL_NAMES
from footpalm.pace import PACE_NAMES
from footpalm.predict import fit_tabpfn
from footpalm.research import _fit_temperature, apply
from footpalm.specials import SPECIAL_NAMES
from footpalm.suite import _row, _tabpfn_probs
from footpalm.trees import _fit_lgbm, _fit_xgb, _load_features, _metrics, _repo_root, _write_json

INNER = 2024
HOLDOUT = 2025
CANDIDATES = PACE_NAMES + SPECIAL_NAMES
GROUPS = {
    "pace": list(PACE_NAMES),
    "specials": list(SPECIAL_NAMES),
    "clocks": ["play_speed_diff", "sec_per_play_diff"],
    "rush": ["ypc_diff"],
    "ball": ["to_margin_diff"],
    "punts": ["punt_rate_diff", "punt_yds_diff"],
    "kicks": ["fg_avg_make_diff", "fg_make_adj_diff"],
    "form": ["margin_momentum_diff", "win_streak_diff"],
    "plays": ["plays_pg_diff"],
    "all": list(CANDIDATES),
}


def _year_split(season: np.ndarray, fbs: np.ndarray, y: np.ndarray, m: np.ndarray, year: int):
    train = season < year
    hold = (season == year) & fbs
    return train, hold, y[train], m[train], y[hold]


def _cols(names: list[str], extra: list[str]) -> list[int]:
    idx = {n: i for i, n in enumerate(names)}
    return list(range(len(ALL_NAMES))) + [idx[n] for n in extra]


def _score(fit, X: np.ndarray, y_tr: np.ndarray, m_tr: np.ndarray, y_te: np.ndarray, train: np.ndarray, hold: np.ndarray) -> dict:
    clf, _reg = fit(X[train], y_tr, m_tr)
    p = clf.predict_proba(X[hold])[:, 1]
    return _metrics(y_te, p)


def _scan(payload: dict, year: int) -> dict:
    season = np.asarray(payload["season"])
    fbs = np.asarray(payload["fbs"])
    y = np.asarray(payload["y_win"], dtype=float)
    m = np.asarray(payload["y_margin"], dtype=float)
    X = np.concatenate([payload["X_full"], payload["pace"], payload["specials"]], axis=1)
    names = ALL_NAMES + CANDIDATES
    train, hold, y_tr, m_tr, y_te = _year_split(season, fbs, y, m, year)
    extras = X[:, : len(ALL_NAMES)]
    rows = []

    def add(label: str, cols: list[int], kind: str) -> None:
        for family, fit in (("lightgbm", _fit_lgbm), ("xgboost", _fit_xgb)):
            met = _score(fit, X[:, cols], y_tr, m_tr, y_te, train, hold)
            rows.append({"id": f"{family}/{label}", "family": family, "set": label, "kind": kind, **met})
            print(f"  {year} {family}/{label:<28} brier {met['brier']:.4f}  n={met['n']}", flush=True)

    add("extras", list(range(len(ALL_NAMES))), "baseline")
    for label, extra in GROUPS.items():
        add(label, _cols(names, extra), "group")
    for name in CANDIDATES:
        add(name, _cols(names, [name]), "single")
    for name in CANDIDATES:
        keep = [n for n in CANDIDATES if n != name]
        add(f"all-{name}", _cols(names, keep), "drop")
    return {"year": year, "n": int(hold.sum()), "rows": rows, "X": X, "names": names, "extras": extras}


def _delta(rows: list[dict], family: str, label: str, baseline: str = "extras") -> float | None:
    base = next((r for r in rows if r["id"] == f"{family}/{baseline}"), None)
    row = next((r for r in rows if r["id"] == f"{family}/{label}"), None)
    if not base or not row:
        return None
    return round(row["brier"] - base["brier"], 4)


def _best_group(rows: list[dict]) -> str:
    scored = []
    for label in GROUPS:
        ds = [_delta(rows, fam, label) for fam in ("lightgbm", "xgboost")]
        if any(d is None for d in ds):
            continue
        scored.append((sum(ds) / 2, label))
    scored.sort()
    return scored[0][1] if scored else "all"


def _tabpfn_batch(X: np.ndarray, season: np.ndarray, fbs: np.ndarray, y: np.ndarray, m: np.ndarray, label: str) -> list[dict]:
    train = season < HOLDOUT
    hold = (season == HOLDOUT) & fbs
    print(f"subset tabpfn batch {label}", flush=True)
    model = fit_tabpfn(X[train], y[train], m[train])
    if not model.ready:
        print(f"  tabpfn/{label} failed {model.error}", flush=True)
        return []
    tail = min(400, model.n_train)
    p_tr = _tabpfn_probs(model, X[train][-tail:])
    p_te = _tabpfn_probs(model, X[hold])
    row = _row(f"tabpfn/{label}", y[train][-tail:], p_tr, y[hold], p_te, {"engine": "tabpfn", "set": label})
    t = _fit_temperature(y[train][-tail:], p_tr)
    p_cal = apply("temperature", p_te, p_te, {"T": t})
    row_t = _row(
        f"tabpfn/{label}+T",
        y[train][-tail:],
        apply("temperature", p_tr, p_tr, {"T": t}),
        y[hold],
        p_cal,
        {"engine": "tabpfn", "set": label, "T": round(float(t), 4)},
    )
    print(
        f"  tabpfn/{label:<16} 2025 brier {row['holdout']['brier']:.4f}  "
        f"+T {row_t['holdout']['brier']:.4f}",
        flush=True,
    )
    return [row, row_t]


def run() -> dict:
    root = _repo_root()
    payload = _load_features(root)
    print(f"subset inner {INNER} (select)", flush=True)
    inner = _scan(payload, INNER)
    pick = _best_group(inner["rows"])
    print(f"  2024-best group={pick}", flush=True)
    print(f"subset holdout {HOLDOUT} (frozen after 2024 pick)", flush=True)
    hold = _scan(payload, HOLDOUT)
    season = np.asarray(payload["season"])
    fbs = np.asarray(payload["fbs"])
    y = np.asarray(payload["y_win"], dtype=float)
    m = np.asarray(payload["y_margin"], dtype=float)
    tabpfn: list[dict] = []

    def pack_year(scan: dict) -> list[dict]:
        out = []
        for label in ["extras", *GROUPS, *CANDIDATES, *[f"all-{n}" for n in CANDIDATES]]:
            item = {"set": label, "kind": "baseline" if label == "extras" else "group" if label in GROUPS else "drop" if label.startswith("all-") else "single"}
            for family in ("lightgbm", "xgboost"):
                row = next((r for r in scan["rows"] if r["id"] == f"{family}/{label}"), None)
                if not row:
                    continue
                item[family] = {"brier": row["brier"], "logloss": row["logloss"], "n": row["n"]}
                if label != "extras":
                    item[f"{family}_delta"] = _delta(scan["rows"], family, label)
            out.append(item)
        return out

    report = {
        "protocol": (
            f"Subset scan. Groups locked before the score. Select on {INNER} "
            f"(train < {INNER}). Score {HOLDOUT} after the {INNER} ranking. Not live."
        ),
        "candidates": CANDIDATES,
        "groups": GROUPS,
        "inner_year": INNER,
        "holdout_year": HOLDOUT,
        "inner_best_group": pick,
        "inner": pack_year(inner),
        "holdout": pack_year(hold),
        "tabpfn": tabpfn,
        "promoted": False,
        "note": "Do not promote a 2025 winner that was not the 2024 pick. Do not carve further.",
    }
    _write_json(root / "data" / "processed" / "subsetpass.json", report)
    _write_json(root / "web" / "public" / "data" / "subsetpass.json", report)
    _append_log(root, report)
    return report


def _append_log(root: Path, report: dict) -> None:
    log = root / "research" / "LOG.md"
    lines = [
        "",
        "## Subsets (diagnostic, not live)",
        "",
        f"2024 selects. 2025 scores after. 2024-best group=`{report['inner_best_group']}`.",
        "",
        "| set | 2024 LGBM Δ | 2024 XGB Δ | 2025 LGBM Δ | 2025 XGB Δ |",
        "|---|---|---|---|---|",
    ]
    inner = {r["set"]: r for r in report["inner"]}
    hold = {r["set"]: r for r in report["holdout"]}
    for label in ["extras", *GROUPS, *CANDIDATES]:
        a, b = inner.get(label, {}), hold.get(label, {})
        lines.append(
            f"| {label} | {a.get('lightgbm_delta', '')} | {a.get('xgboost_delta', '')} | "
            f"{b.get('lightgbm_delta', '')} | {b.get('xgboost_delta', '')} |"
        )
    lines += ["", "Drop-one from all (Δ vs extras):", "", "| dropped | 2024 LGBM Δ | 2025 LGBM Δ |", "|---|---|---|"]
    for name in CANDIDATES:
        label = f"all-{name}"
        lines.append(
            f"| {name} | {inner.get(label, {}).get('lightgbm_delta', '')} | "
            f"{hold.get(label, {}).get('lightgbm_delta', '')} |"
        )
    lines += ["", f"promoted={report.get('promoted')}. Not a promotion pass.", "",]
    existing = log.read_text() if log.exists() else ""
    marker = "## Subsets (diagnostic"
    if marker in existing:
        start = existing.index(marker)
        existing = existing[:start].rstrip() + "\n"
    log.write_text(existing + "\n".join(lines))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
