from __future__ import annotations

from pathlib import Path

import numpy as np

from footpalm.featurize import load_prediction_games
from footpalm.predict import MAX_TABPFN_ROWS, fit_tabpfn
from footpalm.quarters import (
    ALL_SLICES,
    HALVES,
    HALVES_THREE,
    PAIRS,
    THREE_CONSEC,
    expand_rows,
    load_quarters,
    take_block,
)
from footpalm.research import PROMOTE_BRIER, _fit_temperature, apply
from footpalm.suite import _row, _tabpfn_probs
from footpalm.trees import HOLDOUT_SEASON, _load_features, _repo_root, _write_json

# Experiment. Do not promote. Score real 2025 FBS–FBS only.
MENUS = {
    "qscale": {
        "stem": "qscalepass",
        "heading": "QScale (experiment, not live)",
        "blurb": (
            "Synthetic train rows from regulation quarter scores, scaled to 4 quarters. "
            "Same extras X. TabPFN-3 batch, 2014–2024 train, 2025 FBS–FBS once. "
            "T fit on real train labels in the fitted window."
        ),
        "protocol": (
            "QScale experiment. Extra train rows from C(4,1)×4 and C(4,2)×2 regulation "
            "quarter scores. Same extras X. TabPFN-3 last 8000. Score real 2025 FBS–FBS. Not live."
        ),
        "variants": (
            ("extras", None, "real"),
            ("qscale", ALL_SLICES, "tail"),
            ("qscale-pairs", PAIRS, "tail"),
            ("qscale-block", ALL_SLICES, "block"),
        ),
    },
    "consec": {
        "stem": "qscale2pass",
        "heading": "QScale2 (experiment, not live)",
        "blurb": (
            "Consecutive clock only: real halves (×2) and consecutive three-quarter groups (×4/3). "
            "Same extras X. TabPFN-3 batch, 2014–2024 train, 2025 FBS–FBS once. "
            "T fit on real train labels in the fitted window."
        ),
        "protocol": (
            "QScale2 experiment. Halves Q1+Q2 / Q3+Q4 and consecutive threes Q1–Q3 / Q2–Q4. "
            "Same extras X. TabPFN-3 last 8000. Score real 2025 FBS–FBS. Not live."
        ),
        "variants": (
            ("extras", None, "real"),
            ("qscale-halves", HALVES, "tail"),
            ("qscale-three", THREE_CONSEC, "tail"),
            ("qscale-halves+three", HALVES_THREE, "tail"),
        ),
    },
}


def _game_ids(root: Path, n: int) -> np.ndarray:
    games = load_prediction_games(root)
    if len(games) != n:
        raise SystemExit(f"prediction games {len(games)} != features {n}")
    ids = np.full(n, np.nan)
    for i, game in enumerate(games):
        gid = game.get("game_id")
        if gid is not None:
            ids[i] = int(gid)
    return ids


def _pack_train(
    X: np.ndarray,
    y: np.ndarray,
    m: np.ndarray,
    ids: np.ndarray,
    quarters: dict[int, tuple[np.ndarray, np.ndarray]],
    slices: tuple[tuple[int, ...], ...] | None,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if slices is None:
        real = np.ones(len(X), dtype=bool)
        return X, y, m, real
    Xe, ye, me, real = expand_rows(X, y, m, ids, quarters, slices=slices)
    if mode == "block":
        return take_block(Xe, ye, me, real, cap=MAX_TABPFN_ROWS)
    return Xe, ye, me, real


def _score_variant(
    name: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_m: np.ndarray,
    real: np.ndarray,
    hold_X: np.ndarray,
    hold_y: np.ndarray,
) -> list[dict]:
    print(f"qscale tabpfn {name} n={len(train_X)} real={int(real.sum())}", flush=True)
    model = fit_tabpfn(train_X, train_y, train_m)
    if not model.ready:
        return [{"id": f"tabpfn/{name}", "error": model.error, "engine": "tabpfn", "set": name, "experiment": True}]
    used = min(len(train_X), model.n_train)
    window = slice(-used, None)
    real_window = real[window]
    X_win, y_win = train_X[window], train_y[window]
    p_all = _tabpfn_probs(model, X_win)
    if np.any(real_window):
        p_tr, y_tr = p_all[real_window], y_win[real_window]
    else:
        p_tr, y_tr = p_all[-400:], y_win[-400:]
    p_te = _tabpfn_probs(model, hold_X)
    row = _row(f"tabpfn/{name}", y_tr, p_tr, hold_y, p_te, {"engine": "tabpfn", "set": name, "experiment": True})
    row["n_train"] = int(model.n_train)
    row["n_train_real"] = int(real_window.sum())
    t = _fit_temperature(y_tr, p_tr)
    p_te_t = apply("temperature", p_te, p_te, {"T": t})
    cal = _row(
        f"tabpfn/{name}+T",
        y_tr,
        apply("temperature", p_tr, p_tr, {"T": t}),
        hold_y,
        p_te_t,
        {"engine": "tabpfn", "set": name, "T": round(float(t), 4), "experiment": True},
    )
    print(
        f"  tabpfn/{name:<16} 2025 brier {row['holdout']['brier']:.4f}  "
        f"logloss {row['holdout']['logloss']:.4f}  real_in_fit={row['n_train_real']}",
        flush=True,
    )
    return [row, cal]


def _compare(rows: list[dict]) -> list[dict]:
    base = next((r for r in rows if r["id"] == "tabpfn/extras"), None)
    out = []
    if not base or "holdout" not in base:
        return out
    for row in rows:
        if row["id"] == "tabpfn/extras" or "holdout" not in row or row["id"].endswith("+T"):
            continue
        delta = row["holdout"]["brier"] - base["holdout"]["brier"]
        ll_ok = row["holdout"]["logloss"] <= base["holdout"]["logloss"]
        out.append(
            {
                "family": "tabpfn",
                "id": row["id"],
                "locked_brier": base["holdout"]["brier"],
                "full_brier": row["holdout"]["brier"],
                "delta_brier": round(float(delta), 4),
                "pass": bool(delta <= -PROMOTE_BRIER and ll_ok),
            }
        )
    return out


def _append_log(root: Path, report: dict) -> None:
    log = root / "research" / "LOG.md"
    heading = report["heading"]
    lines = [
        "",
        f"## {heading}",
        "",
        report["blurb"],
        "",
        "| model | 2025 Brier | logloss | n_train | n_real |",
        "|---|---|---|---|---|",
    ]
    for row in report.get("rows", []):
        hold = row.get("holdout")
        if not hold:
            continue
        lines.append(
            f"| {row['id']} | {hold['brier']} | {hold['logloss']} | "
            f"{row.get('n_train', '')} | {row.get('n_train_real', '')} |"
        )
    lines += ["", "| id | extras | candidate | Δ | pass |", "|---|---|---|---|---|"]
    for row in report.get("comparisons", []):
        lines.append(
            f"| {row['id']} | {row['locked_brier']} | {row['full_brier']} | "
            f"{row['delta_brier']} | {row['pass']} |"
        )
    lines += [
        "",
        f"would_promote={report.get('would_promote')} live_promoted={report.get('promoted')}. "
        "Do not carve a subset after seeing 2025.",
        "",
    ]
    existing = log.read_text() if log.exists() else ""
    marker = f"## {heading}"
    if marker in existing:
        start = existing.index(marker)
        existing = existing[:start].rstrip() + "\n"
    log.write_text(existing + "\n".join(lines))


def run(*, rebuild_quarters: bool = False, menu: str = "qscale") -> dict:
    root = _repo_root()
    payload = _load_features(root)
    season = np.asarray(payload["season"])
    fbs = np.asarray(payload["fbs"])
    X = np.asarray(payload["X_full"], dtype=float)
    y = np.asarray(payload["y_win"], dtype=float)
    m = np.asarray(payload["y_margin"], dtype=float)
    ids = _game_ids(root, len(X))
    quarters = load_quarters(root, rebuild=rebuild_quarters)

    train = season < HOLDOUT_SEASON
    hold = (season == HOLDOUT_SEASON) & fbs
    hold_X, hold_y = X[hold], y[hold]
    matched = sum(1 for gid in ids[train] if np.isfinite(gid) and int(gid) in quarters)

    spec = MENUS[menu]
    rows = []
    for name, slices, mode in spec["variants"]:
        X_tr, y_tr, m_tr, real = _pack_train(X[train], y[train], m[train], ids[train], quarters, slices, mode)
        rows.extend(_score_variant(name, X_tr, y_tr, m_tr, real, hold_X, hold_y))

    comparisons = _compare(rows)
    would = any(c["pass"] for c in comparisons)
    report = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "experiment": True,
        "menu": menu,
        "heading": spec["heading"],
        "blurb": spec["blurb"],
        "promoted": False,
        "would_promote": would,
        "protocol": spec["protocol"],
        "coverage": {
            "train_games": int(train.sum()),
            "train_with_quarters": int(matched),
            "quarter_games": len(quarters),
            "hold_fbs": int(hold.sum()),
        },
        "rows": rows,
        "comparisons": comparisons,
        "note": "Experiment. Do not promote. Do not carve after 2025.",
    }
    dest = root / "web" / "public" / "data" / f"{spec['stem']}.json"
    _write_json(dest, report)
    _write_json(root / "data" / "processed" / f"{spec['stem']}.json", report)
    _append_log(root, report)
    print(f"wrote {dest} would_promote={would} live_promoted=False")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="QScale TabPFN-3 experiment (not live)")
    parser.add_argument("--rebuild-quarters", action="store_true")
    parser.add_argument("--menu", choices=sorted(MENUS), default="qscale")
    args = parser.parse_args()
    run(rebuild_quarters=args.rebuild_quarters, menu=args.menu)


if __name__ == "__main__":
    main()
