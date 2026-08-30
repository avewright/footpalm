from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from footpalm.form import ALL_NAMES, THIN_ALL, THIN_NAMES, thin_feature
from footpalm.predict import FEATURE_NAMES, MARGIN_SIGMA, MAX_TABPFN_ROWS
from footpalm.research import PROMOTE_BRIER
from footpalm.trees import _compare_pair, _fit_lgbm, _fit_xgb, _load_features, _metrics, _pack, _repo_root, _split, _write_json

HOME_GAMES = FEATURE_NAMES.index("home_games")
AWAY_GAMES = FEATURE_NAMES.index("away_games")
MAE_BAR = 0.3
BLEND_STEP = 0.1


def thin_column(X_locked: np.ndarray) -> np.ndarray:
    return np.array(
        [thin_feature(row[HOME_GAMES], row[AWAY_GAMES]) for row in X_locked],
        dtype=float,
    )


def fit_blend_weights(y: np.ndarray, parts: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    parts = np.asarray(parts, dtype=float)
    best_w = np.ones(parts.shape[1]) / parts.shape[1]
    best = np.inf
    grid = np.arange(0.0, 1.0 + 1e-9, BLEND_STEP)
    for w0 in grid:
        for w1 in grid:
            w2 = 1.0 - w0 - w1
            if w2 < -1e-9:
                continue
            w = np.array([w0, w1, max(w2, 0.0)])
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            p = np.clip(parts @ w, 1e-6, 1 - 1e-6)
            loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            if loss < best:
                best = loss
                best_w = w
    return best_w


def _tabpfn_job(train_p: Path, pred_p: Path, out_p: Path, heads: str) -> None:
    from footpalm.predict import fit_tabpfn

    train = np.load(train_p)
    model = fit_tabpfn(train["X"], train["y"], train["m"], heads=heads)
    if not model.ready:
        raise SystemExit(model.error or "tabpfn not ready")
    X_te = np.load(pred_p)
    rows = model.predict_many(X_te, 27.0)
    p = np.array([r["home_win_prob"] for r in rows], dtype=float)
    m = np.array([r["pred_margin"] for r in rows], dtype=float)
    derived = m
    if model.clf is not None:
        raw = np.clip(model.clf.predict_proba(X_te)[:, 1], 0.01, 0.99)
        derived = MARGIN_SIGMA * np.log(raw / (1 - raw))
    np.savez(
        out_p,
        p=p,
        m=m,
        derived=derived,
        n_train=np.array(model.n_train),
        has_reg=np.array(int(model.reg is not None)),
        has_clf=np.array(int(model.clf is not None)),
    )


def _spawn_tabpfn(X_tr, y_tr, m_tr, X_te, heads: str) -> dict[str, np.ndarray]:
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
                "footpalm.nextpass",
                "--tabpfn-job",
                "--train",
                str(train_p),
                "--pred",
                str(pred_p),
                "--out",
                str(out_p),
                "--heads",
                heads,
            ],
            check=True,
        )
        packed = np.load(out_p)
        return {key: packed[key] for key in packed.files}


def _score_thin(payload: dict) -> tuple[list[dict], list[dict]]:
    e_tr, e_y, e_m, e_ho, e_hy, e_hm, e_fbs = _split(payload, "X_full")
    l_tr, _, _, l_ho, _, _, _ = _split(payload, "X_locked")
    thin_tr = thin_column(l_tr)
    thin_ho = thin_column(l_ho)
    t_tr = np.column_stack([e_tr, thin_tr])
    t_ho = np.column_stack([e_ho, thin_ho])
    rows: list[dict] = []
    models: list[dict] = []
    for family, fit in (("lightgbm", _fit_lgbm), ("xgboost", _fit_xgb)):
        clf, reg = fit(e_tr, e_y, e_m)
        extra = _pack(
            f"{family}-extras",
            "baseline extras",
            list(ALL_NAMES),
            clf,
            reg,
            e_tr,
            e_y,
            e_ho,
            e_hy,
            e_hm,
            e_fbs,
            clf.feature_importances_,
        )
        clf_t, reg_t = fit(t_tr, e_y, e_m)
        thin = _pack(
            f"{family}-thin",
            "extras plus thin",
            list(THIN_ALL),
            clf_t,
            reg_t,
            t_tr,
            e_y,
            t_ho,
            e_hy,
            e_hm,
            e_fbs,
            clf_t.feature_importances_,
        )
        row = _compare_pair(extra, thin, "thin")
        row["family"] = family
        row["thin_rate_train"] = round(float(thin_tr.mean()), 4)
        row["thin_rate_hold"] = round(float(thin_ho[e_fbs].mean()), 4)
        rows.append(row)
        models.extend([extra, thin])
        print(
            f"  {family} extras {extra['holdout']['brier']:.4f} → thin {thin['holdout']['brier']:.4f} "
            f"Δ {row['delta_brier']:+.4f} pass={row['pass']}",
            flush=True,
        )
    return rows, models


def run() -> dict:
    root = _repo_root()
    payload = _load_features(root)
    train_X, train_y, train_m, hold_X, hold_y, hold_m, hold_fbs = _split(payload, "X_full")
    hold_X_fbs = hold_X[hold_fbs]
    hold_y_fbs = hold_y[hold_fbs]
    hold_m_fbs = hold_m[hold_fbs]

    print("thin: extras vs extras+thin", flush=True)
    thin_rows, thin_models = _score_thin(payload)
    thin_pass = any(row["pass"] for row in thin_rows)

    print("blend + tabpfn_margin: one TabPFN pair job", flush=True)
    lgbm_clf, _ = _fit_lgbm(train_X, train_y, train_m)
    xgb_clf, _ = _fit_xgb(train_X, train_y, train_m)
    n_tail = min(MAX_TABPFN_ROWS, len(train_y))
    X_pred = np.vstack([train_X[-n_tail:], hold_X_fbs])
    try:
        pair = _spawn_tabpfn(train_X, train_y, train_m, X_pred, heads="pair")
    except subprocess.CalledProcessError:
        print("  pair head unavailable (no local classifier). falling back to auto.", flush=True)
        pair = _spawn_tabpfn(train_X, train_y, train_m, X_pred, heads="auto")
    pred_m_all = np.asarray(pair["m"], dtype=float)
    has_reg = bool(int(np.asarray(pair["has_reg"])))
    has_clf = bool(int(np.asarray(pair["has_clf"])))
    if has_reg:
        p_board = 1 / (1 + np.exp(-pred_m_all / MARGIN_SIGMA))
    else:
        p_board = np.asarray(pair["p"], dtype=float)
    p_t_tr, p_t_te = p_board[:n_tail], p_board[n_tail:]
    p_clf_te = np.asarray(pair["p"], dtype=float)[n_tail:]
    derived = np.asarray(pair["derived"], dtype=float)[n_tail:]
    pred_m = pred_m_all[n_tail:]

    p_l_tr = lgbm_clf.predict_proba(train_X[-n_tail:])[:, 1]
    p_l_te = lgbm_clf.predict_proba(hold_X_fbs)[:, 1]
    p_x_tr = xgb_clf.predict_proba(train_X[-n_tail:])[:, 1]
    p_x_te = xgb_clf.predict_proba(hold_X_fbs)[:, 1]
    w = fit_blend_weights(train_y[-n_tail:], np.column_stack([p_l_tr, p_x_tr, p_t_tr]))
    equal_te = np.clip((p_l_te + p_x_te + p_t_te) / 3, 1e-6, 1 - 1e-6)
    weighted_te = np.clip(np.column_stack([p_l_te, p_x_te, p_t_te]) @ w, 1e-6, 1 - 1e-6)
    eq = _metrics(hold_y_fbs, equal_te)
    wt = _metrics(hold_y_fbs, weighted_te)
    blend_pass = eq["brier"] - wt["brier"] >= PROMOTE_BRIER and wt["logloss"] <= eq["logloss"]
    blend = {
        "weights": {
            "lightgbm": round(float(w[0]), 4),
            "xgboost": round(float(w[1]), 4),
            "tabpfn": round(float(w[2]), 4),
        },
        "fit_on": f"train tail {n_tail} log loss",
        "tabpfn": "board-shaped: sigmoid(regressor margin / 14.5)",
        "equal": eq,
        "weighted": wt,
        "delta_brier": round(wt["brier"] - eq["brier"], 4),
        "pass": blend_pass,
    }
    print(
        f"  weights {blend['weights']} equal {eq['brier']:.4f} → weighted {wt['brier']:.4f} "
        f"Δ {blend['delta_brier']:+.4f} pass={blend_pass}",
        flush=True,
    )

    brier = _metrics(hold_y_fbs, p_clf_te)
    if has_clf and has_reg:
        mae_derived = float(np.mean(np.abs(derived - hold_m_fbs)))
        mae_reg = float(np.mean(np.abs(pred_m - hold_m_fbs)))
        margin_pass = (mae_derived - mae_reg) >= MAE_BAR
        margin_note = "P(win) is the classifier. Promote Us margin only."
    else:
        mae_derived = float("nan")
        mae_reg = float("nan")
        margin_pass = False
        margin_note = (
            "Skipped. No local TabPFN classifier checkpoint; download needs a Prior Labs token. "
            "Did not write a classifier into the live cache."
        )
    margin = {
        "mae_derived": None if mae_derived != mae_derived else round(mae_derived, 3),
        "mae_regressor": None if mae_reg != mae_reg else round(mae_reg, 3),
        "delta_mae": None if mae_derived != mae_derived else round(mae_reg - mae_derived, 3),
        "brier": brier["brier"] if has_clf else None,
        "logloss": brier["logloss"] if has_clf else None,
        "has_regressor": has_reg,
        "has_classifier": has_clf,
        "pass": margin_pass,
        "note": margin_note,
    }
    if has_clf and has_reg:
        print(
            f"  MAE derived {mae_derived:.3f} → reg {mae_reg:.3f} Δ {mae_reg - mae_derived:+.3f} "
            f"brier {brier['brier']:.4f} pass={margin_pass}",
            flush=True,
        )
    else:
        print(f"  {margin_note}", flush=True)

    report = {
        "protocol": (
            "Next pass locked before the score. extras vs thin; equal blend vs train weights; "
            "TabPFN derived margin vs regressor. Not live."
        ),
        "thin": {
            "features": THIN_NAMES,
            "rows": thin_rows,
            "would_promote": thin_pass,
            "models": thin_models,
        },
        "blend_train": blend,
        "tabpfn_margin": margin,
        "promoted": False,
        "would_promote": thin_pass or blend_pass or margin_pass,
    }
    _write_json(root / "data" / "processed" / "nextpass.json", report)
    _write_json(root / "web" / "public" / "data" / "nextpass.json", report)
    print(f"  would_promote={report['would_promote']} live_promoted={report['promoted']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabpfn-job", action="store_true")
    parser.add_argument("--train")
    parser.add_argument("--pred")
    parser.add_argument("--out")
    parser.add_argument("--heads", default="auto")
    args = parser.parse_args()
    if args.tabpfn_job:
        if args.heads == "pair":
            cache = _repo_root() / "data" / "cache" / "tabpfn-research"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["TABPFN_MODEL_CACHE_DIR"] = str(cache)
        _tabpfn_job(Path(args.train), Path(args.pred), Path(args.out), args.heads)
        return
    run()


if __name__ == "__main__":
    main()
