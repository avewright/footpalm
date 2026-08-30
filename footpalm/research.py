from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise SystemExit("could not find repo root")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")

EPS = 1e-6
PROMOTE_BRIER = 0.002


HOLDOUT_SEASON = 2025
LIVE_SEASON = 2026


def _prediction_seasons(root: Path) -> list[int]:
    seasons = []
    for path in (root / "web" / "public" / "data").glob("predictions-*.json"):
        try:
            seasons.append(int(path.stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(s for s in seasons if s < LIVE_SEASON)


def _raw_prob(game: dict) -> float:
    return float(game.get("home_win_prob_raw", game["home_win_prob"]))


def _load_games(root: Path, season: int) -> list[dict]:
    payload = json.loads((root / "web" / "public" / "data" / f"predictions-{season}.json").read_text())
    games = []
    for game in payload["games"]:
        if not game.get("fbs_fbs"):
            continue
        row = dict(game)
        row["home_win_prob"] = _raw_prob(game)
        games.append(row)
    return games


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, EPS, 1 - EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = _clip(p)
    return {
        "n": int(len(y)),
        "accuracy": round(float(np.mean((p >= 0.5) == y)), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
        "sharpness": round(float(np.mean(np.abs(p - 0.5))), 4),
        "reliability": round(float(np.mean(p * (1 - p))), 4),
    }


def _arrays(games: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.array([g["home_won"] for g in games], dtype=float)
    p = np.array([g["home_win_prob"] for g in games], dtype=float)
    m = np.array([g["pred_margin"] for g in games], dtype=float)
    return y, p, m


def _fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    z = _logit(p)

    def loss(t: float) -> float:
        if t <= 0.2:
            return 1e9
        q = _clip(_sigmoid(z / t))
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    result = minimize_scalar(loss, bounds=(0.4, 2.5), method="bounded")
    return float(result.x)


def _fit_sigma(y: np.ndarray, margin: np.ndarray) -> float:
    def loss(s: float) -> float:
        if s <= 1:
            return 1e9
        q = _clip(_sigmoid(margin / s))
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    result = minimize_scalar(loss, bounds=(6.0, 28.0), method="bounded")
    return float(result.x)


def _fit_shrink(y: np.ndarray, p: np.ndarray, rate: float) -> float:
    def loss(w: float) -> float:
        q = _clip((1 - w) * p + w * rate)
        return float(np.mean((q - y) ** 2))

    result = minimize_scalar(loss, bounds=(0.0, 0.5), method="bounded")
    return float(result.x)


def _fit_platt(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    z = _logit(p)

    def loss(ab: np.ndarray) -> float:
        q = _clip(_sigmoid(ab[0] + ab[1] * z))
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    result = minimize(loss, x0=np.array([0.0, 1.0]), method="Nelder-Mead")
    return float(result.x[0]), float(result.x[1])


def apply(name: str, p: np.ndarray, margin: np.ndarray, params: dict) -> np.ndarray:
    if name == "identity":
        return p
    if name == "clip_05":
        return np.clip(p, 0.05, 0.95)
    if name == "clip_10":
        return np.clip(p, 0.10, 0.90)
    if name == "temperature":
        return _sigmoid(_logit(p) / params["T"])
    if name == "sigma":
        return _sigmoid(margin / params["s"])
    if name == "shrink":
        return (1 - params["w"]) * p + params["w"] * params["rate"]
    if name == "platt":
        return _sigmoid(params["a"] + params["b"] * _logit(p))
    raise ValueError(name)


def run() -> dict:
    root = _repo_root()
    seasons = _prediction_seasons(root)
    train_seasons = [s for s in seasons if s < HOLDOUT_SEASON]
    if HOLDOUT_SEASON not in seasons or not train_seasons:
        raise SystemExit(f"need predictions for years before {HOLDOUT_SEASON} and {HOLDOUT_SEASON}")

    train_games = [g for s in train_seasons for g in _load_games(root, s)]
    test_games = _load_games(root, HOLDOUT_SEASON)
    y_tr, p_tr, m_tr = _arrays(train_games)
    y_te, p_te, m_te = _arrays(test_games)
    rate = float(y_tr.mean())
    train_label = ",".join(str(s) for s in train_seasons)
    hold_label = str(HOLDOUT_SEASON)

    T = _fit_temperature(y_tr, p_tr)
    s = _fit_sigma(y_tr, m_tr)
    w = _fit_shrink(y_tr, p_tr, rate)
    a, b = _fit_platt(y_tr, p_tr)

    catalog = [
        ("identity", {}, "raw walk-forward probabilities"),
        ("clip_05", {}, "lock p into [0.05, 0.95]"),
        ("clip_10", {}, "lock p into [0.10, 0.90]"),
        ("temperature", {"T": round(T, 4)}, "sharpen/flatten via one temperature"),
        ("sigma", {"s": round(s, 3)}, "rebuild p from predicted margin with a fitted sigma"),
        ("shrink", {"w": round(w, 4), "rate": round(rate, 4)}, f"mix toward {train_label} home-win rate"),
        ("platt", {"a": round(a, 4), "b": round(b, 4)}, "two-parameter logistic calibration"),
    ]

    experiments = []
    baseline = None
    for name, params, note in catalog:
        train_p = apply(name, p_tr, m_tr, params)
        test_p = apply(name, p_te, m_te, params)
        row = {
            "id": name,
            "params_fit_on": train_label,
            "params": params,
            "note": note,
            "train": _metrics(y_tr, train_p),
            "holdout": _metrics(y_te, test_p),
        }
        if name == "identity":
            baseline = row["holdout"]
        else:
            row["delta_holdout_brier"] = round(row["holdout"]["brier"] - baseline["brier"], 4)
            row["delta_holdout_logloss"] = round(row["holdout"]["logloss"] - baseline["logloss"], 4)
            row["pass"] = (
                row["delta_holdout_brier"] <= -PROMOTE_BRIER
                and row["delta_holdout_logloss"] <= 0
            )
        experiments.append(row)

    passed = [e for e in experiments if e.get("pass")]
    passed.sort(key=lambda e: (len(e["params"]), e["holdout"]["brier"]))
    promoted = passed[0]["id"] if passed else "identity"

    T_oracle = _fit_temperature(y_te, p_te)
    a_oracle, b_oracle = _fit_platt(y_te, p_te)
    oracle = [
        {
            "id": "oracle_temperature",
            "params_fit_on": f"{hold_label} (cheat)",
            "params": {"T": round(T_oracle, 4)},
            "note": f"ceiling only. fit and score on {hold_label}",
            "holdout": _metrics(y_te, apply("temperature", p_te, m_te, {"T": T_oracle})),
            "pass": False,
        },
        {
            "id": "oracle_platt",
            "params_fit_on": f"{hold_label} (cheat)",
            "params": {"a": round(a_oracle, 4), "b": round(b_oracle, 4)},
            "note": f"ceiling only. fit and score on {hold_label}",
            "holdout": _metrics(y_te, apply("platt", p_te, m_te, {"a": a_oracle, "b": b_oracle})),
            "pass": False,
        },
    ]
    for row in oracle:
        row["delta_holdout_brier"] = round(row["holdout"]["brier"] - baseline["brier"], 4)
        row["train"] = _metrics(y_tr, apply(row["id"].replace("oracle_", ""), p_tr, m_tr, row["params"]))

    expected_if_calibrated = float(np.mean(p_te * (1 - p_te)))
    if promoted == "identity":
        conclusion = (
            f"Leave the raw probabilities. Fit on {train_label} did not beat {hold_label} "
            f"by the promotion rule."
        )
    else:
        winner = next(e for e in experiments if e["id"] == promoted)
        conclusion = (
            f"Promoted {promoted}. {hold_label} Brier {baseline['brier']} → {winner['holdout']['brier']}."
        )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "research/PROTOCOL.md",
        "train_seasons": train_seasons,
        "holdout_season": HOLDOUT_SEASON,
        "train_n": int(len(y_tr)),
        "holdout_n": int(len(y_te)),
        "promote_if": f"{hold_label} Brier drops by >= 0.002 and log loss does not rise",
        "baseline_holdout_brier": baseline["brier"] if baseline else None,
        "holdout_expected_brier_if_calibrated": round(expected_if_calibrated, 4),
        "note": (
            "mean(p(1-p)) is the Brier you would expect if holdout outcomes were as noisy as the "
            "probabilities claim. Actual Brier below that means the year was more predictable "
            "than the model admitted."
        ),
        "conclusion": conclusion,
        "promoted": promoted,
        "experiments": experiments,
        "diagnostics": oracle,
        "trees": None,
    }
    try:
        from footpalm.trees import run as run_trees

        print("trees: LightGBM + XGBoost on the locked feature vector")
        report["trees"] = run_trees()
    except Exception as exc:
        report["trees"] = {"error": str(exc)}
        print(f"trees: {exc}")

    _write_json(root / "data" / "processed" / "research.json", report)
    _write_json(root / "web" / "public" / "data" / "research.json", report)
    (root / "research" / "LOG.md").write_text(_markdown(report))
    print(f"promoted {promoted}  train={train_label} n={len(y_tr)}  holdout={hold_label} n={len(y_te)}")
    for e in experiments + oracle:
        hold = e["holdout"]
        extra = f"  Δbrier {e['delta_holdout_brier']:+.4f}" if e.get("delta_holdout_brier") is not None else ""
        print(f"  {e['id']:<20} {hold_label} brier {hold['brier']:.4f} logloss {hold['logloss']:.4f}{extra}")
    if promoted != "identity":
        _publish_calibrated(root, report)
    return report


def _markdown(report: dict) -> str:
    train = ",".join(str(s) for s in report.get("train_seasons", []))
    hold = report.get("holdout_season", HOLDOUT_SEASON)
    lines = [
        "# Research log",
        "",
        f"Generated {report['generated_at']}. Fit on {train} (n={report.get('train_n')}), scored on {hold} (n={report.get('holdout_n')}).",
        "",
        f"Promoted: **{report['promoted']}**.",
        "",
        report.get("conclusion", ""),
        "",
        f"| id | params | {train} Brier | {hold} Brier | {hold} logloss | Δ Brier | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in report["experiments"] + report.get("diagnostics", []):
        params = e["params"] or "—"
        delta = e.get("delta_holdout_brier", "—")
        passed = e.get("pass", "baseline")
        lines.append(
            f"| {e['id']} | {params} | {e['train']['brier']} | {e['holdout']['brier']} | "
            f"{e['holdout']['logloss']} | {delta} | {passed} |"
        )
    lines += [
        "",
        f"{hold} expected Brier if calibrated: {report.get('holdout_expected_brier_if_calibrated')}.",
        "Promotion rule: holdout Brier must fall by at least 0.002 and log loss must not rise.",
        "oracle_* rows are a ceiling. They fit on the holdout year. Do not ship them.",
        "No NIL. No market line.",
        "",
    ]
    trees = report.get("trees") or {}
    if trees.get("models"):
        lines += ["## Trees (diagnostic)", ""]
        if trees.get("comparison", {}).get("rows"):
            lines += [
                "| family | locked Brier | set Brier | Δ Brier | pass |",
                "|---|---|---|---|---|",
            ]
            for row in trees["comparison"]["rows"]:
                lines.append(
                    f"| {row['family']} | {row['locked_brier']} | {row['full_brier']} | "
                    f"{row['delta_brier']} | {row['pass']} |"
                )
            lines.append("")
        for model in trees["models"]:
            top = ", ".join(f"{r['feature']} {r['brier_increase']:+.4f}" for r in model["permutation"][:3])
            lines.append(f"- {model['id']}: 2025 Brier {model['holdout']['brier']}. perm {top}")
        lines += ["", "Trees are not the live model.", ""]
    return "\n".join(lines) + "\n"


def apply_promoted(games: list[dict], report: dict) -> list[dict]:
    name = report["promoted"]
    params = next(e["params"] for e in report["experiments"] if e["id"] == name)
    if name == "identity":
        return games
    p = np.array([_raw_prob(g) for g in games], dtype=float)
    m = np.array([g["pred_margin"] for g in games], dtype=float)
    q = apply(name, p, m, params)
    out = []
    for game, raw, prob in zip(games, p, q, strict=True):
        row = dict(game)
        row["home_win_prob_raw"] = round(float(raw), 4)
        row["home_win_prob"] = round(float(prob), 4)
        row["calibration"] = name
        out.append(row)
    return out


def _refresh_backtest(bt: dict, games: list[dict]) -> dict:
    from footpalm.backtest import _calibration, _metrics

    fbs = [g for g in games if g.get("fbs_fbs")]
    slates = sorted({g["slate"] for g in fbs})
    by_week = []
    for slate in slates:
        part = [g for g in fbs if g["slate"] == slate]
        if not part:
            continue
        by_week.append({"slate": slate, "week": part[0]["week"], **_metrics(part, "week")})
    bt["all_fbs"] = _metrics(fbs, "fbs")
    bt["tabpfn"] = _metrics([g for g in fbs if g["engine"] == "tabpfn-3"], "tabpfn")
    bt["logistic"] = _metrics([g for g in fbs if g["engine"] == "logistic"], "logistic")
    bt["calibration"] = _calibration(fbs)
    bt["by_week"] = by_week
    return bt


def _publish_calibrated(root: Path, report: dict) -> None:
    summary_path = root / "web" / "public" / "data" / "backtest-summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"seasons": []}
    by_season = {row["season"]: row for row in summary.get("seasons", [])}
    for season in _prediction_seasons(root):
        pred_name = f"predictions-{season}.json"
        payload = json.loads((root / "web" / "public" / "data" / pred_name).read_text())
        payload["games"] = apply_promoted(payload["games"], report)
        payload["calibration"] = {"id": report["promoted"], "params": next(e["params"] for e in report["experiments"] if e["id"] == report["promoted"])}
        for dest in (root / "data" / "processed" / pred_name, root / "web" / "public" / "data" / pred_name):
            _write_json(dest, payload)
        bt_name = f"backtest-{season}.json"
        bt_path = root / "web" / "public" / "data" / bt_name
        if not bt_path.exists():
            continue
        bt = _refresh_backtest(json.loads(bt_path.read_text()), payload["games"])
        bt["calibration_applied"] = payload["calibration"]
        for dest in (root / "data" / "processed" / bt_name, bt_path):
            _write_json(dest, bt)
        if season in by_season:
            by_season[season]["fbs"] = bt["all_fbs"]
            by_season[season]["tabpfn"] = bt["tabpfn"]
            by_season[season]["logistic"] = bt["logistic"]
    summary["seasons"] = [by_season[s] for s in sorted(by_season)]
    summary["calibration"] = {
        "id": report["promoted"],
        "params": next(e["params"] for e in report["experiments"] if e["id"] == report["promoted"]),
        "fit_on": report.get("train_seasons"),
    }
    for dest in (root / "data" / "processed" / "backtest-summary.json", summary_path):
        _write_json(dest, summary)
    print(f"applied {report['promoted']} {summary['calibration']['params']} to displayed probabilities")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
