from __future__ import annotations

import argparse
import json
from pathlib import Path

from footpalm.backtest import nil_residual_check, walk_forward
from footpalm.fetch import DEFAULT_SEASONS, LIVE_SEASON, ensure_pbp
from footpalm.graph import build_graph
from footpalm.nil import attach_money, dump_money, harvest_nil, money_map
from footpalm.plays import game_observations, load_pbp
from footpalm.project import project_live, save_book, save_history
from footpalm.form import FormBook, apply_sides
from footpalm.rate import fit_ratings, publish_table


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise SystemExit("could not find repo root")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in payload.items() if not str(k).startswith("_")}
    path.write_text(json.dumps(clean, indent=2) + "\n")
    n = len(payload.get("teams") or payload.get("games") or payload.get("nodes") or [])
    print(f"wrote {path} ({n})")


def _publish(root: Path, name: str, payload: dict) -> None:
    write_json(root / "data" / "processed" / name, payload)
    write_json(root / "web" / "public" / "data" / name, payload)


def build(seasons: list[int], use_tabpfn: bool = True) -> None:
    root = repo_root()
    print("harvesting NIL / spend", flush=True)
    harvested = harvest_nil()
    print(f"NIL harvest: {len(harvested['roster'])} published rosters, {len(harvested['spend'])} spend rows", flush=True)

    prior = None
    prior_history = None
    form = FormBook()
    pred_form = FormBook()
    backtests = []
    money_by_season: dict[int, list[dict]] = {}

    for season in seasons:
        path = ensure_pbp(root, season)
        print(f"rating {season} from {path.name}")
        plays = load_pbp(path)
        sides, fbs, conferences = game_observations(plays)
        form.new_season()
        book = fit_ratings(sides, fbs, conferences, season, prior=prior)
        apply_sides(form, sides, book)
        table = publish_table(book, sides, elo=form.snapshot())
        table["teams"] = attach_money(table["teams"], harvested)
        money_by_season[season] = table["teams"]
        _publish(root, f"ratings-{season}.json", table)
        top = table["teams"][:5]
        print("  " + ", ".join(f"{t['rank']}. {t['team']} {t['pom']:+.1f}" for t in top))

        graph = build_graph(sides, fbs, conferences, table, money_map(table["teams"]))
        _publish(root, f"graph-{season}.json", graph)

        print(f"walk-forward {season} (tabpfn={use_tabpfn})")
        pred_form.new_season()
        bt = walk_forward(
            sides,
            fbs,
            conferences,
            season,
            prior=prior,
            prior_Xy=prior_history,
            use_tabpfn=use_tabpfn,
            form=pred_form,
        )
        prior_history = bt.pop("_history", prior_history)
        backtests.append(bt)
        _publish(root, f"predictions-{season}.json", {"season": season, "games": bt["games"]})
        _publish(root, f"backtest-{season}.json", {k: v for k, v in bt.items() if k != "games"})
        print(f"  FBS {bt['all_fbs']} engines {bt['engine_counts']}")
        if bt.get("tabpfn_error"):
            print(f"  tabpfn: {bt['tabpfn_error']}")

        save_book(root, book)
        prior = book

    if money_by_season:
        latest = money_by_season[max(money_by_season)]
        dump_money(root / "data" / "processed" / "money.json", harvested, latest)
        dump_money(root / "web" / "public" / "data" / "money.json", harvested, latest)

    by_season = {bt["season"]: bt for bt in backtests}
    if 2024 in by_season and 2025 in by_season:
        residual = nil_residual_check(
            [g for g in by_season[2024]["games"] if g["fbs_fbs"]],
            [g for g in by_season[2025]["games"] if g["fbs_fbs"]],
            money_map(money_by_season[2025]),
        )
    else:
        residual = {"used": False, "reason": "need 2024 and 2025"}

    summary = {
        "seasons": [
            {
                "season": bt["season"],
                "engines": bt["engine_counts"],
                "fbs": bt["all_fbs"],
                "tabpfn": bt["tabpfn"],
                "logistic": bt["logistic"],
            }
            for bt in backtests
        ],
        "nil_residual": residual,
        "rule": "TabPFN-3 is the live model when it has enough prior games. Logistic is the locked baseline. NIL is a holdout diagnostic only.",
    }
    _publish(root, "backtest-summary.json", summary)
    _publish(root, "index.json", {"seasons": [{"season": s, "teams": len(money_by_season[s])} for s in seasons]})
    pred_dir = root / "web" / "public" / "data"
    if prior_history is not None:
        save_history(root, prior_history[0], prior_history[1], prior_history[2])

    if (pred_dir / "predictions-2025.json").exists() and any(pred_dir.glob("predictions-20*.json")):
        from footpalm.research import run as run_research

        print("research: fit on seasons before 2025, score 2025")
        run_research()

    if prior is not None and book_ready_for_live(root):
        print(f"project {LIVE_SEASON}")
        project_live(root, season=LIVE_SEASON, refresh=False, use_tabpfn=use_tabpfn)


def book_ready_for_live(root: Path) -> bool:
    from footpalm.project import book_path, history_path

    return book_path(root, LIVE_SEASON - 1).exists() and history_path(root).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FootPom ratings, predictions, and backtests")
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    parser.add_argument("--no-tabpfn", action="store_true", help="logistic baseline only")
    args = parser.parse_args()
    build(args.seasons, use_tabpfn=not args.no_tabpfn)


if __name__ == "__main__":
    main()
