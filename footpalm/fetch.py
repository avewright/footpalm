from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from footpalm.cfbd import pull as pull_cfbd

PBP_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "cfbfastR_cfb_pbp/play_by_play_{season}.parquet"
)

# cfbfastR has 2014–2025. 2026 is live projections, not a parquet dump.
DEFAULT_SEASONS = list(range(2014, 2026))
# 2026 is live. Cache it, but refetch with --refresh. Do not rate it from CFBD.
LIVE_SEASON = 2026


def raw_dir(root: Path) -> Path:
    path = root / "data" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pbp_path(root: Path, season: int) -> Path:
    return raw_dir(root) / f"play_by_play_{season}.parquet"


def ensure_pbp(root: Path, season: int) -> Path:
    dest = pbp_path(root, season)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    url = PBP_URL.format(season=season)
    print(f"downloading {url}")
    tmp = dest.with_suffix(".partial")
    urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise SystemExit("could not find repo root")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache cfbfastR play-by-play and complementary CFBD facts"
    )
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    parser.add_argument("--live", action="store_true", help=f"also pull {LIVE_SEASON}")
    parser.add_argument("--pbp-only", action="store_true")
    parser.add_argument("--cfbd-only", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="refetch CFBD even if cached")
    args = parser.parse_args()
    if args.pbp_only and args.cfbd_only:
        raise SystemExit("pick one of --pbp-only or --cfbd-only")

    root = repo_root()
    seasons = list(args.seasons)
    if args.live and LIVE_SEASON not in seasons:
        seasons.append(LIVE_SEASON)

    if not args.cfbd_only:
        for season in seasons:
            if season > 2025:
                print(f"skip pbp {season}: cfbfastR dump not published yet")
                continue
            path = ensure_pbp(root, season)
            print(f"pbp {season}: {path.name} ({path.stat().st_size:,} bytes)")

    if not args.pbp_only:
        pull_cfbd(root, seasons, refresh=args.refresh)
        from footpalm.draft import pull_and_write
        from footpalm.people import pull_and_write as pull_people

        pull_and_write(root, refresh=args.refresh)
        pull_people(root, refresh=args.refresh)


if __name__ == "__main__":
    main()
