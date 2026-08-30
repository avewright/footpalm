# FootPom

Opponent-adjusted EPA ratings for FBS, with walk-forward **TabPFN-3** game predictions.

Pom is the number of points a team would beat an average FBS team by on a neutral field.

## Don’t overfit

- Ratings use EPA and opponent adjustment only.
- Predictions use locked Pom plus extras (Elo, form, SOS). Walk-forward TabPFN on extras cleared the 2025 bar (0.1926 → 0.1865). No NIL. No spread. No week dummy. Logistic still uses the locked 10.
- TabPFN-3 is fit walk-forward: it only sees games from earlier slates.
- Logistic Pom + 2.5-point home field is the locked baseline and the fallback.
- NIL is harvested as context. A one-coefficient residual test is fit on 2024 and scored on 2025. It is not in the live model.
- ATS vs the market number is a holdout, never a training target.

## Run

```bash
uv sync
uv run python -m footpalm.build --seasons 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
cd web && npm install && npm run dev
```

That trains TabPFN on 2014–2025 and writes 2026 preseason projections from the CFBD slate.

Weekly, after the historical build exists:

```bash
uv run python -m footpalm.project --refresh
```

`--refresh` refetches 2026 games and lines only. It does not rerun twelve years of walk-forward.

Scores (no TabPFN refit):

```bash
uv run python -m footpalm.score
uv run python -m footpalm.score --install
```

`score` pulls CFBD games/lines, stamps finals onto the frozen projections, and writes `live.json`. `--install` loads a macOS launchd job every 2 hours.

`fetch` keeps cfbfastR play-by-play in `data/raw/` and caches CFBD facts in `data/raw/cfbd/`. Ratings still read only the parquet. Put `CFBD_API_KEY` in `.env`.

TabPFN-3 uses the local v3 checkpoint when it is already cached. To download weights, set `TABPFN_TOKEN` from [platform.priorlabs.ai](https://platform.priorlabs.ai). Without weights, the suite falls back to logistic.

`uv run python -m footpalm.build --no-tabpfn` skips the foundation model.

## Tests

```bash
uv run pytest
```
