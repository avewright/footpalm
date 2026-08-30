from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

LUCK_SIGMA = 9.5
ITERATIONS = 40
PRIOR_GAMES = 4


def win_prob(margin: np.ndarray | pd.Series, sigma: float = LUCK_SIGMA) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(margin, dtype=float) / sigma))


@dataclass
class TeamRating:
    team: str
    off: float
    deff: float
    st: float
    tempo: float
    pom: float
    adjo: float
    adjd: float
    games: int


@dataclass
class RatingBook:
    season: int
    through_slate: int | None
    teams: dict[str, TeamRating]
    home_adv_epa: float
    league_epa: float
    plays_pg: float
    league_ppg: float
    fbs: set[str]
    conferences: dict[str, str] = field(default_factory=dict)

    def pom(self, team: str, default: float = -8.0) -> float:
        rating = self.teams.get(team)
        return rating.pom if rating else default

    def adjo(self, team: str) -> float:
        rating = self.teams.get(team)
        return rating.adjo if rating else -4.0

    def adjd(self, team: str) -> float:
        rating = self.teams.get(team)
        return rating.adjd if rating else 4.0

    def st(self, team: str) -> float:
        rating = self.teams.get(team)
        return rating.st if rating else 0.0

    def tempo_pg(self, team: str) -> float:
        rating = self.teams.get(team)
        return rating.tempo / 2 if rating else 65.0

    def games_played(self, team: str) -> int:
        rating = self.teams.get(team)
        return rating.games if rating else 0


def fit_ratings(
    sides: pd.DataFrame,
    fbs: set[str],
    conferences: dict[str, str],
    season: int,
    through_slate: int | None = None,
    prior: RatingBook | None = None,
    prior_games: int = PRIOR_GAMES,
) -> RatingBook:
    usable = sides.dropna(subset=["off_epa", "def_epa"]).copy()
    teams = sorted(set(usable["team"]) | set(usable["opponent"]) | (set(prior.teams) if prior else set()))
    if not teams:
        teams = sorted(fbs)
    if not teams:
        raise SystemExit(f"no teams for {season}")
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    off = np.zeros(n)
    deff = np.zeros(n)
    st = np.zeros(n)
    tempo = np.zeros(n)
    prior_off = np.zeros(n)
    prior_def = np.zeros(n)
    prior_st = np.zeros(n)
    prior_tempo = np.zeros(n)
    has_prior = np.zeros(n)

    if prior is not None:
        for team, rating in prior.teams.items():
            if team not in idx:
                continue
            i = idx[team]
            prior_off[i] = rating.off
            prior_def[i] = rating.deff
            prior_st[i] = rating.st
            prior_tempo[i] = rating.tempo
            has_prior[i] = 1.0

    k = float(prior_games)
    home_adv = 0.02
    if usable.empty:
        home_adv = prior.home_adv_epa if prior else 0.02
        league_epa = prior.league_epa if prior else 0.0
        league_plays = (prior.plays_pg * 2) if prior else 130.0
        league_ppg = prior.league_ppg if prior else 27.0
    else:
        league_epa = float(usable["off_epa"].mean())
        league_plays = float(usable["game_plays"].mean())
        league_ppg = float(usable["points"].mean())
        t_idx = usable["team"].map(idx).to_numpy()
        o_idx = usable["opponent"].map(idx).to_numpy()
        raw_off = usable["off_epa"].to_numpy()
        raw_def = usable["def_epa"].to_numpy()
        raw_st = usable["st_epa"].to_numpy()
        raw_tempo = usable["game_plays"].to_numpy()
        home_sign = np.where(usable["is_home"].to_numpy(), 1.0, -1.0)
        home_sign = np.where(usable["neutral_site"].to_numpy(), 0.0, home_sign)

        fbs_mask = np.array([t in fbs for t in teams])

        for _ in range(ITERATIONS):
            exp_home = home_adv * home_sign
            off_t = raw_off - deff[o_idx] - league_epa - exp_home
            def_t = raw_def - off[o_idx] - league_epa + exp_home
            st_t = raw_st + st[o_idx]
            tempo_t = raw_tempo + league_plays - tempo[o_idx]

            new_off = np.zeros(n)
            new_def = np.zeros(n)
            new_st = np.zeros(n)
            new_tempo = np.zeros(n)
            counts = np.zeros(n)

            np.add.at(new_off, t_idx, off_t)
            np.add.at(new_def, t_idx, def_t)
            np.add.at(new_st, t_idx, st_t)
            np.add.at(new_tempo, t_idx, tempo_t)
            np.add.at(counts, t_idx, 1.0)

            denom = counts + k * has_prior
            denom = np.maximum(denom, 1.0)
            off = (new_off + k * has_prior * prior_off) / denom
            deff = (new_def + k * has_prior * prior_def) / denom
            st = (new_st + k * has_prior * prior_st) / denom
            tempo = (new_tempo + k * has_prior * prior_tempo) / denom

            if fbs_mask.any():
                off -= off[fbs_mask].mean()
                deff -= deff[fbs_mask].mean()
                st -= st[fbs_mask].mean()
                tempo = tempo - tempo[fbs_mask].mean() + league_plays

            resid = raw_off - off[t_idx] - deff[o_idx] - league_epa
            home_rows = home_sign > 0
            away_rows = home_sign < 0
            if home_rows.any() and away_rows.any():
                home_adv = float((resid[home_rows].mean() - resid[away_rows].mean()) / 2)

    if usable.empty:
        counts = k * has_prior
        off = prior_off
        deff = prior_def
        st = prior_st
        tempo = prior_tempo

    plays_pg = league_plays / 2.0
    pom = (off - deff) * plays_pg + st
    games = (
        usable.groupby("team")["game_id"].nunique().to_dict()
        if not usable.empty
        else {}
    )

    book_teams = {}
    for team, i in idx.items():
        book_teams[team] = TeamRating(
            team=team,
            off=float(off[i]),
            deff=float(deff[i]),
            st=float(st[i]),
            tempo=float(tempo[i]),
            pom=float(pom[i]),
            adjo=float(off[i] * plays_pg),
            adjd=float(deff[i] * plays_pg),
            games=int(games.get(team, 0)),
        )

    return RatingBook(
        season=season,
        through_slate=through_slate,
        teams=book_teams,
        home_adv_epa=float(home_adv),
        league_epa=float(league_epa),
        plays_pg=float(plays_pg),
        league_ppg=float(league_ppg),
        fbs=set(fbs),
        conferences=conferences,
    )


def publish_table(
    book: RatingBook,
    sides: pd.DataFrame,
    min_games: int = 8,
    elo: dict[str, float] | None = None,
) -> dict:
    records = sides.groupby("team", as_index=False).agg(
        wins=("won", "sum"),
        losses=("lost", "sum"),
        games=("game_id", "nunique"),
    )
    scored = sides.copy()
    scored["exp_win"] = win_prob(scored["points"] - scored["opp_points"])
    luck = scored.groupby("team").agg(wins=("won", "mean"), exp=("exp_win", "mean"))
    luck["luck"] = luck["wins"] - luck["exp"]

    rows = []
    for team, rating in book.teams.items():
        if team not in book.fbs:
            continue
        rec = records.loc[records["team"].eq(team)]
        played = int(rec["games"].iloc[0]) if not rec.empty else rating.games
        if played < min_games:
            continue
        opp = sides.loc[sides["team"].eq(team), "opponent"]
        sos = float(np.mean([book.pom(o) for o in opp])) if len(opp) else 0.0
        rows.append(
            {
                "team": team,
                "conf": book.conferences.get(team, ""),
                "wins": int(rec["wins"].iloc[0]) if not rec.empty else 0,
                "losses": int(rec["losses"].iloc[0]) if not rec.empty else 0,
                "games": played,
                "pom": round(rating.pom, 2),
                "adjo": round(rating.adjo, 2),
                "adjd": round(rating.adjd, 2),
                "adjst": round(rating.st, 2),
                "tempo": round(rating.tempo / 2, 1),
                "sos": round(sos, 2),
                "luck": round(float(luck.loc[team, "luck"]) if team in luck.index else 0.0, 3),
            }
        )
        if elo is not None:
            rows[-1]["elo"] = round(float(elo.get(team, 1500.0)))

    rows.sort(key=lambda r: r["pom"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return {
        "season": book.season,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": (
            "Opponent-adjusted EPA/play on scrimmage downs, plus net special-teams EPA. "
            "Pom is points vs an average FBS team over one game. "
            "Garbage-time plays dropped. Each game weighted equally. "
            f"In-season ratings shrink toward last season with {PRIOR_GAMES} prior games."
        ),
        "home_adv_epa": round(book.home_adv_epa, 4),
        "league_epa": round(book.league_epa, 4),
        "plays_per_game": round(book.plays_pg, 1),
        "teams": rows,
    }


def dump_book(book: RatingBook) -> dict:
    return {
        "season": book.season,
        "through_slate": book.through_slate,
        "home_adv_epa": book.home_adv_epa,
        "league_epa": book.league_epa,
        "plays_pg": book.plays_pg,
        "league_ppg": book.league_ppg,
        "fbs": sorted(book.fbs),
        "conferences": book.conferences,
        "teams": {name: asdict(rating) for name, rating in book.teams.items()},
    }


def load_book(payload: dict) -> RatingBook:
    fields = TeamRating.__dataclass_fields__
    teams = {}
    for name, row in payload["teams"].items():
        data = dict(row)
        if "pom" not in data and "palm" in data:
            data["pom"] = data["palm"]
        teams[name] = TeamRating(**{k: data[k] for k in fields if k in data})
    return RatingBook(
        season=int(payload["season"]),
        through_slate=payload.get("through_slate"),
        teams=teams,
        home_adv_epa=float(payload["home_adv_epa"]),
        league_epa=float(payload["league_epa"]),
        plays_pg=float(payload["plays_pg"]),
        league_ppg=float(payload["league_ppg"]),
        fbs=set(payload.get("fbs") or []),
        conferences=dict(payload.get("conferences") or {}),
    )


_WEEK0_CARRY = (
    "wins",
    "losses",
    "games",
    "pom",
    "adjo",
    "adjd",
    "adjst",
    "tempo",
    "sos",
    "luck",
    "elo",
    "nil_roster",
    "nil_quality",
    "athletic_spend",
    "staff_payroll",
)


def publish_preseason(
    book: RatingBook,
    *,
    prior_season: int,
    prior_table: dict | None = None,
    elo: dict[str, float] | None = None,
) -> dict:
    empty = pd.DataFrame(columns=["team", "won", "lost", "game_id", "opponent", "points", "opp_points"])
    table = publish_table(book, empty, min_games=0, elo=elo)
    prior_rows = {row["team"]: row for row in (prior_table or {}).get("teams", [])}
    for row in table["teams"]:
        prev = prior_rows.get(row["team"])
        if not prev:
            continue
        if "pom" not in prev and "palm" in prev:
            prev = {**prev, "pom": prev["palm"]}
        for key in _WEEK0_CARRY:
            if prev.get(key) is not None:
                row[key] = prev[key]
    table["teams"].sort(key=lambda r: r["pom"], reverse=True)
    for rank, row in enumerate(table["teams"], start=1):
        row["rank"] = rank
    table["week"] = 0
    table["method"] = (
        f"Week 0 {book.season} board. Pom, Elo, record, SoS, and luck are {prior_season} finals. "
        f"No {book.season} games in the rating yet. When games start, ratings shrink toward that prior "
        f"with {PRIOR_GAMES} games."
    )
    return table


def rate_season(
    sides: pd.DataFrame,
    fbs: set[str],
    conferences: dict[str, str],
    season: int,
    prior: RatingBook | None = None,
    elo: dict[str, float] | None = None,
) -> dict:
    book = fit_ratings(sides, fbs, conferences, season, prior=prior)
    return publish_table(book, sides, elo=elo)
