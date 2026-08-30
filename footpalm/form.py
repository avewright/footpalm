from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from footpalm.predict import FEATURE_NAMES, game_features
from footpalm.rate import LUCK_SIGMA, RatingBook

# March Madness 2nd-place 2026: MOV Elo, last-N form, SOS, season diffs.
# Locked before the 2025 score. No spread. No NIL. No week dummy.
ELO_MEAN = 1500.0
ELO_K = 20.0
ELO_HOME = 55.0
ELO_REVERT = 0.75
FORM_N = 4
FORM_SHORT = 2
FORM_LONG = 8
FORM_10 = 10
CLOSE_MARGIN = 8.0
EWMA_DECAY = 0.45
MOMENTUM_N = 4
ELO_TANH = 200.0
PROD_SCALE = 400.0
FORM_SOS_SCALE = 20.0
COLLEY_DEFAULT = 0.5
LATE_SLATE = 8
TRIM_PROP = 0.1
GLM_RIDGE = 1.0
GLM_ITERS = 25
TIER_ELITE = 10.0
TIER_PLUS = 0.0
TIER_OK = -15.0

_POM_DIFF = FEATURE_NAMES.index("pom_diff")
_HOME_POM = FEATURE_NAMES.index("home_pom")
_AWAY_POM = FEATURE_NAMES.index("away_pom")
_TEMPO_DIFF = FEATURE_NAMES.index("tempo_diff")

EXTRA_NAMES = [
    "elo_diff",
    "form4_win_diff",
    "form4_margin_diff",
    "winpct_diff",
    "avg_margin_diff",
    "close_win_diff",
    "sos_diff",
    "rest_diff",
    "luck_diff",
    "quality_win_diff",
]

# Locked from extras permutation before this score. Signal axes only.
SIGNAL_NAMES = [
    "elo_diff",
    "elo_momentum_diff",
    "avg_margin_diff",
    "median_margin_diff",
    "form2_margin_diff",
    "form4_margin_diff",
    "form8_margin_diff",
    "ewma_margin_diff",
    "pf_diff",
    "pa_diff",
    "form4_pf_diff",
    "form4_pa_diff",
    "pythag_diff",
    "venue_margin_diff",
    "sos_diff",
    "residual_margin_diff",
    "form4_residual_diff",
    "h2h_margin",
]

# Locked from 2025 1st / 2026 2nd+3rd before this score. Graph ratings, ncsos, sums, nonlinear.
CRAFT_NAMES = [
    "srs_diff",
    "colley_diff",
    "ncsos_diff",
    "margin_std_diff",
    "pom_sum",
    "pom_abs",
    "tempo_abs",
    "log_pom_diff",
    "tanh_elo_diff",
    "pom_elo_prod",
    "form_sos_prod",
    "srs_elo_prod",
]

# Locked from ten 2025/2026 March Madness writeups before the LOSO score.
LOSO_NAMES = [
    "glm_quality_diff",
    "glm_sum",
    "elo_ratio",
    "log_margin_diff",
    "conf_pom_diff",
    "tier_win_diff",
    "form10_win_diff",
    "close_x_margin",
    "late_win_diff",
    "yoy_margin_diff",
]

ALL_NAMES = FEATURE_NAMES + EXTRA_NAMES
SIGNAL_ALL = FEATURE_NAMES + SIGNAL_NAMES
CRAFT_ALL = FEATURE_NAMES + CRAFT_NAMES
FULL_CRAFT_ALL = ALL_NAMES + CRAFT_NAMES
LOSO_ALL = ALL_NAMES + LOSO_NAMES

# Locked from LOSO permutation before this score. Conference axis only.
POWER_CONFS = frozenset({"ACC", "B12", "B1G", "SEC", "P12", "Big 12", "Big Ten", "Pac-12", "Pac-10"})
CONF_NAMES = [
    "conf_pom_diff",
    "conf_elo_diff",
    "p4_diff",
    "same_conf",
    "conf_win_diff",
    "conf_margin_diff",
    "ooc_win_diff",
    "ooc_margin_diff",
]
CONF_ALL = ALL_NAMES + CONF_NAMES

TIME_ORIGIN = 2014
TIME_WEEK_DENOM = 52
TIME_NAMES = ["year_idx", "week52"]
TIME_ALL = ALL_NAMES + TIME_NAMES

THIN_GAMES = 3
THIN_NAMES = ["thin"]
THIN_ALL = ALL_NAMES + THIN_NAMES


def thin_feature(home_games: float, away_games: float) -> float:
    return 1.0 if min(float(home_games), float(away_games)) < THIN_GAMES else 0.0


def same_conf(home_conf: str, away_conf: str) -> bool:
    return bool(home_conf) and bool(away_conf) and home_conf == away_conf


def is_power(conference: str) -> float:
    return 1.0 if conference in POWER_CONFS else 0.0


def signed_log(value: float) -> float:
    if value == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(value)), value)


def tier_win_points(opp_pom: float) -> float:
    if opp_pom > TIER_ELITE:
        return 6.0
    if opp_pom > TIER_PLUS:
        return 4.0
    if opp_pom > TIER_OK:
        return 2.0
    return 0.25


def trim_mean(values: list[float], proportion: float = TRIM_PROP) -> float:
    if not values:
        return 0.0
    if len(values) < 5:
        return float(np.mean(values))
    drop = max(1, int(len(values) * proportion))
    ordered = sorted(values)
    kept = ordered[drop : len(ordered) - drop]
    if not kept:
        return float(np.mean(values))
    return float(np.mean(kept))


def time_features(season: int, week: int) -> np.ndarray:
    return np.array([float(int(season) - TIME_ORIGIN), float(int(week)) / TIME_WEEK_DENOM], dtype=float)


@dataclass
class GameResult:
    win: float
    margin: float
    opp_pom: float
    slate: int
    pf: float
    pa: float
    at_home: bool
    elo_before: float
    opp: str = ""
    same_conf: bool = False


@dataclass
class TeamForm:
    elo: float = ELO_MEAN
    season_start_elo: float = ELO_MEAN
    results: list[GameResult] = field(default_factory=list)
    last_slate: int | None = None

    def _recent(self, n: int | None = None) -> list[GameResult]:
        if n is None:
            return self.results
        return self.results[-n:]

    def winpct(self, n: int | None = None) -> float:
        rows = self._recent(n)
        if not rows:
            return 0.5
        return float(np.mean([r.win for r in rows]))

    def avg_margin(self, n: int | None = None) -> float:
        rows = self._recent(n)
        if not rows:
            return 0.0
        return float(np.mean([r.margin for r in rows]))

    def median_margin(self) -> float:
        if not self.results:
            return 0.0
        return float(np.median([r.margin for r in self.results]))

    def avg_pf(self, n: int | None = None) -> float:
        rows = self._recent(n)
        if not rows:
            return 0.0
        return float(np.mean([r.pf for r in rows]))

    def avg_pa(self, n: int | None = None) -> float:
        rows = self._recent(n)
        if not rows:
            return 0.0
        return float(np.mean([r.pa for r in rows]))

    def ewma_margin(self, decay: float = EWMA_DECAY) -> float:
        if not self.results:
            return 0.0
        value = 0.0
        for row in self.results:
            value = decay * value + (1.0 - decay) * row.margin
        return float(value)

    def residual(self, n: int | None = None) -> float:
        rows = self._recent(n)
        if not rows:
            return 0.0
        return float(np.mean([r.margin - r.opp_pom for r in rows]))

    def pythag(self) -> float:
        pf = self.avg_pf()
        pa = self.avg_pa()
        if pf <= 0.0 and pa <= 0.0:
            return 0.5
        return float((pf * pf) / (pf * pf + pa * pa + 1e-9))

    def venue_margin(self, at_home: bool) -> float:
        rows = [r for r in self.results if r.at_home == at_home]
        if not rows:
            return 0.0
        return float(np.mean([r.margin for r in rows]))

    def elo_momentum(self, n: int = MOMENTUM_N) -> float:
        if not self.results:
            return 0.0
        start = self.results[-min(n, len(self.results))].elo_before
        return float(self.elo - start)

    def close_winpct(self) -> float:
        close = [r for r in self.results if abs(r.margin) <= CLOSE_MARGIN]
        if not close:
            return 0.5
        return float(np.mean([r.win for r in close]))

    def sos(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.opp_pom for r in self.results]))

    def luck(self) -> float:
        if not self.results:
            return 0.0
        exp = [1 / (1 + math.exp(-r.margin / LUCK_SIGMA)) for r in self.results]
        return self.winpct() - float(np.mean(exp))

    def quality_winpct(self) -> float:
        quality = [r for r in self.results if r.opp_pom > 0]
        if not quality:
            return 0.5
        return float(np.mean([r.win for r in quality]))

    def ncsos(self) -> float:
        rows = [r for r in self.results if not r.same_conf]
        if not rows:
            return 0.0
        return float(np.mean([r.opp_pom for r in rows]))

    def split_winpct(self, conference: bool) -> float:
        rows = [r for r in self.results if r.same_conf == conference]
        if not rows:
            return 0.5
        return float(np.mean([r.win for r in rows]))

    def split_margin(self, conference: bool) -> float:
        rows = [r for r in self.results if r.same_conf == conference]
        if not rows:
            return 0.0
        return float(np.mean([r.margin for r in rows]))

    def margin_std(self) -> float:
        if len(self.results) < 2:
            return 0.0
        return float(np.std([r.margin for r in self.results], ddof=0))

    def rest(self, slate: int) -> float:
        if self.last_slate is None:
            return 0.0
        return float(slate - self.last_slate)

    def log_margin(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([signed_log(r.margin) for r in self.results]))

    def trim_margin(self) -> float:
        return trim_mean([r.margin for r in self.results])

    def tier_wins(self) -> float:
        return float(sum(tier_win_points(r.opp_pom) for r in self.results if r.win))

    def late_winpct(self) -> float:
        rows = [r for r in self.results if r.slate >= LATE_SLATE]
        if not rows:
            return 0.5
        return float(np.mean([r.win for r in rows]))

    def elo_slope(self) -> float:
        if not self.results:
            return 0.0
        return float((self.elo - self.season_start_elo) / len(self.results))


class FormBook:
    def __init__(self) -> None:
        self.teams: dict[str, TeamForm] = {}
        self.h2h: dict[tuple[str, str], float] = {}
        self.pom: dict[str, float] = {}
        self.conf: dict[str, str] = {}
        self.prev_margin: dict[str, float] = {}
        self._srs: dict[str, float] | None = None
        self._colley: dict[str, float] | None = None
        self._glm: dict[str, float] | None = None

    def get(self, team: str) -> TeamForm:
        if team not in self.teams:
            self.teams[team] = TeamForm()
        return self.teams[team]

    def touch(self, team: str, pom: float, conference: str = "") -> None:
        self.pom[team] = pom
        if conference:
            self.conf[team] = conference

    def new_season(self) -> None:
        for team, form in self.teams.items():
            if form.results:
                self.prev_margin[team] = form.avg_margin()
            form.elo = form.elo * ELO_REVERT + ELO_MEAN * (1 - ELO_REVERT)
            form.season_start_elo = form.elo
            form.results = []
            form.last_slate = None
        self._srs = None
        self._colley = None
        self._glm = None

    def _bump_ratings(self) -> None:
        self._srs = None
        self._colley = None
        self._glm = None

    def _solve_srs(self) -> dict[str, float]:
        teams = [team for team, form in self.teams.items() if form.results]
        if not teams:
            return {}
        if len(teams) == 1:
            return {teams[0]: self.teams[teams[0]].avg_margin()}
        idx = {team: i for i, team in enumerate(teams)}
        n = len(teams)
        matrix = np.zeros((n, n))
        rhs = np.zeros(n)
        for team, form in self.teams.items():
            if team not in idx:
                continue
            i = idx[team]
            for row in form.results:
                matrix[i, i] += 1.0
                rhs[i] += row.margin
                if row.opp in idx:
                    matrix[i, idx[row.opp]] -= 1.0
        matrix[-1, :] = 1.0
        rhs[-1] = 0.0
        try:
            solved = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        return {team: float(solved[i]) for team, i in idx.items()}

    def _solve_colley(self) -> dict[str, float]:
        teams = [team for team, form in self.teams.items() if form.results]
        if len(teams) < 2:
            return {}
        idx = {team: i for i, team in enumerate(teams)}
        n = len(teams)
        matrix = np.eye(n) * 2.0
        rhs = np.ones(n)
        for team, form in self.teams.items():
            if team not in idx:
                continue
            i = idx[team]
            for row in form.results:
                if row.opp not in idx:
                    continue
                j = idx[row.opp]
                matrix[i, i] += 1.0
                matrix[i, j] -= 1.0
                rhs[i] += 0.5 if row.win else -0.5
        try:
            solved = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        return {team: float(solved[i]) for team, i in idx.items()}

    def srs_of(self, team: str) -> float:
        if self._srs is None:
            self._srs = self._solve_srs()
        return float(self._srs.get(team, 0.0))

    def colley_of(self, team: str) -> float:
        if self._colley is None:
            self._colley = self._solve_colley()
        return float(self._colley.get(team, COLLEY_DEFAULT))

    def _solve_glm(self) -> dict[str, float]:
        teams = [team for team, form in self.teams.items() if form.results]
        if len(teams) < 2:
            return {}
        idx = {team: i for i, team in enumerate(teams)}
        seen: set[tuple[str, str, int]] = set()
        pairs: list[tuple[int, int, float]] = []
        for team, form in self.teams.items():
            if team not in idx:
                continue
            for row in form.results:
                if row.opp not in idx:
                    continue
                key = (min(team, row.opp), max(team, row.opp), row.slate)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((idx[team], idx[row.opp], row.win))
        if not pairs:
            return {}
        n = len(teams)
        design = np.zeros((len(pairs), n))
        y = np.zeros(len(pairs))
        for k, (i, j, win) in enumerate(pairs):
            design[k, i] = 1.0
            design[k, j] = -1.0
            y[k] = win
        beta = np.zeros(n)
        ridge = np.eye(n) * GLM_RIDGE
        for _ in range(GLM_ITERS):
            z = np.clip(design @ beta, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            weight = np.clip(p * (1.0 - p), 1e-6, None)
            hessian = design.T @ (weight[:, None] * design) + ridge
            grad = design.T @ (y - p) - GLM_RIDGE * beta
            try:
                step = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
            beta = beta + step
            if float(np.max(np.abs(step))) < 1e-6:
                break
        beta = beta - float(beta.mean())
        return {team: float(beta[i]) for team, i in idx.items()}

    def glm_of(self, team: str) -> float:
        if self._glm is None:
            self._glm = self._solve_glm()
        return float(self._glm.get(team, 0.0))

    def conf_pom_of(self, team: str) -> float:
        conference = self.conf.get(team, "")
        if not conference:
            return 0.0
        vals = [self.pom[name] for name, conf in self.conf.items() if conf == conference and name in self.pom]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def conf_elo_of(self, team: str) -> float:
        conference = self.conf.get(team, "")
        if not conference:
            return ELO_MEAN
        vals = [self.get(name).elo for name, conf in self.conf.items() if conf == conference]
        if not vals:
            return ELO_MEAN
        return float(np.mean(vals))

    def conference(self, home: str, away: str, home_conf: str = "", away_conf: str = "") -> np.ndarray:
        if home_conf:
            self.conf[home] = home_conf
        if away_conf:
            self.conf[away] = away_conf
        h, a = self.get(home), self.get(away)
        hc = home_conf or self.conf.get(home, "")
        ac = away_conf or self.conf.get(away, "")
        return np.array(
            [
                self.conf_pom_of(home) - self.conf_pom_of(away),
                self.conf_elo_of(home) - self.conf_elo_of(away),
                is_power(hc) - is_power(ac),
                1.0 if same_conf(hc, ac) else 0.0,
                h.split_winpct(True) - a.split_winpct(True),
                h.split_margin(True) - a.split_margin(True),
                h.split_winpct(False) - a.split_winpct(False),
                h.split_margin(False) - a.split_margin(False),
            ],
            dtype=float,
        )

    def yoy_margin_of(self, team: str) -> float:
        prior = self.prev_margin.get(team)
        if prior is None:
            return 0.0
        return self.get(team).avg_margin() - prior

    def extras(self, home: str, away: str, slate: int) -> np.ndarray:
        h, a = self.get(home), self.get(away)
        return np.array(
            [
                h.elo - a.elo,
                h.winpct(FORM_N) - a.winpct(FORM_N),
                h.avg_margin(FORM_N) - a.avg_margin(FORM_N),
                h.winpct() - a.winpct(),
                h.avg_margin() - a.avg_margin(),
                h.close_winpct() - a.close_winpct(),
                h.sos() - a.sos(),
                h.rest(slate) - a.rest(slate),
                h.luck() - a.luck(),
                h.quality_winpct() - a.quality_winpct(),
            ],
            dtype=float,
        )

    def signal(self, home: str, away: str, slate: int) -> np.ndarray:
        h, a = self.get(home), self.get(away)
        return np.array(
            [
                h.elo - a.elo,
                h.elo_momentum() - a.elo_momentum(),
                h.avg_margin() - a.avg_margin(),
                h.median_margin() - a.median_margin(),
                h.avg_margin(FORM_SHORT) - a.avg_margin(FORM_SHORT),
                h.avg_margin(FORM_N) - a.avg_margin(FORM_N),
                h.avg_margin(FORM_LONG) - a.avg_margin(FORM_LONG),
                h.ewma_margin() - a.ewma_margin(),
                h.avg_pf() - a.avg_pf(),
                h.avg_pa() - a.avg_pa(),
                h.avg_pf(FORM_N) - a.avg_pf(FORM_N),
                h.avg_pa(FORM_N) - a.avg_pa(FORM_N),
                h.pythag() - a.pythag(),
                h.venue_margin(True) - a.venue_margin(False),
                h.sos() - a.sos(),
                h.residual() - a.residual(),
                h.residual(FORM_N) - a.residual(FORM_N),
                self.h2h.get((home, away), 0.0),
            ],
            dtype=float,
        )

    def craft(self, home: str, away: str, slate: int, locked: np.ndarray) -> np.ndarray:
        h, a = self.get(home), self.get(away)
        pom_diff = float(locked[_POM_DIFF])
        elo_diff = h.elo - a.elo
        srs_diff = self.srs_of(home) - self.srs_of(away)
        form4 = h.avg_margin(FORM_N) - a.avg_margin(FORM_N)
        sos = h.sos() - a.sos()
        return np.array(
            [
                srs_diff,
                self.colley_of(home) - self.colley_of(away),
                h.ncsos() - a.ncsos(),
                h.margin_std() - a.margin_std(),
                float(locked[_HOME_POM]) + float(locked[_AWAY_POM]),
                abs(pom_diff),
                abs(float(locked[_TEMPO_DIFF])),
                signed_log(pom_diff),
                math.tanh(elo_diff / ELO_TANH),
                pom_diff * elo_diff / PROD_SCALE,
                form4 * sos / FORM_SOS_SCALE,
                srs_diff * elo_diff / PROD_SCALE,
            ],
            dtype=float,
        )

    def loso(self, home: str, away: str, slate: int) -> np.ndarray:
        h, a = self.get(home), self.get(away)
        glm_h, glm_a = self.glm_of(home), self.glm_of(away)
        away_elo = a.elo if a.elo != 0.0 else 1.0
        close_diff = h.close_winpct() - a.close_winpct()
        margin_diff = h.avg_margin() - a.avg_margin()
        return np.array(
            [
                glm_h - glm_a,
                glm_h + glm_a,
                h.elo / away_elo - 1.0,
                h.log_margin() - a.log_margin(),
                self.conf_pom_of(home) - self.conf_pom_of(away),
                h.tier_wins() - a.tier_wins(),
                h.winpct(FORM_10) - a.winpct(FORM_10),
                close_diff * margin_diff,
                h.late_winpct() - a.late_winpct(),
                self.yoy_margin_of(home) - self.yoy_margin_of(away),
            ],
            dtype=float,
        )

    def apply_game(
        self,
        home: str,
        away: str,
        *,
        home_won: bool,
        margin: float,
        home_pom: float,
        away_pom: float,
        slate: int,
        neutral: bool,
        home_points: float = 0.0,
        away_points: float = 0.0,
        home_conf: str = "",
        away_conf: str = "",
    ) -> None:
        h, a = self.get(home), self.get(away)
        self.pom[home] = home_pom
        self.pom[away] = away_pom
        if home_conf:
            self.conf[home] = home_conf
        if away_conf:
            self.conf[away] = away_conf
        conference = same_conf(home_conf, away_conf)
        h_elo, a_elo = h.elo, a.elo
        if not neutral:
            h_elo += ELO_HOME
        expected = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        winner_elo = h_elo if home_won else a_elo
        loser_elo = a_elo if home_won else h_elo
        mov = math.log(abs(margin) + 1) if margin != 0 else 0.0
        denom = (winner_elo - loser_elo) * 0.001 + 2.2
        mov_mult = mov * (2.2 / denom) if denom != 0 else mov
        delta = ELO_K * mov_mult * ((1.0 if home_won else 0.0) - expected)
        h_before, a_before = h.elo, a.elo
        h.elo += delta
        a.elo -= delta
        h.results.append(
            GameResult(
                win=1.0 if home_won else 0.0,
                margin=margin,
                opp_pom=away_pom,
                slate=slate,
                pf=float(home_points),
                pa=float(away_points),
                at_home=not neutral,
                elo_before=h_before,
                opp=away,
                same_conf=conference,
            )
        )
        a.results.append(
            GameResult(
                win=0.0 if home_won else 1.0,
                margin=-margin,
                opp_pom=home_pom,
                slate=slate,
                pf=float(away_points),
                pa=float(home_points),
                at_home=False,
                elo_before=a_before,
                opp=home,
                same_conf=conference,
            )
        )
        h.last_slate = slate
        a.last_slate = slate
        self.h2h[(home, away)] = margin
        self.h2h[(away, home)] = -margin
        self._bump_ratings()

    def snapshot(self) -> dict[str, float]:
        return {team: form.elo for team, form in self.teams.items()}


def revert_elo(elo: float) -> float:
    return elo * ELO_REVERT + ELO_MEAN * (1 - ELO_REVERT)


def apply_sides(form: FormBook, sides, book: RatingBook | None = None) -> None:
    from footpalm.plays import listed_games

    games = listed_games(sides)
    if games.empty:
        return
    for slate in sorted(int(s) for s in games["slate"].unique()):
        part = games.loc[games["slate"].eq(slate)]
        for row in part.itertuples(index=False):
            form.apply_game(
                row.home_team,
                row.away_team,
                home_won=bool(row.won),
                margin=float(row.points - row.opp_points),
                home_pom=book.pom(row.home_team) if book is not None else 0.0,
                away_pom=book.pom(row.away_team) if book is not None else 0.0,
                slate=slate,
                neutral=bool(row.neutral_site),
                home_points=float(row.points),
                away_points=float(row.opp_points),
                home_conf=str(getattr(row, "home_conf", "") or ""),
                away_conf=str(getattr(row, "away_conf", "") or ""),
            )


def elo_snapshots_from_games(games: list[dict]) -> tuple[dict[int, dict[str, float]], FormBook]:
    form = FormBook()
    out: dict[int, dict[str, float]] = {}
    i = 0
    while i < len(games):
        season = int(games[i]["season"])
        if i == 0 or int(games[i - 1]["season"]) != season:
            form.new_season()
        slate = int(games[i]["slate"])
        start = i
        while i < len(games) and int(games[i]["season"]) == season and int(games[i]["slate"]) == slate:
            i += 1
        for j in range(start, i):
            game = games[j]
            if game.get("actual_margin") is None:
                continue
            form.apply_game(
                game["home"],
                game["away"],
                home_won=bool(game.get("home_won")),
                margin=float(game["actual_margin"]),
                home_pom=0.0,
                away_pom=0.0,
                slate=slate,
                neutral=bool(game.get("neutral")),
                home_points=float(game.get("actual_home") or 0.0),
                away_points=float(game.get("actual_away") or 0.0),
                home_conf=str(game.get("home_conf") or ""),
                away_conf=str(game.get("away_conf") or ""),
            )
        if i == len(games) or int(games[i]["season"]) != season:
            out[season] = form.snapshot()
    return out, form


def write_ratings_elo(root, snapshots: dict[int, dict[str, float]]) -> None:
    import json
    from pathlib import Path

    root = Path(root)
    dests = [root / "data" / "processed", root / "web" / "public" / "data"]
    for season, elos in snapshots.items():
        for dest in dests:
            path = dest / f"ratings-{season}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            payload["method"] = str(payload.get("method", "")).replace("Palm is", "Pom is")
            for team in payload.get("teams", []):
                if "pom" not in team and "palm" in team:
                    team["pom"] = team["palm"]
                team.pop("palm", None)
                team["elo"] = round(float(elos.get(team["team"], ELO_MEAN)))
            path.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"wrote elo onto {path}")


def attach_elo(root=None) -> dict[int, dict[str, float]]:
    import json
    from pathlib import Path

    from footpalm.fetch import LIVE_SEASON

    if root is None:
        here = Path(__file__).resolve()
        root = next(parent for parent in here.parents if (parent / "pyproject.toml").exists())
    else:
        root = Path(root)
    games = []
    for path in sorted((root / "web" / "public" / "data").glob("predictions-*.json")):
        try:
            season = int(path.stem.split("-")[1])
        except (IndexError, ValueError):
            continue
        if season >= LIVE_SEASON:
            continue
        games.extend(json.loads(path.read_text())["games"])
    snapshots, form = elo_snapshots_from_games(games)
    write_ratings_elo(root, snapshots)
    return snapshots


def replay_form(games: list[dict], X_locked: np.ndarray) -> FormBook:
    """Walk completed games slate-by-slate. Used to carry Elo into the next season."""
    if len(games) != len(X_locked):
        raise SystemExit(f"replay games n={len(games)} != locked n={len(X_locked)}")
    form = FormBook()
    i = 0
    while i < len(games):
        season = int(games[i]["season"])
        if i == 0 or int(games[i - 1]["season"]) != season:
            form.new_season()
        slate = int(games[i]["slate"])
        start = i
        while i < len(games) and int(games[i]["season"]) == season and int(games[i]["slate"]) == slate:
            i += 1
        for j in range(start, i):
            game = games[j]
            if game.get("actual_margin") is None:
                continue
            form.apply_game(
                game["home"],
                game["away"],
                home_won=bool(game.get("home_won")),
                margin=float(game["actual_margin"]),
                home_pom=float(X_locked[j, _HOME_POM]),
                away_pom=float(X_locked[j, _AWAY_POM]),
                slate=slate,
                neutral=bool(game.get("neutral")),
                home_points=float(game.get("actual_home") or 0.0),
                away_points=float(game.get("actual_away") or 0.0),
                home_conf=str(game.get("home_conf") or ""),
                away_conf=str(game.get("away_conf") or ""),
            )
    return form


def extras_matrix(book: RatingBook, form: FormBook, games: list[dict]) -> np.ndarray:
    """Locked Pom + extras. Form sees only earlier slates. Completed games update after the slate."""
    if not games:
        return np.zeros((0, len(ALL_NAMES)), dtype=float)
    X = np.zeros((len(games), len(ALL_NAMES)), dtype=float)
    i = 0
    while i < len(games):
        slate = int(games[i].get("slate") or 0)
        start = i
        while i < len(games) and int(games[i].get("slate") or 0) == slate:
            i += 1
        for j in range(start, i):
            game = games[j]
            locked = game_features(book, game["home"], game["away"], neutral=bool(game.get("neutral")))
            X[j] = np.concatenate([locked, form.extras(game["home"], game["away"], slate)])
        for j in range(start, i):
            game = games[j]
            if game.get("actual_margin") is None:
                continue
            form.apply_game(
                game["home"],
                game["away"],
                home_won=bool(game.get("home_won")),
                margin=float(game["actual_margin"]),
                home_pom=book.pom(game["home"]),
                away_pom=book.pom(game["away"]),
                slate=slate,
                neutral=bool(game.get("neutral")),
                home_points=float(game.get("actual_home") or 0.0),
                away_points=float(game.get("actual_away") or 0.0),
                home_conf=str(game.get("home_conf") or ""),
                away_conf=str(game.get("away_conf") or ""),
            )
    return X


def game_features_full(
    book: RatingBook,
    form: FormBook,
    home: str,
    away: str,
    *,
    slate: int,
    neutral: bool = False,
) -> np.ndarray:
    base = game_features(book, home, away, neutral=neutral)
    return np.concatenate([base, form.extras(home, away, slate)])


def game_features_craft(
    book: RatingBook,
    form: FormBook,
    home: str,
    away: str,
    *,
    slate: int,
    neutral: bool = False,
) -> np.ndarray:
    base = game_features(book, home, away, neutral=neutral)
    return np.concatenate([base, form.craft(home, away, slate, base)])


def game_features_signal(
    book: RatingBook,
    form: FormBook,
    home: str,
    away: str,
    *,
    slate: int,
    neutral: bool = False,
) -> np.ndarray:
    base = game_features(book, home, away, neutral=neutral)
    return np.concatenate([base, form.signal(home, away, slate)])


def main() -> None:
    attach_elo()


if __name__ == "__main__":
    main()
