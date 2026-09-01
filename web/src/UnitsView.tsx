import { useMemo } from "react";
import { signed } from "./format";
import { pctile, Radar } from "./Radar";
import { TeamLink } from "./TeamView";
import type { TeamRow, UnitPpa, UnitTeam, UnitsFile } from "./types";

const DOWNS = [
  {
    key: "firstDown" as const,
    label: "1st",
    title: "Offensive PPA on first down. Bar height is FBS percentile among offenses.",
  },
  {
    key: "secondDown" as const,
    label: "2nd",
    title: "Offensive PPA on second down. Bar height is FBS percentile among offenses.",
  },
  {
    key: "thirdDown" as const,
    label: "3rd",
    title: "Offensive PPA on third down. Bar height is FBS percentile among offenses.",
  },
];

function pool(teams: UnitTeam[], side: "offense" | "defense", key: keyof UnitPpa) {
  return teams.map((t) => t[side][key]).filter((n): n is number => n != null && !Number.isNaN(n));
}

function div(num: number | null | undefined, den: number | null | undefined) {
  if (num == null || den == null || !den) return null;
  const n = num / den;
  return Number.isFinite(n) ? n : null;
}

function ypaOf(team: UnitTeam, side: "offense" | "defense") {
  const s = team.stats;
  return side === "offense"
    ? div(s.netPassingYards, s.passAttempts)
    : div(s.netPassingYardsOpponent, s.passAttemptsOpponent);
}

function ypcOf(team: UnitTeam, side: "offense" | "defense") {
  const s = team.stats;
  return side === "offense"
    ? div(s.rushingYards, s.rushingAttempts)
    : div(s.rushingYardsOpponent, s.rushingAttemptsOpponent);
}

function ratePool(teams: UnitTeam[], get: (team: UnitTeam) => number | null) {
  return teams.map(get).filter((n): n is number => n != null && !Number.isNaN(n));
}

function toneOf(cls: "mux-poly-away" | "mux-poly-home") {
  return cls === "mux-poly-home" ? "home" : "away";
}

function radarAxis(
  label: string,
  title: string,
  offVal: number | null,
  defVal: number | null,
  offPool: number[],
  defPool: number[],
) {
  return {
    label,
    title,
    a: offVal == null ? 0.04 : pctile(offVal, offPool, true),
    b: defVal == null ? 0.04 : pctile(defVal, defPool, false),
  };
}

function ppa(n: number | null | undefined) {
  return n == null ? "—" : n.toFixed(2);
}

function per(n: number | null | undefined, games: number | undefined, digits = 1) {
  if (n == null || !games) return "—";
  return (n / games).toFixed(digits);
}

function rate(num: number | null | undefined, den: number | null | undefined) {
  if (num == null || !den) return "—";
  return (num / den).toFixed(2);
}

function pctRate(num: number | null | undefined, den: number | null | undefined) {
  if (num == null || !den) return "—";
  return `${((num / den) * 100).toFixed(0)}%`;
}

function Row({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="unit-stat" title={hint}>
      <span className="quiet">{label}</span>
      <span>{value}</span>
    </div>
  );
}

function DownChart({
  off,
  league,
  tone,
}: {
  off: UnitPpa;
  league: UnitTeam[];
  tone: "away" | "home";
}) {
  const cols = DOWNS.map((down) => {
    const value = off[down.key];
    return {
      ...down,
      value,
      pct: value == null ? 0 : pctile(value, pool(league, "offense", down.key), true),
    };
  });

  return (
    <div className="unit-downs">
      <div className="unit-dive-head">Offensive PPA by down</div>
      <div className="unit-down-grid" role="list">
        {cols.map((col) => (
          <div key={col.key} className="unit-down-col" title={col.title} role="listitem">
            <span className="unit-down-lab">{col.label}</span>
            <div className="unit-down-track">
              <i
                className={`unit-down-bar mux-${tone}`}
                style={{ height: col.value == null ? 0 : `${Math.max(col.pct * 100, 3)}%` }}
              />
            </div>
            <span className={`unit-down-val mux-${tone}`}>{ppa(col.value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Clash({
  offName,
  defName,
  off,
  def,
  offRow,
  defRow,
  league,
  predPts,
  offClass,
  defClass,
  onOpenTeam,
}: {
  offName: string;
  defName: string;
  off: UnitTeam;
  def: UnitTeam;
  offRow: TeamRow | null;
  defRow: TeamRow | null;
  league: UnitTeam[];
  predPts: number;
  offClass: "mux-poly-away" | "mux-poly-home";
  defClass: "mux-poly-away" | "mux-poly-home";
  onOpenTeam: (team: string) => void;
}) {
  const offTone = toneOf(offClass);
  const axes = [
    radarAxis(
      "PPA",
      "Predicted points added per play. Offense: higher is better. Defense: lower allowed is better (inverted on the chart).",
      off.offense.overall ?? null,
      def.defense.overall ?? null,
      pool(league, "offense", "overall"),
      pool(league, "defense", "overall"),
    ),
    radarAxis(
      "Pass PPA",
      "PPA per passing play. Offense: how much the pass game adds. Defense: PPA allowed through the air (inverted).",
      off.offense.passing ?? null,
      def.defense.passing ?? null,
      pool(league, "offense", "passing"),
      pool(league, "defense", "passing"),
    ),
    radarAxis(
      "Rush PPA",
      "PPA per rushing play. Offense: how much the run game adds. Defense: PPA allowed on the ground (inverted).",
      off.offense.rushing ?? null,
      def.defense.rushing ?? null,
      pool(league, "offense", "rushing"),
      pool(league, "defense", "rushing"),
    ),
    radarAxis(
      "YPA",
      "Net passing yards per attempt. Offense: higher is better. Defense: yards allowed (inverted).",
      ypaOf(off, "offense"),
      ypaOf(def, "defense"),
      ratePool(league, (t) => ypaOf(t, "offense")),
      ratePool(league, (t) => ypaOf(t, "defense")),
    ),
    radarAxis(
      "YPC",
      "Rushing yards per carry. Offense: higher is better. Defense: yards allowed (inverted).",
      ypcOf(off, "offense"),
      ypcOf(def, "defense"),
      ratePool(league, (t) => ypcOf(t, "offense")),
      ratePool(league, (t) => ypcOf(t, "defense")),
    ),
  ].filter((axis) => axis.a > 0.04 || axis.b > 0.04);

  const os = off.stats;
  const ds = def.stats;
  const og = os.games;
  const dg = ds.games;
  const offPlays = (os.passAttempts ?? 0) + (os.rushingAttempts ?? 0);
  const defPlays = (ds.passAttemptsOpponent ?? 0) + (ds.rushingAttemptsOpponent ?? 0);
  const look = offRow && defRow ? offRow.adjo + defRow.adjd : null;
  const offLead =
    off.offense.overall != null && def.defense.overall != null && off.offense.overall > def.defense.overall;

  return (
    <div className="unit-clash">
      <h3>
        <TeamLink team={offName} onOpen={onOpenTeam} /> offense vs <TeamLink team={defName} onOpen={onOpenTeam} />{" "}
        defense
      </h3>
      <p className="quiet unit-clash-kicker">
        {offLead
          ? `${offName} PPA is ahead of ${defName}'s allowed PPA.`
          : `${defName} defense PPA is stingier than ${offName}'s offense.`}
        {look != null ? ` EPA look ${signed(look)}.` : ""} Pred {predPts.toFixed(1)} pts.
      </p>
      {axes.length >= 3 && (
        <Radar
          axes={axes}
          classA={offClass}
          classB={defClass}
          label={`${offName} offense versus ${defName} defense`}
        />
      )}
      <DownChart off={off.offense} league={league} tone={offTone} />
      <div className="unit-dive">
        <div>
          <div className="unit-dive-head">Offense</div>
          <Row
            label="AdjO"
            value={offRow ? signed(offRow.adjo) : "—"}
            hint="FootPalm opponent-adjusted offensive EPA, scaled to points per game against an average defense. Higher is better."
          />
          <Row
            label="PPA"
            value={ppa(off.offense.overall)}
            hint="CFBD predicted points added per offensive play, garbage time excluded. Higher is better."
          />
          <Row
            label="Pass PPA"
            value={ppa(off.offense.passing)}
            hint="PPA per passing play. How much the pass game adds relative to a replacement play."
          />
          <Row
            label="Rush PPA"
            value={ppa(off.offense.rushing)}
            hint="PPA per rushing play. How much the run game adds relative to a replacement play."
          />
          <Row
            label="Yds/play"
            value={rate(os.totalYards, offPlays)}
            hint="Total yards divided by pass attempts plus rush attempts. Unadjusted; includes sacks in the pass total."
          />
          <Row
            label="YPA / YPC"
            value={`${rate(os.netPassingYards, os.passAttempts)} / ${rate(os.rushingYards, os.rushingAttempts)}`}
            hint="Net passing yards per attempt, then rushing yards per carry. Unadjusted box-score rates."
          />
          <Row
            label="3rd down"
            value={pctRate(os.thirdDownConversions, os.thirdDowns)}
            hint="Third-down conversion rate: conversions divided by attempts."
          />
          <Row
            label="Pass rate"
            value={pctRate(os.passAttempts, offPlays)}
            hint="Share of scrimmage plays that are pass attempts. Higher means a more pass-heavy offense."
          />
          <Row
            label="TD / game"
            value={per((os.passingTDs ?? 0) + (os.rushingTDs ?? 0), og)}
            hint="Passing plus rushing touchdowns per game. Does not include returns."
          />
          <Row
            label="TO / game"
            value={per(os.turnovers, og)}
            hint="Giveaways per game: interceptions plus fumbles lost."
          />
        </div>
        <div>
          <div className="unit-dive-head">Defense</div>
          <Row
            label="AdjD"
            value={defRow ? signed(defRow.adjd) : "—"}
            hint="FootPalm opponent-adjusted defensive EPA, points allowed per game vs an average offense. Lower (more negative) is better."
          />
          <Row
            label="PPA allowed"
            value={ppa(def.defense.overall)}
            hint="CFBD predicted points added allowed per play. Lower is stingier."
          />
          <Row
            label="Pass PPA"
            value={ppa(def.defense.passing)}
            hint="PPA allowed per opponent passing play. Lower means the pass defense is tighter."
          />
          <Row
            label="Rush PPA"
            value={ppa(def.defense.rushing)}
            hint="PPA allowed per opponent rushing play. Lower means the run defense is tighter."
          />
          <Row
            label="Yds/play"
            value={rate(ds.totalYardsOpponent, defPlays)}
            hint="Yards allowed per opponent pass attempt plus rush attempt. Unadjusted."
          />
          <Row
            label="YPA / YPC"
            value={`${rate(ds.netPassingYardsOpponent, ds.passAttemptsOpponent)} / ${rate(ds.rushingYardsOpponent, ds.rushingAttemptsOpponent)}`}
            hint="Net passing yards allowed per attempt, then rushing yards allowed per carry."
          />
          <Row
            label="3rd down"
            value={pctRate(ds.thirdDownConversionsOpponent, ds.thirdDownsOpponent)}
            hint="Opponent third-down conversion rate. Lower is better."
          />
          <Row
            label="Sacks / TFL"
            value={`${per(ds.sacks, dg)} / ${per(ds.tacklesForLoss, dg)}`}
            hint="Sacks per game, then tackles for loss per game."
          />
          <Row
            label="Takeaways / g"
            value={per(ds.turnoversOpponent, dg)}
            hint="Opponent turnovers per game — interceptions and fumbles recovered."
          />
        </div>
      </div>
    </div>
  );
}

export function UnitsView({
  homeName,
  awayName,
  homeRow,
  awayRow,
  predHome,
  predAway,
  files,
  onOpenTeam,
}: {
  homeName: string;
  awayName: string;
  homeRow: TeamRow | null;
  awayRow: TeamRow | null;
  predHome: number;
  predAway: number;
  files: UnitsFile[];
  onOpenTeam: (team: string) => void;
}) {
  const found = useMemo(() => {
    function pick(team: string) {
      const ordered = [...files].sort((a, b) => b.n - a.n);
      for (const file of ordered) {
        const row = file.teams.find((t) => t.team === team);
        if (row) return { row, league: file.teams, season: file.season };
      }
      return null;
    }
    return { home: pick(homeName), away: pick(awayName) };
  }, [files, homeName, awayName]);

  if (!found.home || !found.away) return null;
  const league = found.home.league.length >= found.away.league.length ? found.home.league : found.away.league;
  const year = found.home.season;

  return (
    <div className="units">
      <h2 className="team-sched-title">Offense vs defense</h2>
      <p className="lede-note">
        {year} PPA and yards per play (CFBD, garbage time out) as FBS percentiles. Outward is better — defense is
        inverted. Offensive PPA by down sits under each radar. Not a live-model feature.
      </p>
      <div className="unit-legend">
        <span className="mux-swatch mux-away" />
        <TeamLink team={awayName} onOpen={onOpenTeam} />
        <span className="mux-swatch mux-home" />
        <TeamLink team={homeName} onOpen={onOpenTeam} />
      </div>
      <div className="unit-grid">
        <Clash
          offName={awayName}
          defName={homeName}
          off={found.away.row}
          def={found.home.row}
          offRow={awayRow}
          defRow={homeRow}
          league={league}
          predPts={predAway}
          offClass="mux-poly-away"
          defClass="mux-poly-home"
          onOpenTeam={onOpenTeam}
        />
        <Clash
          offName={homeName}
          defName={awayName}
          off={found.home.row}
          def={found.away.row}
          offRow={homeRow}
          defRow={awayRow}
          league={league}
          predPts={predHome}
          offClass="mux-poly-home"
          defClass="mux-poly-away"
          onOpenTeam={onOpenTeam}
        />
      </div>
    </div>
  );
}
