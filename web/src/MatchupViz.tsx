import { money, signed } from "./format";
import { pomOf } from "./game";
import { pctile, Radar } from "./Radar";
import { TeamLink } from "./TeamView";
import type { TeamRow } from "./types";

type Stat = {
  key: string;
  label: string;
  title: string;
  home: number | null;
  away: number | null;
  format: (n: number) => string;
  higherBetter: boolean;
  radar?: boolean;
};

function poolOf(league: TeamRow[], get: (row: TeamRow) => number | null | undefined) {
  return league.map(get).filter((n): n is number => n != null && !Number.isNaN(n));
}

function statsOf(home: TeamRow, away: TeamRow): Stat[] {
  return [
    {
      key: "rank",
      label: "Rank",
      title: "Pom rank on the board. Shorter bar is worse (rank 1 fills the track).",
      home: home.rank,
      away: away.rank,
      format: (n) => String(n),
      higherBetter: false,
    },
    {
      key: "pom",
      label: "Pom",
      title: "Points vs an average FBS team on a neutral field.",
      home: pomOf(home),
      away: pomOf(away),
      format: (n) => signed(n),
      higherBetter: true,
      radar: true,
    },
    {
      key: "elo",
      label: "Elo",
      title: "538-style margin-of-victory Elo.",
      home: home.elo ?? null,
      away: away.elo ?? null,
      format: (n) => String(Math.round(n)),
      higherBetter: true,
      radar: true,
    },
    {
      key: "adjo",
      label: "AdjO",
      title: "Opponent-adjusted offense, points per game vs an average defense.",
      home: home.adjo,
      away: away.adjo,
      format: (n) => signed(n),
      higherBetter: true,
      radar: true,
    },
    {
      key: "adjd",
      label: "AdjD",
      title: "Opponent-adjusted defense, points allowed vs an average offense. Lower is better.",
      home: home.adjd,
      away: away.adjd,
      format: (n) => signed(n),
      higherBetter: false,
      radar: true,
    },
    {
      key: "adjst",
      label: "AdjST",
      title: "Opponent-adjusted special teams per game.",
      home: home.adjst,
      away: away.adjst,
      format: (n) => signed(n),
      higherBetter: true,
    },
    {
      key: "tempo",
      label: "Tempo",
      title: "Scrimmage plays per team per game.",
      home: home.tempo,
      away: away.tempo,
      format: (n) => n.toFixed(1),
      higherBetter: true,
      radar: true,
    },
    {
      key: "sos",
      label: "SoS",
      title: "Average opponent Pom in games already played.",
      home: home.sos,
      away: away.sos,
      format: (n) => signed(n),
      higherBetter: true,
      radar: true,
    },
    {
      key: "luck",
      label: "Luck",
      title: "Actual win rate minus expected from score margins.",
      home: home.luck,
      away: away.luck,
      format: (n) => `${signed(n * 100, 1)}%`,
      higherBetter: true,
    },
    {
      key: "roster",
      label: "Roster",
      title: "Football roster estimate. Context only.",
      home: home.nil_roster ?? null,
      away: away.nil_roster ?? null,
      format: (n) => money(n),
      higherBetter: true,
    },
  ];
}

function Bars({
  rows,
}: {
  rows: { key: string; label: string; title: string; home: string; away: string; pHome: number; pAway: number }[];
}) {
  return (
    <div className="mux-bars" role="list">
      {rows.map((row) => (
        <div key={row.key} className="mux-row" title={row.title} role="listitem">
          <span className="mux-val mux-away">{row.away}</span>
          <span className="mux-track mux-left">
            <i style={{ width: `${Math.max(row.pAway * 100, 2)}%` }} />
          </span>
          <span className="mux-lab" title={row.title}>
            {row.label}
          </span>
          <span className="mux-track mux-right">
            <i style={{ width: `${Math.max(row.pHome * 100, 2)}%` }} />
          </span>
          <span className="mux-val mux-home">{row.home}</span>
        </div>
      ))}
    </div>
  );
}

export function MatchupViz({
  home,
  away,
  homeName,
  awayName,
  league,
  onOpenTeam,
}: {
  home: TeamRow;
  away: TeamRow;
  homeName: string;
  awayName: string;
  league: TeamRow[];
  onOpenTeam: (team: string) => void;
}) {
  const stats = statsOf(home, away).filter((s) => s.home != null || s.away != null);
  const getters: Record<string, (row: TeamRow) => number | null | undefined> = {
    rank: (t) => t.rank,
    pom: pomOf,
    elo: (t) => t.elo,
    adjo: (t) => t.adjo,
    adjd: (t) => t.adjd,
    adjst: (t) => t.adjst,
    tempo: (t) => t.tempo,
    sos: (t) => t.sos,
    luck: (t) => t.luck,
    roster: (t) => t.nil_roster,
  };

  const rows = stats
    .map((s) => {
      const pool = poolOf(league, getters[s.key] ?? (() => null));
      const pHome = s.home == null ? 0 : pctile(s.home, pool, s.higherBetter);
      const pAway = s.away == null ? 0 : pctile(s.away, pool, s.higherBetter);
      return {
        ...s,
        home: s.home == null ? "—" : s.format(s.home),
        away: s.away == null ? "—" : s.format(s.away),
        pHome,
        pAway,
        rawHome: s.home,
        rawAway: s.away,
      };
    })
    .filter((s) => s.home !== "—" || s.away !== "—");

  const radarAxes = rows
    .filter((s) => s.radar && s.rawHome != null && s.rawAway != null)
    .map((s) => ({ label: s.label, title: s.title, home: s.pHome, away: s.pAway }));

  const edges = rows
    .filter((s) => s.rawHome != null && s.rawAway != null && s.key !== "rank")
    .map((s) => ({
      label: s.label,
      gap: s.pHome - s.pAway,
      who: s.pHome === s.pAway ? null : s.pHome > s.pAway ? homeName : awayName,
    }))
    .filter((e) => e.who && Math.abs(e.gap) >= 0.08)
    .sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap))
    .slice(0, 2);

  return (
    <div className="mux">
      <div className="mux-legend">
        <span className="mux-swatch mux-away" />
        <TeamLink team={awayName} onOpen={onOpenTeam} />
        <span className="mux-swatch mux-home" />
        <TeamLink team={homeName} onOpen={onOpenTeam} />
        <span className="quiet mux-legend-note">Bar length is FBS percentile. Outer radar ring is 100th.</span>
      </div>
      <div className="mux-grid-main">
        <div className="mux-radar-wrap">
          {radarAxes.length >= 3 && (
            <Radar
              axes={radarAxes.map((s) => ({
                label: s.label,
                a: s.away,
                b: s.home,
                title: s.title,
              }))}
              label="Team profile radar"
            />
          )}
          {edges.length > 0 && (
            <p className="mux-edge-note">
              {edges.map((e, i) => (
                <span key={e.label}>
                  {i ? " · " : ""}
                  {e.who} leads {e.label}
                </span>
              ))}
            </p>
          )}
        </div>
        <Bars rows={rows} />
      </div>
    </div>
  );
}
