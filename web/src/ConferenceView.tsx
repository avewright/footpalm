import { useMemo } from "react";
import { signed } from "./format";
import { gameKey, pomOf } from "./game";
import { etDay, isFinal, prettyDay } from "./score";
import { TeamLink } from "./TeamView";
import type { GamePred, RatingsFile, TeamRow } from "./types";

function avg(rows: TeamRow[], value: (row: TeamRow) => number | null | undefined) {
  const vals = rows.map(value).filter((n): n is number => n != null && !Number.isNaN(n));
  if (!vals.length) return null;
  return vals.reduce((sum, n) => sum + n, 0) / vals.length;
}

function confRecord(team: string, games: GamePred[], members: Set<string>) {
  let w = 0;
  let l = 0;
  for (const g of games) {
    if (!isFinal(g) || g.home_won == null) continue;
    if (!members.has(g.home) || !members.has(g.away)) continue;
    if (g.home !== team && g.away !== team) continue;
    if (g.home === team ? Boolean(g.home_won) : !g.home_won) w += 1;
    else l += 1;
  }
  return { w, l };
}

export function ConferenceView({
  conference,
  seasons,
  season,
  onSeason,
  onOpenTeam,
  onOpenGame,
  onClose,
  ratings,
  games,
}: {
  conference: string;
  seasons: number[];
  season: number;
  onSeason: (season: number) => void;
  onOpenTeam: (team: string) => void;
  onOpenGame: (key: string) => void;
  onClose: () => void;
  ratings: RatingsFile | null;
  games: GamePred[];
}) {
  const teams = useMemo(() => {
    return (ratings?.teams.filter((row) => row.conf === conference) ?? [])
      .slice()
      .sort((a, b) => pomOf(b) - pomOf(a) || a.rank - b.rank);
  }, [ratings, conference]);
  const members = useMemo(() => new Set(teams.map((row) => row.team)), [teams]);
  const records = useMemo(() => {
    const map = new Map<string, { w: number; l: number }>();
    for (const row of teams) map.set(row.team, confRecord(row.team, games, members));
    return map;
  }, [teams, games, members]);
  const schedule = useMemo(() => {
    return games
      .filter((g) => g.fbs_fbs && members.has(g.home) && members.has(g.away))
      .sort((a, b) => a.week - b.week || (a.start ?? "").localeCompare(b.start ?? ""));
  }, [games, members]);

  const meanPom = avg(teams, pomOf);
  const meanElo = avg(teams, (row) => row.elo);
  const meanAdjO = avg(teams, (row) => row.adjo);
  const meanAdjD = avg(teams, (row) => row.adjd);
  const meanTempo = avg(teams, (row) => row.tempo);
  const wins = teams.reduce((sum, row) => sum + row.wins, 0);
  const losses = teams.reduce((sum, row) => sum + row.losses, 0);
  const confGames = schedule.filter(isFinal).length;

  return (
    <div>
      <div className="team-hero">
        <button type="button" className="team-back" onClick={onClose}>
          Back
        </button>
        <h1>{conference}</h1>
        <span className="quiet">{teams.length} teams</span>
        <select value={season} onChange={(e) => onSeason(Number(e.target.value))} aria-label="Season">
          {seasons.map((year) => (
            <option key={year} value={year}>
              {year}
            </option>
          ))}
        </select>
      </div>

      {teams.length ? (
        <dl className="team-stats">
          <div>
            <dt>Record</dt>
            <dd>
              {wins}-{losses}
            </dd>
          </div>
          <div>
            <dt>Conf games</dt>
            <dd>{confGames}</dd>
          </div>
          <div>
            <dt>Mean Pom</dt>
            <dd>{meanPom == null ? "—" : signed(meanPom)}</dd>
          </div>
          <div>
            <dt>Mean Elo</dt>
            <dd>{meanElo == null ? "—" : Math.round(meanElo)}</dd>
          </div>
          <div>
            <dt>Mean AdjO</dt>
            <dd>{meanAdjO == null ? "—" : signed(meanAdjO)}</dd>
          </div>
          <div>
            <dt>Mean AdjD</dt>
            <dd>{meanAdjD == null ? "—" : signed(meanAdjD)}</dd>
          </div>
          <div>
            <dt>Mean Tempo</dt>
            <dd>{meanTempo == null ? "—" : meanTempo.toFixed(1)}</dd>
          </div>
        </dl>
      ) : (
        <p className="lede-note">
          No {season} ratings for {conference}. Pick another year or a different conference.
        </p>
      )}

      {teams.length > 0 && (
        <>
          <h2 className="team-sched-title">{season} standings</h2>
          <div className="table-wrap team-sched">
            <table>
              <thead>
                <tr>
                  <th>Rk</th>
                  <th className="team">Team</th>
                  <th className="record">W-L</th>
                  <th className="record">Conf</th>
                  <th>Pom</th>
                  <th>Elo</th>
                  <th>AdjO</th>
                  <th>AdjD</th>
                  <th>Tempo</th>
                  <th>SoS</th>
                </tr>
              </thead>
              <tbody>
                {teams.map((row) => {
                  const conf = records.get(row.team);
                  return (
                    <tr key={row.team}>
                      <td>{row.rank}</td>
                      <td className="team">
                        <TeamLink team={row.team} onOpen={onOpenTeam} />
                      </td>
                      <td className="record">
                        {row.wins}-{row.losses}
                      </td>
                      <td className="record">{conf ? `${conf.w}-${conf.l}` : "0-0"}</td>
                      <td>{signed(pomOf(row))}</td>
                      <td>{row.elo == null ? "—" : Math.round(row.elo)}</td>
                      <td>{signed(row.adjo)}</td>
                      <td>{signed(row.adjd)}</td>
                      <td>{row.tempo.toFixed(1)}</td>
                      <td>{signed(row.sos)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      <h2 className="team-sched-title">{season} conference games</h2>
      {schedule.length === 0 ? (
        <p className="lede-note">No conference games in the {season} file.</p>
      ) : (
        <div className="table-wrap team-sched">
          <table>
            <thead>
              <tr>
                <th>Wk</th>
                <th className="left">Date</th>
                <th className="team">Away</th>
                <th className="team">Home</th>
                <th>Pred</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {schedule.map((g) => {
                const day = etDay(g.start);
                const done = isFinal(g);
                return (
                  <tr key={gameKey(g)} className="game-row" onClick={() => onOpenGame(gameKey(g))}>
                    <td>{g.week}</td>
                    <td className="left">{day ? prettyDay(day) : "—"}</td>
                    <td className="team">
                      <TeamLink team={g.away} onOpen={onOpenTeam} />
                    </td>
                    <td className="team">
                      <TeamLink team={g.home} onOpen={onOpenTeam} />
                    </td>
                    <td>
                      {g.pred_away.toFixed(0)}–{g.pred_home.toFixed(0)}
                    </td>
                    <td>
                      {done ? `${g.actual_away?.toFixed(0)}–${g.actual_home?.toFixed(0)}` : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
