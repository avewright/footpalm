import { useMemo } from "react";
import { signed } from "./format";
import { isFinal, prettyDay, etDay } from "./score";
import type { GamePred, RatingsFile, TeamRow } from "./types";

export function TeamLink({ team, onOpen }: { team: string; onOpen: (team: string) => void }) {
  return (
    <button type="button" className="team-link" onClick={() => onOpen(team)}>
      {team}
    </button>
  );
}

function pomOf(row: TeamRow) {
  return Number(row.pom ?? row.palm ?? 0);
}

function pct(p: number) {
  return `${(p * 100).toFixed(0)}%`;
}

function gameLine(g: GamePred, team: string) {
  const home = g.home === team;
  const opp = home ? g.away : g.home;
  const loc = home ? "vs" : "@";
  const predMargin = home ? g.pred_margin : -g.pred_margin;
  const winProb = home ? g.home_win_prob : 1 - g.home_win_prob;
  const ours = home ? g.actual_home : g.actual_away;
  const theirs = home ? g.actual_away : g.actual_home;
  const won = isFinal(g) && g.home_won != null ? (home ? Boolean(g.home_won) : !g.home_won) : null;
  const day = etDay(g.start);
  return { opp, loc, predMargin, winProb, ours, theirs, won, day, week: g.week };
}

export function TeamView({
  team,
  seasons,
  season,
  onSeason,
  onOpen,
  onClose,
  ratings,
  games,
}: {
  team: string;
  seasons: number[];
  season: number;
  onSeason: (season: number) => void;
  onOpen: (team: string) => void;
  onClose: () => void;
  ratings: RatingsFile | null;
  games: GamePred[];
}) {
  const row = ratings?.teams.find((t) => t.team === team) ?? null;
  const schedule = useMemo(() => {
    return games
      .filter((g) => g.home === team || g.away === team)
      .map((g) => ({ g, ...gameLine(g, team) }))
      .sort((a, b) => a.week - b.week || (a.g.start ?? "").localeCompare(b.g.start ?? ""));
  }, [games, team]);

  const finals = schedule.filter((row) => row.won != null);
  const suW = finals.filter((row) => row.won).length;

  return (
    <div>
      <div className="team-hero">
        <button type="button" className="team-back" onClick={onClose}>
          Back
        </button>
        <h1>{team}</h1>
        {row && <span className="quiet">{row.conf}</span>}
        <select value={season} onChange={(e) => onSeason(Number(e.target.value))} aria-label="Season">
          {seasons.map((year) => (
            <option key={year} value={year}>
              {year}
            </option>
          ))}
        </select>
      </div>

      {row ? (
        <>
          <dl className="team-stats">
            <div>
              <dt>Rank</dt>
              <dd>{row.rank}</dd>
            </div>
            <div>
              <dt>Record</dt>
              <dd>
                {row.wins}-{row.losses}
              </dd>
            </div>
            <div>
              <dt>Pom</dt>
              <dd>{signed(pomOf(row))}</dd>
            </div>
            <div>
              <dt>Elo</dt>
              <dd>{row.elo == null ? "—" : Math.round(row.elo)}</dd>
            </div>
            <div>
              <dt>AdjO</dt>
              <dd>{signed(row.adjo)}</dd>
            </div>
            <div>
              <dt>AdjD</dt>
              <dd>{signed(row.adjd)}</dd>
            </div>
            <div>
              <dt>AdjST</dt>
              <dd>{signed(row.adjst)}</dd>
            </div>
            <div>
              <dt>Tempo</dt>
              <dd>{row.tempo.toFixed(1)}</dd>
            </div>
            <div>
              <dt>SoS</dt>
              <dd>{signed(row.sos)}</dd>
            </div>
            <div>
              <dt>Luck</dt>
              <dd>{`${signed(row.luck * 100, 1)}%`}</dd>
            </div>
            {row.nil_roster != null && (
              <div>
                <dt>Roster</dt>
                <dd>
                  ${(row.nil_roster / 1_000_000).toFixed(1)}M{row.nil_quality === "modeled" ? "*" : ""}
                </dd>
              </div>
            )}
          </dl>
          {ratings?.method && <p className="lede-note">{ratings.method}</p>}
        </>
      ) : (
        <p className="lede-note">
          No {season} ratings for {team}. Pick another year or a different team.
        </p>
      )}

      <h2 className="team-sched-title">
        {season} schedule
        {finals.length ? ` · ${suW}–${finals.length - suW}` : ""}
      </h2>
      {schedule.length === 0 ? (
        <p className="lede-note">No games in the {season} file.</p>
      ) : (
        <div className="table-wrap team-sched">
          <table>
            <thead>
              <tr>
                <th>Wk</th>
                <th className="left">Date</th>
                <th className="team">Opponent</th>
                <th>Pred</th>
                <th>Win%</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {schedule.map(({ g, opp, loc, predMargin, winProb, ours, theirs, won, day }) => (
                <tr key={`${g.week}-${g.home}-${g.away}-${g.game_id ?? ""}`}>
                  <td>{g.week}</td>
                  <td className="left">{day ? prettyDay(day) : "—"}</td>
                  <td className="team">
                    {loc} <TeamLink team={opp} onOpen={onOpen} />
                    {g.neutral ? " (N)" : ""}
                  </td>
                  <td>{signed(predMargin)}</td>
                  <td>{pct(winProb)}</td>
                  <td className={won == null ? undefined : won ? "good" : "bad"}>
                    {won == null ? "—" : `${won ? "W" : "L"} ${ours?.toFixed(0)}–${theirs?.toFixed(0)}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
