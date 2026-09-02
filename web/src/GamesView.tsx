import { useMemo, useState } from "react";
import { signed } from "./format";
import { gameAnalytics, gameKey, hitTone, pct, teamMap } from "./game";
import { etDay, isFinal, prettyDay } from "./score";
import { TeamLink } from "./TeamView";
import type { GamePred, RatingsFile } from "./types";

type SortKey = "week" | "date" | "away" | "home" | "win" | "margin" | "pom" | "tempo" | "residual";

export function GamesView({
  games,
  ratings,
  onOpenTeam,
  onOpenGame,
}: {
  games: GamePred[];
  ratings: RatingsFile | null;
  onOpenTeam: (team: string) => void;
  onOpenGame: (key: string) => void;
}) {
  const teams = useMemo(() => teamMap(ratings), [ratings]);
  const weeks = useMemo(() => [...new Set(games.map((g) => g.week))].sort((a, b) => a - b), [games]);
  const hasFinals = useMemo(() => games.some((g) => g.fbs_fbs && isFinal(g)), [games]);

  const [week, setWeek] = useState<number | "all">("all");
  const [q, setQ] = useState("");
  const [view, setView] = useState<"final" | "upcoming" | "all">("all");
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const out = games.filter((g) => {
      if (!g.fbs_fbs) return false;
      const done = isFinal(g);
      if (view === "final" && !done) return false;
      if (view === "upcoming" && done) return false;
      if (week !== "all" && g.week !== week) return false;
      if (needle && !`${g.away} ${g.home}`.toLowerCase().includes(needle)) return false;
      return true;
    });
    const scored = out.map((g) => {
      const a = gameAnalytics(g, teams);
      return { g, a, start: g.start ? Date.parse(g.start) : 0 };
    });
    scored.sort((x, y) => {
      let cmp = 0;
      if (sortKey === "week") cmp = x.g.week - y.g.week;
      else if (sortKey === "date") cmp = x.start - y.start || x.g.week - y.g.week;
      else if (sortKey === "away") cmp = x.g.away.localeCompare(y.g.away);
      else if (sortKey === "home") cmp = x.g.home.localeCompare(y.g.home);
      else if (sortKey === "win") cmp = x.g.home_win_prob - y.g.home_win_prob;
      else if (sortKey === "margin") cmp = x.g.pred_margin - y.g.pred_margin;
      else if (sortKey === "pom") cmp = (x.a.pomGap ?? -999) - (y.a.pomGap ?? -999);
      else if (sortKey === "tempo") cmp = (x.a.tempo ?? -1) - (y.a.tempo ?? -1);
      else cmp = (x.a.residual ?? -999) - (y.a.residual ?? -999);
      if (cmp === 0) cmp = x.start - y.start;
      return sortDir === "asc" ? cmp : -cmp;
    });
    return scored;
  }, [games, teams, week, q, view, sortKey, sortDir]);

  function onSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(key);
    setSortDir(key === "residual" ? "desc" : "asc");
  }

  function mark(key: SortKey, label: string) {
    return `${label}${sortKey === key ? (sortDir === "asc" ? " ↑" : " ↓") : ""}`;
  }

  return (
    <div>
      <p className="lede-note">
        {view === "final"
          ? "Pre-game projection against the final. Winner is straight-up accuracy. Bias is mean residual (actual home margin minus predicted). Click a row for the analytics breakdown."
          : view === "upcoming"
            ? "Ensemble projection, Pom gap, and tempo from the ratings board. Click a row for the analytics preview."
            : "Finals and upcoming on one slate. Click a row for the analytics preview or breakdown."}
      </p>
      <div className="toolbar">
        <div className="seg" role="group" aria-label="Games view">
          <button type="button" aria-pressed={view === "all"} onClick={() => setView("all")}>
            All
          </button>
          <button type="button" aria-pressed={view === "final"} onClick={() => setView("final")} disabled={!hasFinals}>
            Final
          </button>
          <button type="button" aria-pressed={view === "upcoming"} onClick={() => setView("upcoming")}>
            Upcoming
          </button>
        </div>
        <input
          type="search"
          placeholder="Team"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          aria-label="Filter games"
        />
        <select
          value={week}
          onChange={(e) => setWeek(e.target.value === "all" ? "all" : Number(e.target.value))}
          aria-label="Week"
        >
          <option value="all">All weeks</option>
          {weeks.map((w) => (
            <option key={w} value={w}>
              Week {w}
            </option>
          ))}
        </select>
        <span className="lede-note" style={{ margin: 0 }}>
          {rows.length} games
        </span>
      </div>
      <div className="table-wrap">
        <table className="games-table">
          <thead>
            <tr>
              <th colSpan={5} />
              <th className="group split" colSpan={3}>
                Projection
              </th>
              <th className="group split" colSpan={2}>
                Ratings
              </th>
              <th className="group split">Result</th>
            </tr>
            <tr>
              <th>
                <button type="button" onClick={() => onSort("week")}>
                  {mark("week", "Wk")}
                </button>
              </th>
              <th className="left">
                <button type="button" onClick={() => onSort("date")}>
                  {mark("date", "Date")}
                </button>
              </th>
              <th className="left">
                <button type="button" onClick={() => onSort("away")}>
                  {mark("away", "Away")}
                </button>
              </th>
              <th className="left">
                <button type="button" onClick={() => onSort("home")}>
                  {mark("home", "Home")}
                </button>
              </th>
              <th>Score</th>
              <th className="split" title="Predicted away–home score">
                Pred
              </th>
              <th title="Home win probability">
                <button type="button" onClick={() => onSort("win")}>
                  {mark("win", "Win%")}
                </button>
              </th>
              <th title="Predicted home margin">
                <button type="button" onClick={() => onSort("margin")}>
                  {mark("margin", "Margin")}
                </button>
              </th>
              <th className="split" title="Home Pom minus away Pom">
                <button type="button" onClick={() => onSort("pom")}>
                  {mark("pom", "Pom Δ")}
                </button>
              </th>
              <th title="Average of the two teams' scrimmage plays per game">
                <button type="button" onClick={() => onSort("tempo")}>
                  {mark("tempo", "Tempo")}
                </button>
              </th>
              <th className="split" title="Actual home margin minus predicted home margin">
                <button type="button" onClick={() => onSort("residual")}>
                  {mark("residual", "Resid")}
                </button>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ g, a }) => {
              const kick = etDay(g.start);
              return (
                <tr key={gameKey(g)} className="game-row" onClick={() => onOpenGame(gameKey(g))}>
                  <td>{g.week}</td>
                  <td className="left">{kick ? prettyDay(kick) : "—"}</td>
                  <td className="left">
                    <TeamLink team={g.away} onOpen={onOpenTeam} />
                    {g.neutral ? " (N)" : ""}
                  </td>
                  <td className="left">
                    <TeamLink team={g.home} onOpen={onOpenTeam} />
                  </td>
                  <td className={hitTone(a.su)}>{a.score ?? "—"}</td>
                  <td className="split">{a.pred}</td>
                  <td>{pct(g.home_win_prob)}</td>
                  <td>{signed(g.pred_margin)}</td>
                  <td className="split">{a.pomGap == null ? "—" : signed(a.pomGap)}</td>
                  <td>{a.tempo == null ? "—" : a.tempo.toFixed(1)}</td>
                  <td className="split">{a.residual == null ? "—" : signed(a.residual)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <details className="glossary">
        <summary>Column notes</summary>
        <dl>
          <dt>Win%</dt>
          <dd>Calibrated home win probability from the ensemble (TabPFN-3, with logistic fallback).</dd>
          <dt>Margin</dt>
          <dd>Predicted home scoring margin. Positive means the home team is projected in front.</dd>
          <dt>Pom Δ</dt>
          <dd>Home Pom minus away Pom. Points either side would beat an average FBS team by on a neutral field.</dd>
          <dt>Tempo</dt>
          <dd>Mean of the two teams' scrimmage plays per game from the ratings board.</dd>
          <dt>Resid</dt>
          <dd>Actual home margin minus predicted home margin. Positive means the home side outperformed the model.</dd>
          <dt>Bias</dt>
          <dd>Mean residual over scored games. Positive means home teams are running hotter than the projection.</dd>
        </dl>
      </details>
    </div>
  );
}
