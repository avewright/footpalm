import { useEffect, useMemo, useState } from "react";
import { evPct, gameFacts, gameKey, hitTone, lineOf, mlPair, pp, tone } from "./game";
import { etDay, isFinal, lastDay, prettyDay, record, summarize } from "./score";
import { TeamLink } from "./TeamView";
import type { GamePred } from "./types";

function Card({ title, card }: { title: string; card: ReturnType<typeof summarize> }) {
  if (!card.n) return null;
  return (
    <div className="score-card">
      <h3>{title}</h3>
      <dl className="metrics">
        <dt>Straight up</dt>
        <dd>{record(card.suW, card.suL)}</dd>
        <dt>ATS</dt>
        <dd>{record(card.atsW, card.atsL)}</dd>
        <dt>Brier</dt>
        <dd>{card.brier == null ? "—" : card.brier.toFixed(3)}</dd>
        <dt>Margin MAE</dt>
        <dd>{card.mae == null ? "—" : card.mae.toFixed(1)}</dd>
      </dl>
    </div>
  );
}

type SortKey = "week" | "date" | "away" | "home" | "spread" | "ev";

export function GamesView({
  games,
  onOpenTeam,
  onOpenGame,
}: {
  games: GamePred[];
  onOpenTeam: (team: string) => void;
  onOpenGame: (key: string) => void;
}) {
  const weeks = useMemo(() => [...new Set(games.map((g) => g.week))].sort((a, b) => a - b), [games]);
  const finals = useMemo(() => games.filter((g) => g.fbs_fbs && isFinal(g)), [games]);
  const hasFinals = finals.length > 0;
  const hasBooks = games.some((g) => g.books?.kalshi || g.books?.polymarket);
  const day = lastDay(finals);
  const last = useMemo(() => summarize(day ? finals.filter((g) => etDay(g.start) === day) : []), [finals, day]);
  const season = useMemo(() => summarize(finals), [finals]);

  const [week, setWeek] = useState<number | "all">("all");
  const [q, setQ] = useState("");
  const [listed, setListed] = useState(true);
  const [view, setView] = useState<"final" | "upcoming" | "all">("upcoming");
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  useEffect(() => {
    if (hasFinals) setView("final");
  }, [hasFinals]);

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const out = games.filter((g) => {
      if (!g.fbs_fbs) return false;
      const done = isFinal(g);
      if (view === "final" && !done) return false;
      if (view === "upcoming" && done) return false;
      if (listed && hasBooks && view !== "final" && !g.books?.kalshi && !g.books?.polymarket) return false;
      if (week !== "all" && g.week !== week) return false;
      if (needle && !`${g.away} ${g.home}`.toLowerCase().includes(needle)) return false;
      return true;
    });
    const scored = out.map((g) => {
      const facts = gameFacts(g);
      return { g, ...facts, start: g.start ? Date.parse(g.start) : 0 };
    });
    scored.sort((a, b) => {
      let cmp = 0;
      if (sortKey === "week") cmp = a.g.week - b.g.week;
      else if (sortKey === "date") cmp = a.start - b.start || a.g.week - b.g.week;
      else if (sortKey === "away") cmp = a.g.away.localeCompare(b.g.away);
      else if (sortKey === "home") cmp = a.g.home.localeCompare(b.g.home);
      else if (sortKey === "spread") cmp = (a.mkt ?? 99) - (b.mkt ?? 99);
      else cmp = (a.play?.ev ?? -99) - (b.play?.ev ?? -99);
      if (cmp === 0) cmp = a.start - b.start;
      return sortDir === "asc" ? cmp : -cmp;
    });
    return scored;
  }, [games, week, q, listed, hasBooks, view, sortKey, sortDir]);

  function onSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(key);
    setSortDir(key === "ev" ? "desc" : "asc");
  }

  function mark(key: SortKey, label: string) {
    return `${label}${sortKey === key ? (sortDir === "asc" ? " ↑" : " ↓") : ""}`;
  }

  return (
    <div>
      {hasFinals && (
        <div className="score-strip">
          {day && <Card title={prettyDay(day)} card={last} />}
          <Card title="Season" card={season} />
        </div>
      )}
      <p className="lede-note">
        {view === "final"
          ? "Frozen pre-game Vegas vs us, then the final. ATS is a holdout, not a training target. Click a row for the game."
          : "Vegas is the listed book (Kalshi / Polymarket). Our Prediction is the ensemble. Line is the home number. ML is home/away. Δ is who we like more than Vegas, in points and win probability. Click a row."}
      </p>
      <div className="toolbar">
        <div className="seg" role="group" aria-label="Games view">
          <button type="button" aria-pressed={view === "final"} onClick={() => setView("final")} disabled={!hasFinals}>
            Final
          </button>
          <button type="button" aria-pressed={view === "upcoming"} onClick={() => setView("upcoming")}>
            Upcoming
          </button>
          <button type="button" aria-pressed={view === "all"} onClick={() => setView("all")}>
            All
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
        {hasBooks && view !== "final" && (
          <label>
            <input type="checkbox" checked={listed} onChange={(e) => setListed(e.target.checked)} />
            Listed only
          </label>
        )}
        <span className="lede-note" style={{ margin: 0 }}>
          {rows.length} games
        </span>
      </div>
      <div className="table-wrap">
        <table className="games-table">
          <thead>
            <tr>
              <th colSpan={5} />
              <th className="group split" colSpan={2}>
                Vegas
              </th>
              <th className="group split" colSpan={2}>
                Our Prediction
              </th>
              <th className="group split">Difference</th>
              <th colSpan={2} />
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
              <th className="split">
                <button type="button" onClick={() => onSort("spread")}>
                  {mark("spread", "Line")}
                </button>
              </th>
              <th>ML</th>
              <th className="split">Line</th>
              <th>ML</th>
              <th className="split left">Δ</th>
              <th className="left">Pick</th>
              <th>
                <button type="button" onClick={() => onSort("ev")}>
                  {mark("ev", view === "final" ? "ATS" : "EV")}
                </button>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ g, play, us, mkt, mktAm, ourAm, dSpread, dMl, lean, hit, su, pick, score }) => {
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
                  <td className={hitTone(su)}>{score ?? "—"}</td>
                  <td className="split">{lineOf(mkt)}</td>
                  <td>{mlPair(mktAm)}</td>
                  <td className="split">{lineOf(us)}</td>
                  <td>{mlPair(ourAm)}</td>
                  <td className={`split left ${tone(play?.ev) ?? ""}`}>
                    {lean == null && dSpread == null && dMl == null ? (
                      "—"
                    ) : lean == null ? (
                      "Even"
                    ) : (
                      <>
                        {lean}
                        {dSpread != null ? ` ${Math.abs(dSpread).toFixed(1)}` : ""}
                        {dMl != null && Math.abs(dMl) >= 0.005 ? <div className="quiet">{pp(Math.abs(dMl))}</div> : null}
                      </>
                    )}
                  </td>
                  <td className="left">{pick ?? "—"}</td>
                  <td className={view === "final" ? hitTone(hit) : tone(play?.ev)}>
                    {view === "final" ? (hit == null ? "—" : hit ? "Cover" : "Miss") : play ? evPct(play.ev) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
