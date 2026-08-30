import { useEffect, useState } from "react";
import { AskView } from "./AskView";
import { GamesView } from "./GamesView";
import { GameView } from "./GameView";
import { GraphView } from "./GraphView";
import { MoneyView } from "./MoneyView";
import { RatingsTable } from "./RatingsTable";
import { TeamView } from "./TeamView";
import { findGame } from "./game";
import { signed } from "./format";
import type { GamePred, GraphFile, IndexFile, MoneyFile, RatingsFile } from "./types";

const TABS = ["ratings", "games", "graph", "ask", "money"] as const;
type Tab = (typeof TABS)[number];
type View = Tab | "team" | "game";

const TAB_LABEL: Record<Tab, string> = {
  ratings: "Ratings",
  games: "Games",
  graph: "Graph",
  ask: "Ask",
  money: "Money",
};

function search() {
  return new URLSearchParams(window.location.search);
}

async function loadJson<T>(url: string, fallback: T): Promise<T> {
  const response = await fetch(url);
  const type = response.headers.get("content-type") ?? "";
  if (!response.ok || !type.includes("json")) return fallback;
  return response.json() as Promise<T>;
}

export function App() {
  const [seasons, setSeasons] = useState<number[]>([2026, 2025]);
  const [season, setSeason] = useState(() => Number(search().get("season")) || 2026);
  const [tab, setTab] = useState<View>(() => {
    const q = search().get("tab");
    if (q === "team" && search().get("team")) return "team";
    if (q === "game" && search().get("game")) return "game";
    return TABS.includes(q as Tab) ? (q as Tab) : "games";
  });
  const [back, setBack] = useState<View>("games");
  const [team, setTeam] = useState(() => search().get("team") ?? "");
  const [game, setGame] = useState(() => search().get("game") ?? "");
  const [ratings, setRatings] = useState<RatingsFile | null>(null);
  const [games, setGames] = useState<GamePred[]>([]);
  const [graph, setGraph] = useState<GraphFile | null>(null);
  const [money, setMoney] = useState<MoneyFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [askTick, setAskTick] = useState(0);

  useEffect(() => {
    loadJson<IndexFile>("/data/index.json", { seasons: [] }).then((idx) => {
      const years = idx.seasons.map((s) => s.season).sort((a, b) => b - a);
      if (!years.length) return;
      setSeasons(years);
      setSeason((current) => (years.includes(current) ? current : years[0]));
    });
  }, []);

  useEffect(() => {
    const q = search();
    q.set("tab", tab);
    q.set("season", String(season));
    if (team) q.set("team", team);
    else q.delete("team");
    if (game) q.set("game", game);
    else q.delete("game");
    const next = `?${q}`;
    if (window.location.search !== next) window.history.replaceState(null, "", next);
  }, [tab, season, team, game]);

  useEffect(() => {
    setError(null);
    Promise.all([
      loadJson<RatingsFile | null>(`/data/ratings-${season}.json`, null),
      loadJson<{ games: GamePred[] }>(`/data/predictions-${season}.json`, { games: [] }),
      loadJson<GraphFile | null>(`/data/graph-${season}.json`, null),
      loadJson<MoneyFile | null>("/data/money.json", null),
    ])
      .then(([rt, pred, g, m]) => {
        setRatings(rt);
        setGames(pred?.games ?? []);
        setGraph(g);
        setMoney(m);
        if (!rt) setError(`missing ${season} ratings — run the build`);
      })
      .catch((err: Error) => setError(err.message));
  }, [season]);

  function openTeam(name: string) {
    if (tab !== "team") setBack(tab);
    setTeam(name);
    setTab("team");
  }

  function closeTeam() {
    setTeam("");
    setTab(back === "team" ? "games" : back);
  }

  function openGame(key: string) {
    if (tab !== "game") setBack(tab);
    setGame(key);
    setTab("game");
  }

  function closeGame() {
    setGame("");
    setTab(back === "game" ? "games" : back);
  }

  const selected = findGame(games, game);

  return (
    <div className={`page${tab === "ask" ? " page-ask" : ""}`}>
      <header className="top">
        <div className="brand">
          <img className="brand-mark" src="/logo.svg" width="22" height="22" alt="" />
          FootPalm
        </div>
        <nav className="tabs">
          {TABS.map((id) => (
            <button key={id} type="button" aria-pressed={tab === id} onClick={() => setTab(id)}>
              {TAB_LABEL[id]}
            </button>
          ))}
        </nav>
        <div className="top-right">
          {tab === "ask" && (
            <button type="button" className="ask-new" onClick={() => setAskTick((n) => n + 1)}>
              New chat
            </button>
          )}
          {tab !== "team" && tab !== "game" && (
            <select value={season} onChange={(e) => setSeason(Number(e.target.value))} aria-label="Season">
              {seasons.map((year) => (
                <option key={year} value={year}>
                  {year}
                </option>
              ))}
            </select>
          )}
        </div>
      </header>

      {tab !== "ask" && tab !== "team" && tab !== "game" && (
        <p className="meta">
          {ratings
            ? `${ratings.week === 0 ? "Week 0 · " : ""}${ratings.teams.length} teams · ${ratings.plays_per_game} plays/game · home-field ${signed(ratings.home_adv_epa * ratings.plays_per_game, 1)} pts`
            : (error ?? "Loading…")}
        </p>
      )}
      {tab === "ratings" && ratings?.method && <p className="lede-note">{ratings.method}</p>}

      {tab === "ratings" && ratings && <RatingsTable data={ratings} onOpenTeam={openTeam} />}
      {tab === "games" && <GamesView games={games} onOpenTeam={openTeam} onOpenGame={openGame} />}
      {tab === "team" && team && (
        <TeamView
          team={team}
          seasons={seasons}
          season={season}
          onSeason={setSeason}
          onOpen={openTeam}
          onOpenGame={openGame}
          onClose={closeTeam}
          ratings={ratings}
          games={games}
        />
      )}
      {tab === "game" && selected && <GameView game={selected} onOpenTeam={openTeam} onClose={closeGame} />}
      {tab === "game" && game && !selected && (
        <p className="lede-note">
          No game {game} in {season}.{" "}
          <button type="button" className="team-link" onClick={closeGame}>
            Back
          </button>
        </p>
      )}
      {tab === "graph" && graph && (
        <GraphView graph={graph} team={team} onSelectTeam={setTeam} onOpenTeam={openTeam} />
      )}
      {tab === "graph" && !graph && <p className="lede-note">No graph file for {season} yet.</p>}
      {tab === "ask" && <AskView key={askTick} season={season} />}
      {tab === "money" && money && <MoneyView data={money} onOpenTeam={openTeam} />}

      {tab === "ratings" && (
        <details className="glossary">
          <summary>Column notes</summary>
          <dl>
            <dt>Rk</dt>
            <dd>Rank by Pom. Elo is a second rating, not the sort key.</dd>
            <dt>Pom</dt>
            <dd>
              Points this team would beat an average FBS team by on a neutral field. AdjO − AdjD + AdjST, from
              opponent-adjusted EPA. This is the board.
            </dd>
            <dt>Elo</dt>
            <dd>
              538-style margin-of-victory Elo. Everyone starts at 1500. K=20, home field is +55 Elo, each season
              reverts 75% of the way back to 1500. Blowouts move it more than one-score games. It only sees wins,
              losses, and margins — not EPA.
            </dd>
            <dt>AdjO</dt>
            <dd>Opponent-adjusted offensive EPA, scaled to points per game against an average defense. Higher is better.</dd>
            <dt>AdjD</dt>
            <dd>
              Opponent-adjusted defensive EPA, scaled to points allowed against an average offense. Lower (more
              negative) is better.
            </dd>
            <dt>AdjST</dt>
            <dd>Opponent-adjusted special-teams EPA per game.</dd>
            <dt>Tempo</dt>
            <dd>Scrimmage plays per team per game. Faster teams run more plays.</dd>
            <dt>SoS</dt>
            <dd>Average opponent Pom in games already played.</dd>
            <dt>Luck</dt>
            <dd>
              Actual win rate minus the win rate implied by score margins (σ=9.5). Positive means they won more
              close games than the scores deserved.
            </dd>
            <dt>Roster</dt>
            <dd>
              2026 football payroll estimate: school revenue share to football plus third-party NIL.
              Median of nil-ncaa.com, College Front Office, NIL Standard, and Sideline football
              splits, then floored by the CBS/247 industry tiers. Sideline’s Texas $73.9M is
              all-sports, not football. G6 marked * is modeled. Context only. Not a model feature.
            </dd>
          </dl>
        </details>
      )}
    </div>
  );
}
