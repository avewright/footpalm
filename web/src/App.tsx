import { useEffect, useState } from "react";
import { AskView } from "./AskView";
import { AuthBar } from "./AuthBar";
import { ConferenceView } from "./ConferenceView";
import { GamesView } from "./GamesView";
import { GameView } from "./GameView";
import { ModelsView } from "./ModelsView";
import { MoneyView } from "./MoneyView";
import { MyModelView } from "./MyModelView";
import { RatingsTable } from "./RatingsTable";
import { TeamView } from "./TeamView";
import {
  createModel,
  deleteModel,
  fetchActive,
  fetchCatalog,
  fetchMe,
  fetchMine,
  login,
  logout,
  patchModel,
  type ModelCard,
  type SessionUser,
} from "./accounts";
import { findGame } from "./game";
import { signed } from "./format";
import { clearUserModel, loadUserModel, type UserModel } from "./mymodel";
import type { GamePred, IndexFile, MoneyFile, RatingsFile } from "./types";

const TABS = ["ratings", "games", "models", "mine", "ask", "money"] as const;
type Tab = (typeof TABS)[number];
type View = Tab | "team" | "game" | "conference";

const TAB_LABEL: Record<Tab, string> = {
  ratings: "Ratings",
  games: "Games",
  models: "Models",
  mine: "My Model",
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
    if (q === "conference" && search().get("conference")) return "conference";
    return TABS.includes(q as Tab) ? (q as Tab) : "ratings";
  });
  const [back, setBack] = useState<View>("ratings");
  const [team, setTeam] = useState(() => search().get("team") ?? "");
  const [game, setGame] = useState(() => search().get("game") ?? "");
  const [conference, setConference] = useState(() => search().get("conference") ?? "");
  const [ratings, setRatings] = useState<RatingsFile | null>(null);
  const [games, setGames] = useState<GamePred[]>([]);
  const [money, setMoney] = useState<MoneyFile | null>(null);
  const [user, setUser] = useState<SessionUser | null>(null);
  const [userModels, setUserModels] = useState<UserModel[]>([]);
  const [userModel, setUserModel] = useState<UserModel | null>(null);
  const [catalog, setCatalog] = useState<ModelCard[]>([]);
  const [authError, setAuthError] = useState<string | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [authTick, setAuthTick] = useState(0);
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
    if (conference) q.set("conference", conference);
    else q.delete("conference");
    const next = `?${q}`;
    if (window.location.search !== next) window.history.replaceState(null, "", next);
  }, [tab, season, team, game, conference]);

  useEffect(() => {
    setError(null);
    Promise.all([
      loadJson<RatingsFile | null>(`/data/ratings-${season}.json`, null),
      loadJson<{ games: GamePred[] }>(`/data/predictions-${season}.json`, { games: [] }),
      loadJson<MoneyFile | null>("/data/money.json", null),
    ])
      .then(([rt, pred, m]) => {
        setRatings(rt);
        setGames(pred?.games ?? []);
        setMoney(m);
        if (!rt) setError(`missing ${season} ratings — run the build`);
      })
      .catch((err: Error) => setError(err.message));
  }, [season]);

  useEffect(() => {
    fetchMe().then(setUser);
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setModelsError(null);
      try {
        const [active, mine, board] = await Promise.all([
          fetchActive(season),
          user ? fetchMine(season) : Promise.resolve([] as UserModel[]),
          fetchCatalog(season),
        ]);
        if (cancelled) return;
        setUserModel(active);
        setUserModels(mine);
        setCatalog(board);
        if (user && !mine.length) {
          const local = loadUserModel(season);
          if (local) {
            const saved = await createModel(local);
            clearUserModel(season);
            if (cancelled) return;
            setUserModel(saved);
            setUserModels([saved]);
            setCatalog(await fetchCatalog(season));
          }
        }
      } catch (err) {
        if (!cancelled) {
          setModelsError(err instanceof Error ? err.message : "Could not load models.");
          if (!user) {
            setUserModel(loadUserModel(season));
            setUserModels([]);
            setCatalog([]);
          }
        }
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [season, user]);

  function openTeam(name: string) {
    if (tab !== "team") setBack(tab);
    setTeam(name);
    setTab("team");
  }

  function closeTeam() {
    setTeam("");
    setTab(back === "team" ? "ratings" : back);
  }

  function openGame(key: string) {
    if (tab !== "game") setBack(tab);
    setGame(key);
    setTab("game");
  }

  function closeGame() {
    setGame("");
    setTab(back === "game" ? "ratings" : back);
  }

  function openConference(name: string) {
    if (!name) return;
    if (tab !== "conference") setBack(tab);
    setConference(name);
    setTab("conference");
  }

  function closeConference() {
    setConference("");
    setTab(back === "conference" ? "ratings" : back);
  }

  function goHome() {
    setTeam("");
    setGame("");
    setConference("");
    setTab("ratings");
  }

  async function refreshModels() {
    const [active, mine, board] = await Promise.all([
      fetchActive(season),
      user ? fetchMine(season) : Promise.resolve([] as UserModel[]),
      fetchCatalog(season),
    ]);
    setUserModel(active);
    setUserModels(mine);
    setCatalog(board);
  }

  async function onLogin(username: string) {
    setAuthError(null);
    try {
      setUser(await login(username));
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Could not set name.");
      throw err;
    }
  }

  async function onLogout() {
    try {
      await logout();
    } catch {
      /* still sign out locally */
    }
    setUser(null);
    setUserModel(null);
    setUserModels([]);
  }

  async function onUploadModel(next: UserModel) {
    const saved = await createModel(next);
    setUserModel(saved);
    await refreshModels();
  }

  async function onActivateModel(id: string) {
    const saved = await patchModel(id, { active: true });
    setUserModel(saved);
    await refreshModels();
  }

  async function onPublishModel(id: string, published: boolean) {
    await patchModel(id, { published });
    await refreshModels();
  }

  async function onRemoveModel(id: string) {
    await deleteModel(id);
    await refreshModels();
  }

  const selected = findGame(games, game);

  return (
    <div className={`page${tab === "ask" ? " page-ask" : ""}`}>
      <header className="top">
        <button type="button" className="brand" onClick={goHome} aria-label="FootPalm home">
          <img className="brand-mark" src="/logo.svg" width="22" height="22" alt="" />
          FootPalm
        </button>
        <nav className="tabs">
          {TABS.filter((id) => id !== "mine" && id !== "models").map((id) => (
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
          <button type="button" className="tab-link" aria-pressed={tab === "models"} onClick={() => setTab("models")}>
            Models
          </button>
          <button type="button" className="tab-link" aria-pressed={tab === "mine"} onClick={() => setTab("mine")}>
            My Model
          </button>
          <AuthBar
            user={user}
            error={authError}
            openSignal={authTick}
            onLogin={onLogin}
            onLogout={() => void onLogout()}
          />
          {tab !== "team" && tab !== "game" && tab !== "conference" && (
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

      {tab !== "ask" && tab !== "team" && tab !== "game" && tab !== "mine" && tab !== "models" && tab !== "conference" && (
        <p className="meta">
          {ratings
            ? `${ratings.week === 0 ? "Week 0 · " : ""}${ratings.teams.length} teams · ${ratings.plays_per_game} plays/game · home-field ${signed(ratings.home_adv_epa * ratings.plays_per_game, 1)} pts`
            : (error ?? "Loading…")}
        </p>
      )}
      {tab === "ratings" && ratings?.method && <p className="lede-note">{ratings.method}</p>}

      {tab === "ratings" && ratings && (
        <RatingsTable data={ratings} onOpenTeam={openTeam} onOpenConference={openConference} />
      )}
      {tab === "games" && (
        <GamesView games={games} ratings={ratings} onOpenTeam={openTeam} onOpenGame={openGame} />
      )}
      {tab === "models" && (
        <ModelsView
          season={season}
          user={user}
          catalog={catalog}
          error={modelsError}
          onNeedLogin={() => setAuthTick((n) => n + 1)}
        />
      )}
      {tab === "mine" && (
        <MyModelView
          season={season}
          games={games}
          user={user}
          models={userModels}
          activeId={userModel?.id ?? null}
          onUpload={onUploadModel}
          onActivate={onActivateModel}
          onPublish={onPublishModel}
          onRemove={onRemoveModel}
          onNeedLogin={() => setAuthTick((n) => n + 1)}
          onOpenTeam={openTeam}
          onOpenGame={openGame}
        />
      )}
      {tab === "team" && team && (
        <TeamView
          team={team}
          seasons={seasons}
          season={season}
          onSeason={setSeason}
          onOpen={openTeam}
          onOpenConference={openConference}
          onOpenGame={openGame}
          onClose={closeTeam}
          ratings={ratings}
          games={games}
        />
      )}
      {tab === "game" && selected && (
        <GameView
          game={selected}
          ratings={ratings}
          userModel={userModel}
          userModels={userModels}
          onOpenTeam={openTeam}
          onOpenConference={openConference}
          onClose={closeGame}
        />
      )}
      {tab === "game" && game && !selected && (
        <p className="lede-note">
          No game {game} in {season}.{" "}
          <button type="button" className="team-link" onClick={closeGame}>
            Back
          </button>
        </p>
      )}
      {tab === "conference" && conference && (
        <ConferenceView
          conference={conference}
          seasons={seasons}
          season={season}
          onSeason={setSeason}
          onOpenTeam={openTeam}
          onOpenGame={openGame}
          onClose={closeConference}
          ratings={ratings}
          games={games}
        />
      )}
      {tab === "ask" && <AskView key={askTick} season={season} onOpenTeam={openTeam} />}
      {tab === "money" && money && (
        <MoneyView data={money} onOpenTeam={openTeam} onOpenConference={openConference} />
      )}

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
