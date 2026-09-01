import { useEffect, useState } from "react";
import { signed } from "./format";
import { gameAnalytics, hitTone, MODEL_IDS, pct, pickOf, teamMap } from "./game";
import { marketSpread } from "./ev";
import { MatchupViz } from "./MatchupViz";
import { prettyWhen } from "./score";
import { TeamLink } from "./TeamView";
import { UnitsView } from "./UnitsView";
import { pickOfModel, predMargin, type UserModel } from "./mymodel";
import type { EspnQb, GamePred, RatingsFile, UnitsFile } from "./types";

function qbLine(rows?: EspnQb[]) {
  if (!rows?.length) return "—";
  return rows
    .slice(0, 3)
    .map((q) => {
      const year = q.year ? ` ${q.year}` : "";
      const jersey = q.jersey ? ` #${q.jersey}` : "";
      return `${q.name}${jersey}${year}`;
    })
    .join(" · ");
}

function favoriteLine(home: string, away: string, homeSpread: number | null | undefined) {
  if (homeSpread == null || Number.isNaN(homeSpread)) return "—";
  if (Math.abs(homeSpread) < 0.05) return "PK";
  return homeSpread < 0 ? `${home} ${signed(homeSpread)}` : `${away} ${signed(-homeSpread)}`;
}

function weatherBits(game: GamePred) {
  const w = game.espn?.weather;
  if (!w) return [];
  const bits: string[] = [];
  if (w.temperature != null) bits.push(`${Math.round(w.temperature)}°`);
  if (w.gust != null) bits.push(`${Math.round(w.gust)} mph wind`);
  if (w.precipitation != null) bits.push(`${Math.round(w.precipitation)}% rain`);
  if (w.condition && !/^\d+$/.test(w.condition)) bits.push(w.condition);
  if (w.city) bits.push([w.city, w.state].filter(Boolean).join(", "));
  return bits;
}

function Row({
  label,
  value,
  tone: t,
  hint,
}: {
  label: string;
  value: string;
  tone?: string;
  hint?: string;
}) {
  return (
    <div className="slip-row" title={hint}>
      <span className="quiet">{label}</span>
      <span className={t}>{value}</span>
    </div>
  );
}

function WinMeter({ game }: { game: GamePred }) {
  const homeP = game.home_win_prob;
  const awayP = 1 - homeP;
  return (
    <div
      className="win-meter"
      role="img"
      aria-label={`${game.away} ${pct(awayP)}, ${game.home} ${pct(homeP)}`}
    >
      <div className="win-meter-away" style={{ flexGrow: Math.max(awayP, 0.04), flexBasis: 0 }}>
        {awayP >= 0.12 ? (
          <span>
            {game.away} {pct(awayP)}
          </span>
        ) : null}
      </div>
      <div className="win-meter-home" style={{ flexGrow: Math.max(homeP, 0.04), flexBasis: 0 }}>
        {homeP >= 0.12 ? (
          <span>
            {game.home} {pct(homeP)}
          </span>
        ) : null}
      </div>
    </div>
  );
}

export function GameView({
  game,
  ratings,
  userModel,
  onOpenTeam,
  onClose,
}: {
  game: GamePred;
  ratings: RatingsFile | null;
  userModel?: UserModel | null;
  onOpenTeam: (team: string) => void;
  onClose: () => void;
}) {
  const teams = teamMap(ratings);
  const a = gameAnalytics(game, teams);
  const yours = pickOfModel(userModel, game);
  const when = prettyWhen(game.start);
  const models = MODEL_IDS.map((id) => ({ id, pick: pickOf(game, id) })).filter((row) => row.pick);
  const vegas = marketSpread(game) ?? game.espn?.odds?.spread ?? null;
  const weather = weatherBits(game);
  const field = [
    game.espn?.weather?.venue || game.espn?.venue?.venue,
    game.espn?.weather?.grass === false || game.espn?.venue?.grass === false
      ? "turf"
      : game.espn?.weather?.grass || game.espn?.venue?.grass
        ? "grass"
        : null,
  ]
    .filter(Boolean)
    .join(" · ");
  const kind = a.done ? "Analytics Breakdown" : "Analytics Preview";
  const lede = a.done
    ? `${game.away} ${a.score ?? ""} ${game.home}. Model had ${a.favorite} at ${pct(a.favP)} / ${signed(a.homeFav ? game.pred_margin : -game.pred_margin)}. Residual ${a.residual == null ? "—" : signed(a.residual)}${a.su === false ? ". Upset." : "."}`
    : `${a.favorite} ${pct(a.favP)} to win, projected ${a.pred} (${signed(game.pred_margin)} home).${a.pomGap != null ? ` Pom gap ${signed(a.pomGap)}.` : ""}`;
  const [units, setUnits] = useState<UnitsFile[]>([]);

  useEffect(() => {
    const years = [game.season, game.season - 1];
    Promise.all(
      years.map((year) =>
        fetch(`/data/units-${year}.json`)
          .then((res) => (res.ok ? res.json() : null))
          .catch(() => null),
      ),
    ).then((rows) => {
      setUnits((rows as (UnitsFile | null)[]).filter((row): row is UnitsFile => Boolean(row?.teams?.length)));
    });
  }, [game.season]);

  return (
    <div className="game-page">
      <div className="team-hero">
        <button type="button" className="team-back" onClick={onClose}>
          Back
        </button>
        <h1>{kind}</h1>
        <span className="quiet">
          Week {game.week}
          {game.neutral ? " · Neutral" : ""}
          {when ? ` · ${when}` : ""}
          {game.engine ? ` · ${game.engine}` : ""}
        </span>
      </div>

      <div className="game-board">
        <div className="game-side">
          <TeamLink team={game.away} onOpen={onOpenTeam} />
          {game.away_conf && <span className="quiet">{game.away_conf}</span>}
          <div className="game-score">{a.done ? game.actual_away?.toFixed(0) : "—"}</div>
          <div className="quiet">Pred {game.pred_away.toFixed(1)}</div>
          {game.away_pom != null && <div className="quiet">Pom {signed(game.away_pom)}</div>}
        </div>
        <div className="game-at">
          <div>{game.neutral ? "vs" : "@"}</div>
          <div className="quiet">{a.done ? "Final" : "Upcoming"}</div>
        </div>
        <div className="game-side">
          <TeamLink team={game.home} onOpen={onOpenTeam} />
          {game.home_conf && <span className="quiet">{game.home_conf}</span>}
          <div className="game-score">{a.done ? game.actual_home?.toFixed(0) : "—"}</div>
          <div className="quiet">Pred {game.pred_home.toFixed(1)}</div>
          {game.home_pom != null && <div className="quiet">Pom {signed(game.home_pom)}</div>}
        </div>
      </div>

      <WinMeter game={game} />
      <p className="lede-note">{lede}</p>

      <dl className="team-stats">
        <div title="Actual final score, away then home.">
          <dt>Final</dt>
          <dd className={a.done ? hitTone(a.su) : undefined}>{a.score ?? "—"}</dd>
        </div>
        <div title="Model predicted score, away then home.">
          <dt>Projected</dt>
          <dd>{a.pred}</dd>
        </div>
        <div title="Listed spread, shown as the favorite.">
          <dt>Vegas Line</dt>
          <dd>{favoriteLine(game.home, game.away, vegas)}</dd>
        </div>
        <div
          title={
            yours
              ? `${userModel?.name ?? "Your model"} implied spread, from the uploaded slate.`
              : "Upload a score slate on My Model."
          }
        >
          <dt>Your Prediction</dt>
          <dd>{yours ? favoriteLine(game.home, game.away, -predMargin(yours)) : "—"}</dd>
        </div>
      </dl>

      {a.home && a.away && (
        <>
          <h2 className="team-sched-title">Matchup</h2>
          <MatchupViz
            home={a.home}
            away={a.away}
            homeName={game.home}
            awayName={game.away}
            league={ratings?.teams ?? []}
            onOpenTeam={onOpenTeam}
          />
        </>
      )}

      {units.length > 0 && (
        <UnitsView
          homeName={game.home}
          awayName={game.away}
          homeRow={a.home}
          awayRow={a.away}
          predHome={game.pred_home}
          predAway={game.pred_away}
          files={units}
          onOpenTeam={onOpenTeam}
        />
      )}

      <h2 className="team-sched-title">Forecasts</h2>
      <div className="table-wrap team-sched">
        <table>
          <thead>
            <tr>
              <th className="left">Source</th>
              <th title="Predicted probability the home team wins.">Home win%</th>
              <th title="Predicted home minus away. Positive means home is favored.">Home margin</th>
              <th title="Predicted combined points.">Total</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="left">FootPalm</td>
              <td>{pct(game.home_win_prob)}</td>
              <td>{signed(game.pred_margin)}</td>
              <td>{a.predTotal.toFixed(1)}</td>
            </tr>
            {yours && (
              <tr>
                <td className="left">{userModel?.name ?? "Your model"}</td>
                <td>{yours.home_win_prob == null ? "—" : pct(yours.home_win_prob)}</td>
                <td>{signed(predMargin(yours))}</td>
                <td>{(yours.pred_away + yours.pred_home).toFixed(1)}</td>
              </tr>
            )}
            {a.fpiWin != null && (
              <tr>
                <td className="left">ESPN FPI</td>
                <td>{pct(a.fpiWin)}</td>
                <td>{a.fpiMargin == null ? "—" : signed(a.fpiMargin)}</td>
                <td>—</td>
              </tr>
            )}
            {(a.consensusWin != null || a.consensusMargin != null) && (
              <tr>
                <td className="left">Consensus</td>
                <td>{a.consensusWin == null ? "—" : pct(a.consensusWin)}</td>
                <td>{a.consensusMargin == null ? "—" : signed(a.consensusMargin)}</td>
                <td>{a.listedTotal == null ? "—" : a.listedTotal.toFixed(1)}</td>
              </tr>
            )}
            {a.done && (
              <tr>
                <td className="left">Actual</td>
                <td>{game.home_won == null ? "—" : pct(game.home_won)}</td>
                <td>{game.actual_margin == null ? "—" : signed(game.actual_margin)}</td>
                <td>{a.actualTotal == null ? "—" : a.actualTotal.toFixed(0)}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {(a.consensusWin != null || a.consensusMargin != null || game.espn?.locked) && (
        <p className="lede-note">
          {a.consensusWin != null || a.consensusMargin != null
            ? "Consensus is the listed market number, shown as another forecast — not a recommendation."
            : ""}
          {game.espn?.locked ? " ESPN context was locked at kickoff." : ""}
        </p>
      )}

      {(weather.length > 0 || field || game.espn?.qbs) && (
        <div className="slip-cols">
          <section>
            <h3>Context</h3>
            <Row
              label="Field"
              value={field || a.venue || "—"}
              hint="Venue and surface from ESPN, locked at kickoff when available."
            />
            <Row
              label="Weather"
              value={weather.length ? weather.join(" · ") : "—"}
              hint="Kickoff weather from ESPN: temperature, wind, rain chance, and condition."
            />
          </section>
          <section>
            <h3>Quarterbacks</h3>
            <Row
              label={game.home}
              value={qbLine(game.espn?.qbs?.home)}
              hint={`${game.home} quarterbacks listed by ESPN for this game.`}
            />
            <Row
              label={game.away}
              value={qbLine(game.espn?.qbs?.away)}
              hint={`${game.away} quarterbacks listed by ESPN for this game.`}
            />
          </section>
        </div>
      )}

      {models.length > 0 && (
        <>
          <h2 className="team-sched-title">Models</h2>
          <div className="table-wrap team-sched">
            <table>
              <thead>
                <tr>
                  <th className="left">Model</th>
                  <th title="That model's predicted home minus away.">Home margin</th>
                  <th title="That model's predicted home win probability.">Home win</th>
                </tr>
              </thead>
              <tbody>
                {models.map(({ id, pick: m }) => (
                  <tr key={id}>
                    <td className="left">{id}</td>
                    <td>{m ? signed(m.pred_margin) : "—"}</td>
                    <td>{m ? pct(m.home_win_prob) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
