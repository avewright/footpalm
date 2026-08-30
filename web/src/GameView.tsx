import { formatAmerican, signed } from "./format";
import { evPct, gameFacts, hitTone, MODEL_IDS, pct, pickOf, pp, tone } from "./game";
import { prettyWhen } from "./score";
import { TeamLink } from "./TeamView";
import type { GamePred } from "./types";

function Stat({ label, value, tone: t }: { label: string; value: string; tone?: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd className={t}>{value}</dd>
    </div>
  );
}

function Row({ label, value, tone: t }: { label: string; value: string; tone?: string }) {
  return (
    <div className="slip-row">
      <span className="quiet">{label}</span>
      <span className={t}>{value}</span>
    </div>
  );
}

export function GameView({
  game,
  onOpenTeam,
  onClose,
}: {
  game: GamePred;
  onOpenTeam: (team: string) => void;
  onClose: () => void;
}) {
  const { ens, play, ml, us, mkt, mktMl, agree, hit, su, pick, done, score, pred } = gameFacts(game);
  const when = prettyWhen(game.start);
  const models = MODEL_IDS.map((id) => ({ id, pick: pickOf(game, id) })).filter((row) => row.pick);

  return (
    <div className="game-page">
      <div className="team-hero">
        <button type="button" className="team-back" onClick={onClose}>
          Back
        </button>
      </div>

      <div className="game-board">
        <div className="game-side">
          <TeamLink team={game.away} onOpen={onOpenTeam} />
          {game.away_conf && <span className="quiet">{game.away_conf}</span>}
          <div className="game-score">{done ? game.actual_away?.toFixed(0) : "—"}</div>
          <div className="quiet">Pred {game.pred_away.toFixed(1)}</div>
          {game.away_pom != null && <div className="quiet">Pom {signed(game.away_pom)}</div>}
        </div>
        <div className="game-at">
          <div>{game.neutral ? "vs" : "@"}</div>
          <div className="quiet">
            Week {game.week}
            {game.neutral ? " · Neutral" : ""}
          </div>
        </div>
        <div className="game-side">
          <TeamLink team={game.home} onOpen={onOpenTeam} />
          {game.home_conf && <span className="quiet">{game.home_conf}</span>}
          <div className="game-score">{done ? game.actual_home?.toFixed(0) : "—"}</div>
          <div className="quiet">Pred {game.pred_home.toFixed(1)}</div>
          {game.home_pom != null && <div className="quiet">Pom {signed(game.home_pom)}</div>}
        </div>
      </div>

      <p className="lede-note">
        {when ?? `Week ${game.week}`}
        {score ? ` · Final ${score}` : " · Upcoming"}
        {game.engine ? ` · ${game.engine}` : ""}
      </p>

      <dl className="team-stats">
        <Stat label="Pred" value={pred} />
        <Stat label="Home win" value={pct(ens?.home_win_prob ?? game.home_win_prob)} />
        <Stat label="Pred margin" value={`${game.home} ${signed(ens?.pred_margin ?? game.pred_margin)}`} />
        <Stat label="Market" value={mkt == null ? "—" : `${game.home} ${signed(mkt)}`} />
        <Stat label="Us" value={us == null ? "—" : `${game.home} ${signed(us)}`} />
        <Stat label="Spread pick" value={pick ?? "—"} tone={hitTone(hit)} />
        {done ? (
          <>
            <Stat label="SU" value={su == null ? "—" : su ? "Hit" : "Miss"} tone={hitTone(su)} />
            <Stat label="ATS" value={hit == null ? "—" : hit ? "Covered" : "Missed"} tone={hitTone(hit)} />
          </>
        ) : (
          <Stat label="Spread EV" value={play ? evPct(play.ev) : "—"} tone={tone(play?.ev)} />
        )}
      </dl>

      <div className="slip-cols">
        <section>
          <h3>Spread</h3>
          <Row label="Bet" value={pick ?? "—"} tone={hitTone(hit)} />
          {done ? (
            <Row label="Result" value={hit == null ? "—" : hit ? "Covered" : "Missed"} tone={hitTone(hit)} />
          ) : (
            <Row label="EV" value={play ? evPct(play.ev) : "—"} tone={tone(play?.ev)} />
          )}
          <Row label="Market" value={mkt == null ? "—" : `${game.home} ${signed(mkt)}`} />
          <Row label="Us" value={us == null ? "—" : `${game.home} ${signed(us)}`} />
          {!done && <Row label="Agree" value={play ? `${agree} of 3 models` : "—"} tone={agree === 3 ? "good" : agree <= 1 && play ? "bad" : undefined} />}
        </section>
        <section>
          <h3>Moneyline</h3>
          {done ? (
            <>
              <Row
                label="Pick"
                value={
                  ens
                    ? `${ens.home_win_prob >= 0.5 ? game.home : game.away} ${pct(ens.home_win_prob >= 0.5 ? ens.home_win_prob : 1 - ens.home_win_prob)}`
                    : "—"
                }
                tone={hitTone(su)}
              />
              <Row label="Result" value={su == null ? "—" : su ? "Won" : "Lost"} tone={hitTone(su)} />
            </>
          ) : (
            <>
              <Row
                label="Bet"
                value={ml ? `${ml.who} ${ml.mktAmerican > 0 ? "+" : ""}${ml.mktAmerican}` : "—"}
                tone={tone(ml?.edge, "pp")}
              />
              <Row label="Edge" value={ml ? pp(ml.edge) : "—"} tone={tone(ml?.edge, "pp")} />
            </>
          )}
          <Row label="Market" value={mktMl == null ? "—" : `${game.home} ${pct(mktMl)}`} />
          <Row label="Us" value={ens ? `${game.home} ${pct(ens.home_win_prob)}` : "—"} />
          {ml && <Row label="Us (Am)" value={formatAmerican(ml.ourAmerican)} />}
        </section>
      </div>

      {models.length > 0 && (
        <>
          <h2 className="team-sched-title">Models</h2>
          <div className="table-wrap team-sched">
            <table>
              <thead>
                <tr>
                  <th className="left">Model</th>
                  <th>Home margin</th>
                  <th>Home win</th>
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
