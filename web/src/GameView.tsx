import { formatAmerican, signed } from "./format";
import { evPct, gameFacts, hitTone, lineOf, MODEL_IDS, pct, pickOf, pp, tone } from "./game";
import { prettyWhen } from "./score";
import { TeamLink } from "./TeamView";
import type { GamePred } from "./types";

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
  const { ens, play, ml, us, mkt, mktAm, ourAm, dSpread, dMl, lean, agree, hit, su, pick, done, score } =
    gameFacts(game);
  const when = prettyWhen(game.start);
  const models = MODEL_IDS.map((id) => ({ id, pick: pickOf(game, id) })).filter((row) => row.pick);
  const book = game.books?.kalshi ? "Kalshi" : game.books?.polymarket ? "Polymarket" : null;

  function side(home: boolean) {
    const flip = home ? 1 : -1;
    const vegasLine = mkt == null ? null : mkt * flip;
    const ourLine = us == null ? null : us * flip;
    const vegasMl = mktAm ? (home ? mktAm.home : mktAm.away) : null;
    const ourMl = ourAm ? (home ? ourAm.home : ourAm.away) : null;
    const lineGap = dSpread == null ? null : dSpread * flip;
    const mlGap = dMl == null ? null : dMl * flip;
    return { vegasLine, ourLine, vegasMl, ourMl, lineGap, mlGap };
  }

  const away = side(false);
  const home = side(true);

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
        {book ? ` · Vegas via ${book}` : ""}
      </p>

      <div className="table-wrap team-sched odds-compare">
        <table>
          <thead>
            <tr>
              <th className="left" rowSpan={2} />
              <th className="group split" colSpan={2}>
                Vegas
              </th>
              <th className="group split" colSpan={2}>
                Our Prediction
              </th>
              <th className="group split" colSpan={2}>
                Difference
              </th>
            </tr>
            <tr>
              <th className="split">Line</th>
              <th>ML</th>
              <th className="split">Line</th>
              <th>ML</th>
              <th className="split">Line</th>
              <th>ML</th>
            </tr>
          </thead>
          <tbody>
            {(
              [
                [game.away, away],
                [game.home, home],
              ] as const
            ).map(([team, row]) => (
              <tr key={team} className={lean === team ? "lean" : undefined}>
                <td className="left">
                  <TeamLink team={team} onOpen={onOpenTeam} />
                </td>
                <td className="split">{lineOf(row.vegasLine)}</td>
                <td>{row.vegasMl == null ? "—" : formatAmerican(row.vegasMl)}</td>
                <td className="split">{lineOf(row.ourLine)}</td>
                <td>{row.ourMl == null ? "—" : formatAmerican(row.ourMl)}</td>
                <td className={`split${lean === team ? " good" : ""}`}>{lineOf(row.lineGap)}</td>
                <td className={lean === team ? "good" : undefined}>{row.mlGap == null ? "—" : pp(row.mlGap)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {lean && (
        <p className="odds-lean">
          We are {dSpread != null ? `${Math.abs(dSpread).toFixed(1)} pts` : ""}
          {dSpread != null && dMl != null ? " and " : ""}
          {dMl != null ? pp(Math.abs(dMl)) : ""} higher on {lean} than Vegas.
        </p>
      )}

      <div className="slip-cols">
        <section>
          <h3>Spread play</h3>
          <Row label="Bet" value={pick ?? "—"} tone={hitTone(hit)} />
          {done ? (
            <Row label="Result" value={hit == null ? "—" : hit ? "Covered" : "Missed"} tone={hitTone(hit)} />
          ) : (
            <Row label="EV" value={play ? evPct(play.ev) : "—"} tone={tone(play?.ev)} />
          )}
          {!done && (
            <Row
              label="Agree"
              value={play ? `${agree} of 3 models` : "—"}
              tone={agree === 3 ? "good" : agree <= 1 && play ? "bad" : undefined}
            />
          )}
        </section>
        <section>
          <h3>Moneyline play</h3>
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
