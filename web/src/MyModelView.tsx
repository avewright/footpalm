import { useMemo, useRef, useState } from "react";
import { signed } from "./format";
import { gameKey, pct } from "./game";
import {
  parseUserModel,
  pickOfModel,
  predMargin,
  scoreUserModel,
  slateCsv,
  type UserModel,
  type UserScore,
} from "./mymodel";
import { isFinal, prettyWhen, record, summarize } from "./score";
import { TeamLink } from "./TeamView";
import type { GamePred } from "./types";

function Card({ title, card }: { title: string; card: UserScore | ReturnType<typeof summarize> }) {
  if (!card.n) return null;
  return (
    <div className="score-card">
      <h3>{title}</h3>
      <dl className="metrics">
        <dt title="Correct winner picks versus misses on finals this slate covers.">Winner</dt>
        <dd>{record(card.suW, card.suL)}</dd>
        <dt title="Mean squared error of home win probability, when the file includes one. Lower is better.">
          Brier
        </dt>
        <dd>{card.brier == null ? "—" : card.brier.toFixed(3)}</dd>
        <dt title="Mean absolute error of the predicted home margin, in points.">Margin MAE</dt>
        <dd>{card.mae == null ? "—" : card.mae.toFixed(1)}</dd>
        <dt title="Mean residual: actual home margin minus predicted.">Bias</dt>
        <dd>{card.residual == null ? "—" : signed(card.residual)}</dd>
      </dl>
    </div>
  );
}

export function MyModelView({
  season,
  games,
  model,
  onChange,
  onOpenTeam,
  onOpenGame,
}: {
  season: number;
  games: GamePred[];
  model: UserModel | null;
  onChange: (next: UserModel | null) => void;
  onOpenTeam: (team: string) => void;
  onOpenGame: (key: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [drag, setDrag] = useState(false);
  const slate = useMemo(() => games.filter((g) => g.fbs_fbs), [games]);

  const yours = useMemo(() => (model ? scoreUserModel(slate, model) : null), [slate, model]);
  const foot = useMemo(() => {
    if (!model) return null;
    const covered = slate.filter((g) => pickOfModel(model, g) && isFinal(g));
    return summarize(covered);
  }, [slate, model]);

  const rows = useMemo(() => {
    if (!model) return [];
    return slate
      .filter((g) => pickOfModel(model, g))
      .sort((a, b) => a.week - b.week || a.away.localeCompare(b.away));
  }, [slate, model]);

  function downloadSlate() {
    const blob = new Blob([slateCsv(slate, season)], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `footpalm-slate-${season}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async function ingest(file: File) {
    setError(null);
    setWarnings([]);
    try {
      const text = await file.text();
      const { model: next, warnings: notes } = parseUserModel(text, file.name, season, slate);
      onChange(next);
      setWarnings(notes);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not read that file.");
    }
  }

  function onFiles(files: FileList | null) {
    const file = files?.[0];
    if (file) void ingest(file);
  }

  return (
    <div className="mine">
      <h2 className="team-sched-title">My Model</h2>
      <p className="lede-note">
        Bring a whole-slate score model, not a game-by-game pick sheet. Download the {season} FBS slate, fill it from
        your model, and upload the results. Matched games show up as Your Prediction on each game page. Kept in this
        browser only.
      </p>

      <ol className="mine-steps">
        <li>
          <strong>Download the slate.</strong> Every FBS game FootPalm is scoring, with <code>game_id</code> already
          filled.
        </li>
        <li>
          <strong>Run your model.</strong> Write <code>pred_away</code> and <code>pred_home</code>. Optional:{" "}
          <code>home_win_prob</code> (0–1), or <code>pred_margin</code> plus <code>pred_total</code> instead of scores.
        </li>
        <li>
          <strong>Upload the file.</strong> CSV or JSON. A new upload replaces the previous {season} slate.
        </li>
      </ol>

      <div className="toolbar">
        <button type="button" onClick={downloadSlate}>
          Download {season} slate
        </button>
        <button type="button" onClick={() => input.current?.click()}>
          Upload results
        </button>
        {model && (
          <button
            type="button"
            onClick={() => {
              onChange(null);
              setWarnings([]);
              setError(null);
            }}
          >
            Remove slate
          </button>
        )}
        <input
          ref={input}
          type="file"
          accept=".csv,.json,text/csv,application/json"
          hidden
          onChange={(e) => {
            onFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      <div
        className={`mine-drop${drag ? " is-drag" : ""}`}
        onDragEnter={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          onFiles(e.dataTransfer.files);
        }}
      >
        Drop a CSV or JSON results file here.
      </div>

      {error && <p className="warn">{error}</p>}
      {warnings.map((w) => (
        <p key={w} className="lede-note">
          {w}
        </p>
      ))}

      {model && (
        <>
          <p className="lede-note">
            {model.name} · {model.matched} of {slate.length} FBS games · from {model.source}
            {model.unmatched ? ` · ${model.unmatched} unmatched` : ""}
          </p>
          <div className="score-strip">
            {yours && <Card title={model.name} card={yours} />}
            {foot && <Card title="FootPalm on the same games" card={foot} />}
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Wk</th>
                  <th className="left">Away</th>
                  <th className="left">Home</th>
                  <th>Your pred</th>
                  <th>FootPalm</th>
                  <th>Final</th>
                  <th title="Your predicted home minus away.">Your margin</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((g) => {
                  const pick = pickOfModel(model, g);
                  if (!pick) return null;
                  const done = isFinal(g);
                  return (
                    <tr key={gameKey(g)} className="game-row" onClick={() => onOpenGame(gameKey(g))}>
                      <td>{g.week}</td>
                      <td className="left">
                        <TeamLink team={g.away} onOpen={onOpenTeam} />
                      </td>
                      <td className="left">
                        <TeamLink team={g.home} onOpen={onOpenTeam} />
                      </td>
                      <td>
                        {pick.pred_away.toFixed(1)}–{pick.pred_home.toFixed(1)}
                      </td>
                      <td>
                        {g.pred_away.toFixed(1)}–{g.pred_home.toFixed(1)}
                      </td>
                      <td>{done ? `${g.actual_away?.toFixed(0)}–${g.actual_home?.toFixed(0)}` : prettyWhen(g.start) ?? "—"}</td>
                      <td>
                        {signed(predMargin(pick))}
                        {pick.home_win_prob != null ? ` · ${pct(pick.home_win_prob)}` : ""}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
