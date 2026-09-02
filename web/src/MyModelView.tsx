import { useMemo, useRef, useState } from "react";
import type { SessionUser } from "./accounts";
import { signed } from "./format";
import { gameKey, pct } from "./game";
import {
  exportUserModel,
  hasScores,
  mergeUserModel,
  parseUserModel,
  pickLabel,
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
        {card.atsW + card.atsL > 0 && (
          <>
            <dt title="Against the spread, using the line locked on the pick.">ATS</dt>
            <dd>{record(card.atsW, card.atsL)}</dd>
          </>
        )}
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
  user,
  models,
  activeId,
  onUpload,
  onActivate,
  onPublish,
  onRemove,
  onNeedLogin,
  onOpenTeam,
  onOpenGame,
}: {
  season: number;
  games: GamePred[];
  user: SessionUser | null;
  models: UserModel[];
  activeId: string | null;
  onUpload: (model: UserModel) => Promise<void>;
  onActivate: (id: string) => void;
  onPublish: (id: string, published: boolean) => void;
  onRemove: (id: string) => void;
  onNeedLogin: () => void;
  onOpenTeam: (team: string) => void;
  onOpenGame: (key: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [drag, setDrag] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const slate = useMemo(() => games.filter((g) => g.fbs_fbs), [games]);
  const selected =
    models.find((m) => m.id === selectedId) ?? models.find((m) => m.id === activeId) ?? models[0] ?? null;

  const yours = useMemo(() => (selected ? scoreUserModel(slate, selected) : null), [slate, selected]);
  const foot = useMemo(() => {
    if (!selected) return null;
    const covered = slate.filter((g) => pickOfModel(selected, g) && isFinal(g));
    return summarize(covered);
  }, [slate, selected]);

  const rows = useMemo(() => {
    if (!selected) return [];
    return slate
      .filter((g) => pickOfModel(selected, g))
      .sort((a, b) => a.week - b.week || a.away.localeCompare(b.away));
  }, [slate, selected]);

  function downloadSlate() {
    const blob = new Blob([slateCsv(slate, season)], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `footpalm-slate-${season}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function downloadJson(book: UserModel) {
    const blob = new Blob([exportUserModel(book)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `footpalm-picks-${book.season}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async function askLeftover(book: UserModel, leftover: string): Promise<UserModel> {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        season,
        picks: book,
        messages: [{ role: "user", content: `Add these picks to my book:\n${leftover.slice(0, 11000)}` }],
      }),
    });
    const payload = (await response.json()) as { error?: string; picks?: UserModel };
    if (!response.ok || !payload.picks) {
      throw new Error(payload.error || "Ask could not read the leftover rows.");
    }
    return payload.picks;
  }

  async function ingest(file: File) {
    if (!user) {
      onNeedLogin();
      return;
    }
    setError(null);
    setWarnings([]);
    try {
      const text = await file.text();
      const { model: parsed, warnings: notes, leftover } = parseUserModel(text, file.name, season, slate);
      let next = parsed;
      const extra = [...notes];
      if (leftover) {
        try {
          next = mergeUserModel(next, await askLeftover(next, leftover));
        } catch (err) {
          extra.push(err instanceof Error ? err.message : "Ask could not read leftover rows.");
        }
      }
      await onUpload(next);
      setWarnings(extra);
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
        Upload one or more whole-slate score models. The active model fills Your Prediction on each game page and
        sits on the Models board. Kept under your name. You can also tell Ask a side, or drop a pick list — those
        rows merge in.
      </p>

      {!user ? (
        <p className="lede-note">
          <button type="button" className="team-link" onClick={onNeedLogin}>
            Log in
          </button>{" "}
          to keep your uploads separate.
        </p>
      ) : (
        <>
          <ol className="mine-steps">
            <li>
              <strong>Download the slate.</strong> Every FBS game FootPalm is scoring, with <code>game_id</code>{" "}
              already filled.
            </li>
            <li>
              <strong>Run your model.</strong> Write <code>pred_away</code> and <code>pred_home</code>. Optional:{" "}
              <code>home_win_prob</code> (0–1), or <code>pred_margin</code> plus <code>pred_total</code> instead of
              scores.
            </li>
            <li>
              <strong>Upload another file</strong> whenever you have a new version. It is added to this season’s
              library; it does not replace the others.
            </li>
          </ol>

          <div className="toolbar">
            <button type="button" onClick={downloadSlate}>
              Download {season} slate
            </button>
            <button type="button" onClick={() => input.current?.click()}>
              Upload model
            </button>
            {selected && (
              <button type="button" onClick={() => downloadJson(selected)}>
                Export
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
            Drop a CSV or JSON results file to add another {season} model.
          </div>
        </>
      )}

      {error && <p className="warn">{error}</p>}
      {warnings.map((w) => (
        <p key={w} className="lede-note">
          {w}
        </p>
      ))}

      {user && models.length > 0 && (
        <div className="table-wrap team-sched">
          <table>
            <thead>
              <tr>
                <th className="left">Model</th>
                <th>Games</th>
                <th className="left">Source</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr
                  key={m.id}
                  className={`game-row${m.id === selected?.id ? " is-active-model" : ""}`}
                  onClick={() => m.id && setSelectedId(m.id)}
                >
                  <td className="left">
                    {m.name}
                    {m.id === activeId ? " · active" : ""}
                    {m.published === false ? " · private" : ""}
                  </td>
                  <td>
                    {m.matched}/{slate.length}
                  </td>
                  <td className="left">{m.source}</td>
                  <td className="left">
                    {m.id && m.id !== activeId && (
                      <button
                        type="button"
                        className="team-link"
                        onClick={(e) => {
                          e.stopPropagation();
                          onActivate(m.id as string);
                        }}
                      >
                        Use on games
                      </button>
                    )}{" "}
                    {m.id && (
                      <button
                        type="button"
                        className="team-link"
                        onClick={(e) => {
                          e.stopPropagation();
                          onPublish(m.id as string, m.published === false);
                        }}
                      >
                        {m.published === false ? "Publish" : "Hide"}
                      </button>
                    )}{" "}
                    {m.id && (
                      <button
                        type="button"
                        className="team-link"
                        onClick={(e) => {
                          e.stopPropagation();
                          onRemove(m.id as string);
                        }}
                      >
                        Remove
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selected && (
        <>
          <p className="lede-note">
            {selected.name} · {selected.matched} of {slate.length} FBS games · from {selected.source}
            {selected.unmatched ? ` · ${selected.unmatched} unmatched` : ""}
          </p>
          <div className="score-strip">
            {yours && <Card title={selected.name} card={yours} />}
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
                  const pick = pickOfModel(selected, g);
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
                        {hasScores(pick)
                          ? `${pick.pred_away?.toFixed(1)}–${pick.pred_home?.toFixed(1)}`
                          : pickLabel(g, pick)}
                      </td>
                      <td>
                        {g.pred_away.toFixed(1)}–{g.pred_home.toFixed(1)}
                      </td>
                      <td>{done ? `${g.actual_away?.toFixed(0)}–${g.actual_home?.toFixed(0)}` : prettyWhen(g.start) ?? "—"}</td>
                      <td>
                        {hasScores(pick)
                          ? `${signed(predMargin(pick))}${pick.home_win_prob != null ? ` · ${pct(pick.home_win_prob)}` : ""}`
                          : pickLabel(g, pick)}
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
