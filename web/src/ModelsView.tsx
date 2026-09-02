import { useMemo, useState } from "react";
import type { ModelCard, ModelKind, SessionUser } from "./accounts";
import { signed } from "./format";
import { record } from "./score";
import type { UserScore } from "./mymodel";

const KIND_LABEL: Record<ModelKind, string> = {
  you: "Yours",
  admin: "FootPalm",
  community: "Community",
};

function Card({ title, card }: { title: string; card: UserScore }) {
  if (!card.n) return null;
  return (
    <div className="score-card">
      <h3>{title}</h3>
      <dl className="metrics">
        <dt>Winner</dt>
        <dd>{record(card.suW, card.suL)}</dd>
        <dt>Brier</dt>
        <dd>{card.brier == null ? "—" : card.brier.toFixed(3)}</dd>
        <dt>Margin MAE</dt>
        <dd>{card.mae == null ? "—" : card.mae.toFixed(1)}</dd>
        <dt>Bias</dt>
        <dd>{card.residual == null ? "—" : signed(card.residual)}</dd>
      </dl>
    </div>
  );
}

export function ModelsView({
  season,
  user,
  catalog,
  error,
  onNeedLogin,
}: {
  season: number;
  user: SessionUser | null;
  catalog: ModelCard[];
  error: string | null;
  onNeedLogin: () => void;
}) {
  const [filter, setFilter] = useState<"all" | ModelKind>("all");
  const rows = useMemo(() => {
    const copy = catalog.filter((row) => (filter === "all" ? true : row.kind === filter));
    copy.sort((a, b) => {
      const ar = a.score?.brier ?? 9;
      const br = b.score?.brier ?? 9;
      if (ar !== br) return ar - br;
      const aw = a.score ? a.score.suW / Math.max(a.score.n, 1) : -1;
      const bw = b.score ? b.score.suW / Math.max(b.score.n, 1) : -1;
      return bw - aw || a.name.localeCompare(b.name);
    });
    return copy;
  }, [catalog, filter]);
  const yours = catalog.filter((row) => row.kind === "you");
  const house = catalog.filter((row) => row.kind === "admin");

  return (
    <div className="mine">
      <h2 className="team-sched-title">Models</h2>
      <p className="lede-note">
        {season} score models on the same FBS slate. FootPalm is the house board. Community models are published
        uploads. Your active model is the one game pages use as Your Prediction.
      </p>
      {!user && (
        <p className="lede-note">
          <button type="button" className="team-link" onClick={onNeedLogin}>
            Log in
          </button>{" "}
          to add yours.
        </p>
      )}
      {error && <p className="warn">{error}</p>}
      <div className="score-strip">
        {yours[0]?.score && <Card title={yours.find((m) => m.active)?.name ?? yours[0].name} card={(yours.find((m) => m.active) ?? yours[0]).score!} />}
        {house[0]?.score && <Card title="FootPalm" card={house[0].score} />}
      </div>
      <div className="toolbar">
        <div className="seg" role="group" aria-label="Model filter">
          {(["all", "you", "community", "admin"] as const).map((id) => (
            <button key={id} type="button" aria-pressed={filter === id} onClick={() => setFilter(id)}>
              {id === "all" ? "All" : KIND_LABEL[id]}
            </button>
          ))}
        </div>
        <span className="lede-note" style={{ margin: 0 }}>
          {rows.length} models
        </span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th className="left">Model</th>
              <th className="left">Owner</th>
              <th className="left">Kind</th>
              <th>Winner</th>
              <th>Brier</th>
              <th>MAE</th>
              <th>Bias</th>
              <th>Games</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} className={row.active && row.kind === "you" ? "is-active-model" : undefined}>
                <td className="left">
                  {row.name}
                  {row.active && row.kind === "you" ? " · active" : ""}
                </td>
                <td className="left">{row.owner}</td>
                <td className="left">{KIND_LABEL[row.kind]}</td>
                <td>{row.score?.n ? record(row.score.suW, row.score.suL) : "—"}</td>
                <td>{row.score?.brier == null ? "—" : row.score.brier.toFixed(3)}</td>
                <td>{row.score?.mae == null ? "—" : row.score.mae.toFixed(1)}</td>
                <td>{row.score?.residual == null ? "—" : signed(row.score.residual)}</td>
                <td>{row.score?.n ?? 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
