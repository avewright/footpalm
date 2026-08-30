import type { ResearchFile, TreeReport } from "./types";

function fmtParams(params: Record<string, number>) {
  const keys = Object.keys(params);
  if (!keys.length) return "—";
  return keys.map((k) => `${k}=${params[k]}`).join(" ");
}

function Rows({
  rows,
  promoted,
}: {
  rows: ResearchFile["experiments"];
  promoted: string;
}) {
  return (
    <tbody>
      {rows.map((e) => {
        const delta = e.delta_holdout_brier;
        const passed = e.pass;
        const label = e.id.startsWith("oracle_")
          ? "ceiling"
          : e.id === promoted
            ? "promoted"
            : passed == null
              ? "baseline"
              : passed
                ? "yes"
                : "no";
        return (
          <tr key={e.id}>
            <td className="team">{e.id}</td>
            <td className="team">{fmtParams(e.params)}</td>
            <td>{e.train.brier}</td>
            <td>{e.holdout.brier}</td>
            <td>{e.holdout.logloss}</td>
            <td className={delta != null && delta < 0 ? "good" : delta != null && delta > 0 ? "bad" : undefined}>
              {delta ?? "—"}
            </td>
            <td className="team">{label}</td>
          </tr>
        );
      })}
    </tbody>
  );
}

function Importance({ rows, valueKey }: { rows: { feature: string; gain?: number; share?: number; brier_increase?: number }[]; valueKey: "share" | "brier_increase" }) {
  const max = Math.max(...rows.map((r) => Math.abs(Number(r[valueKey] ?? 0))), 1e-9);
  return (
    <div>
      {rows.map((r) => {
        const v = Number(r[valueKey] ?? 0);
        return (
          <div className="imp-row" key={r.feature}>
            <span>{r.feature}</span>
            <div className="imp-bar">
              <i style={{ width: `${(Math.abs(v) / max) * 100}%` }} />
            </div>
            <span>{valueKey === "share" ? `${Math.round((r.share ?? 0) * 100)}%` : v.toFixed(4)}</span>
          </div>
        );
      })}
    </div>
  );
}

export function ResearchView({ data, trees }: { data: ResearchFile | null; trees?: TreeReport | null }) {
  if (!data) return <p className="lede-note">No research log yet. Run the build or `python -m footpalm.research`.</p>;
  const train = data.train_seasons?.join(", ") ?? "train";
  const hold = data.holdout_season ?? 2025;
  const floor = data.holdout_expected_brier_if_calibrated ?? data.holdout_2025_expected_brier_if_calibrated;
  const baseline = data.baseline_holdout_brier ?? data.baseline_2025_brier;
  return (
    <div>
      <p className="lede-note">{data.conclusion ?? `Fit on ${train}. Score ${hold} once.`}</p>
      <p className="lede-note">
        Promoted: <strong>{data.promoted}</strong>. {data.promote_if}.
        {data.train_n != null && ` Train n=${data.train_n}.`}
        {data.holdout_n != null && ` Holdout n=${data.holdout_n}.`}
        {floor != null && ` ${hold} expected Brier if calibrated: ${floor}. Actual: ${baseline}.`}
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th className="team">id</th>
              <th className="team">params</th>
              <th>{train} Brier</th>
              <th>{hold} Brier</th>
              <th>{hold} logloss</th>
              <th>Δ Brier</th>
              <th className="team">pass</th>
            </tr>
          </thead>
          <Rows rows={data.experiments} promoted={data.promoted} />
        </table>
      </div>
      {(trees ?? data.trees)?.models && (
        <div className="metric-block">
          <h3>What the trees use</h3>
          <p className="lede-note">
            {(trees ?? data.trees)?.protocol} Permutation is the honest one: 2025 Brier rise when that column is
            shuffled. Gain is what the trees split on in 2014–2024.
          </p>
          {(trees ?? data.trees)?.comparison && (
            <div className="table-wrap">
              <p className="lede-note">
                {(trees ?? data.trees)?.comparison?.rule}{" "}
                {(trees ?? data.trees)?.comparison?.would_promote
                  ? "Extras would pass. Live model is still the locked 10."
                  : "Extras do not pass. Live model stays on the locked 10."}
              </p>
              <table>
                <thead>
                  <tr>
                    <th className="team">family</th>
                    <th>locked Brier</th>
                    <th>set Brier</th>
                    <th>Δ Brier</th>
                    <th className="team">pass</th>
                  </tr>
                </thead>
                <tbody>
                  {(trees ?? data.trees)?.comparison?.rows.map((row) => (
                    <tr key={row.family}>
                      <td className="team">{row.family}</td>
                      <td>{row.locked_brier}</td>
                      <td>{row.full_brier}</td>
                      <td className={row.delta_brier < 0 ? "good" : row.delta_brier > 0 ? "bad" : undefined}>
                        {row.delta_brier}
                      </td>
                      <td className="team">{row.pass ? "yes" : "no"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="imp-grid">
            {(trees ?? data.trees)?.models?.map((model) => (
              <div key={model.id}>
                <h3>
                  {model.id} · 2025 Brier {model.holdout.brier}
                </h3>
                <p className="lede-note">Permutation (holdout)</p>
                <Importance rows={model.permutation} valueKey="brier_increase" />
                <p className="lede-note">Train gain</p>
                <Importance rows={model.gain} valueKey="share" />
              </div>
            ))}
          </div>
        </div>
      )}
      {data.diagnostics && data.diagnostics.length > 0 && (
        <div className="metric-block">
          <h3>Oracle ceiling — do not ship</h3>
          <p className="lede-note">These fit on {hold}. They measure temptation, not a model.</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th className="team">id</th>
                  <th className="team">params</th>
                  <th>{train} Brier</th>
                  <th>{hold} Brier</th>
                  <th>{hold} logloss</th>
                  <th>Δ Brier</th>
                  <th className="team">pass</th>
                </tr>
              </thead>
              <Rows rows={data.diagnostics} promoted={data.promoted} />
            </table>
          </div>
        </div>
      )}
      <details className="glossary">
        <summary>What this is</summary>
        <p className="lede-note">{data.note}</p>
        <p className="lede-note">Protocol: {data.protocol}.</p>
      </details>
    </div>
  );
}
