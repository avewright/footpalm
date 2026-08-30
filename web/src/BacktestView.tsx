import { Fragment } from "react";
import type { BacktestFile, BacktestSummary } from "./types";

function metricTable(title: string, m: Record<string, number>) {
  const keys = Object.keys(m);
  if (!keys.length) return null;
  return (
    <div className="metric-block">
      <h3>{title}</h3>
      <dl className="metrics">
        {keys.map((k) => (
          <Fragment key={k}>
            <dt>{k.replace(/^(fbs|tabpfn|logistic)_/, "")}</dt>
            <dd>{m[k]}</dd>
          </Fragment>
        ))}
      </dl>
    </div>
  );
}

export function BacktestView({ season, backtest, summary }: { season: number; backtest: BacktestFile | null; summary: BacktestSummary | null }) {
  const row = summary?.seasons.find((s) => s.season === season);
  return (
    <div>
      <p className="lede-note">{backtest?.protocol ?? summary?.rule}</p>
      {backtest?.tabpfn_error && <p className="warn">TabPFN: {backtest.tabpfn_error}</p>}
      <div className="metric-grid">
        {row && metricTable(`${season} FBS`, row.fbs)}
        {row && metricTable("TabPFN-3", row.tabpfn)}
        {row && metricTable("Logistic baseline", row.logistic)}
      </div>
      {summary?.nil_residual && (
        <div className="metric-block">
          <h3>NIL residual check</h3>
          <p>
            {summary.nil_residual.used
              ? `${summary.nil_residual.note} Helped out-of-sample: ${summary.nil_residual.helped ? "yes" : "no"}. MSE ${summary.nil_residual.test_mse_before} → ${summary.nil_residual.test_mse_after}.`
              : summary.nil_residual.reason}
          </p>
        </div>
      )}
      {backtest && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Bucket</th>
                <th>N</th>
                <th>Pred</th>
                <th>Actual</th>
              </tr>
            </thead>
            <tbody>
              {backtest.calibration.map((c) => (
                <tr key={c.bucket}>
                  <td>{c.bucket}</td>
                  <td>{c.n}</td>
                  <td>{c.pred}</td>
                  <td>{c.actual}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
