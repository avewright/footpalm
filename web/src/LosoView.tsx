import { useMemo, useState } from "react";
import type { LosoFile, LosoRow } from "./types";

type Trial = {
  i: number;
  score: number;
  kept: boolean;
  label: string;
  pass: string;
  features: string[];
  row: LosoRow;
};

const SET_BLURB: Record<string, string> = {
  locked: "the locked 10 Pom features only",
  extras: "locked Pom plus extras — Elo, form, SOS, rest, and luck",
  signal: "locked Pom plus the signal menu (margins, Pythag, H2H)",
  "extras+loso": "extras plus the ten LOSO columns (GLM quality, conference Pom, late form, year-over-year)",
  "extras+conf": "extras plus the conference axis (conf Pom, P4, same-conference, out-of-conference)",
};

type BackSeason = {
  season: number;
  tabpfn?: { tabpfn_n?: number; tabpfn_brier?: number; tabpfn_logloss?: number; tabpfn_accuracy?: number };
};

export function tabpfnFile(backtest: { seasons?: BackSeason[] } | null): LosoFile | null {
  const rows: LosoRow[] = [];
  const folds = (backtest?.seasons || [])
    .map((s) => {
      const t = s.tabpfn;
      if (!t?.tabpfn_n || t.tabpfn_brier == null) return null;
      return {
        season: s.season,
        n: t.tabpfn_n,
        brier: t.tabpfn_brier,
        logloss: t.tabpfn_logloss ?? 0,
      };
    })
    .filter((s): s is NonNullable<typeof s> => Boolean(s));
  if (folds.length) {
    const last = folds[folds.length - 1];
    const mean = folds.reduce((a, s) => a + s.brier, 0) / folds.length;
    rows.push({
      id: "tabpfn/live",
      engine: "tabpfn",
      set: "extras",
      pooled: {
        n: folds.reduce((a, s) => a + s.n, 0),
        accuracy: backtest?.seasons?.find((s) => s.season === last.season)?.tabpfn?.tabpfn_accuracy ?? 0,
        brier: last.brier,
        logloss: last.logloss,
      },
      mean_season_brier: Math.round(mean * 10000) / 10000,
      seasons: folds,
    });
  }
  if (!rows.length) return null;
  return {
    protocol: "Walk-forward TabPFN-3 extras, 2014–2025. Not a LOSO fit.",
    rows,
  };
}

function engineName(engine: string) {
  if (engine === "tabpfn") return "TabPFN-3";
  if (engine === "logistic") return "Ridge logistic";
  if (engine === "lightgbm") return "LightGBM";
  if (engine === "xgboost") return "XGBoost";
  return engine;
}

function knobs(trial: Trial) {
  const t = trial.row.seasons.find((s) => s.T != null)?.T;
  if (trial.row.engine === "tabpfn") {
    const loso = trial.pass === "loso" || trial.pass === "conference";
    const fit = loso ? "LOSO, last 8000 of other seasons" : "walk-forward extras, last 8000";
    return t != null ? `TabPFN-3 ${fit}. Temperature T=${t}` : `TabPFN-3 ${fit}`;
  }
  return trial.row.engine === "logistic" ? "Ridge 1.0, 40 IRLS steps" : "200 trees, depth 4, learning rate 0.05";
}

function trialsOf(files: { pass: string; data: LosoFile }[]): Trial[] {
  const out: Trial[] = [];
  let best = Infinity;
  for (const { pass, data } of files) {
    const extra = data.loso_features || data.features || [];
    for (const row of data.rows) {
      const score = row.mean_season_brier;
      const kept = score < best - 1e-9;
      if (kept) best = score;
      out.push({
        i: out.length,
        score,
        kept,
        label: row.id,
        pass,
        features: extra,
        row,
      });
    }
  }
  return out;
}

function lastKeptBefore(trials: Trial[], i: number) {
  return [...trials].filter((t) => t.i < i && t.kept).pop();
}

function blurb(trial: Trial, trials: Trial[]) {
  const who = engineName(trial.row.engine);
  const features = SET_BLURB[trial.row.set] ?? trial.row.set;
  const setup = `${who} on ${features}. ${knobs(trial)}.`;
  const prev = lastKeptBefore(trials, trial.i);
  const last = [...trials].reverse().find((t) => t.kept);
  const scoreName = "Mean season Brier";
  if (trial.kept && !prev) {
    return `${setup} First trial, so it opens the running best at ${trial.score.toFixed(4)}.`;
  }
  if (trial.kept && prev) {
    const drop = prev.score - trial.score;
    const crown = last?.i === trial.i ? " This is still the low." : "";
    return `${setup} Beat ${prev.label} by ${drop.toFixed(4)} (${prev.score.toFixed(4)} → ${trial.score.toFixed(4)}).${crown}`;
  }
  const best = prev ?? trials.find((t) => t.kept);
  if (!best) return setup;
  const worse = trial.score - best.score;
  return `${setup} ${scoreName} ${trial.score.toFixed(4)}, ${worse.toFixed(4)} behind ${best.label}. Not a new low.`;
}

function Chart({
  trials,
  selected,
  onPick,
}: {
  trials: Trial[];
  selected: number | null;
  onPick: (i: number) => void;
}) {
  const w = 920;
  const h = 280;
  const pad = { l: 48, r: 20, t: 24, b: 48 };
  const xs = trials.map((t) => t.i);
  const ys = trials.map((t) => t.score);
  const x0 = -0.5;
  const x1 = Math.max(...xs, 1) + 0.5;
  const yLo = Math.min(...ys);
  const yHi = Math.max(...ys);
  const yPad = (yHi - yLo) * 0.22 || 0.002;
  const y0 = yLo - yPad * 0.25;
  const y1 = yHi + yPad;
  const x = (v: number) => pad.l + ((v - x0) / (x1 - x0)) * (w - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (v - y0) / (y1 - y0 || 1)) * (h - pad.t - pad.b);
  let run = trials[0]?.score ?? 0;
  const step = [`M ${x(0).toFixed(1)} ${y(run).toFixed(1)}`];
  for (const t of trials.slice(1)) {
    step.push(`L ${x(t.i).toFixed(1)} ${y(run).toFixed(1)}`);
    if (t.kept) {
      run = t.score;
      step.push(`L ${x(t.i).toFixed(1)} ${y(run).toFixed(1)}`);
    }
  }
  const last = trials.length - 1;
  const xticks = [0, last].filter((n, i, a) => a.indexOf(n) === i && n >= 0);
  const yticks = [yLo, yHi];

  return (
    <svg className="loso-svg" viewBox={`0 0 ${w} ${h}`} role="img">
      {yticks.map((v, i) => (
        <text key={`y${i}`} x={pad.l - 8} y={y(v) + 4} textAnchor="end" className="loso-num">
          {v.toFixed(3)}
        </text>
      ))}
      {xticks.map((v) => (
        <text key={`x${v}`} x={x(v)} y={h - 22} textAnchor="middle" className="loso-num">
          {v}
        </text>
      ))}
      <text x={pad.l} y={14} className="loso-axis">
        Brier
      </text>
      <text x={(pad.l + w - pad.r) / 2} y={h - 6} textAnchor="middle" className="loso-axis">
        Experiment
      </text>
      <path d={step.join(" ")} className="loso-step" fill="none" />
      {trials.map((t) => (
        <g key={t.i} className="loso-hit" onClick={() => onPick(t.i)}>
          <circle cx={x(t.i)} cy={y(t.score)} r="14" fill="transparent" />
          <circle
            className={`loso-dot${t.kept ? " is-kept" : ""}${selected === t.i ? " is-on" : ""}`}
            cx={x(t.i)}
            cy={y(t.score)}
            r={selected === t.i ? 7 : t.kept ? 5.5 : 3.5}
          />
        </g>
      ))}
      {selected != null &&
        (() => {
          const t = trials[selected];
          if (!t) return null;
          const left = t.i > last * 0.7;
          return (
            <text
              className="loso-lab is-on"
              x={x(t.i) + (left ? -10 : 10)}
              y={y(t.score) - 14}
              textAnchor={left ? "end" : "start"}
            >
              {t.label}
            </text>
          );
        })()}
    </svg>
  );
}

function Stats({ row }: { row: LosoRow }) {
  const items = [
    { label: "Mean Brier", value: row.mean_season_brier.toFixed(4) },
    { label: "Pooled", value: row.pooled.brier.toFixed(4) },
    { label: "Log loss", value: row.pooled.logloss.toFixed(3) },
    { label: "Accuracy", value: `${(row.pooled.accuracy * 100).toFixed(1)}%` },
    { label: "Games", value: String(row.pooled.n) },
  ];
  return (
    <dl className="loso-stats">
      {items.map((item) => (
        <div key={item.label}>
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function LosoView({ files }: { files: { pass: string; data: LosoFile }[] }) {
  const trials = useMemo(() => trialsOf(files), [files]);
  const lastKept = [...trials].reverse().find((t) => t.kept)?.i ?? 0;
  const [picked, setPicked] = useState<number | null>(lastKept);
  const trial = trials.find((t) => t.i === picked) ?? null;

  return (
    <div className="loso">
      <p className="lede-note">
        {trials.length} trials. Mean of season Briers. Green is a new low. Click a point.
      </p>
      <Chart trials={trials} selected={picked} onPick={setPicked} />
      {trial && (
        <section className="loso-pick">
          <div>
            <p className="loso-kicker">{trial.kept ? "Kept" : "Discarded"} · #{trial.i}</p>
            <h2 className="loso-title">{trial.label}</h2>
            <p className="loso-blurb">{blurb(trial, trials)}</p>
            <Stats row={trial.row} />
          </div>
          <div>
            <div className="table-wrap loso-years">
              <table>
                <thead>
                  <tr>
                    <th className="left">Season</th>
                    <th>n</th>
                    <th>Brier</th>
                    <th>Log loss</th>
                  </tr>
                </thead>
                <tbody>
                  {trial.row.seasons.map((s) => (
                    <tr key={s.season}>
                      <td className="left">{s.season}</td>
                      <td>{s.n}</td>
                      <td>{s.brier.toFixed(4)}</td>
                      <td>{s.logloss.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
