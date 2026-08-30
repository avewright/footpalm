import { useEffect, useMemo, useState } from "react";
import { atsHit, atsSide, betLabel, marketSpread, mlPlay } from "./ev";
import { signed } from "./format";
import { atsPickLabel, etDay, isFinal, lastDay, prettyDay, record, suHit, summarize } from "./score";
import type { GamePred, ModelPick } from "./types";

const MODEL_IDS = ["lightgbm", "xgboost", "tabpfn"] as const;

function pickOf(g: GamePred, id: (typeof MODEL_IDS)[number] | "ensemble"): ModelPick | null {
  return g.models?.[id] ?? (id === "ensemble" ? { pred_margin: g.pred_margin, home_win_prob: g.home_win_prob } : null);
}

function evPct(ev: number) {
  return `${ev > 0 ? "+" : ""}${(ev * 100).toFixed(0)}%`;
}

function pct(p: number) {
  return `${(p * 100).toFixed(0)}%`;
}

function pp(n: number) {
  return `${n > 0 ? "+" : ""}${(n * 100).toFixed(0)}pp`;
}

function tone(n: number | null | undefined, kind: "ev" | "pp" = "ev") {
  if (n == null) return undefined;
  const cut = kind === "pp" ? 0.02 : 0;
  if (n > cut) return "good";
  if (n < -cut) return "bad";
  return undefined;
}

function hitTone(hit: boolean | null) {
  if (hit == null) return undefined;
  return hit ? "good" : "bad";
}

function Row({ label, value, tone: t }: { label: string; value: string; tone?: string }) {
  return (
    <div className="slip-row">
      <span className="quiet">{label}</span>
      <span className={t}>{value}</span>
    </div>
  );
}

function Card({ title, card }: { title: string; card: ReturnType<typeof summarize> }) {
  if (!card.n) return null;
  return (
    <div className="score-card">
      <h3>{title}</h3>
      <dl className="metrics">
        <dt>Straight up</dt>
        <dd>{record(card.suW, card.suL)}</dd>
        <dt>ATS</dt>
        <dd>{record(card.atsW, card.atsL)}</dd>
        <dt>Brier</dt>
        <dd>{card.brier == null ? "—" : card.brier.toFixed(3)}</dd>
        <dt>Margin MAE</dt>
        <dd>{card.mae == null ? "—" : card.mae.toFixed(1)}</dd>
      </dl>
    </div>
  );
}

export function GamesView({ games }: { games: GamePred[] }) {
  const weeks = useMemo(() => [...new Set(games.map((g) => g.week))].sort((a, b) => a - b), [games]);
  const finals = useMemo(() => games.filter((g) => g.fbs_fbs && isFinal(g)), [games]);
  const hasFinals = finals.length > 0;
  const hasBooks = games.some((g) => g.books?.polymarket);
  const day = lastDay(finals);
  const last = useMemo(() => summarize(day ? finals.filter((g) => etDay(g.start) === day) : []), [finals, day]);
  const season = useMemo(() => summarize(finals), [finals]);

  const [week, setWeek] = useState<number | "all">("all");
  const [q, setQ] = useState("");
  const [listed, setListed] = useState(true);
  const [sort, setSort] = useState<"spread" | "ml">("spread");
  const [view, setView] = useState<"final" | "upcoming" | "all">("upcoming");

  useEffect(() => {
    if (hasFinals) setView("final");
  }, [hasFinals]);

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const out = games.filter((g) => {
      if (!g.fbs_fbs) return false;
      const done = isFinal(g);
      if (view === "final" && !done) return false;
      if (view === "upcoming" && done) return false;
      if (listed && hasBooks && view !== "final" && !g.books?.polymarket) return false;
      if (week !== "all" && g.week !== week) return false;
      if (needle && !`${g.away} ${g.home}`.toLowerCase().includes(needle)) return false;
      return true;
    });
    const scored = out.map((g) => {
      const ens = pickOf(g, "ensemble");
      const play = atsSide(g);
      const ml = ens ? mlPlay(g, ens.home_win_prob) : null;
      const fairs = MODEL_IDS.map((id) => pickOf(g, id));
      const agree = play ? fairs.filter((p) => p && atsSide(g, p)?.who === play.who).length : 0;
      const start = g.start ? Date.parse(g.start) : 0;
      return {
        g,
        ens,
        play,
        ml,
        us: ens ? -ens.pred_margin : null,
        mkt: marketSpread(g),
        agree,
        hit: play ? atsHit(g, play) : null,
        su: suHit(g),
        sprKey: play?.ev ?? -99,
        mlKey: ml?.edge ?? -99,
        start,
      };
    });
    scored.sort((a, b) => {
      if (view === "final") return b.start - a.start;
      return sort === "ml" ? b.mlKey - a.mlKey : b.sprKey - a.sprKey;
    });
    return scored.map((row, i) => ({ ...row, rank: row.play || row.ml ? i + 1 : null }));
  }, [games, week, q, listed, hasBooks, sort, view]);

  return (
    <div>
      {hasFinals && (
        <div className="score-strip">
          {day && <Card title={prettyDay(day)} card={last} />}
          <Card title="Season" card={season} />
        </div>
      )}
      <p className="lede-note">
        {view === "final"
          ? "Frozen pre-game numbers vs the final score. ATS is a holdout, not a training target."
          : "Spread = cover the number at −110. Moneyline = win the game. Market is Polymarket when listed. Us is our ensemble. Not a training target."}
      </p>
      <div className="toolbar">
        <div className="seg" role="group" aria-label="Games view">
          <button type="button" aria-pressed={view === "final"} onClick={() => setView("final")} disabled={!hasFinals}>
            Final
          </button>
          <button type="button" aria-pressed={view === "upcoming"} onClick={() => setView("upcoming")}>
            Upcoming
          </button>
          <button type="button" aria-pressed={view === "all"} onClick={() => setView("all")}>
            All
          </button>
        </div>
        <input
          type="search"
          placeholder="Team"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          aria-label="Filter games"
        />
        <select
          value={week}
          onChange={(e) => setWeek(e.target.value === "all" ? "all" : Number(e.target.value))}
          aria-label="Week"
        >
          <option value="all">All weeks</option>
          {weeks.map((w) => (
            <option key={w} value={w}>
              Week {w}
            </option>
          ))}
        </select>
        {view !== "final" && (
          <select value={sort} onChange={(e) => setSort(e.target.value as "spread" | "ml")} aria-label="Rank by">
            <option value="spread">Rank by spread EV</option>
            <option value="ml">Rank by moneyline edge</option>
          </select>
        )}
        {hasBooks && view !== "final" && (
          <label>
            <input type="checkbox" checked={listed} onChange={(e) => setListed(e.target.checked)} />
            Listed only
          </label>
        )}
        <span className="lede-note" style={{ margin: 0 }}>
          {rows.length} games
        </span>
      </div>
      <div className="slips">
        {rows.map(({ g, ens, play, ml, us, mkt, agree, hit, su, rank }) => (
          <article key={`${g.week}-${g.home}-${g.away}`} className="slip">
            <header className="slip-head">
              <div className="slip-rank">{view === "final" ? (su ? "W" : su === false ? "L" : "—") : (rank ?? "—")}</div>
              <div>
                <div className="slip-game">
                  {g.away} @ {g.home}
                  {g.neutral ? " (N)" : ""}
                  {isFinal(g) ? ` · ${g.actual_away?.toFixed(0)}–${g.actual_home?.toFixed(0)}` : ""}
                </div>
                <div className="quiet">
                  Week {g.week}
                  {su == null ? "" : su ? " · SU hit" : " · SU miss"}
                  {hit == null ? "" : hit ? " · ATS hit" : " · ATS miss"}
                </div>
              </div>
            </header>
            <div className="slip-cols">
              <section>
                <h3>Spread</h3>
                <Row label="Bet" value={play ? betLabel(play) : atsPickLabel(g) ?? "—"} tone={hitTone(hit)} />
                {view === "final" ? (
                  <Row label="Result" value={hit == null ? "—" : hit ? "Covered" : "Missed"} tone={hitTone(hit)} />
                ) : (
                  <Row label="EV" value={play ? evPct(play.ev) : "—"} tone={tone(play?.ev)} />
                )}
                <Row label="Market" value={mkt == null ? "—" : `${g.home} ${signed(mkt)}`} />
                <Row label="Us" value={us == null ? "—" : `${g.home} ${signed(us)}`} />
                {view !== "final" && (
                  <Row label="Agree" value={play ? `${agree} of 3 models` : "—"} tone={agree === 3 ? "good" : agree <= 1 && play ? "bad" : undefined} />
                )}
              </section>
              <section>
                <h3>Moneyline</h3>
                {view === "final" ? (
                  <>
                    <Row label="Pick" value={ens ? `${ens.home_win_prob >= 0.5 ? g.home : g.away} ${pct(ens.home_win_prob >= 0.5 ? ens.home_win_prob : 1 - ens.home_win_prob)}` : "—"} tone={hitTone(su)} />
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
                <Row label="Market" value={ml ? `${g.home} ${pct(ml.mktHome)}` : mkt == null ? "—" : "—"} />
                <Row label="Us" value={ens ? `${g.home} ${pct(ens.home_win_prob)}` : "—"} />
              </section>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
