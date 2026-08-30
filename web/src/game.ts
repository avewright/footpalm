import { atsHit, atsSide, betLabel, marketMl, marketSpread, mlPlay } from "./ev";
import { isFinal, suHit } from "./score";
import type { GamePred, ModelPick } from "./types";

export const MODEL_IDS = ["lightgbm", "xgboost", "tabpfn"] as const;

export function gameKey(g: GamePred) {
  return g.game_id != null ? String(g.game_id) : `${g.season}-${g.week}-${g.away}-${g.home}`;
}

export function findGame(games: GamePred[], key: string) {
  return games.find((g) => gameKey(g) === key) ?? null;
}

export function pickOf(g: GamePred, id: (typeof MODEL_IDS)[number] | "ensemble"): ModelPick | null {
  return g.models?.[id] ?? (id === "ensemble" ? { pred_margin: g.pred_margin, home_win_prob: g.home_win_prob } : null);
}

export function pct(p: number) {
  return `${(p * 100).toFixed(0)}%`;
}

export function evPct(ev: number) {
  return `${ev > 0 ? "+" : ""}${(ev * 100).toFixed(0)}%`;
}

export function pp(n: number) {
  return `${n > 0 ? "+" : ""}${(n * 100).toFixed(0)}pp`;
}

export function tone(n: number | null | undefined, kind: "ev" | "pp" = "ev") {
  if (n == null) return undefined;
  const cut = kind === "pp" ? 0.02 : 0;
  if (n > cut) return "good";
  if (n < -cut) return "bad";
  return undefined;
}

export function hitTone(hit: boolean | null) {
  if (hit == null) return undefined;
  return hit ? "good" : "bad";
}

export function gameFacts(g: GamePred) {
  const ens = pickOf(g, "ensemble");
  const play = atsSide(g);
  const ml = ens ? mlPlay(g, ens.home_win_prob) : null;
  const fairs = MODEL_IDS.map((id) => pickOf(g, id));
  const agree = play ? fairs.filter((p) => p && atsSide(g, p)?.who === play.who).length : 0;
  return {
    ens,
    play,
    ml,
    us: ens ? -ens.pred_margin : null,
    mkt: marketSpread(g),
    mktMl: marketMl(g),
    agree,
    hit: play ? atsHit(g, play) : null,
    su: suHit(g),
    pick: play ? betLabel(play) : null,
    done: isFinal(g),
    score: isFinal(g) ? `${g.actual_away?.toFixed(0)}–${g.actual_home?.toFixed(0)}` : null,
    pred: `${g.pred_away.toFixed(0)}–${g.pred_home.toFixed(0)}`,
  };
}
