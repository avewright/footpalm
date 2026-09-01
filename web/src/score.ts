import { atsHit, atsSide, betLabel, marketSpread } from "./ev";
import type { GamePred } from "./types";

export type ScoreCard = {
  n: number;
  suW: number;
  suL: number;
  atsW: number;
  atsL: number;
  brier: number | null;
  mae: number | null;
  residual: number | null;
};

export function etDay(start?: string | null): string | null {
  if (!start) return null;
  return new Date(start).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
}

export function prettyWhen(start?: string | null): string | null {
  if (!start) return null;
  return new Date(start).toLocaleString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/New_York",
  });
}

export function prettyDay(iso: string): string {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d, 16)).toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

export function isFinal(g: GamePred): boolean {
  return g.actual_home != null && g.actual_away != null;
}

export function suHit(g: GamePred): boolean | null {
  if (!isFinal(g) || g.home_won == null) return null;
  return g.home_win_prob >= 0.5 === Boolean(g.home_won);
}

export function summarize(games: GamePred[]): ScoreCard {
  const played = games.filter(isFinal);
  let suW = 0;
  let atsW = 0;
  let atsN = 0;
  let brier = 0;
  let mae = 0;
  let residual = 0;
  let residualN = 0;
  for (const g of played) {
    const su = suHit(g);
    if (su) suW += 1;
    const play = atsSide(g);
    const hit = play ? atsHit(g, play) : null;
    if (hit != null) {
      atsN += 1;
      if (hit) atsW += 1;
    }
    const y = g.home_won ?? 0;
    brier += (g.home_win_prob - y) ** 2;
    if (g.actual_margin != null) {
      mae += Math.abs(g.pred_margin - g.actual_margin);
      residual += g.actual_margin - g.pred_margin;
      residualN += 1;
    }
  }
  const n = played.length;
  return {
    n,
    suW,
    suL: n - suW,
    atsW,
    atsL: atsN - atsW,
    brier: n ? brier / n : null,
    mae: n ? mae / n : null,
    residual: residualN ? residual / residualN : null,
  };
}

export function lastDay(games: GamePred[]): string | null {
  const days = games.filter(isFinal).map((g) => etDay(g.start)).filter((d): d is string => Boolean(d));
  days.sort();
  return days.at(-1) ?? null;
}

export function record(w: number, l: number): string {
  return `${w}–${l}`;
}

export function atsPickLabel(g: GamePred): string | null {
  const play = atsSide(g);
  return play ? betLabel(play) : marketSpread(g) == null ? null : "—";
}
