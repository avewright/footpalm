import { american, signed } from "./format";
import type { GamePred } from "./types";

export function marketSpread(g: GamePred): number | null {
  const poly = g.books?.polymarket?.spread;
  if (poly != null && !Number.isNaN(poly)) return poly;
  if (g.spread == null || Number.isNaN(g.spread)) return null;
  return g.spread;
}

export const JUICE = 110;
export const BREAKEVEN = JUICE / (JUICE + 100);
export const BIG_NUMBER = 17.5;

export type Side = {
  who: string;
  line: number;
  pCover: number;
  ev: number;
  edge: number;
  home: boolean;
};

function clip(p: number) {
  return Math.min(1 - 1e-6, Math.max(1e-6, p));
}

export function impliedSigma(predMargin: number, pHome: number) {
  const z = Math.log(clip(pHome) / (1 - clip(pHome)));
  if (Math.abs(z) < 1e-9 || Math.abs(predMargin) < 1e-6) return 14.5;
  if (Math.sign(predMargin) !== Math.sign(z)) return 14.5;
  return Math.min(22, Math.max(10, predMargin / z));
}

export function evAtJuice(p: number, juice = JUICE) {
  return p * (100 / juice) - (1 - p);
}

export function atsSideFrom(
  predMargin: number,
  pHomeWin: number,
  spread: number,
  home: string,
  away: string,
): Side {
  const sigma = impliedSigma(predMargin, pHomeWin);
  const pHome = 1 / (1 + Math.exp(-(predMargin + spread) / sigma));
  const takeHome = pHome >= 0.5;
  const pCover = takeHome ? pHome : 1 - pHome;
  return {
    who: takeHome ? home : away,
    line: takeHome ? spread : -spread,
    pCover,
    ev: evAtJuice(pCover),
    edge: Math.abs(predMargin + spread),
    home: takeHome,
  };
}

export function atsSide(g: GamePred, pick?: { pred_margin: number; home_win_prob: number }): Side | null {
  const spread = marketSpread(g);
  if (spread == null) return null;
  if (!pick && g.models?.lightgbm && g.models?.xgboost && g.models?.tabpfn) {
    const parts = [g.models.lightgbm, g.models.xgboost, g.models.tabpfn].map((m) =>
      atsSideFrom(m.pred_margin, m.home_win_prob, spread, g.home, g.away),
    );
    const pHome = parts.reduce((sum, side) => sum + (side.home ? side.pCover : 1 - side.pCover), 0) / parts.length;
    const takeHome = pHome >= 0.5;
    const pCover = takeHome ? pHome : 1 - pHome;
    return {
      who: takeHome ? g.home : g.away,
      line: takeHome ? spread : -spread,
      pCover,
      ev: evAtJuice(pCover),
      edge: Math.abs((g.models.ensemble?.pred_margin ?? g.pred_margin) + spread),
      home: takeHome,
    };
  }
  const src = pick ?? g.models?.ensemble ?? g;
  return atsSideFrom(src.pred_margin, src.home_win_prob, spread, g.home, g.away);
}

export function atsHit(g: GamePred, side: Side): boolean | null {
  if (g.actual_margin == null) return null;
  const spread = marketSpread(g);
  if (spread == null) return null;
  const margin = g.actual_margin + spread;
  if (Math.abs(margin) < 1e-9) return null;
  const homeCovers = margin > 0;
  return side.home === homeCovers;
}

export function betLabel(side: Side) {
  return `${side.who} ${signed(side.line)}`;
}

export type MlPlay = {
  who: string;
  ourHome: number;
  mktHome: number;
  edge: number;
  home: boolean;
  ourAmerican: number;
  mktAmerican: number;
};

export function mlPlay(g: GamePred, pHome: number): MlPlay | null {
  const mkt = g.books?.polymarket?.ml_home;
  if (mkt == null || Number.isNaN(mkt)) return null;
  const takeHome = pHome >= mkt;
  const edge = takeHome ? pHome - mkt : mkt - pHome;
  const mktAway = g.books?.polymarket?.ml_away_american;
  const mktHomeAm = g.books?.polymarket?.ml_home_american;
  return {
    who: takeHome ? g.home : g.away,
    ourHome: pHome,
    mktHome: mkt,
    edge,
    home: takeHome,
    ourAmerican: american(takeHome ? pHome : 1 - pHome),
    mktAmerican: takeHome ? (mktHomeAm ?? 0) : (mktAway ?? 0),
  };
}
