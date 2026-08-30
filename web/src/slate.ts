import { signed } from "./format";

export type SlateTeam = {
  id: string;
  team: string;
  conf: string;
  pom: number;
  wins: number;
  losses: number;
};

export type SlateGame = { opp: SlateTeam; won: boolean; margin: number };

export type SlateRow = {
  team: SlateTeam;
  games: SlateGame[];
  sos: number;
  lo: number;
  hi: number;
  span: number;
  confs: number;
  x: number;
  y: number;
};

export type SlateChart = {
  rows: SlateRow[];
  kind: "scatter" | "rows";
  pad: { l: number; r: number; t: number; b: number };
  pomLo: number;
  pomHi: number;
};

type Edge = { source: string; target: string; margin: number };

const WIDTH = 960;
const HEIGHT = 700;
const PAPER = "#ffffff";
const INK = "#111111";
const MUTED = "#666666";
const LINE = "#e5e5e5";
const WIN = "#1a7f37";
const LOSS = "#b42318";
const FONT = "12px ui-sans-serif, system-ui, sans-serif";

function map(v: number, a: number, b: number, lo: number, hi: number) {
  if (b === a) return (lo + hi) / 2;
  return lo + ((v - a) / (b - a)) * (hi - lo);
}

function short(name: string, n = 12) {
  return name.length > n ? `${name.slice(0, n - 1)}…` : name;
}

export function buildSlate(teams: SlateTeam[], edges: Edge[]): SlateRow[] {
  const byId = new Map(teams.map((t) => [t.id, t]));
  const games = new Map<string, SlateGame[]>();
  for (const t of teams) games.set(t.id, []);
  for (const e of edges) {
    const a = byId.get(e.source);
    const b = byId.get(e.target);
    if (!a || !b) continue;
    games.get(a.id)?.push({ opp: b, won: true, margin: e.margin });
    games.get(b.id)?.push({ opp: a, won: false, margin: e.margin });
  }
  return teams
    .map((team) => {
      const slate = games.get(team.id) ?? [];
      const poms = slate.map((g) => g.opp.pom);
      const lo = poms.length ? Math.min(...poms) : team.pom;
      const hi = poms.length ? Math.max(...poms) : team.pom;
      const sos = poms.length ? poms.reduce((s, v) => s + v, 0) / poms.length : team.pom;
      return {
        team,
        games: slate,
        sos,
        lo,
        hi,
        span: hi - lo,
        confs: new Set(slate.map((g) => g.opp.conf)).size,
        x: 0,
        y: 0,
      };
    })
    .sort((a, b) => b.team.pom - a.team.pom);
}

export function placeSlate(rows: SlateRow[], conference: string): SlateChart {
  if (conference !== "all" && rows.length <= 24) {
    const pad = { l: 108, r: 48, t: 36, b: 44 };
    const poms = rows.flatMap((r) => [r.team.pom, r.lo, r.hi]);
    const pomLo = Math.min(...poms, -20);
    const pomHi = Math.max(...poms, 20);
    const n = Math.max(rows.length, 1);
    const xOf = (v: number) => map(v, pomLo, pomHi, pad.l, WIDTH - pad.r);
    const placed = rows.map((r, i) => ({
      ...r,
      x: (xOf(r.lo) + xOf(r.hi)) / 2,
      y: map(i, 0, n - 1, pad.t + 8, HEIGHT - pad.b - 8),
    }));
    return { rows: placed, kind: "rows", pad, pomLo, pomHi };
  }
  const pad = { l: 58, r: 24, t: 28, b: 48 };
  const poms = rows.map((r) => r.team.pom);
  const soss = rows.map((r) => r.sos);
  const yLo = Math.min(...poms, -20);
  const yHi = Math.max(...poms, 20);
  const xLo = Math.min(...soss, -12);
  const xHi = Math.max(...soss, 12);
  const placed = rows.map((r) => ({
    ...r,
    x: map(r.sos, xLo, xHi, pad.l, WIDTH - pad.r),
    y: map(r.team.pom, yHi, yLo, pad.t, HEIGHT - pad.b),
  }));
  return { rows: placed, kind: "scatter", pad, pomLo: xLo, pomHi: xHi };
}

function axis(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number) {
  ctx.strokeStyle = LINE;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

function pickLabels(rows: SlateRow[], hover: string, max = 8) {
  const named = new Set<string>();
  if (hover) named.add(hover);
  const add = (row?: SlateRow) => {
    if (!row || named.has(row.team.id) || named.size >= max) return;
    named.add(row.team.id);
  };
  add(rows[0]);
  add([...rows].sort((a, b) => b.sos - a.sos)[0]);
  add([...rows].sort((a, b) => a.sos - b.sos)[0]);
  add([...rows].sort((a, b) => b.span - a.span)[0]);
  const leftover = rows.filter((r) => r.team.pom > 8).sort((a, b) => a.sos - b.sos);
  add(leftover[0]);
  const gauntlet = rows.filter((r) => r.team.pom < 2).sort((a, b) => b.sos - a.sos);
  add(gauntlet[0]);
  return named;
}

export function drawSlate(ctx: CanvasRenderingContext2D, chart: SlateChart, hover: string) {
  ctx.fillStyle = PAPER;
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  ctx.font = FONT;
  if (chart.kind === "rows") drawRows(ctx, chart, hover);
  else drawScatter(ctx, chart, hover);
}

function drawScatter(ctx: CanvasRenderingContext2D, chart: SlateChart, hover: string) {
  const { rows, pad, pomLo, pomHi } = chart;
  const poms = rows.map((r) => r.team.pom);
  const yLo = Math.min(...poms, -20);
  const yHi = Math.max(...poms, 20);
  axis(ctx, pad.l, pad.t, pad.l, HEIGHT - pad.b);
  axis(ctx, pad.l, HEIGHT - pad.b, WIDTH - pad.r, HEIGHT - pad.b);
  ctx.fillStyle = MUTED;
  ctx.textAlign = "center";
  ctx.fillText("Mean opponent Pom →", WIDTH / 2, HEIGHT - 14);
  ctx.save();
  ctx.translate(16, HEIGHT / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Pom (quality)", 0, 0);
  ctx.restore();
  ctx.textAlign = "right";
  ctx.fillText(signed(yHi, 0), pad.l - 8, pad.t + 4);
  ctx.fillText(signed(yLo, 0), pad.l - 8, HEIGHT - pad.b + 4);
  ctx.textAlign = "center";
  ctx.fillText(signed(pomLo, 0), pad.l, HEIGHT - pad.b + 16);
  ctx.fillText(signed(pomHi, 0), WIDTH - pad.r, HEIGHT - pad.b + 16);

  const xRaw = (v: number) => map(v, pomLo, pomHi, pad.l, WIDTH - pad.r);
  const xOf = (v: number) => Math.max(pad.l, Math.min(WIDTH - pad.r, xRaw(v)));
  const yOf = (v: number) => map(v, yHi, yLo, pad.t, HEIGHT - pad.b);
  const diagLo = Math.max(yLo, pomLo);
  const diagHi = Math.min(yHi, pomHi);
  ctx.strokeStyle = "rgba(102,102,102,0.35)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(xRaw(diagLo), yOf(diagLo));
  ctx.lineTo(xRaw(diagHi), yOf(diagHi));
  ctx.stroke();
  ctx.setLineDash([]);

  for (const r of rows) {
    const on = hover === r.team.id;
    const dim = hover && !on;
    const x0 = xOf(r.lo);
    const x1 = xOf(r.hi);
    ctx.beginPath();
    ctx.strokeStyle = dim ? "rgba(17,17,17,0.06)" : on ? INK : "rgba(17,17,17,0.22)";
    ctx.lineWidth = on ? 2 : 1;
    ctx.moveTo(x0, r.y);
    ctx.lineTo(x1, r.y);
    ctx.stroke();
    if (on) {
      for (const g of r.games) {
        ctx.beginPath();
        ctx.fillStyle = g.won ? WIN : LOSS;
        ctx.arc(xOf(g.opp.pom), r.y, 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.beginPath();
    ctx.fillStyle = dim ? "rgba(17,17,17,0.18)" : INK;
    ctx.arc(r.x, r.y, on ? 5.2 : 3.4, 0, Math.PI * 2);
    ctx.fill();
  }

  const named = pickLabels(rows, hover, rows.length < 22 ? rows.length : 8);
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  ctx.fillStyle = INK;
  for (const r of rows) {
    if (!named.has(r.team.id)) continue;
    ctx.textAlign = r.x > WIDTH / 2 ? "right" : "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(r.team.team, r.x + (r.x > WIDTH / 2 ? -7 : 7), r.y - 6);
  }
  ctx.textBaseline = "alphabetic";
}

function drawRows(ctx: CanvasRenderingContext2D, chart: SlateChart, hover: string) {
  const { rows, pad, pomLo, pomHi } = chart;
  const xOf = (v: number) => map(v, pomLo, pomHi, pad.l, WIDTH - pad.r);
  axis(ctx, pad.l, HEIGHT - pad.b, WIDTH - pad.r, HEIGHT - pad.b);
  ctx.fillStyle = MUTED;
  ctx.textAlign = "center";
  ctx.fillText("Opponent Pom →", WIDTH / 2, HEIGHT - 14);
  ctx.fillText(signed(pomLo, 0), pad.l, HEIGHT - pad.b + 16);
  ctx.fillText(signed(pomHi, 0), WIDTH - pad.r, HEIGHT - pad.b + 16);

  for (const r of rows) {
    const on = hover === r.team.id;
    const dim = hover && !on;
    const x0 = xOf(r.lo);
    const x1 = xOf(r.hi);
    ctx.beginPath();
    ctx.strokeStyle = dim ? "rgba(17,17,17,0.1)" : "rgba(17,17,17,0.35)";
    ctx.lineWidth = on ? 5 : 3.2;
    ctx.lineCap = "round";
    ctx.moveTo(x0, r.y);
    ctx.lineTo(x1, r.y);
    ctx.stroke();
    ctx.lineCap = "butt";
    for (const g of r.games) {
      ctx.beginPath();
      ctx.fillStyle = dim ? "rgba(17,17,17,0.15)" : g.won ? WIN : LOSS;
      ctx.arc(xOf(g.opp.pom), r.y, on ? 4 : 3, 0, Math.PI * 2);
      ctx.fill();
    }
    const selfX = xOf(r.team.pom);
    ctx.beginPath();
    ctx.strokeStyle = dim ? "#bbbbbb" : INK;
    ctx.lineWidth = 1.4;
    ctx.moveTo(selfX, r.y - 7);
    ctx.lineTo(selfX, r.y + 7);
    ctx.stroke();
    ctx.fillStyle = dim ? "#999999" : INK;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(short(r.team.team), pad.l - 8, r.y);
    ctx.textAlign = "left";
    ctx.fillStyle = dim ? "#bbbbbb" : MUTED;
    ctx.fillText(`${r.confs}c`, WIDTH - pad.r + 8, r.y);
  }
  ctx.textBaseline = "alphabetic";
}

export function slateLine(row: SlateRow) {
  return `SoS ${signed(row.sos)} · opponents ${signed(row.lo, 0)} to ${signed(row.hi, 0)} · ${row.confs} conferences`;
}
