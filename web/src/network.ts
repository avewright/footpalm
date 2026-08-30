import { confFill } from "./clusters";

export type NetworkStats = {
  n: number;
  undirected_edges: number;
  degree_mean: number;
  degree_min: number;
  degree_max: number;
  components: number;
  diameter: number;
  radius: number;
  average_path: number;
  bound_path: number;
  algebraic_connectivity?: number;
  distances: Record<string, number>;
  triangles: number;
  bridges: number;
  mst: { source: string; target: string; margin: number }[];
  cycles: { teams: string[]; margins: number[]; tension: number }[];
};

export type GraphEdge = { source: string; target: string; margin: number };

export type LaidTeam = {
  id: string;
  team: string;
  conf: string;
  nx: number;
  ny: number;
  betweenness: number;
  degree: number;
  eccentricity: number;
  x: number;
  y: number;
  r: number;
  group?: string;
  ray?: number;
};

const WIDTH = 960;
const HEIGHT = 700;
const PAPER = "#ffffff";
const INK = "#111111";
const MUTED = "#666666";
const LINE = "#e5e5e5";
const FONT = "12px ui-sans-serif, system-ui, sans-serif";
const P4 = new Set(["SEC", "B1G", "B12", "ACC"]);
const G5 = new Set(["AAC", "MWC", "MAC", "SBC", "CUSA"]);

function arrow(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, color: string, width = 1.3) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const dist = Math.hypot(dx, dy) || 1;
  const ux = dx / dist;
  const uy = dy / dist;
  const sx = x1 + ux * 8;
  const sy = y1 + uy * 8;
  const ex = x2 - ux * 8;
  const ey = y2 - uy * 8;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - ux * 8 + uy * 3.4, ey - uy * 8 - ux * 3.4);
  ctx.lineTo(ex - ux * 8 - uy * 3.4, ey - uy * 8 + ux * 3.4);
  ctx.closePath();
  ctx.fill();
}

function paintLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number) {
  ctx.lineWidth = 4;
  ctx.strokeStyle = PAPER;
  ctx.lineJoin = "round";
  ctx.strokeText(text, x, y);
  ctx.fillText(text, x, y);
}

function paper(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = PAPER;
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
}

function hiveGroup(conf: string) {
  if (P4.has(conf)) return "P4";
  if (G5.has(conf)) return "G5";
  return "other";
}

export function placeLayout(
  teams: Omit<LaidTeam, "x" | "y" | "r">[],
  xKey: "nx" | "ny" = "nx",
  yKey: "nx" | "ny" = "ny",
): LaidTeam[] {
  const xs = teams.map((t) => t[xKey]);
  const ys = teams.map((t) => t[yKey]);
  const x0 = xs.length ? Math.min(...xs) : -1;
  const x1 = xs.length ? Math.max(...xs) : 1;
  const y0 = ys.length ? Math.min(...ys) : -1;
  const y1 = ys.length ? Math.max(...ys) : 1;
  const pad = 36;
  const bet = teams.map((t) => t.betweenness);
  const b0 = Math.min(...bet, 0);
  const b1 = Math.max(...bet, 0.001);
  return teams.map((t) => ({
    ...t,
    x: pad + ((t[xKey] - x0) / Math.max(x1 - x0, 1e-6)) * (WIDTH - pad * 2),
    y: pad + ((t[yKey] - y0) / Math.max(y1 - y0, 1e-6)) * (HEIGHT - pad * 2 - 18),
    r: 3.2 + ((t.betweenness - b0) / (b1 - b0)) * 7,
  }));
}

export function placeMap(teams: Omit<LaidTeam, "x" | "y" | "r">[]): LaidTeam[] {
  return placeLayout(teams, "nx", "ny");
}

export function placeHive(teams: Omit<LaidTeam, "x" | "y" | "r">[]): LaidTeam[] {
  const groups = ["P4", "G5", "other"] as const;
  const buckets = new Map<string, Omit<LaidTeam, "x" | "y" | "r">[]>();
  for (const t of teams) {
    const g = hiveGroup(t.conf);
    const list = buckets.get(g) ?? [];
    list.push(t);
    buckets.set(g, list);
  }
  const present = groups.filter((g) => (buckets.get(g) ?? []).length);
  const cx = WIDTH / 2;
  const cy = HEIGHT / 2 - 6;
  const bet = teams.map((t) => t.betweenness);
  const b0 = Math.min(...bet, 0);
  const b1 = Math.max(...bet, 0.001);
  const sized = (t: Omit<LaidTeam, "x" | "y" | "r">) => 3.2 + ((t.betweenness - b0) / (b1 - b0)) * 6;

  if (present.length <= 1) {
    const list = [...teams].sort((a, b) => a.team.localeCompare(b.team));
    const n = Math.max(list.length, 1);
    const ring = Math.min(WIDTH, HEIGHT) * 0.36;
    return list.map((t, i) => {
      const a = (i / n) * Math.PI * 2 - Math.PI / 2;
      return { ...t, group: hiveGroup(t.conf), ray: a, x: cx + Math.cos(a) * ring, y: cy + Math.sin(a) * ring, r: sized(t) };
    });
  }

  const placed: LaidTeam[] = [];
  present.forEach((g, gi) => {
    const list = [...(buckets.get(g) ?? [])].sort((a, b) => a.conf.localeCompare(b.conf) || a.team.localeCompare(b.team));
    const ray = -Math.PI / 2 + (gi * 2 * Math.PI) / present.length;
    const inner = 72;
    const outer = Math.min(WIDTH, HEIGHT) * 0.42;
    list.forEach((t, i) => {
      const u = list.length === 1 ? 0.5 : i / (list.length - 1);
      const rad = inner + u * (outer - inner);
      placed.push({
        ...t,
        group: g,
        ray,
        x: cx + Math.cos(ray) * rad,
        y: cy + Math.sin(ray) * rad,
        r: sized(t),
      });
    });
  });
  return placed;
}

function drawDots(ctx: CanvasRenderingContext2D, placed: LaidTeam[], hover: string, named: Set<string>) {
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  for (const t of placed) {
    ctx.beginPath();
    ctx.fillStyle = confFill(t.conf);
    ctx.globalAlpha = hover && hover !== t.id ? 0.22 : 1;
    ctx.arc(t.x, t.y, hover === t.id ? t.r + 1.5 : t.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  ctx.fillStyle = INK;
  for (const t of placed) {
    if (!named.has(t.id)) continue;
    ctx.textAlign = t.x > WIDTH / 2 ? "left" : "right";
    ctx.textBaseline = "middle";
    paintLabel(ctx, t.team, t.x + (t.x > WIDTH / 2 ? t.r + 5 : -(t.r + 5)), t.y);
  }
  ctx.textBaseline = "alphabetic";
}

function caption(ctx: CanvasRenderingContext2D, text: string) {
  ctx.fillStyle = MUTED;
  ctx.textAlign = "center";
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText(text, WIDTH / 2, HEIGHT - 14);
}

function drawEdges(
  ctx: CanvasRenderingContext2D,
  placed: LaidTeam[],
  edges: GraphEdge[],
  hover: string,
  opts: { minMargin?: number; maxIdle?: number },
) {
  const at = new Map(placed.map((t) => [t.id, t]));
  const minMargin = opts.minMargin ?? 0;
  const maxIdle = opts.maxIdle ?? 90;
  let idle = 0;
  const ranked = [...edges].sort((a, b) => b.margin - a.margin);
  for (const e of ranked) {
    const a = at.get(e.source);
    const b = at.get(e.target);
    if (!a || !b) continue;
    const hot = hover && (e.source === hover || e.target === hover);
    if (!hot) {
      if (e.margin < minMargin) continue;
      idle += 1;
      if (idle > maxIdle) continue;
    }
    const t = Math.min(1, e.margin / 40);
    ctx.beginPath();
    ctx.strokeStyle = hot ? `rgba(17,17,17,${0.45 + t * 0.45})` : `rgba(17,17,17,${0.05 + t * 0.22})`;
    ctx.lineWidth = hot ? 1.4 + t * 2.8 : 0.6 + t * 1.8;
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
}

export function drawPaths(ctx: CanvasRenderingContext2D, net: NetworkStats) {
  paper(ctx);
  ctx.fillStyle = INK;
  ctx.font = "28px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(net.average_path.toFixed(2), 48, 72);
  ctx.font = FONT;
  ctx.fillStyle = MUTED;
  ctx.fillText("average shortest path", 48, 94);

  ctx.fillStyle = INK;
  ctx.font = "28px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText(net.bound_path.toFixed(2), 280, 72);
  ctx.font = FONT;
  ctx.fillStyle = MUTED;
  ctx.fillText("if leftover pairs were distance 2", 280, 94);

  ctx.fillStyle = INK;
  ctx.font = "28px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText(String(net.diameter), 560, 72);
  ctx.font = FONT;
  ctx.fillStyle = MUTED;
  ctx.fillText("diameter", 560, 94);

  const lambda = net.algebraic_connectivity;
  if (lambda != null) {
    ctx.fillStyle = INK;
    ctx.font = "28px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText(lambda.toFixed(2), 720, 72);
    ctx.font = FONT;
    ctx.fillStyle = MUTED;
    ctx.fillText("λ₂  algebraic connectivity", 720, 94);
  }

  const pairs = Object.values(net.distances).reduce((s, n) => s + n, 0);
  const ideal: Record<string, number> = {
    "1": net.undirected_edges,
    "2": Math.max(0, pairs - net.undirected_edges),
  };
  const keys = ["1", "2", "3", "4", "5", "6", "7", "8"].filter((k) => (net.distances[k] || 0) > 0 || (ideal[k] || 0) > 0);
  const max = Math.max(...keys.map((k) => Math.max(net.distances[k] || 0, ideal[k] || 0)), 1);
  const left = 80;
  const top = 160;
  const bottom = HEIGHT - 70;
  const slot = (WIDTH - left - 48) / Math.max(keys.length, 1);

  ctx.strokeStyle = LINE;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.lineTo(WIDTH - 36, bottom);
  ctx.stroke();

  keys.forEach((key, i) => {
    const x = left + i * slot + slot * 0.22;
    const w = slot * 0.28;
    const actual = net.distances[key] || 0;
    const hope = ideal[key] || 0;
    const h1 = ((bottom - top) * actual) / max;
    const h2 = ((bottom - top) * hope) / max;
    ctx.fillStyle = "rgba(17,17,17,0.14)";
    ctx.fillRect(x + w + 4, bottom - h2, w, h2);
    ctx.fillStyle = INK;
    ctx.fillRect(x, bottom - h1, w, h1);
    ctx.fillStyle = MUTED;
    ctx.textAlign = "center";
    ctx.fillText(key, x + w + 2, bottom + 18);
  });

  ctx.textAlign = "left";
  ctx.fillStyle = INK;
  ctx.fillRect(left, HEIGHT - 36, 10, 10);
  ctx.fillStyle = MUTED;
  ctx.fillText("this schedule", left + 16, HEIGHT - 27);
  ctx.fillStyle = "rgba(17,17,17,0.14)";
  ctx.fillRect(left + 130, HEIGHT - 36, 10, 10);
  ctx.fillStyle = MUTED;
  ctx.fillText("diameter-2 bound", left + 146, HEIGHT - 27);

  ctx.textAlign = "right";
  ctx.fillText(`${net.n} teams · ${net.undirected_edges} games · mean degree ${net.degree_mean}`, WIDTH - 36, HEIGHT - 27);
}

export function drawMap(ctx: CanvasRenderingContext2D, placed: LaidTeam[], mst: NetworkStats["mst"], hover: string) {
  paper(ctx);
  const at = new Map(placed.map((t) => [t.id, t]));
  for (const e of mst) {
    const a = at.get(e.source);
    const b = at.get(e.target);
    if (!a || !b) continue;
    const t = Math.min(1, e.margin / 40);
    ctx.beginPath();
    ctx.strokeStyle = `rgba(17,17,17,${0.18 + t * 0.45})`;
    ctx.lineWidth = 1 + t * 2.4;
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  if (hover) {
    for (const e of mst) {
      if (e.source !== hover && e.target !== hover) continue;
      const a = at.get(e.source);
      const b = at.get(e.target);
      if (!a || !b) continue;
      ctx.beginPath();
      ctx.strokeStyle = INK;
      ctx.lineWidth = 2;
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }
  const named = new Set(
    [...placed]
      .sort((a, b) => b.betweenness - a.betweenness)
      .slice(0, 7)
      .map((t) => t.id),
  );
  if (hover) named.add(hover);
  drawDots(ctx, placed, hover, named);
  caption(ctx, "Maximum spanning tree by margin. Size is betweenness.");
}

export function drawEmbedded(
  ctx: CanvasRenderingContext2D,
  placed: LaidTeam[],
  edges: GraphEdge[],
  hover: string,
  note: string,
  opts: { minMargin?: number; maxIdle?: number } = {},
) {
  paper(ctx);
  drawEdges(ctx, placed, edges, hover, opts);
  const named = new Set(
    [...placed]
      .sort((a, b) => b.betweenness - a.betweenness)
      .slice(0, placed.length <= 24 ? placed.length : 8)
      .map((t) => t.id),
  );
  if (hover) named.add(hover);
  drawDots(ctx, placed, hover, named);
  caption(ctx, note);
}

export function drawHive(ctx: CanvasRenderingContext2D, placed: LaidTeam[], edges: GraphEdge[], hover: string) {
  paper(ctx);
  const cx = WIDTH / 2;
  const cy = HEIGHT / 2 - 6;
  const groups = [...new Set(placed.map((t) => t.group).filter(Boolean))] as string[];
  const at = new Map(placed.map((t) => [t.id, t]));

  if (groups.length > 1) {
    for (const g of groups) {
      const members = placed.filter((t) => t.group === g);
      const ray = members[0]?.ray ?? 0;
      const outer = Math.min(WIDTH, HEIGHT) * 0.42;
      ctx.beginPath();
      ctx.strokeStyle = LINE;
      ctx.lineWidth = 1;
      ctx.moveTo(cx + Math.cos(ray) * 56, cy + Math.sin(ray) * 56);
      ctx.lineTo(cx + Math.cos(ray) * (outer + 8), cy + Math.sin(ray) * (outer + 8));
      ctx.stroke();
      ctx.fillStyle = MUTED;
      ctx.font = FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      paintLabel(ctx, g, cx + Math.cos(ray) * (outer + 22), cy + Math.sin(ray) * (outer + 22));
    }
  }

  for (const e of edges) {
    const a = at.get(e.source);
    const b = at.get(e.target);
    if (!a || !b) continue;
    const cross = a.group !== b.group;
    const hot = hover && (e.source === hover || e.target === hover);
    if (!hover && !cross && placed.length > 24) continue;
    if (hover && !hot) continue;
    const t = Math.min(1, e.margin / 40);
    ctx.beginPath();
    ctx.strokeStyle = hot ? `rgba(17,17,17,${0.4 + t * 0.5})` : `rgba(17,17,17,${0.08 + t * 0.28})`;
    ctx.lineWidth = hot ? 1.3 + t * 2.6 : 0.7 + t * 1.6;
    if (a.ray != null && b.ray != null && a.ray !== b.ray) {
      const ra = Math.hypot(a.x - cx, a.y - cy);
      const rb = Math.hypot(b.x - cx, b.y - cy);
      ctx.moveTo(a.x, a.y);
      ctx.bezierCurveTo(
        cx + Math.cos(a.ray) * ra * 0.35,
        cy + Math.sin(a.ray) * ra * 0.35,
        cx + Math.cos(b.ray) * rb * 0.35,
        cy + Math.sin(b.ray) * rb * 0.35,
        b.x,
        b.y,
      );
    } else {
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();
  }

  const named = new Set<string>();
  if (hover) named.add(hover);
  else if (placed.length <= 20) for (const t of placed) named.add(t.id);
  drawDots(ctx, placed, hover, named);
  caption(ctx, "Hive: P4 / G5 / other. Curves are games between groups. Thickness is margin.");
}

export function drawCircuits(
  ctx: CanvasRenderingContext2D,
  placed: LaidTeam[],
  cycles: NetworkStats["cycles"],
  hover: string,
) {
  paper(ctx);
  const at = new Map(placed.map((t) => [t.id, t]));
  const active = cycles.filter((c) => c.teams.every((id) => at.has(id)));
  const inCycle = new Set(active.flatMap((c) => c.teams));

  if (!active.length) {
    ctx.fillStyle = MUTED;
    ctx.font = FONT;
    ctx.textAlign = "center";
    ctx.fillText("No directed 3-cycles in this slice.", WIDTH / 2, HEIGHT / 2);
    return;
  }

  for (const t of placed) {
    if (inCycle.has(t.id)) continue;
    ctx.beginPath();
    ctx.fillStyle = "rgba(17,17,17,0.08)";
    ctx.arc(t.x, t.y, 2.4, 0, Math.PI * 2);
    ctx.fill();
  }

  for (const cycle of active) {
    const on = !hover || cycle.teams.includes(hover);
    for (let k = 0; k < 3; k++) {
      const a = at.get(cycle.teams[k]);
      const b = at.get(cycle.teams[(k + 1) % 3]);
      if (!a || !b) continue;
      arrow(ctx, a.x, a.y, b.x, b.y, on ? INK : "rgba(17,17,17,0.12)", on ? 1.5 : 1);
      if (on) {
        ctx.fillStyle = MUTED;
        ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(cycle.margins[k]), (a.x + b.x) / 2, (a.y + b.y) / 2);
      }
    }
  }

  const named = new Set(hover ? [hover] : [...inCycle].slice(0, 10));
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  for (const t of placed) {
    if (!inCycle.has(t.id)) continue;
    const on = !hover || hover === t.id || active.some((c) => c.teams.includes(hover) && c.teams.includes(t.id));
    ctx.beginPath();
    ctx.fillStyle = on ? confFill(t.conf) : "rgba(17,17,17,0.18)";
    ctx.arc(t.x, t.y, hover === t.id ? t.r + 1.5 : Math.max(4, t.r * 0.85), 0, Math.PI * 2);
    ctx.fill();
    if (!named.has(t.id) || !on) continue;
    ctx.fillStyle = INK;
    ctx.textAlign = t.x > WIDTH / 2 ? "left" : "right";
    ctx.textBaseline = "middle";
    paintLabel(ctx, t.team, t.x + (t.x > WIDTH / 2 ? t.r + 5 : -(t.r + 5)), t.y);
  }
  ctx.textBaseline = "alphabetic";
  caption(ctx, "Directed 3-cycles on the spectral layout. Ranked by rating contradiction.");
}

export function hitMap(placed: LaidTeam[], x: number, y: number): LaidTeam | null {
  let best: LaidTeam | null = null;
  let bestD = Infinity;
  for (const t of placed) {
    const d = Math.hypot(t.x - x, t.y - y);
    if (d < t.r + 8 && d < bestD) {
      best = t;
      bestD = d;
    }
  }
  return best;
}
