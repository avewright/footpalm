import { useEffect, useMemo, useRef, useState } from "react";
import { signed } from "./format";
import { drawClusters, hitCluster, layoutClusters } from "./clusters";
import { drawCircuits, drawEmbedded, drawHive, drawMap, drawPaths, placeHive, placeMap } from "./network";
import { buildSlate, drawSlate, placeSlate, slateLine } from "./slate";
import type { GraphFile } from "./types";

type Mode = "clusters" | "paths" | "map" | "spectral" | "weights" | "hive" | "circuits" | "scatter" | "slate" | "upsets" | "matrix";
const NETWORK: Mode[] = ["paths", "map", "spectral", "weights", "hive", "circuits", "clusters"];
const EMBED: Mode[] = ["map", "spectral", "weights", "hive", "circuits"];
type Edge = GraphFile["edges"][number];
type Node = GraphFile["nodes"][number];

type Team = {
  id: string;
  team: string;
  conf: string;
  pom: number;
  wins: number;
  losses: number;
  vsNeighbors: number;
  wr: number;
  winningness: number;
  pagerank: number;
  degree: number;
  betweenness: number;
  eccentricity: number;
  fiedler: number;
  nx: number;
  ny: number;
  tx: number;
  ty: number;
  sx: number;
  sy: number;
  wx: number;
  wy: number;
  x: number;
  y: number;
};

type Hit = { id: string; x: number; y: number; r: number };
type Tip = { title: string; line: string };

const WIDTH = 960;
const HEIGHT = 700;
const PAPER = "#ffffff";
const INK = "#111111";
const MUTED = "#666666";
const LINE = "#e5e5e5";
const WIN = "#1a7f37";
const LOSS = "#b42318";
const FONT = "12px ui-sans-serif, system-ui, sans-serif";
const UPSET_GAP = 3;

function winRate(wins: number, losses: number) {
  const games = wins + losses;
  return games ? wins / games : 0;
}

function fromRow(n: Node): Team {
  const wr = winRate(n.wins, n.losses);
  return {
    id: n.id,
    team: n.team,
    conf: n.conf,
    pom: n.pom ?? (n as { palm?: number }).palm ?? 0,
    wins: n.wins,
    losses: n.losses,
    vsNeighbors: n.vs_neighbors ?? 0,
    wr,
    winningness: n.winningness ?? wr,
    pagerank: n.pagerank ?? 0,
    degree: n.degree ?? 0,
    betweenness: n.betweenness ?? 0,
    eccentricity: n.eccentricity ?? 0,
    fiedler: n.fiedler ?? 0,
    nx: n.nx ?? 0,
    ny: n.ny ?? 0,
    tx: n.tx ?? n.nx ?? 0,
    ty: n.ty ?? n.ny ?? 0,
    sx: n.sx ?? n.nx ?? 0,
    sy: n.sy ?? n.ny ?? 0,
    wx: n.wx ?? n.nx ?? 0,
    wy: n.wy ?? n.ny ?? 0,
    x: 0,
    y: 0,
  };
}

function short(name: string, n = 10) {
  return name.length > n ? `${name.slice(0, n - 1)}…` : name;
}

function map(v: number, a: number, b: number, lo: number, hi: number) {
  if (b === a) return (lo + hi) / 2;
  return lo + ((v - a) / (b - a)) * (hi - lo);
}

function isUpset(winner: Team, loser: Team) {
  return loser.pom - winner.pom >= UPSET_GAP;
}

function residual(t: Team) {
  return t.wr - (t.pom + 30) / 60;
}

function pickScatterLabels(placed: Team[], max = 7): Team[] {
  const minAbs = placed.length < 20 ? 0.12 : 0.28;
  const minDist = placed.length < 20 ? 36 : 58;
  const byResid = [...placed].sort((a, b) => Math.abs(residual(b)) - Math.abs(residual(a)));
  const picked: Team[] = [];
  const tryAdd = (t: Team | undefined) => {
    if (!t || picked.some((p) => p.id === t.id)) return;
    if (picked.some((p) => Math.hypot(p.x - t.x, p.y - t.y) < minDist)) return;
    picked.push(t);
  };
  tryAdd([...placed].sort((a, b) => b.pom - a.pom)[0]);
  tryAdd(byResid.find((t) => residual(t) < -minAbs));
  tryAdd(byResid.find((t) => residual(t) > minAbs));
  for (const t of byResid) {
    if (picked.length >= max) break;
    if (Math.abs(residual(t)) < minAbs) continue;
    tryAdd(t);
  }
  return picked;
}

function boxesOverlap(a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

function drawScatterLabel(
  ctx: CanvasRenderingContext2D,
  t: Team,
  occupied: { x: number; y: number; w: number; h: number }[],
) {
  const w = ctx.measureText(t.team).width;
  const h = 14;
  const towardOpen = t.x < WIDTH / 2 ? 1 : -1;
  const spots = [
    { x: t.x + 8 * towardOpen, y: t.y - 7, align: towardOpen > 0 ? "left" : "right" },
    { x: t.x - 8 * towardOpen, y: t.y - 7, align: towardOpen > 0 ? "right" : "left" },
    { x: t.x + 8 * towardOpen, y: t.y + 15, align: towardOpen > 0 ? "left" : "right" },
    { x: t.x - 8 * towardOpen, y: t.y + 15, align: towardOpen > 0 ? "right" : "left" },
    { x: t.x + 8 * towardOpen, y: t.y - 22, align: towardOpen > 0 ? "left" : "right" },
    { x: t.x - 8 * towardOpen, y: t.y + 28, align: towardOpen > 0 ? "right" : "left" },
  ] as const;
  for (const spot of spots) {
    const left = spot.align === "left" ? spot.x : spot.x - w;
    const box = { x: left - 2, y: spot.y - h + 2, w: w + 4, h };
    if (box.x < 50 || box.x + box.w > WIDTH - 6 || box.y < 6 || box.y + box.h > HEIGHT - 28) continue;
    if (occupied.some((o) => boxesOverlap(box, o))) continue;
    occupied.push(box);
    ctx.textAlign = spot.align;
    ctx.fillText(t.team, spot.x, spot.y);
    return;
  }
}

function placeByPom<T extends { team: Team }>(items: T[], x: number, top: number, bottom: number) {
  const poms = items.map((it) => it.team.pom);
  const lo = Math.min(...poms, 0);
  const hi = Math.max(...poms, 1);
  const placed = items
    .map((it) => ({ ...it, y: map(it.team.pom, hi, lo, top, bottom) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < placed.length; i++) {
    if (placed[i].y - placed[i - 1].y < 16) placed[i].y = placed[i - 1].y + 16;
  }
  const extra = placed.length ? placed[placed.length - 1].y - bottom : 0;
  if (extra > 0) for (const p of placed) p.y -= extra * ((p.y - top) / Math.max(bottom - top, 1));
  return placed.map((p) => ({ ...p, team: { ...p.team, x, y: p.y } }));
}

function arrow(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  r1: number,
  r2: number,
  color: string,
  width: number,
) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const dist = Math.hypot(dx, dy) || 1;
  const ux = dx / dist;
  const uy = dy / dist;
  const sx = x1 + ux * (r1 + 2);
  const sy = y1 + uy * (r1 + 2);
  const ex = x2 - ux * (r2 + 5);
  const ey = y2 - uy * (r2 + 5);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  const ah = 8 + width;
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - ux * ah + uy * ah * 0.42, ey - uy * ah - ux * ah * 0.42);
  ctx.lineTo(ex - ux * ah - uy * ah * 0.42, ey - uy * ah + ux * ah * 0.42);
  ctx.closePath();
  ctx.fill();
}

function drawPaper(ctx: CanvasRenderingContext2D) {
  ctx.clearRect(0, 0, WIDTH, HEIGHT);
  ctx.fillStyle = PAPER;
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
}

function axis(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number) {
  ctx.strokeStyle = LINE;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

export function GraphView({ graph }: { graph: GraphFile }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<Mode>(() => {
    const q = new URLSearchParams(window.location.search).get("view");
    return q === "upsets" || q === "matrix" || q === "scatter" || q === "slate" || q === "clusters" || q === "planets" || q === "paths" || q === "map" || q === "spectral" || q === "weights" || q === "hive" || q === "circuits"
      ? q === "planets"
        ? "clusters"
        : q
      : "scatter";
  });
  const [focus, setFocus] = useState(() => new URLSearchParams(window.location.search).get("team") ?? "");
  const [hover, setHover] = useState("");
  const [conf, setConf] = useState(() => new URLSearchParams(window.location.search).get("conf") ?? "all");
  const [cellTip, setCellTip] = useState<Tip | null>(null);

  const conferences = useMemo(
    () => ["all", ...[...new Set(graph.nodes.map((n) => n.conf).filter(Boolean))].sort()],
    [graph],
  );

  const { teams, edges, byId } = useMemo(() => {
    const filtered = graph.nodes.filter((n) => conf === "all" || n.conf === conf).map(fromRow);
    const idSet = new Set(filtered.map((t) => t.id));
    const edges = graph.edges.filter((e) => e.fbs_fbs && idSet.has(e.source) && idSet.has(e.target));
    return { teams: filtered, edges, byId: new Map(filtered.map((t) => [t.id, t])) };
  }, [graph, conf]);

  const clusterChart = useMemo(
    () =>
      layoutClusters(
        teams.map((t) => ({ id: t.id, team: t.team, conf: t.conf, wr: t.winningness })),
        edges,
        WIDTH,
        HEIGHT,
      ),
    [teams, edges],
  );

  const slateAll = useMemo(() => {
    const allTeams = graph.nodes.map(fromRow);
    const allEdges = graph.edges.filter((e) => e.fbs_fbs);
    return buildSlate(allTeams, allEdges);
  }, [graph]);
  const slateChart = useMemo(() => {
    const rows = conf === "all" ? slateAll : slateAll.filter((r) => r.team.conf === conf);
    return placeSlate(rows, conf);
  }, [slateAll, conf]);
  const slateById = useMemo(() => new Map(slateAll.map((r) => [r.team.id, r])), [slateAll]);

  const net = graph.network;
  const laid = useMemo(() => {
    const seed = teams.map((t) => ({
      id: t.id,
      team: t.team,
      conf: t.conf,
      nx: 0,
      ny: 0,
      betweenness: t.betweenness,
      degree: t.degree,
      eccentricity: t.eccentricity,
    }));
    return {
      map: placeMap(seed.map((t, i) => ({ ...t, nx: teams[i].tx, ny: teams[i].ty }))),
      spectral: placeMap(seed.map((t, i) => ({ ...t, nx: teams[i].sx, ny: teams[i].sy }))),
      weights: placeMap(seed.map((t, i) => ({ ...t, nx: teams[i].wx, ny: teams[i].wy }))),
      hive: placeHive(seed),
    };
  }, [teams]);

  const scatter = useMemo(() => {
    const pad = { l: 58, r: 20, t: 28, b: 48 };
    const poms = teams.map((t) => t.pom);
    const lo = Math.min(...poms, -20);
    const hi = Math.max(...poms, 20);
    const placed = teams.map((t) => ({
      ...t,
      x: map(t.wr, 0, 1, pad.l, WIDTH - pad.r),
      y: map(t.pom, hi, lo, pad.t, HEIGHT - pad.b),
    }));
    const labels = new Set(pickScatterLabels(placed).map((t) => t.id));
    return { placed, labels, lo, hi, pad };
  }, [teams]);

  const upsetChart = useMemo(() => {
    const upsets = edges
      .map((e) => {
        const w = byId.get(e.source);
        const l = byId.get(e.target);
        if (!w || !l || !isUpset(w, l)) return null;
        return { e, w, l, gap: l.pom - w.pom };
      })
      .filter((x): x is { e: Edge; w: Team; l: Team; gap: number } => Boolean(x))
      .sort((a, b) => b.gap - a.gap);
    const ids = [...new Set(upsets.flatMap((u) => [u.w.id, u.l.id]))];
    const ranked = ids
      .map((id) => byId.get(id)!)
      .filter(Boolean)
      .sort((a, b) => a.pom - b.pom);
    const pad = 28;
    const y = HEIGHT - 88;
    const placed = ranked.map((t, i) => ({
      ...t,
      x: map(i, 0, Math.max(ranked.length - 1, 1), pad, WIDTH - pad),
      y,
    }));
    const at = new Map(placed.map((t) => [t.id, t]));
    return { upsets, placed, at, y };
  }, [edges, byId]);

  const matrix = useMemo(() => {
    if (conf !== "all") {
      const ranked = [...teams].sort((a, b) => b.pom - a.pom);
      const n = ranked.length;
      const left = 92;
      const top = 92;
      const size = Math.min(36, (WIDTH - left - 16) / Math.max(n, 1), (HEIGHT - top - 16) / Math.max(n, 1));
      const cells: { r: number; c: number; margin: number; upset: boolean; a: Team; b: Team }[] = [];
      for (const e of edges) {
        const a = ranked.findIndex((t) => t.id === e.source);
        const b = ranked.findIndex((t) => t.id === e.target);
        if (a < 0 || b < 0) continue;
        const winner = ranked[a];
        const loser = ranked[b];
        cells.push({ r: a, c: b, margin: e.margin, upset: isUpset(winner, loser), a: winner, b: loser });
      }
      return { kind: "teams" as const, ranked, cells, left, top, size };
    }
    const confs = [...new Set(teams.map((t) => t.conf))].sort();
    const n = confs.length;
    const left = 88;
    const top = 72;
    const size = Math.min(52, (WIDTH - left - 24) / Math.max(n, 1), (HEIGHT - top - 24) / Math.max(n, 1));
    const wins = confs.map(() => confs.map(() => ({ n: 0, margin: 0 })));
    for (const e of edges) {
      const w = byId.get(e.source);
      const l = byId.get(e.target);
      if (!w || !l) continue;
      const i = confs.indexOf(w.conf);
      const j = confs.indexOf(l.conf);
      if (i < 0 || j < 0) continue;
      wins[i][j].n += 1;
      wins[i][j].margin += e.margin;
    }
    return { kind: "confs" as const, confs, wins, left, top, size };
  }, [conf, teams, edges, byId]);

  const ego = useMemo(() => {
    if (!focus) return null;
    const self = byId.get(focus);
    if (!self) return null;
    const wins = edges
      .filter((e) => e.source === focus)
      .map((e) => ({ edge: e, team: byId.get(e.target)! }))
      .filter((x) => x.team);
    const losses = edges
      .filter((e) => e.target === focus)
      .map((e) => ({ edge: e, team: byId.get(e.source)! }))
      .filter((x) => x.team);
    const selfNode = { ...self, x: WIDTH / 2, y: map(self.pom, 32, -28, 80, HEIGHT - 40) };
    return {
      self: selfNode,
      wins: placeByPom(wins, 200, 72, HEIGHT - 36),
      losses: placeByPom(losses, WIDTH - 200, 72, HEIGHT - 36),
    };
  }, [focus, edges, byId]);

  const sky = mode === "clusters" && !ego;

  const hits = useMemo(() => {
    const list: Hit[] = [];
    if (ego) {
      list.push({ id: ego.self.id, x: ego.self.x, y: ego.self.y, r: 18 });
      for (const row of ego.wins) list.push({ id: row.team.id, x: row.team.x, y: row.team.y, r: 10 });
      for (const row of ego.losses) list.push({ id: row.team.id, x: row.team.x, y: row.team.y, r: 10 });
      return list;
    }
    if (mode === "clusters") {
      for (const b of clusterChart.bodies) list.push({ id: b.id, x: b.x, y: b.y, r: b.r });
    } else if (mode === "map") {
      for (const t of laid.map) list.push({ id: t.id, x: t.x, y: t.y, r: t.r });
    } else if (mode === "spectral" || mode === "circuits") {
      for (const t of laid.spectral) list.push({ id: t.id, x: t.x, y: t.y, r: t.r });
    } else if (mode === "weights") {
      for (const t of laid.weights) list.push({ id: t.id, x: t.x, y: t.y, r: t.r });
    } else if (mode === "hive") {
      for (const t of laid.hive) list.push({ id: t.id, x: t.x, y: t.y, r: t.r });
    } else if (mode === "scatter") {
      for (const t of scatter.placed) list.push({ id: t.id, x: t.x, y: t.y, r: 6 });
    } else if (mode === "slate") {
      for (const r of slateChart.rows) {
        list.push({ id: r.team.id, x: r.x, y: r.y, r: slateChart.kind === "rows" ? 16 : 7 });
      }
    } else if (mode === "upsets") {
      for (const t of upsetChart.placed) list.push({ id: t.id, x: t.x, y: t.y, r: 7 });
    } else if (matrix.kind === "teams") {
      for (let i = 0; i < matrix.ranked.length; i++) {
        const t = matrix.ranked[i];
        list.push({ id: t.id, x: 46, y: matrix.top + i * matrix.size + matrix.size / 2, r: 10 });
      }
    }
    return list;
  }, [ego, mode, scatter, slateChart, upsetChart, matrix, clusterChart, laid]);

  const prevFilter = useRef({ conf, mode });
  useEffect(() => {
    if (prevFilter.current.conf === conf && prevFilter.current.mode === mode) return;
    prevFilter.current = { conf, mode };
    setFocus("");
    setHover("");
    setCellTip(null);
  }, [conf, mode]);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    q.set("view", mode);
    if (conf !== "all") q.set("conf", conf);
    else q.delete("conf");
    if (focus) q.set("team", focus);
    else q.delete("team");
    const next = `?${q}`;
    if (window.location.search !== next) window.history.replaceState(null, "", next);
  }, [mode, conf, focus]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (!ego && mode === "paths" && net) {
      drawPaths(ctx, net);
      return;
    }
    if (!ego && mode === "map" && net) {
      drawMap(ctx, laid.map, net.mst, hover);
      return;
    }
    if (!ego && mode === "spectral") {
      drawEmbedded(
        ctx,
        laid.spectral,
        edges,
        hover,
        "Laplacian spectral layout. The Fiedler cut is the long axis. Thickness is margin.",
        { maxIdle: 70 },
      );
      return;
    }
    if (!ego && mode === "weights") {
      drawEmbedded(
        ctx,
        laid.weights,
        edges,
        hover,
        "NetworkX spring with margin as spring weight. Blowouts sit close. Thickness is margin.",
        { minMargin: teams.length > 40 ? 24 : 0, maxIdle: 48 },
      );
      return;
    }
    if (!ego && mode === "hive") {
      drawHive(ctx, laid.hive, edges, hover);
      return;
    }
    if (!ego && mode === "circuits" && net) {
      drawCircuits(ctx, laid.spectral, net.cycles, hover);
      return;
    }
    if (sky) {
      drawClusters(ctx, clusterChart.bodies, clusterChart.links, hover, WIDTH, HEIGHT);
      return;
    }
    if (!ego && mode === "slate") {
      drawSlate(ctx, slateChart, hover);
      return;
    }

    drawPaper(ctx);
    ctx.font = FONT;

    if (ego) {
      ctx.textAlign = "center";
      ctx.fillStyle = WIN;
      ctx.fillText(`Won ${ego.wins.length}`, 200, 32);
      ctx.fillStyle = LOSS;
      ctx.fillText(`Lost ${ego.losses.length}`, WIDTH - 200, 32);
      ctx.fillStyle = INK;
      ctx.font = "16px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(ego.self.team, ego.self.x, 32);
      ctx.font = FONT;
      ctx.fillStyle = MUTED;
      ctx.fillText(
        `${ego.self.conf} · ${ego.self.wins}-${ego.self.losses} · Pom ${signed(ego.self.pom)}${slateById.get(ego.self.id) ? ` · ${slateLine(slateById.get(ego.self.id)!)}` : ""}`,
        ego.self.x,
        50,
      );
      ctx.fillText("better ↑", 28, 80);
      ctx.fillText("worse ↓", 28, HEIGHT - 24);

      for (const row of ego.wins) {
        const surprise = row.team.pom > ego.self.pom + UPSET_GAP;
        arrow(ctx, ego.self.x, ego.self.y, row.team.x, row.team.y, 16, 7, surprise ? WIN : "rgba(26,127,55,0.35)", surprise ? 2.4 : 1.2);
      }
      for (const row of ego.losses) {
        const surprise = row.team.pom < ego.self.pom - UPSET_GAP;
        arrow(ctx, row.team.x, row.team.y, ego.self.x, ego.self.y, 7, 16, surprise ? LOSS : "rgba(180,35,24,0.35)", surprise ? 2.4 : 1.2);
      }
      const dot = (x: number, y: number, r: number, color: string) => {
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      };
      dot(ego.self.x, ego.self.y, 16, INK);
      ctx.textBaseline = "middle";
      for (const row of ego.wins) {
        const surprise = row.team.pom > ego.self.pom + UPSET_GAP;
        dot(row.team.x, row.team.y, 6, surprise ? WIN : INK);
        ctx.fillStyle = surprise ? WIN : MUTED;
        ctx.textAlign = "right";
        ctx.fillText(`${row.team.team}  ${signed(row.team.pom, 0)}  ${row.edge.margin.toFixed(0)}`, row.team.x - 12, row.team.y);
      }
      for (const row of ego.losses) {
        const surprise = row.team.pom < ego.self.pom - UPSET_GAP;
        dot(row.team.x, row.team.y, 6, surprise ? LOSS : INK);
        ctx.fillStyle = surprise ? LOSS : MUTED;
        ctx.textAlign = "left";
        ctx.fillText(`${row.edge.margin.toFixed(0)}  ${signed(row.team.pom, 0)}  ${row.team.team}`, row.team.x + 12, row.team.y);
      }
      ctx.textBaseline = "alphabetic";
      return;
    }

    if (mode === "scatter") {
      const { placed, labels, lo, hi, pad } = scatter;
      axis(ctx, pad.l, pad.t, pad.l, HEIGHT - pad.b);
      axis(ctx, pad.l, HEIGHT - pad.b, WIDTH - pad.r, HEIGHT - pad.b);
      ctx.fillStyle = MUTED;
      ctx.textAlign = "center";
      ctx.fillText("Win rate →", WIDTH / 2, HEIGHT - 14);
      ctx.save();
      ctx.translate(16, HEIGHT / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Pom (quality)", 0, 0);
      ctx.restore();
      ctx.textAlign = "right";
      ctx.fillText(signed(hi, 0), pad.l - 8, pad.t + 4);
      ctx.fillText(signed(lo, 0), pad.l - 8, HEIGHT - pad.b + 4);
      ctx.textAlign = "center";
      ctx.fillText("0%", pad.l, HEIGHT - pad.b + 16);
      ctx.fillText("100%", WIDTH - pad.r, HEIGHT - pad.b + 16);
      ctx.strokeStyle = "rgba(102,102,102,0.35)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(pad.l, HEIGHT - pad.b);
      ctx.lineTo(WIDTH - pad.r, pad.t);
      ctx.stroke();
      ctx.setLineDash([]);
      for (const t of placed) {
        const active = hover === t.id;
        ctx.beginPath();
        ctx.fillStyle = t.vsNeighbors > 2 ? WIN : t.vsNeighbors < -2 ? LOSS : INK;
        ctx.globalAlpha = hover && !active ? 0.22 : 1;
        ctx.arc(t.x, t.y, active ? 7 : 4.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = INK;
      const occupied: { x: number; y: number; w: number; h: number }[] = [];
      const named = placed.filter((t) => labels.has(t.id) || hover === t.id);
      named.sort((a, b) => a.y - b.y || a.x - b.x);
      for (const t of named) drawScatterLabel(ctx, t, occupied);
      return;
    }

    if (mode === "upsets") {
      const { upsets, placed, at, y } = upsetChart;
      ctx.fillStyle = MUTED;
      ctx.textAlign = "left";
      ctx.fillText(`${upsets.length} games where the worse team (by Pom) won`, 24, 28);
      ctx.textAlign = "left";
      ctx.fillText("worse", 24, y + 48);
      ctx.textAlign = "right";
      ctx.fillText("better", WIDTH - 24, y + 48);
      for (const u of upsets) {
        const a = at.get(u.w.id);
        const b = at.get(u.l.id);
        if (!a || !b) continue;
        const on = !hover || hover === u.w.id || hover === u.l.id;
        const mid = (a.x + b.x) / 2;
        const h = 28 + Math.min(160, u.gap * 5);
        ctx.beginPath();
        ctx.strokeStyle = on ? "rgba(180,35,24,0.7)" : "rgba(180,35,24,0.08)";
        ctx.lineWidth = on ? Math.max(1.1, Math.min(3.2, u.e.margin / 14)) : 1;
        ctx.moveTo(a.x, y);
        ctx.quadraticCurveTo(mid, y - h, b.x, y);
        ctx.stroke();
      }
      for (const t of placed) {
        const on = hover === t.id;
        ctx.beginPath();
        ctx.fillStyle = on ? LOSS : INK;
        ctx.arc(t.x, y, on ? 4.5 : 2.4, 0, Math.PI * 2);
        ctx.fill();
      }
      if (hover) {
        const t = at.get(hover);
        if (t) {
          ctx.fillStyle = INK;
          ctx.textAlign = "center";
          ctx.fillText(`${t.team}  ${t.wins}-${t.losses}  ${signed(t.pom)}`, t.x, y + 28);
        }
      } else if (placed[0] && placed[placed.length - 1]) {
        ctx.fillStyle = MUTED;
        ctx.textAlign = "left";
        ctx.fillText(placed[0].team, placed[0].x, y + 28);
        ctx.textAlign = "right";
        ctx.fillText(placed[placed.length - 1].team, placed[placed.length - 1].x, y + 28);
      }
      return;
    }

    if (matrix.kind === "confs") {
      ctx.fillStyle = MUTED;
      ctx.textAlign = "left";
      ctx.fillText("Row beat column. Darker = more wins.", 24, 28);
      const { confs, wins, left, top, size } = matrix;
      let maxN = 1;
      for (const row of wins) for (const c of row) maxN = Math.max(maxN, c.n);
      for (let i = 0; i < confs.length; i++) {
        ctx.fillStyle = MUTED;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(confs[i], left - 8, top + i * size + size / 2);
        ctx.save();
        ctx.translate(left + i * size + size / 2, top - 8);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
        ctx.fillText(confs[i], 0, 0);
        ctx.restore();
        for (let j = 0; j < confs.length; j++) {
          const cell = wins[i][j];
          const x = left + j * size;
          const y = top + i * size;
          const t = cell.n / maxN;
          ctx.fillStyle = i === j ? "rgba(17,17,17,0.08)" : `rgba(180,35,24,${0.08 + t * 0.72})`;
          ctx.fillRect(x + 1, y + 1, size - 2, size - 2);
          if (cell.n && (hover === `${confs[i]}>${confs[j]}` || size > 36)) {
            ctx.fillStyle = t > 0.55 ? PAPER : INK;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
            ctx.fillText(String(cell.n), x + size / 2, y + size / 2);
            ctx.font = FONT;
          }
        }
      }
      ctx.textBaseline = "alphabetic";
      return;
    }

    const { ranked, cells, left, top, size } = matrix;
    ctx.fillStyle = MUTED;
    ctx.textAlign = "left";
    ctx.fillText("Ordered by Pom. Oxblood mark = worse team won.", 24, 28);
    for (let i = 0; i < ranked.length; i++) {
      ctx.fillStyle = hover === ranked[i].id ? INK : MUTED;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(short(ranked[i].team, 11), left - 6, top + i * size + size / 2);
      ctx.save();
      ctx.translate(left + i * size + size / 2, top - 6);
      ctx.rotate(-Math.PI / 3);
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(short(ranked[i].team, 10), 0, 0);
      ctx.restore();
    }
    for (const cell of cells) {
      const x = left + cell.c * size;
      const y = top + cell.r * size;
      const t = Math.min(1, cell.margin / 35);
      ctx.fillStyle = cell.upset ? `rgba(180,35,24,${0.35 + t * 0.5})` : `rgba(26,127,55,${0.2 + t * 0.45})`;
      ctx.fillRect(x + 0.6, y + 0.6, size - 1.2, size - 1.2);
    }
    ctx.textBaseline = "alphabetic";
  }, [ego, mode, scatter, slateChart, upsetChart, matrix, hover, clusterChart, sky, net, laid, edges, teams.length]);

  function pick(ev: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const box = canvas.getBoundingClientRect();
    const x = ((ev.clientX - box.left) * canvas.width) / box.width;
    const y = ((ev.clientY - box.top) * canvas.height) / box.height;

    if (!ego && mode === "matrix" && matrix.kind === "confs") {
      const { confs, wins, left, top, size } = matrix;
      const c = Math.floor((x - left) / size);
      const r = Math.floor((y - top) / size);
      if (r >= 0 && c >= 0 && r < confs.length && c < confs.length) {
        const cell = wins[r][c];
        const id = `${confs[r]}>${confs[c]}`;
        setHover(id);
        setCellTip({
          title: `${confs[r]} over ${confs[c]}`,
          line: cell.n ? `${cell.n} wins · mean margin ${(cell.margin / cell.n).toFixed(0)}` : "no games",
        });
        if (ev.type === "click") setFocus("");
        return;
      }
    }

    if (!ego && mode === "matrix" && matrix.kind === "teams") {
      const { cells, left, top, size } = matrix;
      const c = Math.floor((x - left) / size);
      const r = Math.floor((y - top) / size);
      const hit = cells.find((cell) => cell.r === r && cell.c === c);
      if (hit) {
        setHover(hit.a.id);
        setCellTip({
          title: `${hit.a.team} beat ${hit.b.team}`,
          line: `margin ${hit.margin.toFixed(0)}${hit.upset ? " · upset" : ""}`,
        });
        if (ev.type === "click") setFocus(hit.a.id);
        return;
      }
    }

    if (!ego && mode === "slate" && slateChart.kind === "rows") {
      let best = "";
      let bestD = 11;
      for (const r of slateChart.rows) {
        const d = Math.abs(r.y - y);
        if (d < bestD) {
          best = r.team.id;
          bestD = d;
        }
      }
      if (ev.type === "click") setFocus(best);
      setHover(best);
      setCellTip(null);
      return;
    }

    if (sky) {
      const body = hitCluster(clusterChart.bodies, x, y);
      if (ev.type === "click") setFocus(body?.kind === "team" ? body.id : "");
      setHover(body?.id ?? "");
      return;
    }

    setCellTip(null);
    let best: Hit | null = null;
    let bestD = Infinity;
    for (const n of hits) {
      const d = Math.hypot(n.x - x, n.y - y);
      if (d < n.r + 8 && d < bestD) {
        best = n;
        bestD = d;
      }
    }
    if (ev.type === "click") setFocus(best ? (ego && best.id === focus ? "" : best.id) : "");
    setHover(best?.id ?? "");
  }

  const tipCluster = clusterChart.bodies.find((b) => b.kind === "cluster" && b.id === hover);
  const tipTeam = byId.get(hover) ?? byId.get(focus);
  const lede =
    ego
      ? "Better teams higher. Green = beat a better team. Red = lost to a worse one. Click empty space to go back."
      : mode === "paths"
        ? "How far apart two teams are on the schedule graph. The pale bars are the diameter-2 floor."
        : mode === "map"
          ? "NetworkX maximum spanning tree by margin. Size is betweenness."
          : mode === "spectral"
            ? "Laplacian embedding. Left–right is the Fiedler cut — the weakest place to split the schedule."
            : mode === "weights"
              ? "Margin-weighted spring. A 50-point win pulls harder than a field goal."
              : mode === "hive"
                ? "Three axes: Power 4, Group of 5, everyone else. Curves are cross-group games."
          : mode === "circuits"
            ? "Directed 3-cycles on the spectral map: A beat B beat C beat A."
            : mode === "clusters"
        ? "Who played whom. Color is conference."
        : mode === "scatter"
          ? "Win rate vs Pom. Named points are the mismatches."
          : mode === "slate"
            ? "How good they were vs who they played. The bar is the opponent-quality span. Hover to see each game."
          : mode === "upsets"
            ? "Worse team won. Arc height is the Pom gap."
            : conf === "all"
              ? "Row beat column."
              : "Teams by Pom. Green = expected, red = upset.";

  return (
    <div>
      <p className="lede-note">{lede}</p>
      <div className="toolbar">
        <div className="seg">
          {(["scatter", "slate", "upsets", "matrix"] as Mode[]).map((id) => (
            <button key={id} type="button" aria-pressed={mode === id} onClick={() => setMode(id)}>
              {id}
            </button>
          ))}
          <button type="button" aria-pressed={NETWORK.includes(mode)} onClick={() => setMode("paths")}>
            network
          </button>
        </div>
        {NETWORK.includes(mode) && (
          <div className="seg">
            {NETWORK.map((id) => (
              <button key={id} type="button" aria-pressed={mode === id} onClick={() => setMode(id)}>
                {id}
              </button>
            ))}
          </div>
        )}
        <select value={conf} onChange={(e) => setConf(e.target.value)} aria-label="Conference">
          {conferences.map((c) => (
            <option key={c} value={c}>
              {c === "all" ? "All conferences" : c}
            </option>
          ))}
        </select>
      </div>
      <canvas
        ref={canvasRef}
        width={WIDTH}
        height={HEIGHT}
        className="graph-canvas"
        onClick={pick}
        onMouseMove={pick}
        onMouseLeave={() => {
          setHover("");
          setCellTip(null);
        }}
      />
      {cellTip && !ego && (
        <div className="graph-card">
          <strong>{cellTip.title}</strong> · {cellTip.line}
        </div>
      )}
      {tipCluster && !tipTeam && (
        <div className="graph-card">
          <strong>{tipCluster.label}</strong> · {tipCluster.members} teams
        </div>
      )}
      {tipTeam && !cellTip && (
        <div className="graph-card">
          <strong>{tipTeam.team}</strong> {tipTeam.conf} · {tipTeam.wins}-{tipTeam.losses} · Pom{" "}
          {signed(tipTeam.pom)}
          {mode === "slate" && slateById.get(tipTeam.id)
            ? ` · ${slateLine(slateById.get(tipTeam.id)!)}`
            : EMBED.includes(mode)
            ? ` · degree ${tipTeam.degree} · betweenness ${tipTeam.betweenness.toFixed(3)} · ecc ${tipTeam.eccentricity}`
            : sky
              ? ` · ${(tipTeam.winningness * 100).toFixed(0)}% wins`
              : ` · ${signed(tipTeam.vsNeighbors)} vs neighbors`}
          {!ego && <div className="lede-note">Click for their games.</div>}
        </div>
      )}
    </div>
  );
}
