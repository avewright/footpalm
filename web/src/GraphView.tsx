import { useEffect, useMemo, useRef, useState } from "react";
import { signed } from "./format";
import type { GraphFile } from "./types";

type Node = GraphFile["nodes"][number];

type Team = {
  id: string;
  team: string;
  conf: string;
  pom: number;
  wins: number;
  losses: number;
  x: number;
  y: number;
};

type Hit = { id: string; x: number; y: number; r: number };

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

function fromRow(n: Node): Team {
  return {
    id: n.id,
    team: n.team,
    conf: n.conf,
    pom: n.pom ?? (n as { palm?: number }).palm ?? 0,
    wins: n.wins,
    losses: n.losses,
    x: 0,
    y: 0,
  };
}

function map(v: number, a: number, b: number, lo: number, hi: number) {
  if (b === a) return (lo + hi) / 2;
  return lo + ((v - a) / (b - a)) * (hi - lo);
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function hex(c: string) {
  return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
}

function mix(a: string, b: string, t: number) {
  const A = hex(a);
  const B = hex(b);
  const r = Math.round(lerp(A[0], B[0], t));
  const g = Math.round(lerp(A[1], B[1], t));
  const bl = Math.round(lerp(A[2], B[2], t));
  return `rgb(${r},${g},${bl})`;
}

function pomColor(pom: number, lo: number, hi: number) {
  const t = clamp((pom - lo) / Math.max(hi - lo, 1), 0, 1);
  return t < 0.5 ? mix(LOSS, INK, t * 2) : mix(INK, WIN, (t - 0.5) * 2);
}

function short(name: string, n = 11) {
  return name.length > n ? `${name.slice(0, n - 1)}…` : name;
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
  const ah = 7 + width;
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

function layoutColumns(teams: Team[], pad: { l: number; r: number; t: number; b: number }) {
  const poms = teams.map((t) => t.pom);
  const lo = Math.min(...poms, -20);
  const hi = Math.max(...poms, 20);
  const groups = [...new Set(teams.map((t) => t.conf))].map((conf) => {
    const members = teams.filter((t) => t.conf === conf);
    const mean = members.reduce((s, t) => s + t.pom, 0) / members.length;
    return { conf, members, mean };
  });
  groups.sort((a, b) => b.mean - a.mean);

  const n = Math.max(groups.length, 1);
  const placed: Team[] = [];
  groups.forEach((g, i) => {
    const x = n === 1 ? WIDTH / 2 : map(i, 0, n - 1, pad.l + 28, WIDTH - pad.r - 28);
    const col = g.members
      .map((t) => ({ ...t, x, y: map(t.pom, hi, lo, pad.t, HEIGHT - pad.b) }))
      .sort((a, b) => a.y - b.y);
    for (let j = 1; j < col.length; j++) {
      if (col[j].y - col[j - 1].y < 13) col[j].y = col[j - 1].y + 13;
    }
    const extra = col.length ? col[col.length - 1].y - (HEIGHT - pad.b) : 0;
    if (extra > 0) {
      const span = Math.max(HEIGHT - pad.b - pad.t, 1);
      for (const t of col) t.y -= extra * ((t.y - pad.t) / span);
    }
    placed.push(...col);
  });

  return {
    placed,
    lo,
    hi,
    pad,
    headers: groups.map((g, i) => ({
      conf: g.conf,
      x: n === 1 ? WIDTH / 2 : map(i, 0, n - 1, pad.l + 28, WIDTH - pad.r - 28),
      mean: g.mean,
    })),
  };
}

export function GraphView({ graph }: { graph: GraphFile }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [focus, setFocus] = useState(() => new URLSearchParams(window.location.search).get("team") ?? "");
  const [hover, setHover] = useState("");
  const [conf, setConf] = useState(() => new URLSearchParams(window.location.search).get("conf") ?? "all");
  const [query, setQuery] = useState("");

  const conferences = useMemo(
    () => ["all", ...[...new Set(graph.nodes.map((n) => n.conf).filter(Boolean))].sort()],
    [graph],
  );

  const allTeams = useMemo(() => graph.nodes.map(fromRow), [graph]);
  const allById = useMemo(() => new Map(allTeams.map((t) => [t.id, t])), [allTeams]);
  const allEdges = useMemo(() => graph.edges.filter((e) => e.fbs_fbs), [graph]);

  const overview = useMemo(() => {
    const pad = { l: 52, r: 16, t: 40, b: 36 };
    const chart = layoutColumns(allTeams, pad);
    const byId = new Map(chart.placed.map((t) => [t.id, t]));
    return { ...chart, byId, edges: allEdges };
  }, [allTeams, allEdges]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return new Set<string>();
    return new Set(overview.placed.filter((t) => t.team.toLowerCase().includes(q)).map((t) => t.id));
  }, [overview, query]);

  const active = hover || (matches.size === 1 ? [...matches][0] : "");

  const ego = useMemo(() => {
    if (!focus) return null;
    const self = allById.get(focus);
    if (!self) return null;
    const wins = allEdges
      .filter((e) => e.source === focus)
      .map((e) => ({ edge: e, team: allById.get(e.target)! }))
      .filter((x) => x.team);
    const losses = allEdges
      .filter((e) => e.target === focus)
      .map((e) => ({ edge: e, team: allById.get(e.source)! }))
      .filter((x) => x.team);
    return {
      self: { ...self, x: WIDTH / 2, y: map(self.pom, 32, -28, 80, HEIGHT - 40) },
      wins: placeByPom(wins, 200, 72, HEIGHT - 36),
      losses: placeByPom(losses, WIDTH - 200, 72, HEIGHT - 36),
    };
  }, [focus, allEdges, allById]);

  const hits = useMemo(() => {
    const list: Hit[] = [];
    if (ego) {
      list.push({ id: ego.self.id, x: ego.self.x, y: ego.self.y, r: 18 });
      for (const row of ego.wins) list.push({ id: row.team.id, x: row.team.x, y: row.team.y, r: 10 });
      for (const row of ego.losses) list.push({ id: row.team.id, x: row.team.x, y: row.team.y, r: 10 });
      return list;
    }
    for (const t of overview.placed) list.push({ id: t.id, x: t.x, y: t.y, r: 7 });
    return list;
  }, [ego, overview]);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    if (conf !== "all") q.set("conf", conf);
    else q.delete("conf");
    if (focus) q.set("team", focus);
    else q.delete("team");
    q.delete("view");
    const next = `?${q}`;
    if (window.location.search !== next) window.history.replaceState(null, "", next);
  }, [conf, focus]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
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
      ctx.fillText(`${ego.self.conf} · ${ego.self.wins}-${ego.self.losses} · Pom ${signed(ego.self.pom)}`, ego.self.x, 50);
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

    const { placed, lo, hi, pad, headers, byId, edges } = overview;
    const linked = new Set<string>();
    if (active) {
      linked.add(active);
      for (const e of edges) {
        if (e.source === active) linked.add(e.target);
        if (e.target === active) linked.add(e.source);
      }
    }

    ctx.fillStyle = MUTED;
    ctx.textAlign = "center";
    ctx.fillText("better ↑", 28, pad.t);
    ctx.fillText("worse ↓", 28, HEIGHT - 14);
    ctx.textAlign = "right";
    ctx.fillText(signed(hi, 0), pad.l - 8, pad.t + 4);
    ctx.fillText(signed(lo, 0), pad.l - 8, HEIGHT - pad.b + 4);

    ctx.strokeStyle = LINE;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, HEIGHT - pad.b);
    ctx.stroke();

    ctx.textAlign = "center";
    for (const h of headers) {
      ctx.globalAlpha = conf === "all" || h.conf === conf ? 1 : 0.28;
      ctx.fillStyle = MUTED;
      ctx.fillText(short(h.conf, 8), h.x, 22);
      ctx.globalAlpha = 1;
    }

    if (active) {
      for (const e of edges) {
        if (e.source !== active && e.target !== active) continue;
        const a = byId.get(e.source);
        const b = byId.get(e.target);
        if (!a || !b) continue;
        arrow(
          ctx,
          a.x,
          a.y,
          b.x,
          b.y,
          5,
          5,
          e.source === active ? WIN : LOSS,
          Math.max(1.2, Math.min(2.6, e.margin / 16)),
        );
      }
    }

    const labeled = new Set<string>();
    if (active) for (const id of linked) labeled.add(id);
    else if (matches.size) for (const id of matches) labeled.add(id);
    else if (conf === "all") {
      for (const h of headers) {
        const best = placed.filter((t) => t.conf === h.conf).sort((a, b) => b.pom - a.pom)[0];
        if (best) labeled.add(best.id);
      }
    } else {
      const top = placed.filter((t) => t.conf === conf).sort((a, b) => b.pom - a.pom).slice(0, 10);
      for (const t of top) labeled.add(t.id);
    }

    for (const t of placed) {
      const inConf = conf === "all" || t.conf === conf;
      const on = (!active || linked.has(t.id)) && inConf;
      const connected = Boolean(active && linked.has(t.id));
      const hot = t.id === active;
      const r = hot ? 6.5 : 4.2;
      ctx.beginPath();
      ctx.globalAlpha = on || connected ? 1 : 0.12;
      ctx.fillStyle = pomColor(t.pom, lo, hi);
      ctx.arc(t.x, t.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    for (const t of placed) {
      if (!labeled.has(t.id)) continue;
      ctx.fillStyle = t.id === active ? INK : MUTED;
      ctx.textAlign = t.x < WIDTH / 2 ? "left" : "right";
      const dx = t.x < WIDTH / 2 ? 8 : -8;
      ctx.fillText(short(t.team, 14), t.x + dx, t.y - 8);
    }
    ctx.font = FONT;
  }, [ego, overview, active, matches, conf]);

  function pick(ev: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const box = canvas.getBoundingClientRect();
    const x = ((ev.clientX - box.left) * canvas.width) / box.width;
    const y = ((ev.clientY - box.top) * canvas.height) / box.height;
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

  const tip = allById.get(hover) ?? allById.get(focus);
  const games = tip
    ? allEdges.filter((e) => e.source === tip.id || e.target === tip.id).length
    : 0;

  return (
    <div>
      <p className="lede-note">
        {ego
          ? "Better teams higher. Green = beat a better team. Red = lost to a worse one. Click empty space to go back."
          : "Columns are conferences, strongest on the left. Higher is better. Color is Pom. Hover to see who they played."}
      </p>
      <div className="toolbar">
        <select value={conf} onChange={(e) => setConf(e.target.value)} aria-label="Conference">
          {conferences.map((c) => (
            <option key={c} value={c}>
              {c === "all" ? "All conferences" : c}
            </option>
          ))}
        </select>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Find a team"
          aria-label="Find a team"
        />
        {focus && (
          <button type="button" onClick={() => setFocus("")}>
            All teams
          </button>
        )}
      </div>
      <canvas
        ref={canvasRef}
        width={WIDTH}
        height={HEIGHT}
        className="graph-canvas"
        onClick={pick}
        onMouseMove={pick}
        onMouseLeave={() => setHover("")}
      />
      {tip && (
        <div className="graph-card">
          <strong>{tip.team}</strong> {tip.conf} · {tip.wins}-{tip.losses} · Pom {signed(tip.pom)}
          {games ? ` · ${games} FBS games` : ""}
          {!ego && <div className="lede-note">Click to see every game.</div>}
        </div>
      )}
    </div>
  );
}
