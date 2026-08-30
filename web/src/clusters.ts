export type ClusterTeam = {
  id: string;
  team: string;
  conf: string;
  wr: number;
};

export type ClusterBody = ClusterTeam & {
  x: number;
  y: number;
  r: number;
  kind: "team" | "cluster";
  cluster: number;
  members?: number;
  label?: string;
};

export type ClusterLink = {
  si: number;
  ti: number;
  margin: number;
  same: boolean;
};

export type GameEdge = {
  source: string;
  target: string;
  margin: number;
};

const TAU = Math.PI * 2;
const PAPER = "#ffffff";
const INK = "#111111";
const LINE = "#e5e5e5";

const CONF_HUE: Record<string, number> = {
  SEC: 8,
  B1G: 210,
  B12: 32,
  ACC: 150,
  AAC: 188,
  MWC: 265,
  MAC: 48,
  SBC: 172,
  CUSA: 312,
  Ind: 0,
};

function hash(s: string) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 33 + s.charCodeAt(i)) >>> 0;
  return h;
}

function hueFor(conf: string) {
  return CONF_HUE[conf] ?? hash(conf) % 360;
}

export function confFill(conf: string) {
  return `hsl(${hueFor(conf)} 32% 40%)`;
}

function ringRadius(radii: number[]) {
  const n = radii.length;
  if (n <= 1) return 0;
  const step = Math.sin(Math.PI / n);
  let need = 0;
  for (let i = 0; i < n; i++) need = Math.max(need, (radii[i] + radii[(i + 1) % n]) / (2 * step));
  return need;
}

function placeRing<T extends { x: number; y: number }>(items: T[], cx: number, cy: number, ring: number) {
  const n = Math.max(items.length, 1);
  items.forEach((item, i) => {
    const angle = (i / n) * TAU - Math.PI / 2;
    item.x = cx + Math.cos(angle) * ring;
    item.y = cy + Math.sin(angle) * ring;
  });
}

type Adj = Map<number, number>[];

function graphFromEdges(n: number, edges: { source: string; target: string }[], at: Map<string, number>): { adj: Adj; m: number } {
  const adj: Adj = Array.from({ length: n }, () => new Map());
  let m = 0;
  for (const e of edges) {
    const i = at.get(e.source);
    const j = at.get(e.target);
    if (i == null || j == null || i === j || adj[i].has(j)) continue;
    adj[i].set(j, 1);
    adj[j].set(i, 1);
    m += 1;
  }
  return { adj, m };
}

function degOf(adj: Adj, i: number) {
  let k = 0;
  for (const [j, w] of adj[i]) k += i === j ? 2 * w : w;
  return k;
}

function kin(adj: Adj, i: number, comm: Int32Array, target: number) {
  let w = 0;
  for (const [j, wij] of adj[i]) if (j !== i && comm[j] === target) w += wij;
  return w;
}

function phase(adj: Adj, m: number): Int32Array {
  const n = adj.length;
  const comm = new Int32Array(n);
  const tot = new Float64Array(n);
  const k = adj.map((_, i) => degOf(adj, i));
  for (let i = 0; i < n; i++) {
    comm[i] = i;
    tot[i] = k[i];
  }
  if (m <= 0) return comm;
  let moved = true;
  while (moved) {
    moved = false;
    for (let i = 0; i < n; i++) {
      const cur = comm[i];
      tot[cur] -= k[i];
      comm[i] = -1;
      const cand = new Map<number, number>();
      cand.set(cur, kin(adj, i, comm, cur));
      for (const [j] of adj[i]) {
        if (j === i || comm[j] < 0) continue;
        const c = comm[j];
        if (!cand.has(c)) cand.set(c, kin(adj, i, comm, c));
      }
      let best = cur;
      let bestQ = -Infinity;
      for (const [c, w] of cand) {
        const dq = w / m - (tot[c] * k[i]) / (2 * m * m);
        if (dq > bestQ + 1e-12 || (Math.abs(dq - bestQ) <= 1e-12 && c < best)) {
          bestQ = dq;
          best = c;
        }
      }
      comm[i] = best;
      tot[best] += k[i];
      if (best !== cur) moved = true;
    }
  }
  return comm;
}

function contract(adj: Adj, comm: Int32Array): { adj: Adj; members: number[][] } {
  const ids = [...new Set(comm)].sort((a, b) => a - b);
  const remap = new Map(ids.map((c, i) => [c, i]));
  const members: number[][] = Array.from({ length: ids.length }, () => []);
  comm.forEach((c, i) => members[remap.get(c)!].push(i));
  const next: Adj = Array.from({ length: ids.length }, () => new Map());
  for (let i = 0; i < adj.length; i++) {
    const ci = remap.get(comm[i])!;
    for (const [j, w] of adj[i]) {
      if (j < i) continue;
      const cj = remap.get(comm[j])!;
      if (ci === cj) next[ci].set(ci, (next[ci].get(ci) ?? 0) + w);
      else {
        next[ci].set(cj, (next[ci].get(cj) ?? 0) + w);
        next[cj].set(ci, (next[cj].get(ci) ?? 0) + w);
      }
    }
  }
  return { adj: next, members };
}

/** Louvain communities on the undirected who-played-whom graph. */
export function modularityCommunities(ids: string[], edges: { source: string; target: string }[]): Map<string, number> {
  const assigned = new Map<string, number>();
  const n = ids.length;
  if (!n) return assigned;
  const at = new Map(ids.map((id, i) => [id, i]));
  let { adj, m } = graphFromEdges(n, edges, at);
  let groups = ids.map((_, i) => [i]);

  for (let pass = 0; pass < 8; pass++) {
    const comm = phase(adj, m);
    if (new Set(comm).size === adj.length) break;
    const next = contract(adj, comm);
    groups = next.members.map((nodes) => nodes.flatMap((u) => groups[u]));
    adj = next.adj;
    m = 0;
    for (let i = 0; i < adj.length; i++) for (const [j, w] of adj[i]) if (j >= i) m += w;
    if (adj.length === 1) break;
  }

  const ranked = [...groups].sort((a, b) => b.length - a.length || a[0] - b[0]);
  ranked.forEach((nodes, k) => {
    for (const i of nodes) assigned.set(ids[i], k);
  });
  return assigned;
}

function clusterLabel(members: ClusterTeam[]): string {
  const counts = new Map<string, number>();
  for (const t of members) counts.set(t.conf, (counts.get(t.conf) ?? 0) + 1);
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const [top, n] = ranked[0] ?? ["?", 0];
  if (n === members.length) return top;
  if (n / members.length >= 0.6) return `${top} +`;
  return ranked
    .slice(0, 2)
    .map(([c]) => c)
    .join(" / ");
}

export function layoutClusters(
  teams: ClusterTeam[],
  edges: GameEdge[],
  width: number,
  height: number,
): { bodies: ClusterBody[]; links: ClusterLink[] } {
  const ids = teams.map((t) => t.id);
  const clusterOf = modularityCommunities(ids, edges);
  const groups = new Map<number, ClusterTeam[]>();
  for (const t of teams) {
    const k = clusterOf.get(t.id) ?? 0;
    const list = groups.get(k) ?? [];
    list.push(t);
    groups.set(k, list);
  }

  const wells = [...groups.entries()]
    .sort((a, b) => b[1].length - a[1].length || a[0] - b[0])
    .map(([cluster, members]) => ({
      cluster,
      members: [...members].sort((a, b) => a.team.localeCompare(b.team)),
      label: clusterLabel(members),
      wellR: 22 + Math.sqrt(members.length) * 18,
      x: 0,
      y: 0,
    }));

  const cx = width / 2;
  const cy = height / 2;
  const single = wells.length <= 1;
  if (single && wells[0]) {
    wells[0].x = cx;
    wells[0].y = cy;
    wells[0].wellR = Math.min(width, height) * 0.38;
  } else if (wells.length) {
    const need = ringRadius(wells.map((w) => w.wellR));
    const farthest = need + Math.max(...wells.map((w) => w.wellR));
    const fit = farthest > 0 ? Math.min(1, (Math.min(width, height) / 2 - 30) / farthest) : 1;
    for (const w of wells) w.wellR *= fit;
    placeRing(wells, cx, cy, need * fit);
  }

  const bodies: ClusterBody[] = [];
  for (const w of wells) {
    const n = w.members.length;
    w.members.forEach((t, i) => {
      const angle = (i / Math.max(n, 1)) * TAU - Math.PI / 2;
      const rad = n <= 2 ? 8 : w.wellR * 0.62;
      bodies.push({
        ...t,
        x: w.x + Math.cos(angle) * rad,
        y: w.y + Math.sin(angle) * rad,
        r: single ? 6 + t.wr * 10 : 3.4 + t.wr * 7,
        kind: "team",
        cluster: w.cluster,
      });
    });
    bodies.push({
      id: `cluster:${w.cluster}`,
      team: w.label,
      conf: w.label,
      wr: 0,
      x: w.x,
      y: w.y,
      r: w.wellR,
      kind: "cluster",
      cluster: w.cluster,
      members: n,
      label: w.label,
    });
  }

  const at = new Map(bodies.map((b, i) => [b.id, i]));
  const clusterAt = new Map(bodies.filter((b) => b.kind === "team").map((b) => [b.id, b.cluster]));
  const links: ClusterLink[] = [];
  for (const e of edges) {
    const si = at.get(e.source);
    const ti = at.get(e.target);
    if (si == null || ti == null) continue;
    links.push({
      si,
      ti,
      margin: e.margin,
      same: clusterAt.get(e.source) === clusterAt.get(e.target),
    });
  }
  return { bodies, links };
}

export function hitCluster(bodies: ClusterBody[], x: number, y: number): ClusterBody | null {
  let team: ClusterBody | null = null;
  let teamD = Infinity;
  let well: ClusterBody | null = null;
  let wellD = Infinity;
  for (const b of bodies) {
    const d = Math.hypot(b.x - x, b.y - y);
    if (b.kind === "team") {
      if (d < b.r + 8 && d < teamD) {
        team = b;
        teamD = d;
      }
    } else if (d < b.r && d < wellD) {
      well = b;
      wellD = d;
    }
  }
  return team ?? well;
}

function paintLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number) {
  ctx.lineWidth = 4;
  ctx.strokeStyle = PAPER;
  ctx.lineJoin = "round";
  ctx.strokeText(text, x, y);
  ctx.fillText(text, x, y);
}

export function drawClusters(
  ctx: CanvasRenderingContext2D,
  bodies: ClusterBody[],
  links: ClusterLink[],
  hover: string,
  width: number,
  height: number,
) {
  const cx = width / 2;
  const cy = height / 2;
  ctx.fillStyle = PAPER;
  ctx.fillRect(0, 0, width, height);

  const hoverBody = bodies.find((b) => b.id === hover);
  const teams = bodies.filter((b) => b.kind === "team");
  const wells = bodies.filter((b) => b.kind === "cluster");
  const hoverCluster = hoverBody?.cluster;

  const wellNames = new Set(wells.map((w) => w.label ?? w.team));
  const nameWells = wells.length > 1 && wellNames.size > 1;
  for (const w of wells) {
    const on = hoverCluster == null || hoverCluster === w.cluster;
    ctx.beginPath();
    ctx.strokeStyle = on ? LINE : "rgba(229,229,229,0.35)";
    ctx.lineWidth = 1;
    ctx.arc(w.x, w.y, w.r, 0, TAU);
    ctx.stroke();
    if (!nameWells) continue;
    ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = on ? INK : "#999999";
    paintLabel(ctx, w.label ?? w.team, w.x, w.y);
  }

  const showLink = (link: ClusterLink) => {
    const a = bodies[link.si];
    const b = bodies[link.ti];
    if (!hoverBody) return link.same && teams.length <= 24;
    if (hoverBody.kind === "cluster") return a.cluster === hoverBody.cluster && b.cluster === hoverBody.cluster;
    return a.id === hover || b.id === hover;
  };

  for (const link of links) {
    if (!showLink(link)) continue;
    const a = bodies[link.si];
    const b = bodies[link.ti];
    const hot = hoverBody?.kind === "team";
    const t = Math.min(1, link.margin / 40);
    ctx.beginPath();
    ctx.strokeStyle = hot ? `rgba(17,17,17,${0.2 + t * 0.5})` : "rgba(17,17,17,0.12)";
    ctx.lineWidth = hot ? 1 + t * 2.2 : 1;
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  for (const b of teams) {
    const dim = hoverCluster != null && b.cluster !== hoverCluster;
    ctx.beginPath();
    ctx.fillStyle = confFill(b.conf);
    ctx.globalAlpha = dim ? 0.18 : 1;
    ctx.arc(b.x, b.y, b.r, 0, TAU);
    ctx.fill();
    if (hover === b.id) {
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  const named = new Set<string>();
  if (teams.length <= 18) for (const t of teams) named.add(t.id);
  if (hoverBody?.kind === "team") named.add(hoverBody.id);

  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  ctx.fillStyle = INK;
  const wellAt = new Map(wells.map((w) => [w.cluster, w]));
  for (const b of teams) {
    if (!named.has(b.id)) continue;
    const well = wellAt.get(b.cluster);
    const ox = b.x - (well?.x ?? cx);
    const oy = b.y - (well?.y ?? cy);
    const dist = Math.hypot(ox, oy) || 1;
    const gap = b.r + 8;
    const x = b.x + (ox / dist) * gap;
    const y = b.y + (oy / dist) * gap;
    ctx.textAlign = ox > 6 ? "left" : ox < -6 ? "right" : "center";
    ctx.textBaseline = oy > 6 ? "top" : oy < -6 ? "bottom" : "middle";
    paintLabel(ctx, b.team, x, y);
  }
  ctx.textBaseline = "alphabetic";
}
