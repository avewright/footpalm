import { useMemo, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { signed } from "./format";
import { TeamLink } from "./TeamView";

export type ScatterPoint = { label: string; x: number; y: number; hint?: string };

export type Card = {
  kind: "stats" | "table" | "bars" | "line" | "graph" | "scatter";
  title: string;
  items?: { label: string; value: string | number; tone?: string }[];
  columns?: string[];
  rows?: (string | number)[][];
  nodes?: { id: string; label?: string; value?: string | number }[];
  edges?: { source: string; target: string; label?: string }[];
  points?: ScatterPoint[];
  x_label?: string;
  y_label?: string;
};

const VIZ_FENCE = /```(?:viz|chart|graph)\s*\n([\s\S]*?)```/gi;

export function splitViz(text: string): { text: string; cards: Card[] } {
  const cards: Card[] = [];
  const cleaned = text.replace(VIZ_FENCE, (_, body: string) => {
    try {
      const parsed = JSON.parse(body);
      if (parsed && typeof parsed === "object" && parsed.kind) cards.push(parsed);
    } catch {
      /* leave the fence if it is not JSON */
      return _;
    }
    return "";
  });
  return { text: cleaned.trim(), cards };
}

function cell(value: string | number) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  return value;
}

function Bars({ items }: { items: NonNullable<Card["items"]> }) {
  const values = items.map((item) => Number(item.value) || 0);
  const max = Math.max(...values.map(Math.abs), 1);
  const w = 640;
  const row = 22;
  const h = items.length * row + 8;
  const labelW = 120;
  const barW = w - labelW - 56;
  return (
    <svg className="ask-svg" viewBox={`0 0 ${w} ${h}`} role="img">
      {items.map((item, i) => {
        const n = Number(item.value) || 0;
        const y = i * row + 4;
        const width = (Math.abs(n) / max) * barW;
        return (
          <g key={item.label}>
            <text x={0} y={y + 12} className="ask-svg-label">
              {item.label}
            </text>
            <rect x={labelW} y={y + 4} width={barW} height={10} fill="var(--line)" />
            <rect x={labelW} y={y + 4} width={width} height={10} fill="var(--ink)" />
            <text x={w} y={y + 12} textAnchor="end" className="ask-svg-num">
              {cell(item.value)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function Line({ items }: { items: NonNullable<Card["items"]> }) {
  const w = 640;
  const h = 200;
  const pad = { l: 36, r: 12, t: 12, b: 28 };
  const values = items.map((item) => Number(item.value) || 0);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;
  const x = (i: number) => pad.l + (items.length < 2 ? innerW / 2 : (i / (items.length - 1)) * innerW);
  const y = (v: number) => pad.t + innerH - ((v - min) / (max - min || 1)) * innerH;
  const d = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const zero = min < 0 && max > 0 ? y(0) : null;
  return (
    <svg className="ask-svg" viewBox={`0 0 ${w} ${h}`} role="img">
      {zero != null && <line x1={pad.l} x2={w - pad.r} y1={zero} y2={zero} className="ask-svg-axis" />}
      <line x1={pad.l} x2={pad.l} y1={pad.t} y2={h - pad.b} className="ask-svg-axis" />
      <line x1={pad.l} x2={w - pad.r} y1={h - pad.b} y2={h - pad.b} className="ask-svg-axis" />
      <path d={d} fill="none" stroke="var(--ink)" strokeWidth="2" />
      {items.map((item, i) => (
        <circle key={item.label} cx={x(i)} cy={y(Number(item.value) || 0)} r="3" fill="var(--ink)" />
      ))}
      {items.map((item, i) => {
        const every = Math.ceil(items.length / 6);
        if (i !== 0 && i !== items.length - 1 && i % every !== 0) return null;
        return (
          <text key={`${item.label}-${i}`} x={x(i)} y={h - 8} textAnchor="middle" className="ask-svg-label">
            {item.label}
          </text>
        );
      })}
    </svg>
  );
}

function MiniGraph({ nodes, edges }: { nodes: NonNullable<Card["nodes"]>; edges: NonNullable<Card["edges"]> }) {
  const w = 640;
  const h = 320;
  const cx = w / 2;
  const cy = h / 2;
  const r = Math.min(w, h) / 2 - 48;
  const pos = new Map<string, { x: number; y: number }>();
  if (nodes.length === 2) {
    pos.set(nodes[0].id, { x: w * 0.28, y: cy });
    pos.set(nodes[1].id, { x: w * 0.72, y: cy });
  } else {
    nodes.forEach((node, i) => {
      const a = (i / Math.max(nodes.length, 1)) * Math.PI * 2 - Math.PI / 2;
      pos.set(node.id, { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) });
    });
  }
  return (
    <svg className="ask-svg" viewBox={`0 0 ${w} ${h}`} role="img">
      {edges.map((edge, i) => {
        const a = pos.get(edge.source);
        const b = pos.get(edge.target);
        if (!a || !b) return null;
        return (
          <g key={i}>
            <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="ask-svg-edge" />
            {edge.label && (
              <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 4} textAnchor="middle" className="ask-svg-label">
                {edge.label}
              </text>
            )}
          </g>
        );
      })}
      {nodes.map((node) => {
        const p = pos.get(node.id);
        if (!p) return null;
        return (
          <g key={node.id}>
            <circle cx={p.x} cy={p.y} r="16" fill="var(--bg)" stroke="var(--ink)" strokeWidth="1.5" />
            <text x={p.x} y={p.y + 28} textAnchor="middle" className="ask-svg-label">
              {node.label || node.id}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function padRange(values: number[]) {
  const lo0 = Math.min(...values);
  const hi0 = Math.max(...values);
  const pad = (hi0 - lo0) * 0.08 || 1;
  const lo = lo0 >= 0 ? Math.max(0, lo0 - pad * 0.25) : lo0 - pad;
  return [lo, hi0 + pad] as const;
}

function namedPoints(points: ScatterPoint[]) {
  const byX = [...points].sort((a, b) => b.x - a.x).slice(0, 6);
  const byY = [...points].sort((a, b) => b.y - a.y).slice(0, 6);
  const seen = new Map<string, ScatterPoint>();
  for (const p of [...byY, ...byX]) seen.set(p.label, p);
  return seen;
}

function Scatter({
  points,
  xLabel,
  yLabel,
  onOpen,
}: {
  points: ScatterPoint[];
  xLabel?: string;
  yLabel?: string;
  onOpen?: (team: string) => void;
}) {
  const [hover, setHover] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const w = 640;
  const h = 380;
  const box = { l: 46, r: 16, t: 16, b: 40 };
  const [x0, x1] = padRange(points.map((p) => p.x));
  const [y0, y1] = padRange(points.map((p) => p.y));
  const x = (v: number) => box.l + ((v - x0) / (x1 - x0 || 1)) * (w - box.l - box.r);
  const y = (v: number) => box.t + (1 - (v - y0) / (y1 - y0 || 1)) * (h - box.t - box.b);
  const labels = useMemo(() => namedPoints(points), [points]);
  const needle = q.trim().toLowerCase();
  const active = points.find((p) => p.label === hover) ?? null;
  const xticks = [x0, x0 + (x1 - x0) / 2, x1];
  const yticks = [y0, y0 + (y1 - y0) / 2, y1];

  if (!points.length) return <p className="lede-note">No points.</p>;

  return (
    <div className="ask-scatter">
      <input
        type="search"
        className="ask-scatter-find"
        placeholder="Find a team"
        value={q}
        aria-label="Find a team"
        onChange={(e) => {
          const next = e.target.value;
          setQ(next);
          const hit = points.find((p) => p.label.toLowerCase().includes(next.trim().toLowerCase()));
          setHover(next.trim() && hit ? hit.label : null);
        }}
      />
      <svg className="ask-svg" viewBox={`0 0 ${w} ${h}`} role="img">
        {y0 < 0 && y1 > 0 && <line x1={box.l} x2={w - box.r} y1={y(0)} y2={y(0)} className="ask-svg-axis" />}
        <line x1={box.l} x2={box.l} y1={box.t} y2={h - box.b} className="ask-svg-axis" />
        <line x1={box.l} x2={w - box.r} y1={h - box.b} y2={h - box.b} className="ask-svg-axis" />
        {yticks.map((v, i) => (
          <text key={`y${i}`} x={box.l - 6} y={y(v) + 4} textAnchor="end" className="ask-svg-num">
            {v.toFixed(0)}
          </text>
        ))}
        {xticks.map((v, i) => (
          <text key={`x${i}`} x={x(v)} y={h - 22} textAnchor="middle" className="ask-svg-num">
            {v.toFixed(0)}
          </text>
        ))}
        <text x={box.l} y={12} className="ask-svg-label">
          {yLabel || "Y"}
        </text>
        <text x={(box.l + w - box.r) / 2} y={h - 6} textAnchor="middle" className="ask-svg-label">
          {xLabel || "X"}
        </text>
        {points.map((p) => {
          const on = hover === p.label;
          const dim = Boolean(needle && !p.label.toLowerCase().includes(needle));
          return (
            <g
              key={p.label}
              className={`ask-svg-hit${on ? " is-on" : ""}${dim ? " is-dim" : ""}`}
              onMouseEnter={() => setHover(p.label)}
              onMouseLeave={() => setHover((cur) => (cur === p.label ? null : cur))}
              onClick={() => onOpen?.(p.label)}
            >
              <circle cx={x(p.x)} cy={y(p.y)} r="12" fill="transparent" />
              <circle className="ask-svg-dot" cx={x(p.x)} cy={y(p.y)} r={on ? 6 : 4} />
            </g>
          );
        })}
        {[...labels.values()].map((p) => (
          <text
            key={`lbl-${p.label}`}
            x={x(p.x) + 8}
            y={y(p.y) - 8}
            className={`ask-svg-label${hover === p.label ? " is-on" : ""}`}
          >
            {p.label}
          </text>
        ))}
        {active && !labels.has(active.label) && (
          <text x={x(active.x) + 8} y={y(active.y) - 8} className="ask-svg-label is-on">
            {active.label}
          </text>
        )}
      </svg>
      <div className="ask-scatter-tip">
        {active ? (
          <>
            {onOpen ? <TeamLink team={active.label} onOpen={onOpen} /> : <strong>{active.label}</strong>}
            {active.hint ? ` · ${active.hint}` : ""}
            {` · Pom ${signed(active.y)} · $${active.x.toFixed(1)}M`}
          </>
        ) : (
          <span className="lede-note">Hover a team. Click to open.</span>
        )}
      </div>
    </div>
  );
}

export function CardView({ card, onOpen }: { card: Card; onOpen?: (team: string) => void }) {
  return (
    <section className="ask-card">
      {card.title && <h3>{card.title}</h3>}
      {card.kind === "stats" && (
        <dl className="metrics">
          {(card.items ?? []).map((item) => (
            <div key={item.label} className="ask-stat">
              <dt>{item.label}</dt>
              <dd className={item.tone || undefined}>{cell(item.value)}</dd>
            </div>
          ))}
        </dl>
      )}
      {card.kind === "bars" && <Bars items={card.items ?? []} />}
      {card.kind === "line" && <Line items={card.items ?? []} />}
      {card.kind === "graph" && <MiniGraph nodes={card.nodes ?? []} edges={card.edges ?? []} />}
      {card.kind === "scatter" && (
        <Scatter points={card.points ?? []} xLabel={card.x_label} yLabel={card.y_label} onOpen={onOpen} />
      )}
      {card.kind === "table" && (
        <div className="table-wrap ask-table">
          <table>
            <thead>
              <tr>
                {(card.columns ?? []).map((col) => (
                  <th key={col} className={col === card.columns?.[0] ? "left" : undefined}>
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(card.rows ?? []).map((row, i) => (
                <tr key={i}>
                  {row.map((value, j) => (
                    <td key={j} className={j === 0 ? "left" : undefined}>
                      {cell(value)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export function AskMarkdown({ text }: { text: string }) {
  return (
    <div className="ask-md">
      <Markdown remarkPlugins={[remarkGfm]} components={{ img: () => null }}>
        {text}
      </Markdown>
    </div>
  );
}
