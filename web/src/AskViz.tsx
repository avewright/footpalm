import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

export type Card = {
  kind: "stats" | "table" | "bars" | "line" | "graph";
  title: string;
  items?: { label: string; value: string | number; tone?: string }[];
  columns?: string[];
  rows?: (string | number)[][];
  nodes?: { id: string; label?: string; value?: string | number }[];
  edges?: { source: string; target: string; label?: string }[];
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

export function CardView({ card }: { card: Card }) {
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
