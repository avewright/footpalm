export function pctile(value: number, pool: number[], higherBetter: boolean) {
  if (!pool.length) return 0.5;
  let beats = 0;
  for (const v of pool) {
    if (higherBetter ? value > v : value < v) beats += 1;
    else if (value === v) beats += 0.5;
  }
  return beats / pool.length;
}

function polar(cx: number, cy: number, r: number, i: number, n: number, t: number) {
  const a = -Math.PI / 2 + (i / n) * 2 * Math.PI;
  return { x: cx + r * t * Math.cos(a), y: cy + r * t * Math.sin(a) };
}

export function Radar({
  axes,
  classA = "mux-poly-away",
  classB = "mux-poly-home",
  label,
}: {
  axes: { label: string; a: number; b: number; title?: string }[];
  classA?: string;
  classB?: string;
  label?: string;
}) {
  const w = 460;
  const h = 320;
  const cx = w / 2;
  const cy = h / 2 + 4;
  const r = 100;
  const n = axes.length;
  const rings = [0.25, 0.5, 0.75, 1];

  function ringPath(t: number) {
    return (
      axes
        .map((_, i) => {
          const p = polar(cx, cy, r, i, n, t);
          return `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`;
        })
        .join(" ") + "Z"
    );
  }

  function poly(key: "a" | "b") {
    return (
      axes
        .map((axis, i) => {
          const p = polar(cx, cy, r, i, n, Math.max(0.04, axis[key]));
          return `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`;
        })
        .join(" ") + "Z"
    );
  }

  return (
    <svg className="mux-radar" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label ?? "Radar"}>
      {rings.map((t) => (
        <path key={t} d={ringPath(t)} className="mux-ring" />
      ))}
      {axes.map((_, i) => {
        const p = polar(cx, cy, r, i, n, 1);
        return <line key={i} x1={cx} y1={cy} x2={p.x} y2={p.y} className="mux-ring" />;
      })}
      <path d={poly("a")} className={`mux-poly ${classA}`} />
      <path d={poly("b")} className={`mux-poly ${classB}`} />
      {axes.map((axis, i) => {
        const p = polar(cx, cy, r + 28, i, n, 1);
        const anchor = p.x < cx - 8 ? "end" : p.x > cx + 8 ? "start" : "middle";
        const hitW = Math.max(36, axis.label.length * 8);
        const hitX = anchor === "end" ? p.x - hitW : anchor === "start" ? p.x : p.x - hitW / 2;
        return (
          <g key={axis.label}>
            {axis.title ? <title>{axis.title}</title> : null}
            <rect x={hitX} y={p.y - 10} width={hitW} height={20} fill="transparent" />
            <text x={p.x} y={p.y} textAnchor={anchor} className="mux-radar-lab">
              {axis.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
