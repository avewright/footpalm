export function signed(n: number, digits = 1) {
  return `${n > 0 ? "+" : ""}${n.toFixed(digits)}`;
}

export function american(p: number) {
  const clip = Math.min(1 - 1e-6, Math.max(1e-6, p));
  return clip >= 0.5 ? Math.round((-100 * clip) / (1 - clip)) : Math.round((100 * (1 - clip)) / clip);
}

export function formatAmerican(n: number) {
  return n > 0 ? `+${n}` : `${n}`;
}

export function money(n: number | null | undefined) {
  if (n == null) return "—";
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n}`;
}

export function tone(n: number, invert = false) {
  const v = invert ? -n : n;
  if (v > 0.05) return "good";
  if (v < -0.05) return "bad";
  return;
}
