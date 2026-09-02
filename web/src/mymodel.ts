import { gameKey } from "./game";
import { marketSpread } from "./ev";
import { signed } from "./format";
import { isFinal } from "./score";
import type { GamePred } from "./types";

const STORE = "footpalm.mymodel.v1";

export type PickKind = "score" | "ats" | "ml";
export type PickSide = "home" | "away";

export type UserPick = {
  pred_away?: number;
  pred_home?: number;
  home_win_prob: number | null;
  kind?: PickKind;
  side?: PickSide;
  line?: number | null;
};

export type UserModel = {
  name: string;
  season: number;
  source: string;
  uploaded_at: string;
  picks: Record<string, UserPick>;
  matched: number;
  unmatched: number;
};

export type UserScore = {
  n: number;
  suW: number;
  suL: number;
  atsW: number;
  atsL: number;
  pending: number;
  brier: number | null;
  mae: number | null;
  residual: number | null;
};

type Store = Record<string, UserModel>;

function readStore(): Store {
  try {
    const raw = localStorage.getItem(STORE);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Store;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function loadUserModel(season: number): UserModel | null {
  const row = readStore()[String(season)];
  if (!row?.picks || typeof row.picks !== "object") return null;
  return row;
}

export function saveUserModel(model: UserModel) {
  const store = readStore();
  store[String(model.season)] = model;
  try {
    localStorage.setItem(STORE, JSON.stringify(store));
  } catch {
    throw new Error("Could not save the slate in this browser.");
  }
}

export function clearUserModel(season: number) {
  const store = readStore();
  delete store[String(season)];
  localStorage.setItem(STORE, JSON.stringify(store));
}

export function pickOfModel(model: UserModel | null | undefined, game: GamePred): UserPick | null {
  if (!model) return null;
  return model.picks[gameKey(game)] ?? null;
}

export function hasScores(pick: UserPick) {
  return pick.pred_away != null && pick.pred_home != null;
}

export function predMargin(pick: UserPick) {
  if (pick.pred_home != null && pick.pred_away != null) return pick.pred_home - pick.pred_away;
  if (pick.line != null && pick.side) return pick.side === "home" ? -pick.line : pick.line;
  return 0;
}

export function pickLabel(game: GamePred, pick: UserPick) {
  if (pick.kind === "ml" && pick.side) return `${pick.side === "home" ? game.home : game.away} ML`;
  if ((pick.kind === "ats" || pick.side) && pick.line != null) {
    const who = pick.side === "away" ? game.away : game.home;
    return `${who} ${signed(pick.line)}`;
  }
  if (hasScores(pick)) return `${pick.pred_away?.toFixed(1)}–${pick.pred_home?.toFixed(1)}`;
  return "—";
}

export function gradeAts(game: GamePred, pick: UserPick): boolean | null {
  if (game.actual_margin == null || !pick.side) return null;
  const homeLine = pick.line != null ? (pick.side === "home" ? pick.line : -pick.line) : marketSpread(game);
  if (homeLine == null) return null;
  const cover = game.actual_margin + homeLine;
  if (Math.abs(cover) < 1e-9) return null;
  const homeCovers = cover > 0;
  return pick.side === "home" ? homeCovers : !homeCovers;
}

export function mergeUserModel(base: UserModel | null, incoming: UserModel): UserModel {
  const picks = { ...(base?.picks ?? {}), ...incoming.picks };
  return {
    name: incoming.name || base?.name || "Picks",
    season: incoming.season,
    source: incoming.source || base?.source || "",
    uploaded_at: incoming.uploaded_at || base?.uploaded_at || new Date().toISOString(),
    picks,
    matched: Object.keys(picks).length,
    unmatched: incoming.unmatched,
  };
}

export function exportUserModel(model: UserModel) {
  return `${JSON.stringify(
    {
      name: model.name,
      season: model.season,
      source: model.source,
      uploaded_at: model.uploaded_at,
      picks: Object.entries(model.picks).map(([game_id, pick]) => ({ game_id, ...pick })),
    },
    null,
    2,
  )}\n`;
}

export function emptyBook(season: number, name = "Picks"): UserModel {
  return {
    name,
    season,
    source: "",
    uploaded_at: new Date().toISOString(),
    picks: {},
    matched: 0,
    unmatched: 0,
  };
}

function norm(s: string) {
  return s.trim().toLowerCase().replace(/\s+/g, " ");
}

function num(v: unknown): number | null {
  if (v == null || v === "") return null;
  if (typeof v === "number" && Number.isFinite(v)) return v;
  const n = Number(String(v).trim().replace("%", ""));
  return Number.isFinite(n) ? n : null;
}

function headerKey(raw: string) {
  return raw
    .trim()
    .toLowerCase()
    .replace(/^\ufeff/, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");
}

const ALIAS: Record<string, string> = {
  gameid: "game_id",
  game: "game_id",
  id: "game_id",
  cfbd_id: "game_id",
  cfbdid: "game_id",
  away_team: "away",
  home_team: "home",
  visitor: "away",
  predaway: "pred_away",
  predhome: "pred_home",
  away_pred: "pred_away",
  home_pred: "pred_home",
  away_score: "pred_away",
  home_score: "pred_home",
  predicted_away: "pred_away",
  predicted_home: "pred_home",
  pred_away_score: "pred_away",
  pred_home_score: "pred_home",
  margin: "pred_margin",
  home_margin: "pred_margin",
  predmargin: "pred_margin",
  total: "pred_total",
  predtotal: "pred_total",
  over_under: "pred_total",
  win: "home_win_prob",
  win_prob: "home_win_prob",
  home_win: "home_win_prob",
  home_winpct: "home_win_prob",
  p_home: "home_win_prob",
  name: "name",
  model: "name",
  team: "team",
  pick: "team",
  selection: "team",
  ats: "line",
  spread: "line",
};

function canonField(h: string) {
  const key = headerKey(h);
  return ALIAS[key] ?? key;
}

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let q = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (q && line[i + 1] === '"') {
        cur += '"';
        i += 1;
      } else q = !q;
    } else if (c === "," && !q) {
      out.push(cur);
      cur = "";
    } else cur += c;
  }
  out.push(cur);
  return out;
}

function rowsFromCsv(text: string): Record<string, string>[] {
  const lines = text
    .replace(/^\ufeff/, "")
    .split(/\r?\n/)
    .filter((ln) => ln.trim() && !ln.trim().startsWith("#"));
  if (lines.length < 2) return [];
  const headers = splitCsvLine(lines[0]).map(canonField);
  return lines.slice(1).map((ln) => {
    const cells = splitCsvLine(ln);
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      if (h) row[h] = cells[i] ?? "";
    });
    return row;
  });
}

function rowsFromJson(text: string): { name?: string; season?: number; rows: Record<string, unknown>[] } {
  const raw = JSON.parse(text) as unknown;
  if (Array.isArray(raw)) return { rows: raw as Record<string, unknown>[] };
  if (raw && typeof raw === "object") {
    const obj = raw as Record<string, unknown>;
    const list = obj.predictions ?? obj.games ?? obj.picks ?? obj.rows;
    if (list && typeof list === "object" && !Array.isArray(list)) {
      return {
        name: typeof obj.name === "string" ? obj.name : undefined,
        season: typeof obj.season === "number" ? obj.season : num(obj.season) ?? undefined,
        rows: Object.entries(list as Record<string, unknown>).map(([game_id, pick]) =>
          pick && typeof pick === "object" ? { game_id, ...(pick as object) } : { game_id },
        ),
      };
    }
    if (!Array.isArray(list)) throw new Error("JSON needs a predictions array.");
    return {
      name: typeof obj.name === "string" ? obj.name : typeof obj.model === "string" ? obj.model : undefined,
      season: typeof obj.season === "number" ? obj.season : num(obj.season) ?? undefined,
      rows: list as Record<string, unknown>[],
    };
  }
  throw new Error("JSON must be an object or an array of games.");
}

function scoresOf(row: Record<string, unknown>): { pred_away: number; pred_home: number } | null {
  let away = num(row.pred_away);
  let home = num(row.pred_home);
  const margin = num(row.pred_margin);
  const total = num(row.pred_total);
  if (away == null && home == null && margin != null && total != null) {
    home = (total + margin) / 2;
    away = (total - margin) / 2;
  }
  if (away == null || home == null) return null;
  if (away < 0 || home < 0 || away > 120 || home > 120) return null;
  return { pred_away: away, pred_home: home };
}

function winProbOf(row: Record<string, unknown>): number | null {
  let p = num(row.home_win_prob);
  if (p == null) return null;
  if (p > 1 && p <= 100) p = p / 100;
  if (p < 0 || p > 1) return null;
  return p;
}

function lookupMaps(games: GamePred[]) {
  const byId = new Map<string, GamePred>();
  const byTeams = new Map<string, GamePred>();
  for (const g of games) {
    if (g.game_id != null) byId.set(String(g.game_id), g);
    byTeams.set(`${g.week}|${norm(g.away)}|${norm(g.home)}`, g);
  }
  return { byId, byTeams };
}

function matchTeam(query: string, games: GamePred[]): GamePred | null {
  const q = norm(query);
  if (!q) return null;
  const hits = games.filter((g) => {
    const home = norm(g.home);
    const away = norm(g.away);
    return home === q || away === q || home.startsWith(q) || away.startsWith(q) || home.includes(q) || away.includes(q);
  });
  if (hits.length === 1) return hits[0];
  const open = hits.filter((g) => !isFinal(g));
  if (open.length === 1) return open[0];
  return null;
}

function sideOf(game: GamePred, query: string, raw?: string): PickSide | undefined {
  const s = norm(raw || "");
  if (s === "home") return "home";
  if (s === "away") return "away";
  const q = norm(query);
  if (!q) return undefined;
  if (norm(game.home) === q || norm(game.home).includes(q)) return "home";
  if (norm(game.away) === q || norm(game.away).includes(q)) return "away";
  return undefined;
}

function lockLine(game: GamePred, side: PickSide, line?: number | null) {
  if (line != null) return line;
  const mkt = marketSpread(game);
  if (mkt == null) return null;
  return side === "home" ? mkt : -mkt;
}

function betFromRow(row: Record<string, unknown>): { query: string; kind: PickKind; side?: string; line: number | null } | null {
  const query = String(row.team ?? row.away ?? row.home ?? "").trim();
  const rawSide = String(row.side ?? "").trim();
  const kindRaw = norm(String(row.kind ?? ""));
  let line = num(row.line);
  if (kindRaw === "ml" || kindRaw === "moneyline") {
    if (!query && !rawSide) return null;
    return { query: query || rawSide, kind: "ml", side: rawSide, line: null };
  }
  if (query && (line != null || rawSide)) {
    return { query, kind: "ats", side: rawSide, line };
  }
  return null;
}

const PROSE_ML = /^(.+?)\s+(ml|moneyline)\s*$/i;
const PROSE_LINE = /^(.+?)\s+([+-]\d+(?:\.\d+)?)\s*$/;

function betFromLine(text: string): { query: string; kind: PickKind; line: number | null } | null {
  const t = text.trim().replace(/^[-*•]\s*/, "");
  if (!t || t.startsWith("#")) return null;
  const ml = t.match(PROSE_ML);
  if (ml) return { query: ml[1].trim(), kind: "ml", line: null };
  const line = t.match(PROSE_LINE);
  if (line) return { query: line[1].trim(), kind: "ats", line: Number(line[2]) };
  return null;
}

function applyBet(
  picks: Record<string, UserPick>,
  game: GamePred,
  spec: { query: string; kind: PickKind; side?: string; line: number | null },
) {
  const side = sideOf(game, spec.query, spec.side);
  if (!side) return false;
  const line = spec.kind === "ats" ? lockLine(game, side, spec.line) : null;
  picks[gameKey(game)] = {
    home_win_prob: null,
    kind: spec.kind,
    side,
    line,
  };
  return true;
}

function matchRow(
  row: Record<string, unknown>,
  maps: ReturnType<typeof lookupMaps>,
): GamePred | null {
  const id = row.game_id != null && String(row.game_id).trim() !== "" ? String(row.game_id).trim() : "";
  if (id && maps.byId.has(id)) return maps.byId.get(id) ?? null;
  const away = typeof row.away === "string" ? row.away : String(row.away ?? "");
  const home = typeof row.home === "string" ? row.home : String(row.home ?? "");
  const week = num(row.week);
  if (away && home && week != null) {
    const hit = maps.byTeams.get(`${week}|${norm(away)}|${norm(home)}`);
    if (hit) return hit;
  }
  if (away && home) {
    const hits = [...maps.byTeams.values()].filter(
      (g) => norm(g.away) === norm(away) && norm(g.home) === norm(home),
    );
    if (hits.length === 1) return hits[0];
  }
  return null;
}

function tableHeaders(text: string) {
  const first = text
    .split(/\r?\n/)
    .find((ln) => ln.trim() && !ln.trim().startsWith("#"));
  if (!first) return [];
  return splitCsvLine(first).map(canonField);
}

function looksLikeTable(text: string) {
  const keys = tableHeaders(text);
  return keys.some((k) =>
    ["game_id", "away", "home", "pred_away", "pred_home", "team", "line", "side", "kind"].includes(k),
  );
}

export function parseUserModel(
  text: string,
  filename: string,
  season: number,
  games: GamePred[],
): { model: UserModel; warnings: string[]; leftover: string } {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("File is empty.");
  const maps = lookupMaps(games);
  let name: string | undefined;
  let fileSeason: number | undefined;
  let rawRows: Record<string, unknown>[] = [];
  const leftover: string[] = [];
  const json = trimmed.startsWith("{") || trimmed.startsWith("[");
  if (json) {
    const parsed = rowsFromJson(trimmed);
    name = parsed.name;
    fileSeason = parsed.season;
    rawRows = parsed.rows;
  } else if (looksLikeTable(trimmed)) {
    rawRows = rowsFromCsv(trimmed);
  } else {
    leftover.push(...trimmed.split(/\r?\n/));
  }

  const picks: Record<string, UserPick> = {};
  let unmatched = 0;
  let skipped = 0;
  const warnings: string[] = [];
  if (fileSeason != null && fileSeason !== season) {
    warnings.push(`File season is ${fileSeason}; matching against the ${season} slate.`);
  }

  for (const row of rawRows) {
    const scores = scoresOf(row);
    if (scores) {
      const game = matchRow(row, maps);
      if (!game) {
        unmatched += 1;
        leftover.push([row.away, row.home, row.pred_away, row.pred_home].filter((x) => x !== "").join(" "));
        continue;
      }
      picks[gameKey(game)] = {
        pred_away: scores.pred_away,
        pred_home: scores.pred_home,
        home_win_prob: winProbOf(row),
        kind: "score",
      };
      continue;
    }
    const bet = betFromRow(row);
    if (bet) {
      const game = matchRow(row, maps) ?? matchTeam(bet.query, games);
      if (!game || !applyBet(picks, game, bet)) {
        unmatched += 1;
        leftover.push([bet.query, bet.line ?? "", bet.kind].join(" "));
      }
      continue;
    }
    skipped += 1;
  }

  for (const line of leftover.splice(0)) {
    const bet = betFromLine(line);
    if (!bet) {
      if (line.trim()) leftover.push(line);
      continue;
    }
    const game = matchTeam(bet.query, games);
    if (!game || !applyBet(picks, game, bet)) leftover.push(line);
  }

  const matched = Object.keys(picks).length;
  if (!matched && !leftover.length) {
    throw new Error(
      "No rows matched the current slate. Use game_id, team + line, or names as FootPalm spells them.",
    );
  }
  if (skipped) warnings.push(`${skipped} row${skipped === 1 ? "" : "s"} skipped.`);
  if (unmatched) warnings.push(`${unmatched} row${unmatched === 1 ? "" : "s"} did not match a ${season} game.`);

  const fromFile = filename.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
  return {
    model: {
      name: name?.trim() || fromFile || "Picks",
      season,
      source: filename,
      uploaded_at: new Date().toISOString(),
      picks,
      matched,
      unmatched,
    },
    warnings,
    leftover: leftover.map((ln) => ln.trim()).filter(Boolean).join("\n"),
  };
}

export function slateCsv(games: GamePred[], season: number) {
  const header = "game_id,week,away,home,kickoff,pred_away,pred_home,home_win_prob";
  const body = games
    .filter((g) => g.fbs_fbs)
    .sort((a, b) => a.week - b.week || (a.start ?? "").localeCompare(b.start ?? "") || a.away.localeCompare(b.away))
    .map((g) =>
      [
        g.game_id ?? "",
        g.week,
        csvCell(g.away),
        csvCell(g.home),
        g.start ?? "",
        "",
        "",
        "",
      ].join(","),
    );
  return [`# FootPalm ${season} slate. Fill pred_away and pred_home from your model. Optional: home_win_prob (0–1).`, header, ...body].join(
    "\n",
  );
}

function csvCell(s: string) {
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function scoreUserModel(games: GamePred[], model: UserModel): UserScore {
  const covered = games.filter((g) => model.picks[gameKey(g)]);
  const played = covered.filter((g) => isFinal(g));
  let suW = 0;
  let suN = 0;
  let atsW = 0;
  let atsN = 0;
  let brier = 0;
  let brierN = 0;
  let mae = 0;
  let residual = 0;
  let maeN = 0;
  for (const g of played) {
    const pick = model.picks[gameKey(g)];
    if (hasScores(pick)) {
      suN += 1;
      if (Boolean(pick.pred_home! > pick.pred_away!) === Boolean(g.home_won)) suW += 1;
    } else if (pick.side && pick.kind !== "ats") {
      suN += 1;
      if (Boolean(pick.side === "home") === Boolean(g.home_won)) suW += 1;
    }
    const ats = gradeAts(g, pick);
    if (ats != null) {
      atsN += 1;
      if (ats) atsW += 1;
    }
    if (pick.home_win_prob != null && g.home_won != null) {
      brier += (pick.home_win_prob - g.home_won) ** 2;
      brierN += 1;
    }
    if (g.actual_margin != null && (hasScores(pick) || pick.line != null)) {
      mae += Math.abs(predMargin(pick) - g.actual_margin);
      residual += g.actual_margin - predMargin(pick);
      maeN += 1;
    }
  }
  return {
    n: played.length,
    suW,
    suL: suN - suW,
    atsW,
    atsL: atsN - atsW,
    pending: covered.length - played.length,
    brier: brierN ? brier / brierN : null,
    mae: maeN ? mae / maeN : null,
    residual: maeN ? residual / maeN : null,
  };
}

export function overlayPick(game: GamePred, pick: UserPick): GamePred {
  if (!hasScores(pick)) return game;
  const pred_margin = predMargin(pick);
  return {
    ...game,
    pred_away: pick.pred_away ?? game.pred_away,
    pred_home: pick.pred_home ?? game.pred_home,
    pred_margin,
    home_win_prob: pick.home_win_prob ?? game.home_win_prob,
  };
}
