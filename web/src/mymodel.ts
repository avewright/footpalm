import { gameKey } from "./game";
import { isFinal } from "./score";
import type { GamePred } from "./types";

const STORE = "footpalm.mymodel.v1";

export type UserPick = {
  pred_away: number;
  pred_home: number;
  home_win_prob: number | null;
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

export function predMargin(pick: UserPick) {
  return pick.pred_home - pick.pred_away;
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

export function parseUserModel(
  text: string,
  filename: string,
  season: number,
  games: GamePred[],
): { model: UserModel; warnings: string[] } {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("File is empty.");
  const maps = lookupMaps(games);
  let name: string | undefined;
  let fileSeason: number | undefined;
  let rawRows: Record<string, unknown>[];
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    const parsed = rowsFromJson(trimmed);
    name = parsed.name;
    fileSeason = parsed.season;
    rawRows = parsed.rows;
  } else {
    rawRows = rowsFromCsv(trimmed);
  }
  if (!rawRows.length) throw new Error("No prediction rows found.");

  const picks: Record<string, UserPick> = {};
  let unmatched = 0;
  let skipped = 0;
  const warnings: string[] = [];
  if (fileSeason != null && fileSeason !== season) {
    warnings.push(`File season is ${fileSeason}; matching against the ${season} slate.`);
  }

  for (const row of rawRows) {
    const scores = scoresOf(row);
    if (!scores) {
      skipped += 1;
      continue;
    }
    const game = matchRow(row, maps);
    if (!game) {
      unmatched += 1;
      continue;
    }
    picks[gameKey(game)] = {
      pred_away: scores.pred_away,
      pred_home: scores.pred_home,
      home_win_prob: winProbOf(row),
    };
  }

  const matched = Object.keys(picks).length;
  if (!matched) {
    throw new Error(
      "No rows matched the current slate. Use the template’s game_id column, or week + away + home names as FootPalm spells them.",
    );
  }
  if (skipped) warnings.push(`${skipped} row${skipped === 1 ? "" : "s"} skipped — need pred_away and pred_home (or pred_margin + pred_total).`);
  if (unmatched) warnings.push(`${unmatched} row${unmatched === 1 ? "" : "s"} did not match a ${season} game.`);

  const fromFile = filename.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
  return {
    model: {
      name: name?.trim() || fromFile || "Uploaded model",
      season,
      source: filename,
      uploaded_at: new Date().toISOString(),
      picks,
      matched,
      unmatched,
    },
    warnings,
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
  const played = games.filter((g) => isFinal(g) && model.picks[gameKey(g)]);
  let suW = 0;
  let brier = 0;
  let brierN = 0;
  let mae = 0;
  let residual = 0;
  for (const g of played) {
    const pick = model.picks[gameKey(g)];
    const predHome = pick.pred_home > pick.pred_away;
    const actualHome = Boolean(g.home_won);
    if (predHome === actualHome) suW += 1;
    if (pick.home_win_prob != null && g.home_won != null) {
      brier += (pick.home_win_prob - g.home_won) ** 2;
      brierN += 1;
    }
    if (g.actual_margin != null) {
      const margin = predMargin(pick);
      mae += Math.abs(margin - g.actual_margin);
      residual += g.actual_margin - margin;
    }
  }
  const n = played.length;
  return {
    n,
    suW,
    suL: n - suW,
    brier: brierN ? brier / brierN : null,
    mae: n ? mae / n : null,
    residual: n ? residual / n : null,
  };
}

export function overlayPick(game: GamePred, pick: UserPick): GamePred {
  const pred_margin = predMargin(pick);
  return {
    ...game,
    pred_away: pick.pred_away,
    pred_home: pick.pred_home,
    pred_margin,
    home_win_prob: pick.home_win_prob ?? game.home_win_prob,
  };
}
