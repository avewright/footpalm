export type TeamRow = {
  rank: number;
  team: string;
  conf: string;
  wins: number;
  losses: number;
  games: number;
  pom: number;
  palm?: number;
  elo?: number;
  adjo: number;
  adjd: number;
  adjst: number;
  tempo: number;
  sos: number;
  luck: number;
  nil_roster?: number | null;
  nil_quality?: string;
  nil_all_sports?: number | null;
  athletic_spend?: number | null;
  staff_payroll?: number | null;
};

export type RatingsFile = {
  season: number;
  generated_at: string;
  method: string;
  week?: number;
  home_adv_epa: number;
  league_epa: number;
  plays_per_game: number;
  teams: TeamRow[];
};

export type SortKey = keyof TeamRow;

export type ModelPick = {
  home_win_prob: number;
  pred_margin: number;
};

export type MarketBook = {
  source: string;
  slug?: string;
  url?: string;
  title?: string;
  spread?: number;
  spread_p_home?: number;
  ml_home?: number;
  ml_away?: number;
  ml_home_american?: number;
  ml_away_american?: number;
};

export type PolyBook = MarketBook;

export type EspnQb = {
  name: string;
  jersey?: string | null;
  year?: string | null;
};

export type EspnSnap = {
  logged_at?: string;
  locked?: boolean;
  snaps?: number;
  weather?: {
    venue?: string | null;
    city?: string | null;
    state?: string | null;
    grass?: boolean | null;
    temperature?: number | null;
    gust?: number | null;
    precipitation?: number | null;
    condition?: string | null;
  };
  venue?: { venue?: string; city?: string; state?: string; grass?: boolean };
  qbs?: { home?: EspnQb[]; away?: EspnQb[] };
  fpi?: { home_win?: number; pred_margin?: number; source?: string };
  odds?: { source?: string; spread?: number | null; total?: number | null; details?: string };
};

export type GamePred = {
  season: number;
  slate: number;
  week: number;
  season_type: string;
  game_id?: number;
  start?: string;
  home: string;
  away: string;
  home_conf?: string;
  away_conf?: string;
  home_pom?: number;
  away_pom?: number;
  neutral: boolean;
  fbs_fbs: boolean;
  completed?: boolean;
  pred_margin: number;
  home_win_prob: number;
  pred_home: number;
  pred_away: number;
  actual_home: number | null;
  actual_away: number | null;
  actual_margin: number | null;
  home_won: number | null;
  spread: number | null;
  engine: string;
  books?: { kalshi?: MarketBook; polymarket?: MarketBook };
  espn?: EspnSnap;
  models?: Partial<Record<"lightgbm" | "xgboost" | "tabpfn" | "ensemble", ModelPick>>;
};

export type UnitPpa = {
  overall?: number | null;
  passing?: number | null;
  rushing?: number | null;
  firstDown?: number | null;
  secondDown?: number | null;
  thirdDown?: number | null;
  fourthDown?: number | null;
};

export type UnitStats = {
  games?: number;
  totalYards?: number;
  totalYardsOpponent?: number;
  netPassingYards?: number;
  netPassingYardsOpponent?: number;
  rushingYards?: number;
  rushingYardsOpponent?: number;
  passAttempts?: number;
  passAttemptsOpponent?: number;
  passCompletions?: number;
  passCompletionsOpponent?: number;
  rushingAttempts?: number;
  rushingAttemptsOpponent?: number;
  passingTDs?: number;
  passingTDsOpponent?: number;
  rushingTDs?: number;
  rushingTDsOpponent?: number;
  thirdDowns?: number;
  thirdDownsOpponent?: number;
  thirdDownConversions?: number;
  thirdDownConversionsOpponent?: number;
  sacks?: number;
  sacksOpponent?: number;
  tacklesForLoss?: number;
  tacklesForLossOpponent?: number;
  turnovers?: number;
  turnoversOpponent?: number;
  firstDowns?: number;
  firstDownsOpponent?: number;
};

export type UnitTeam = {
  team: string;
  conf?: string;
  offense: UnitPpa;
  defense: UnitPpa;
  stats: UnitStats;
};

export type UnitsFile = {
  season: number;
  source: string;
  n: number;
  teams: UnitTeam[];
};

export type GraphFile = {
  season: number;
  note: string;
  nodes: {
    id: string;
    team: string;
    conf: string;
    pom: number | null;
    wins: number;
    losses: number;
    nil_roster?: number | null;
    pagerank?: number;
    neighbor_pom?: number;
    vs_neighbors?: number;
    margin_vs?: number;
    winningness?: number;
    size?: number;
    degree?: number;
    betweenness?: number;
    eccentricity?: number | null;
    fiedler?: number | null;
    nx?: number;
    ny?: number;
    tx?: number;
    ty?: number;
    sx?: number;
    sy?: number;
    wx?: number;
    wy?: number;
  }[];
  network?: {
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
  edges: {
    id: string;
    source: string;
    target: string;
    winner: string;
    loser: string;
    home: string;
    away: string;
    margin: number;
    week: number | null;
    fbs_fbs: boolean;
  }[];
};

export type IndexFile = {
  seasons: { season: number; teams: number }[];
};

export type LosoSeason = {
  season: number;
  n: number;
  brier: number;
  logloss: number;
  brier_T?: number;
  brier_clip?: number;
  T?: number;
};

export type LosoRow = {
  id: string;
  engine: string;
  set: string;
  mode?: string;
  pooled: { n: number; accuracy: number; brier: number; logloss: number };
  pooled_T?: { brier: number; logloss: number };
  mean_season_brier: number;
  seasons: LosoSeason[];
  slices?: Record<string, { n: number; brier: number; logloss: number }>;
};

export type LosoFile = {
  protocol: string;
  loso_features?: string[];
  features?: string[];
  rows: LosoRow[];
  perm?: Record<string, Record<string, number>>;
  note?: string;
};

export type MoneyFile = {
  source: string;
  note: string;
  teams: {
    team: string;
    conf: string;
    nil_roster: number | null;
    nil_quality: string;
    nil_all_sports?: number | null;
    athletic_spend: number | null;
    staff_payroll: number | null;
    pom: number;
  }[];
};
