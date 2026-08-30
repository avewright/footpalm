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

export type PolyBook = {
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

export type GamePred = {
  season: number;
  slate: number;
  week: number;
  season_type: string;
  game_id?: number;
  start?: string;
  home: string;
  away: string;
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
  books?: { polymarket?: PolyBook };
  models?: Partial<Record<"lightgbm" | "xgboost" | "tabpfn" | "ensemble", ModelPick>>;
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

export type BacktestFile = {
  season: number;
  protocol: string;
  features: string[];
  engine_counts: Record<string, number>;
  tabpfn_error?: string | null;
  all_fbs: Record<string, number>;
  tabpfn: Record<string, number>;
  logistic: Record<string, number>;
  calibration: { bucket: string; n: number; pred: number; actual: number }[];
  by_week: Record<string, number>[];
};

export type BacktestSummary = {
  seasons: {
    season: number;
    engines: Record<string, number>;
    fbs: Record<string, number>;
    tabpfn: Record<string, number>;
    logistic: Record<string, number>;
  }[];
  nil_residual: {
    used: boolean;
    note?: string;
    helped?: boolean;
    coef?: number;
    test_mse_before?: number;
    test_mse_after?: number;
    reason?: string;
  };
  rule: string;
};

export type ResearchMetrics = { n: number; accuracy: number; brier: number; logloss: number };

export type ResearchExperiment = {
  id: string;
  params: Record<string, number>;
  note: string;
  train: ResearchMetrics;
  holdout: ResearchMetrics;
  delta_holdout_brier?: number;
  delta_holdout_logloss?: number;
  pass?: boolean;
};

export type ResearchFile = {
  generated_at: string;
  protocol: string;
  train_seasons?: number[];
  holdout_season?: number;
  train_n?: number;
  holdout_n?: number;
  promote_if: string;
  baseline_holdout_brier?: number;
  baseline_2025_brier?: number;
  holdout_expected_brier_if_calibrated?: number;
  holdout_2025_expected_brier_if_calibrated?: number;
  note?: string;
  conclusion?: string;
  promoted: string;
  experiments: ResearchExperiment[];
  diagnostics?: ResearchExperiment[];
  trees?: TreeReport | null;
};

export type TreeImportance = { feature: string; gain?: number; share?: number; brier_increase?: number };

export type TreeModel = {
  id: string;
  note: string;
  holdout: ResearchMetrics & { mae?: number };
  gain: TreeImportance[];
  permutation: TreeImportance[];
};

export type TreeComparisonRow = {
  family: string;
  set?: string;
  locked_brier: number;
  full_brier: number;
  locked_logloss: number;
  full_logloss: number;
  delta_brier: number;
  pass: boolean;
};

export type TreeReport = {
  protocol?: string;
  features?: string[];
  extra_features?: string[];
  signal_features?: string[];
  train_n?: number;
  holdout_n?: number;
  comparison?: {
    rule: string;
    extras?: string[];
    signal?: string[];
    rows: TreeComparisonRow[];
    would_promote: boolean;
    would_promote_signal?: boolean;
    promoted: boolean;
    note: string;
  };
  models?: TreeModel[];
  error?: string;
};

export type IndexFile = {
  seasons: { season: number; teams: number }[];
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
