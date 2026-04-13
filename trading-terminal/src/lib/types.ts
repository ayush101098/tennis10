/* ── Shared TypeScript types mirroring the Python models ── */

export type Action = "ENTER" | "HOLD" | "HEDGE" | "SKIP" | "EXIT" | "SCALP" | "EMERGENCY_EXIT";
export type Side = "SERVER" | "RECEIVER";
export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type PositionType = "BACK" | "LAY" | "NONE";
export type PointWinner = "SERVER" | "RECEIVER";
export type HedgeTrigger = "TREND_BREAK" | "ADVERSE_MOVE" | "DEUCE_LOSS" | "MANUAL";

export interface ScoreState {
  server_points: number;
  receiver_points: number;
  server_games: number;
  receiver_games: number;
  server_sets: number;
  receiver_sets: number;
  is_tiebreak: boolean;
  point_display: string;
  game_state_key: string;
}

export interface MatchInfo {
  match_id: string;
  player1_name: string;
  player2_name: string;
  surface: string;
  best_of: number;
  server: number;
  tournament: string;
}

export interface MatchState {
  info: MatchInfo;
  score: ScoreState;
  last_point_winner: PointWinner | null;
  is_live: boolean;
  timestamp: number;
}

export interface ProbabilityState {
  true_probability: number;
  market_probability: number;
  edge_pct: number;
  ev: number;
  server_hold_prob: number;
  break_prob: number;
  match_win_prob_p1: number;
  confidence: number;
}

export interface TradingSignal {
  action: Action;
  side: Side;
  confidence: number;
  bet_size_pct: number;
  reason: string;
  ev: number;
  edge: number;
  timestamp: number;
}

export interface HedgeSignal {
  should_hedge: boolean;
  hedge_size: number;
  trigger: HedgeTrigger | null;
  expected_neutral_pnl: number;
  reason: string;
  urgency: string;
}

export interface TradeEntry {
  id: string;
  account: string;
  side: Side;
  position_type: PositionType;
  entry_odds: number;
  current_odds: number;
  stake: number;
  pnl: number;
  state_at_entry: string;
  timestamp: number;
  is_open: boolean;
}

export interface PositionState {
  current_type: PositionType;
  entry_odds: number;
  current_odds: number;
  pnl: number;
  stake: number;
  account_a_balance: number;
  account_b_balance: number;
  combined_exposure: number;
  open_trades: TradeEntry[];
}

export interface NextAction {
  if_point_won: string;
  if_point_lost: string;
  hedge_size_if_lost: number;
}

export interface DeuceLoopState {
  is_active: boolean;
  cycle_count: number;
  max_cycles: number;
  net_profit: number;
  current_cycle_entry_odds: number;
  cycles: Array<{
    cycle: number;
    entry_odds: number;
    edge: number;
    status: string;
    pnl: number;
  }>;
}

export interface RiskState {
  risk_level: RiskLevel;
  current_exposure_pct: number;
  max_per_trade_pct: number;
  max_per_match_pct: number;
  consecutive_losses: number;
  is_trading_enabled: boolean;
  stop_reason: string | null;
}

export interface ExecutionResult {
  requested_price: number;
  fill_price: number;
  slippage_ticks: number;
  delay_ms: number;
  filled: boolean;
  fill_probability: number;
}

export interface StatePerformanceRow {
  state: string;
  trades: number;
  wins: number;
  win_rate: number;
  avg_ev: number;
  total_pnl: number;
}

/* ── Live Stats ──────────────────────────────────────────────────────────── */

export interface PlayerStatsSnapshot {
  serve_points_played: number;
  first_serves_in: number;
  first_serve_points_won: number;
  second_serve_points_won: number;
  aces: number;
  double_faults: number;
  serve_games: number;
  break_points_faced: number;
  break_points_saved: number;
  return_points_played: number;
  return_points_won: number;
  winners: number;
  unforced_errors: number;
  total_points_won: number;
  total_points_played: number;
  first_serve_pct: number;
  first_serve_win_pct: number;
  second_serve_win_pct: number;
  serve_points_won_pct: number;
  return_points_won_pct: number;
  break_point_save_pct: number;
  aces_per_game: number;
  df_per_game: number;
  win_rate: number;
}

export interface LiveStatsSnapshot {
  points_played: number;
  player1: PlayerStatsSnapshot;
  player2: PlayerStatsSnapshot;
}

export interface ModelContributionData {
  name: string;
  probability: number;
  weight: number;
  available: boolean;
}

export interface EnsembleData {
  models: ModelContributionData[];
  blend_weight: number;
  blended_stats: Record<string, Record<string, number>>;
}

export interface PointStatsInput {
  is_ace?: boolean;
  is_double_fault?: boolean;
  is_first_serve_in?: boolean;
  is_winner?: boolean;
  is_unforced_error?: boolean;
  is_break_point?: boolean;
}

/* ── Aggregated Frame ────────────────────────────────────────────────────── */

export interface TradeBoxFrame {
  match: MatchState;
  probability: ProbabilityState;
  signal: TradingSignal;
  hedge: HedgeSignal;
  position: PositionState;
  next_action: NextAction;
  deuce_loop: DeuceLoopState;
  risk: RiskState;
  execution: ExecutionResult | null;
  state_performance: StatePerformanceRow[];
  trade_log: TradeEntry[];
  live_stats: LiveStatsSnapshot | null;
  ensemble: EnsembleData | null;
  server_ts: number;
}

export interface MatchListItem {
  match_id: string;
  label: string;
  is_live: boolean;
  score_summary: string;
}

export interface MatchSetupPayload {
  player1_name: string;
  player2_name: string;
  surface: string;
  best_of: number;
  tournament: string;
  tournament_level?: string;
  initial_server: number;
  p1_serve_pct: number;
  p2_serve_pct: number;
  p1_return_pct: number;
  p2_return_pct: number;
  p1_rank: number;
  p2_rank: number;
  p1_ranking_points?: number;
  p2_ranking_points?: number;
  p1_first_serve_pct?: number;
  p2_first_serve_pct?: number;
  p1_first_serve_win_pct?: number;
  p2_first_serve_win_pct?: number;
  p1_second_serve_win_pct?: number;
  p2_second_serve_win_pct?: number;
  p1_bp_save_pct?: number;
  p2_bp_save_pct?: number;
  p1_win_rate?: number;
  p2_win_rate?: number;
  bankroll: number;
}
