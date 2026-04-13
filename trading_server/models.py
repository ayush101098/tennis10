"""
Pydantic models for the trading terminal.
All data contracts between engine ↔ API ↔ WebSocket ↔ frontend.
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid


# ─── Enums ────────────────────────────────────────────────────────────────────

class Action(str, Enum):
    ENTER = "ENTER"
    HOLD = "HOLD"
    HEDGE = "HEDGE"
    SKIP = "SKIP"
    EXIT = "EXIT"
    SCALP = "SCALP"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


class Side(str, Enum):
    SERVER = "SERVER"
    RECEIVER = "RECEIVER"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PositionType(str, Enum):
    BACK = "BACK"
    LAY = "LAY"
    NONE = "NONE"


class PointWinner(str, Enum):
    SERVER = "SERVER"
    RECEIVER = "RECEIVER"


class HedgeTrigger(str, Enum):
    TREND_BREAK = "TREND_BREAK"
    ADVERSE_MOVE = "ADVERSE_MOVE"
    DEUCE_LOSS = "DEUCE_LOSS"
    MANUAL = "MANUAL"


# ─── Match State ──────────────────────────────────────────────────────────────

class ScoreState(BaseModel):
    """Point-level score."""
    model_config = {"frozen": False}

    server_points: int = 0
    receiver_points: int = 0
    server_games: int = 0
    receiver_games: int = 0
    server_sets: int = 0
    receiver_sets: int = 0
    is_tiebreak: bool = False
    point_display: str = "0-0"
    game_state_key: str = "0-0"

    def model_post_init(self, __context: object) -> None:
        self._recompute()

    def _recompute(self) -> None:
        # point_display
        if self.is_tiebreak:
            self.point_display = f"{self.server_points}-{self.receiver_points}"
        elif self.server_points >= 3 and self.receiver_points >= 3:
            if self.server_points == self.receiver_points:
                self.point_display = "DEUCE"
            elif self.server_points > self.receiver_points:
                self.point_display = "AD-IN"
            else:
                self.point_display = "AD-OUT"
        else:
            m = {0: "0", 1: "15", 2: "30", 3: "40"}
            self.point_display = f"{m.get(self.server_points, '40')}-{m.get(self.receiver_points, '40')}"
        # game_state_key
        if self.is_tiebreak:
            self.game_state_key = "TIEBREAK"
        else:
            sp, rp = self.server_points, self.receiver_points
            if sp >= 3 and rp >= 3:
                if sp == rp:
                    self.game_state_key = "DEUCE"
                elif sp > rp:
                    self.game_state_key = "AD-IN"
                else:
                    self.game_state_key = "AD-OUT"
            else:
                m = {0: "0", 1: "15", 2: "30", 3: "40"}
                self.game_state_key = f"{m.get(sp, '40')}-{m.get(rp, '40')}"


class MatchInfo(BaseModel):
    match_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    player1_name: str = ""
    player2_name: str = ""
    surface: str = "Hard"
    best_of: int = 3
    server: int = 1  # 1 = player1, 2 = player2
    tournament: str = ""


class MatchState(BaseModel):
    info: MatchInfo = Field(default_factory=MatchInfo)
    score: ScoreState = Field(default_factory=ScoreState)
    last_point_winner: Optional[PointWinner] = None
    is_live: bool = False
    timestamp: float = 0.0


# ─── Probability ──────────────────────────────────────────────────────────────

class ProbabilityState(BaseModel):
    true_probability: float = 0.5       # model
    market_probability: float = 0.5     # from odds
    edge_pct: float = 0.0              # true_p - market_p
    ev: float = 0.0                    # (true_p * odds) - 1
    server_hold_prob: float = 0.65
    break_prob: float = 0.35
    match_win_prob_p1: float = 0.5
    confidence: int = 50               # 0-100


# ─── Trading Signal ──────────────────────────────────────────────────────────

class TradingSignal(BaseModel):
    action: Action = Action.SKIP
    side: Side = Side.SERVER
    confidence: int = 0                 # 0-100
    bet_size_pct: float = 0.0          # % of bankroll
    reason: str = ""
    ev: float = 0.0
    edge: float = 0.0
    timestamp: float = 0.0


# ─── Hedge Signal ─────────────────────────────────────────────────────────────

class HedgeSignal(BaseModel):
    should_hedge: bool = False
    hedge_size: float = 0.0
    trigger: Optional[HedgeTrigger] = None
    expected_neutral_pnl: float = 0.0
    reason: str = ""
    urgency: str = "LOW"               # LOW / HIGH / IMMEDIATE


# ─── Position ─────────────────────────────────────────────────────────────────

class TradeEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    account: str = "A"                 # A=entries, B=hedges
    side: Side = Side.SERVER
    position_type: PositionType = PositionType.BACK
    entry_odds: float = 0.0
    current_odds: float = 0.0
    stake: float = 0.0
    pnl: float = 0.0
    state_at_entry: str = ""
    timestamp: float = 0.0
    is_open: bool = True


class PositionState(BaseModel):
    current_type: PositionType = PositionType.NONE
    entry_odds: float = 0.0
    current_odds: float = 0.0
    pnl: float = 0.0
    stake: float = 0.0
    account_a_balance: float = 10000.0
    account_b_balance: float = 10000.0
    combined_exposure: float = 0.0
    open_trades: List[TradeEntry] = Field(default_factory=list)


# ─── Next Action ──────────────────────────────────────────────────────────────

class NextAction(BaseModel):
    if_point_won: str = "HOLD"
    if_point_lost: str = "HOLD"
    hedge_size_if_lost: float = 0.0


# ─── Deuce Loop ───────────────────────────────────────────────────────────────

class DeuceLoopState(BaseModel):
    is_active: bool = False
    cycle_count: int = 0
    max_cycles: int = 3
    net_profit: float = 0.0
    current_cycle_entry_odds: float = 0.0
    cycles: List[dict] = Field(default_factory=list)


# ─── Risk ─────────────────────────────────────────────────────────────────────

class RiskState(BaseModel):
    risk_level: RiskLevel = RiskLevel.LOW
    current_exposure_pct: float = 0.0
    max_per_trade_pct: float = 1.0
    max_per_match_pct: float = 10.0
    consecutive_losses: int = 0
    is_trading_enabled: bool = True
    stop_reason: Optional[str] = None


# ─── Execution ────────────────────────────────────────────────────────────────

class ExecutionResult(BaseModel):
    requested_price: float = 0.0
    fill_price: float = 0.0
    slippage_ticks: int = 0
    delay_ms: int = 0
    filled: bool = True
    fill_probability: float = 1.0


# ─── State Performance ───────────────────────────────────────────────────────

class StatePerformanceRow(BaseModel):
    state: str
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    avg_ev: float = 0.0
    total_pnl: float = 0.0


# ─── Live Stats ───────────────────────────────────────────────────────────────

class PlayerStatsSnapshot(BaseModel):
    """Serialisable snapshot of one player's live stats."""
    serve_points_played: int = 0
    first_serves_in: int = 0
    first_serve_points_won: int = 0
    second_serve_points_won: int = 0
    aces: int = 0
    double_faults: int = 0
    serve_games: int = 0
    break_points_faced: int = 0
    break_points_saved: int = 0
    return_points_played: int = 0
    return_points_won: int = 0
    winners: int = 0
    unforced_errors: int = 0
    total_points_won: int = 0
    total_points_played: int = 0
    # derived
    first_serve_pct: float = 0.0
    first_serve_win_pct: float = 0.0
    second_serve_win_pct: float = 0.0
    serve_points_won_pct: float = 0.0
    return_points_won_pct: float = 0.0
    break_point_save_pct: float = 0.0
    aces_per_game: float = 0.0
    df_per_game: float = 0.0
    win_rate: float = 0.0


class LiveStatsSnapshot(BaseModel):
    """Both players' live stats in one object."""
    points_played: int = 0
    player1: PlayerStatsSnapshot = Field(default_factory=PlayerStatsSnapshot)
    player2: PlayerStatsSnapshot = Field(default_factory=PlayerStatsSnapshot)


class ModelContributionData(BaseModel):
    """One ML/Markov model's contribution to the ensemble."""
    name: str = ""
    probability: float = 0.5
    weight: float = 0.0
    available: bool = False


class EnsembleData(BaseModel):
    """Ensemble breakdown pushed to the frontend."""
    models: List[ModelContributionData] = Field(default_factory=list)
    blend_weight: float = 0.0  # 0 = all career, 1 = all live data
    blended_stats: dict = Field(default_factory=dict)


# ─── Aggregated Frame (what the WebSocket pushes) ────────────────────────────

class TradeBoxFrame(BaseModel):
    """Single frame pushed to UI every tick."""
    match: MatchState
    probability: ProbabilityState
    signal: TradingSignal
    hedge: HedgeSignal
    position: PositionState
    next_action: NextAction
    deuce_loop: DeuceLoopState
    risk: RiskState
    execution: Optional[ExecutionResult] = None
    state_performance: List[StatePerformanceRow] = Field(default_factory=list)
    trade_log: List[TradeEntry] = Field(default_factory=list)
    live_stats: Optional[LiveStatsSnapshot] = None
    ensemble: Optional[EnsembleData] = None
    server_ts: float = 0.0


# ─── REST payloads ────────────────────────────────────────────────────────────

class MatchSetupRequest(BaseModel):
    player1_name: str
    player2_name: str
    surface: str = "Hard"
    best_of: int = 3
    tournament: str = ""
    tournament_level: str = ""       # "G" = Grand Slam, "M" = Masters
    initial_server: int = 1
    p1_serve_pct: float = 63.0
    p2_serve_pct: float = 63.0
    p1_return_pct: float = 35.0
    p2_return_pct: float = 35.0
    p1_rank: int = 50
    p2_rank: int = 50
    p1_ranking_points: int = 1000
    p2_ranking_points: int = 1000
    p1_first_serve_pct: float = 62.0
    p2_first_serve_pct: float = 62.0
    p1_first_serve_win_pct: float = 70.0
    p2_first_serve_win_pct: float = 70.0
    p1_second_serve_win_pct: float = 50.0
    p2_second_serve_win_pct: float = 50.0
    p1_bp_save_pct: float = 60.0
    p2_bp_save_pct: float = 60.0
    p1_win_rate: float = 50.0
    p2_win_rate: float = 50.0
    bankroll: float = 10000.0


class PointStatsInput(BaseModel):
    """Optional per-point stat annotations from the operator."""
    is_ace: bool = False
    is_double_fault: bool = False
    is_first_serve_in: bool = False
    is_winner: bool = False
    is_unforced_error: bool = False
    is_break_point: bool = False


class PointUpdateRequest(BaseModel):
    winner: PointWinner
    market_odds_server: float = 1.80
    market_odds_receiver: float = 2.10
    stats: Optional[PointStatsInput] = None


class OddsUpdateRequest(BaseModel):
    break_odds: float
    hold_odds: float


class MatchListItem(BaseModel):
    match_id: str
    label: str
    is_live: bool = False
    score_summary: str = ""
