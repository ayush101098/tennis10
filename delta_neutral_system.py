"""
🎾 TENNIS DELTA-NEUTRAL BETTING SYSTEM
=====================================
Exact IF-THEN Rulebook Implementation
Production-ready betting automation

Run: python delta_neutral_system.py
"""

import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import numpy as np
from scipy.special import comb

# ==================== LOGGER CONFIG ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeltaNeutral")

# ==================== SYSTEM PRIMITIVES ====================
class SystemConstants:
    """Fixed constants - never change mid-match"""
    P_EQ_BREAK = 0.25                    # Equilibrium break probability
    O_ENTRY_MIN = 2.80                   # Minimum break odds for entry
    O_ENTRY_MAX = 3.40                   # Maximum break odds for entry
    MAX_HEDGE = 1.0                      # 100% hedge
    PARTIAL_HEDGE = 0.5                  # 50% hedge
    MIN_HOLD_ODDS = 1.25                 # Threshold for server dominance
    BP_ODDS_INCREASE_THRESHOLD = 0.40    # 40% increase for partial hedge
    ODDS_EXPLOSION_THRESHOLD = 8.0       # Emergency exit threshold
    MIN_EDGE_THRESHOLD = 0.02             # Minimum 2% edge for value entry
    MIN_EXIT_EDGE = -0.01                # Exit if edge drops below -1%

# ==================== TRUE PROBABILITY ENGINE ====================

class ProbabilityEngine:
    """Calculate true probability from player statistics"""
    
    @staticmethod
    def p_game_from_points(server_pts: int, returner_pts: int, p_point: float) -> float:
        """Calculate probability that server wins current game from given points."""
        if server_pts >= 4 and server_pts >= returner_pts + 2:
            return 1.0
        if returner_pts >= 4 and returner_pts >= server_pts + 2:
            return 0.0
        
        if server_pts >= 3 and returner_pts >= 3:
            p_d = (p_point ** 2) / (p_point ** 2 + (1 - p_point) ** 2)
            if server_pts == returner_pts:
                return p_d
            elif server_pts > returner_pts:
                return p_point + (1 - p_point) * p_d
            else:
                return p_point * p_d
        
        cache = {}
        def prob_from(s, r):
            if (s, r) in cache:
                return cache[(s, r)]
            if s == 4 and r <= 2:
                return 1.0
            if r == 4 and s <= 2:
                return 0.0
            result = p_point * prob_from(min(s+1, 4), r) + (1-p_point) * prob_from(s, min(r+1, 4))
            cache[(s, r)] = result
            return result
        return prob_from(server_pts, returner_pts)
    
    @staticmethod
    def p_win_game(p_point: float) -> float:
        """Win probability for a single game."""
        p, q = p_point, 1 - p_point
        p_before_deuce = p**4 * (1 + 4*q + 10*q**2)
        p_deuce = comb(6, 3, exact=True) * p**3 * q**3 * (p**2 / (p**2 + q**2))
        return p_before_deuce + p_deuce
    
    @classmethod
    def estimate_true_probability(cls,
                                 p1_serve_pct: float,
                                 p1_return_pct: float,
                                 p2_serve_pct: float,
                                 p2_return_pct: float,
                                 p1_rank: int = 100,
                                 p2_rank: int = 100,
                                 p1_momentum: float = 0.0,
                                 p2_momentum: float = 0.0,
                                 server_starting: bool = True) -> float:
        """
        Estimate true probability for BREAK on next game
        
        Returns probability that server's serve will be broken
        """
        # Normalize percentages to 0-1
        p1_serve = p1_serve_pct / 100.0
        p1_return = p1_return_pct / 100.0
        p2_serve = p2_serve_pct / 100.0
        p2_return = p2_return_pct / 100.0
        
        if server_starting:
            # Server is Player 1 (typically)
            # Break prob = 1 - probability player 1 holds
            p_hold = cls.p_win_game(p1_serve)
            p_break = 1 - p_hold
            
            # Adjust for returner skill (p2_return)
            # Higher return % = higher break probability
            adjustment = (p2_return - 0.35) * 0.2  # -20% to +20%
            p_break = np.clip(p_break + adjustment, 0.15, 0.50)
        else:
            # Server is Player 2
            p_hold = cls.p_win_game(p2_serve)
            p_break = 1 - p_hold
            
            # Adjust for returner skill (p1_return)
            adjustment = (p1_return - 0.35) * 0.2
            p_break = np.clip(p_break + adjustment, 0.15, 0.50)
        
        # Ranking adjustment
        if p1_rank != p2_rank:
            if server_starting:
                rank_favors_server = p1_rank < p2_rank
                if rank_favors_server:
                    p_break *= 0.95  # Server stronger - lower break prob
                else:
                    p_break *= 1.05  # Server weaker - higher break prob
            else:
                rank_favors_server = p2_rank < p1_rank
                if rank_favors_server:
                    p_break *= 0.95
                else:
                    p_break *= 1.05
        
        # Momentum adjustment
        momentum_delta = p1_momentum - p2_momentum if server_starting else p2_momentum - p1_momentum
        if momentum_delta != 0:
            p_break += (momentum_delta * 0.005)  # ±0.5% per momentum point
        
        return np.clip(p_break, 0.10, 0.55)
    
    @staticmethod
    def calculate_value_edge(true_prob: float, odds: float) -> float:
        """
        Calculate expected value edge (EV%)
        
        EV = (true_prob × (odds - 1)) - (1 - true_prob)
        Positive EV = value bet
        """
        implied_prob = 1.0 / odds
        ev = (true_prob * (odds - 1)) - (1 - true_prob)
        return ev
    
    @staticmethod
    def calculate_implied_probability(odds: float) -> float:
        """Calculate implied probability from odds"""
        return 1.0 / odds

# ==================== ENUMS ====================

class GameState(Enum):
    """Game state classification (S0-S8)"""
    S0 = "0-0 (start)"
    S1 = "15-0 or 0-15"
    S2 = "30-15 or 15-30"
    S3 = "30-30 (deuce area)"
    S4 = "Break Point (30-40, 15-40, Ad-Out)"
    S5 = "Deuce (40-40, Ad-In, Ad-Out)"
    S6 = "Server wins 3 straight"
    S7 = "Break Point Saved"
    S8 = "Break Occurs"

class PositionStatus(Enum):
    """Position status tracking"""
    IDLE = "No active position"
    LONG_BREAK = "Long break exposure"
    NEUTRAL = "Delta-neutral (hedged)"
    PARTIAL_HEDGE = "50% hedged"

class Signal(Enum):
    """System signals"""
    ENTRY = "Place initial break bet"
    FULL_HEDGE = "100% hedge"
    PARTIAL_HEDGE = "50% hedge"
    HOLD = "No action"
    EXIT = "Close positions"
    EMERGENCY_EXIT = "Risk control exit"

# ==================== DATA CLASSES ====================

@dataclass
class Position:
    """Account A: Initial break bet"""
    stake: float                    # S₁
    odds: float                     # O₁
    entry_state: GameState         # State when entered
    entry_time: datetime           # When entered
    active: bool = True
    
    def potential_pnl(self, new_odds: float, break_occurred: bool = False) -> float:
        """Calculate unrealized PnL"""
        if break_occurred:
            return (self.odds - 1) * self.stake  # Win
        # Mark-to-market
        fair_value = (self.odds - 1) * self.stake * (1 / new_odds)
        return fair_value - self.stake

@dataclass
class Hedge:
    """Account B: Risk mitigation bet"""
    stake: float                    # S₂
    odds: float                     # O₂
    hedge_type: str                # "FULL" or "PARTIAL"
    trigger_state: GameState       # State that triggered hedge
    trigger_time: datetime
    active: bool = True
    
    def potential_pnl(self, new_odds: float, hold_occurred: bool = False) -> float:
        """Calculate unrealized PnL"""
        if hold_occurred:
            return (self.odds - 1) * self.stake  # Win if hold
        fair_value = (self.odds - 1) * self.stake * (1 / new_odds)
        return fair_value - self.stake

@dataclass
class GameScore:
    """Real-time game score"""
    server_points: int = 0
    returner_points: int = 0
    server_games: int = 0
    returner_games: int = 0
    server_sets: int = 0
    returner_sets: int = 0
    
    def get_point_string(self) -> str:
        """Display points as tennis score"""
        if self.server_points >= 3 and self.returner_points >= 3:
            if self.server_points == self.returner_points:
                return "40-40 (Deuce)"
            elif self.server_points > self.returner_points:
                return "Ad-In (Server)"
            else:
                return "Ad-Out (Returner)"
        
        score_map = {0: "0", 1: "15", 2: "30", 3: "40"}
        s = score_map.get(min(self.server_points, 3), "40")
        r = score_map.get(min(self.returner_points, 3), "40")
        return f"{s}-{r}"

@dataclass
class BettingState:
    """Complete betting system state"""
    position_a: Optional[Position] = None      # Account A (aggressor)
    position_b: Optional[Hedge] = None         # Account B (stabilizer)
    position_status: PositionStatus = PositionStatus.IDLE
    delta: float = 0.0                         # Position delta
    game_score: GameScore = field(default_factory=GameScore)
    current_state: GameState = GameState.S0
    entry_count: int = 0                       # Entries in this set
    break_occurred: bool = False
    pnl_history: List[Dict] = field(default_factory=list)
    # NEW: True probability tracking
    true_prob_break: float = 0.0              # True prob of break
    implied_prob_break: float = 0.0           # Implied from odds
    current_edge: float = 0.0                 # Current EV edge %
    entry_edge: float = 0.0                   # Edge at entry
    
    def get_delta_summary(self) -> str:
        """Explain delta status"""
        delta_meanings = {
            1.0: "Full break exposure (aggressive)",
            0.5: "Partial hedge (balanced)",
            0.0: "Delta-neutral (hedged)",
            -0.5: "Reverse hedge (rare)"
        }
        return delta_meanings.get(self.delta, "Unknown")

# ==================== GAME STATE CLASSIFIER ====================

class GameStateClassifier:
    """
    Classify game state from score
    Rule: Exact state detection after each point
    """
    
    @staticmethod
    def classify(score: GameScore) -> GameState:
        """Classify current game state (S0-S8)"""
        s_pts = score.server_points
        r_pts = score.returner_points
        
        # Check for break (game won by returner)
        if score.returner_games > score.server_games and \
           score.returner_games >= 1 and \
           (score.returner_games - score.server_games) == 1:
            # A break occurred on last game
            return GameState.S8
        
        # S4: Break Point
        if (s_pts == 3 and r_pts >= 3) or (s_pts >= 3 and r_pts == 3):
            if r_pts > s_pts:  # Ad-Out or 15-40 or 30-40
                return GameState.S4
        
        # S5: Deuce situation
        if s_pts >= 3 and r_pts >= 3:
            return GameState.S5
        
        # S6: Server won 3 straight points (15-0, 30-0, or 40-0)
        if s_pts >= 3 and r_pts == 0:
            return GameState.S6
        
        # S3: 30-30
        if s_pts == 2 and r_pts == 2:
            return GameState.S3
        
        # S2: 30-15 or 15-30
        if (s_pts == 2 and r_pts == 1) or (s_pts == 1 and r_pts == 2):
            return GameState.S2
        
        # S1: 15-0 or 0-15
        if (s_pts == 1 and r_pts == 0) or (s_pts == 0 and r_pts == 1):
            return GameState.S1
        
        # S0: Start of game
        if s_pts == 0 and r_pts == 0:
            return GameState.S0
        
        return GameState.S0

# ==================== ENTRY RULE ENGINE ====================

class EntryRuleEngine:
    """
    Rule E1: Initial Long Break Entry
    
    IF server starts game AND break odds ∈ [2.8, 3.4] AND edge ≥ 2% AND no break in set
    THEN place break bet, set delta = +1
    
    Enhanced: Now validates value bet (true_prob vs implied_prob)
    """
    
    @staticmethod
    def check_entry(betting_state: BettingState, break_odds: float, 
                   server_starting: bool = True,
                   p1_serve_pct: float = 65.0,
                   p1_return_pct: float = 35.0,
                   p2_serve_pct: float = 62.0,
                   p2_return_pct: float = 38.0,
                   p1_rank: int = 100,
                   p2_rank: int = 100,
                   p1_momentum: float = 0.0,
                   p2_momentum: float = 0.0) -> Signal:
        """
        E1 Entry Logic with value validation
        """
        # Condition 1: Game is starting
        if betting_state.current_state != GameState.S0:
            logger.info("❌ E1 SKIP: Game not at S0, score not 0-0")
            return Signal.HOLD
        
        # Condition 2: Break odds in valid range
        if not (SystemConstants.O_ENTRY_MIN <= break_odds <= SystemConstants.O_ENTRY_MAX):
            logger.info(f"❌ E1 SKIP: Break odds {break_odds:.2f} outside [{SystemConstants.O_ENTRY_MIN}, {SystemConstants.O_ENTRY_MAX}]")
            return Signal.HOLD
        
        # Condition 3: No existing position in this game
        if betting_state.position_a is not None and betting_state.position_a.active:
            logger.info("❌ E1 SKIP: Already have active break position (no re-entry)")
            return Signal.HOLD
        
        # Condition 4: CALCULATE TRUE PROBABILITY & VALIDATE EDGE ✅ NEW
        true_prob = ProbabilityEngine.estimate_true_probability(
            p1_serve_pct=p1_serve_pct,
            p1_return_pct=p1_return_pct,
            p2_serve_pct=p2_serve_pct,
            p2_return_pct=p2_return_pct,
            p1_rank=p1_rank,
            p2_rank=p2_rank,
            p1_momentum=p1_momentum,
            p2_momentum=p2_momentum,
            server_starting=server_starting
        )
        
        implied_prob = ProbabilityEngine.calculate_implied_probability(break_odds)
        edge = ProbabilityEngine.calculate_value_edge(true_prob, break_odds)
        
        # Store for tracking
        betting_state.true_prob_break = true_prob
        betting_state.implied_prob_break = implied_prob
        betting_state.current_edge = edge
        betting_state.entry_edge = edge
        
        logger.info(f"📊 Probability Analysis:")
        logger.info(f"   True Probability (Break): {true_prob:.1%}")
        logger.info(f"   Implied (Odds {break_odds:.2f}): {implied_prob:.1%}")
        logger.info(f"   Edge: {edge:+.2%}")
        
        # Validate edge is positive and significant
        if edge < SystemConstants.MIN_EDGE_THRESHOLD:
            logger.info(f"❌ E1 SKIP: Edge {edge:+.2%} < minimum {SystemConstants.MIN_EDGE_THRESHOLD:+.2%}")
            return Signal.HOLD
        
        logger.info(f"✅ E1 TRIGGERED: All conditions met! Edge {edge:+.2%} ≥ {SystemConstants.MIN_EDGE_THRESHOLD:+.2%}")
        return Signal.ENTRY

# ==================== HEDGE RULE ENGINE ====================

class HedgeRuleEngine:
    """
    Rules H1, H2, H3: Hedge Trigger Logic
    
    H1: Server dominance (3 straight points, hold odds ≤ 1.25)
    H2: Deuce reached (100% hedge)
    H3: Break point missed (≥40% odds increase, 50% hedge)
    """
    
    @staticmethod
    def check_hedge(betting_state: BettingState, 
                   current_hold_odds: float,
                   current_break_odds: float,
                   entry_break_odds: float) -> Signal:
        """Check all hedge triggers H1-H3"""
        
        # No position to hedge
        if betting_state.position_a is None or not betting_state.position_a.active:
            return Signal.HOLD
        
        # H1: Server Dominance Hedge
        # IF State = S6 (server wins 3 straight) AND hold odds ≤ 1.25
        if betting_state.current_state == GameState.S6:
            if current_hold_odds <= SystemConstants.MIN_HOLD_ODDS:
                logger.info(f"✅ H1 TRIGGERED: Server dominance (S6) + hold odds {current_hold_odds:.2f} ≤ 1.25")
                return Signal.FULL_HEDGE
            else:
                logger.info(f"⚠️  H1 WAIT: S6 but hold odds {current_hold_odds:.2f} > 1.25")
        
        # H2: Deuce Neutralization
        # IF State = S5 (Deuce) AND previously held break exposure
        if betting_state.current_state == GameState.S5:
            logger.info("✅ H2 TRIGGERED: Deuce reached (S5) - highest EV hedge state")
            return Signal.FULL_HEDGE
        
        # H3: Missed Break Point Partial Hedge
        # IF State = S7 (break saved) AND break odds increased ≥40%
        if betting_state.current_state == GameState.S7:
            odds_increase = (current_break_odds - entry_break_odds) / entry_break_odds
            if odds_increase >= SystemConstants.BP_ODDS_INCREASE_THRESHOLD:
                logger.info(f"✅ H3 TRIGGERED: BP saved + odds jumped {odds_increase*100:.1f}% (≥40%)")
                return Signal.PARTIAL_HEDGE
            else:
                logger.info(f"⚠️  H3 WAIT: BP saved but odds only +{odds_increase*100:.1f}% (<40%)")
        
        return Signal.HOLD

# ==================== POSITION MANAGER ====================

class PositionManager:
    """
    Execute betting signals with strict sizing formula:
    
    Full Hedge: S₂ = S₁ × O₁ / O₂
    Partial Hedge: S₂ = 0.5 × S₁ × O₁ / O₂
    """
    
    @staticmethod
    def calculate_hedge_stake(position_stake: float, 
                            position_odds: float,
                            hedge_odds: float,
                            hedge_type: str = "FULL") -> float:
        """
        Calculate hedge stake using exact formula
        
        Full: S₂ = S₁ × O₁ / O₂
        Partial: S₂ = 0.5 × S₁ × O₁ / O₂
        """
        base = (position_stake * position_odds) / hedge_odds
        
        if hedge_type == "FULL":
            return base
        elif hedge_type == "PARTIAL":
            return 0.5 * base
        else:
            raise ValueError(f"Unknown hedge type: {hedge_type}")
    
    @staticmethod
    def execute_entry(betting_state: BettingState, 
                     stake: float, odds: float) -> BettingState:
        """E1: Place initial long break entry"""
        betting_state.position_a = Position(
            stake=stake,
            odds=odds,
            entry_state=betting_state.current_state,
            entry_time=datetime.now()
        )
        betting_state.position_status = PositionStatus.LONG_BREAK
        betting_state.delta = 1.0
        betting_state.entry_count += 1
        
        logger.info(f"📊 ENTRY (A1): Stake ${stake:.2f} @ {odds:.2f} odds on BREAK")
        logger.info(f"   Delta: 0 → +1.0 (aggressive long break)")
        
        return betting_state
    
    @staticmethod
    def execute_full_hedge(betting_state: BettingState,
                          hedge_odds: float,
                          trigger_state: GameState) -> BettingState:
        """H1/H2: Full 100% hedge"""
        if betting_state.position_a is None:
            return betting_state
        
        hedge_stake = PositionManager.calculate_hedge_stake(
            betting_state.position_a.stake,
            betting_state.position_a.odds,
            hedge_odds,
            hedge_type="FULL"
        )
        
        betting_state.position_b = Hedge(
            stake=hedge_stake,
            odds=hedge_odds,
            hedge_type="FULL",
            trigger_state=trigger_state,
            trigger_time=datetime.now()
        )
        betting_state.position_status = PositionStatus.NEUTRAL
        betting_state.delta = 0.0
        
        logger.info(f"🛡️  HEDGE (B1): Stake ${hedge_stake:.2f} @ {hedge_odds:.2f} odds on HOLD")
        logger.info(f"   Delta: +1.0 → 0.0 (neutral, fully hedged)")
        
        return betting_state
    
    @staticmethod
    def execute_partial_hedge(betting_state: BettingState,
                             hedge_odds: float,
                             trigger_state: GameState) -> BettingState:
        """H3: 50% hedge"""
        if betting_state.position_a is None:
            return betting_state
        
        hedge_stake = PositionManager.calculate_hedge_stake(
            betting_state.position_a.stake,
            betting_state.position_a.odds,
            hedge_odds,
            hedge_type="PARTIAL"
        )
        
        betting_state.position_b = Hedge(
            stake=hedge_stake,
            odds=hedge_odds,
            hedge_type="PARTIAL",
            trigger_state=trigger_state,
            trigger_time=datetime.now()
        )
        betting_state.position_status = PositionStatus.PARTIAL_HEDGE
        betting_state.delta = 0.5
        
        logger.info(f"⚠️  PARTIAL HEDGE (B3): Stake ${hedge_stake:.2f} @ {hedge_odds:.2f} odds on HOLD")
        logger.info(f"   Delta: +1.0 → +0.5 (balanced, retains convexity)")
        
        return betting_state

# ==================== RISK CONTROLLER ====================

class RiskController:
    """
    Rule R1: Odds Explosion Emergency Exit
    
    IF break odds > 8.0 AND no break point occurred
    THEN hedge 100% immediately (lock loss)
    """
    
    @staticmethod
    def check_emergency_exit(betting_state: BettingState,
                            current_break_odds: float) -> Signal:
        """R1 Emergency exit logic"""
        
        if betting_state.position_a is None or not betting_state.position_a.active:
            return Signal.HOLD
        
        # Check if odds exploded
        if current_break_odds > SystemConstants.ODDS_EXPLOSION_THRESHOLD:
            # But make sure no break point yet
            if betting_state.current_state not in [GameState.S4, GameState.S8]:
                logger.warning(f"⚠️  R1 EMERGENCY: Break odds exploded to {current_break_odds:.2f} without BP!")
                return Signal.EMERGENCY_EXIT
        
        return Signal.HOLD

# ==================== PnL CALCULATOR ====================

class PnLCalculator:
    """
    Calculate P&L from positions
    
    Outcomes:
    1. Break occurs: Position A wins, Position B loses
    2. Hold occurs: Position A loses, Position B wins
    3. Mid-game exit: Mark-to-market both
    """
    
    @staticmethod
    def calculate_game_pnl(position_a: Optional[Position],
                          position_b: Optional[Hedge],
                          outcome: str) -> Dict:
        """
        Calculate P&L for game outcome
        
        outcome: "BREAK" | "HOLD" | "EXIT"
        """
        result = {
            "outcome": outcome,
            "pnl_a": 0.0,
            "pnl_b": 0.0,
            "total_pnl": 0.0,
            "roi": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        if outcome == "BREAK":
            # Position A (break) wins
            if position_a:
                result["pnl_a"] = (position_a.odds - 1) * position_a.stake
            # Position B (hold) loses
            if position_b:
                result["pnl_b"] = -position_b.stake
        
        elif outcome == "HOLD":
            # Position A loses
            if position_a:
                result["pnl_a"] = -position_a.stake
            # Position B wins
            if position_b:
                result["pnl_b"] = (position_b.odds - 1) * position_b.stake
        
        result["total_pnl"] = result["pnl_a"] + result["pnl_b"]
        
        # ROI calculation
        if position_a:
            total_staked = position_a.stake
            if position_b:
                total_staked += position_b.stake
            result["roi"] = (result["total_pnl"] / total_staked) * 100 if total_staked > 0 else 0
        
        return result

# ==================== MAIN SYSTEM ====================

class DeltaNeutralBettingSystem:
    """
    Complete Delta-Neutral Betting System
    
    Implements exact IF-THEN rulebook with:
    - Deterministic entry rules (E1) with TRUE PROBABILITY validation
    - Adaptive hedge triggers (H1, H2, H3)
    - Emergency risk control (R1)
    - Strict delta tracking
    - Value bet identification
    """
    
    def __init__(self):
        self.betting_state = BettingState()
        self.entry_engine = EntryRuleEngine()
        self.hedge_engine = HedgeRuleEngine()
        self.risk_controller = RiskController()
        self.position_manager = PositionManager()
        self.pnl_calc = PnLCalculator()
        
        # Player statistics storage ✅ NEW
        self.p1_serve_pct = 65.0
        self.p1_return_pct = 35.0
        self.p2_serve_pct = 62.0
        self.p2_return_pct = 38.0
        self.p1_rank = 100
        self.p2_rank = 100
        self.p1_momentum = 0.0
        self.p2_momentum = 0.0
        
        logger.info("🎾 Delta-Neutral Betting System initialized")
        logger.info(f"   P_eq: {SystemConstants.P_EQ_BREAK}")
        logger.info(f"   Entry odds range: [{SystemConstants.O_ENTRY_MIN}, {SystemConstants.O_ENTRY_MAX}]")
        logger.info(f"   Min edge threshold: {SystemConstants.MIN_EDGE_THRESHOLD:+.2%}")
    
    def set_player_stats(self, p1_serve_pct, p1_return_pct, p2_serve_pct, p2_return_pct,
                        p1_rank, p2_rank, p1_momentum=0.0, p2_momentum=0.0):
        """Update player statistics for true probability calculation"""
        self.p1_serve_pct = p1_serve_pct
        self.p1_return_pct = p1_return_pct
        self.p2_serve_pct = p2_serve_pct
        self.p2_return_pct = p2_return_pct
        self.p1_rank = p1_rank
        self.p2_rank = p2_rank
        self.p1_momentum = p1_momentum
        self.p2_momentum = p2_momentum
        logger.info(f"✅ Player stats updated:")
        logger.info(f"   P1: Serve {p1_serve_pct}%, Return {p1_return_pct}%, Rank {p1_rank}")
        logger.info(f"   P2: Serve {p2_serve_pct}%, Return {p2_return_pct}%, Rank {p2_rank}")
    
    def update_score(self, server_pts: int, returner_pts: int,
                    server_games: int, returner_games: int,
                    server_sets: int = 0, returner_sets: int = 0):
        """Update game score and reclassify state"""
        self.betting_state.game_score = GameScore(
            server_points=server_pts,
            returner_points=returner_pts,
            server_games=server_games,
            returner_games=returner_games,
            server_sets=server_sets,
            returner_sets=returner_sets
        )
        
        old_state = self.betting_state.current_state
        self.betting_state.current_state = GameStateClassifier.classify(
            self.betting_state.game_score
        )
        
        if old_state != self.betting_state.current_state:
            logger.info(f"📍 State changed: {old_state.value} → {self.betting_state.current_state.value}")
            logger.info(f"   Score: {self.betting_state.game_score.get_point_string()}")
    
    def process_odds(self, break_odds: float, hold_odds: float):
        """Process live odds and evaluate rules"""
        logger.info(f"\n📊 Market: Break {break_odds:.2f} | Hold {hold_odds:.2f}")
        
        # R1: Emergency exit check
        emergency = self.risk_controller.check_emergency_exit(
            self.betting_state, break_odds
        )
        if emergency == Signal.EMERGENCY_EXIT:
            logger.error("🚨 R1 EMERGENCY EXIT triggered!")
            return emergency
        
        # E1: Entry rule with TRUE PROBABILITY validation ✅ ENHANCED
        if self.betting_state.position_a is None:
            entry_signal = self.entry_engine.check_entry(
                self.betting_state, break_odds,
                server_starting=True,
                p1_serve_pct=self.p1_serve_pct,
                p1_return_pct=self.p1_return_pct,
                p2_serve_pct=self.p2_serve_pct,
                p2_return_pct=self.p2_return_pct,
                p1_rank=self.p1_rank,
                p2_rank=self.p2_rank,
                p1_momentum=self.p1_momentum,
                p2_momentum=self.p2_momentum
            )
            if entry_signal == Signal.ENTRY:
                # Execute entry with standard unit
                min_unit = 10  # $10 minimum unit
                self.betting_state = self.position_manager.execute_entry(
                    self.betting_state, 
                    stake=min_unit * 5,  # $50 standard
                    odds=break_odds
                )
                logger.info(f"   Δ Status: {self.betting_state.get_delta_summary()}")
                return entry_signal
        
        # H1-H3: Hedge rules
        hedge_signal = self.hedge_engine.check_hedge(
            self.betting_state,
            hold_odds,
            break_odds,
            self.betting_state.position_a.odds if self.betting_state.position_a else break_odds
        )
        
        if hedge_signal == Signal.FULL_HEDGE:
            self.betting_state = self.position_manager.execute_full_hedge(
                self.betting_state,
                hedge_odds=hold_odds,
                trigger_state=self.betting_state.current_state
            )
            logger.info(f"   Δ Status: {self.betting_state.get_delta_summary()}")
            return hedge_signal
        
        elif hedge_signal == Signal.PARTIAL_HEDGE:
            self.betting_state = self.position_manager.execute_partial_hedge(
                self.betting_state,
                hedge_odds=hold_odds,
                trigger_state=self.betting_state.current_state
            )
            logger.info(f"   Δ Status: {self.betting_state.get_delta_summary()}")
            return hedge_signal
        
        return Signal.HOLD
    
    def check_exit_opportunity(self, break_odds: float) -> Signal:
        """
        Check if we should exit position based on EDGE DETERIORATION
        
        If edge drops below MIN_EXIT_EDGE (-1%), exit the position
        """
        if self.betting_state.position_a is None or not self.betting_state.position_a.active:
            return Signal.HOLD
        
        # Calculate current edge with new odds
        true_prob = self.betting_state.true_prob_break
        current_edge = ProbabilityEngine.calculate_value_edge(true_prob, break_odds)
        self.betting_state.current_edge = current_edge
        
        # Compare to entry edge
        edge_change = current_edge - self.betting_state.entry_edge
        
        logger.info(f"📊 Edge Check:")
        logger.info(f"   Entry edge: {self.betting_state.entry_edge:+.2%}")
        logger.info(f"   Current edge: {current_edge:+.2%}")
        logger.info(f"   Change: {edge_change:+.2%}")
        
        # If edge turned negative and deteriorated significantly
        if current_edge < SystemConstants.MIN_EXIT_EDGE:
            logger.warning(f"⚠️  EXIT SIGNAL: Edge {current_edge:+.2%} < threshold {SystemConstants.MIN_EXIT_EDGE:+.2%}")
            return Signal.EXIT
        
        # If edge eroded by more than 50% from entry
        if edge_change < -0.01:  # More than 1% deterioration
            logger.info(f"⚠️  EDGE EROSION: {abs(edge_change):.2%} deterioration")
        
        return Signal.HOLD
    
    def execute_emergency_exit(self) -> Dict:
        """
        Execute emergency exit - close both positions immediately
        Mark-to-market settlement
        """
        logger.error("🚨 EXECUTING EMERGENCY EXIT")
        
        if self.betting_state.position_a:
            self.betting_state.position_a.active = False
        if self.betting_state.position_b:
            self.betting_state.position_b.active = False
        
        # Settlement: both positions closed
        result = {
            "outcome": "EMERGENCY_EXIT",
            "status": "Both positions closed",
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def settle_game(self, outcome: str) -> Dict:
        """
        Settle game outcome
        outcome: "BREAK" | "HOLD"
        """
        logger.info(f"\n✅ Game settled: {outcome}")
        
        pnl = self.pnl_calc.calculate_game_pnl(
            self.betting_state.position_a,
            self.betting_state.position_b,
            outcome
        )
        
        self.betting_state.pnl_history.append(pnl)
        
        logger.info(f"   Position A (Break):  {pnl['pnl_a']:+.2f}")
        logger.info(f"   Position B (Hold):   {pnl['pnl_b']:+.2f}")
        logger.info(f"   Total PnL:           {pnl['total_pnl']:+.2f}")
        logger.info(f"   ROI:                 {pnl['roi']:+.1f}%")
        
        # Reset for next game
        self.betting_state.position_a = None
        self.betting_state.position_b = None
        self.betting_state.position_status = PositionStatus.IDLE
        self.betting_state.delta = 0.0
        self.betting_state.game_score = GameScore()
        self.betting_state.current_state = GameState.S0
        
        return pnl
    
    def get_status_report(self) -> Dict:
        """Get complete system status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "current_state": self.betting_state.current_state.value,
            "game_score": self.betting_state.game_score.get_point_string(),
            "position_status": self.betting_state.position_status.value,
            "delta": self.betting_state.delta,
            "delta_explanation": self.betting_state.get_delta_summary(),
            "position_a": {
                "active": self.betting_state.position_a.active if self.betting_state.position_a else None,
                "stake": self.betting_state.position_a.stake if self.betting_state.position_a else None,
                "odds": self.betting_state.position_a.odds if self.betting_state.position_a else None,
            } if self.betting_state.position_a else None,
            "position_b": {
                "active": self.betting_state.position_b.active if self.betting_state.position_b else None,
                "stake": self.betting_state.position_b.stake if self.betting_state.position_b else None,
                "odds": self.betting_state.position_b.odds if self.betting_state.position_b else None,
                "type": self.betting_state.position_b.hedge_type if self.betting_state.position_b else None,
            } if self.betting_state.position_b else None,
            "pnl_history": self.betting_state.pnl_history,
            "cumulative_pnl": sum(p["total_pnl"] for p in self.betting_state.pnl_history),
        }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🎾 DELTA-NEUTRAL BETTING SYSTEM - LIVE DEMO")
    logger.info("=" * 80)
    
    system = DeltaNeutralBettingSystem()
    
    # Scenario 1: Entry + Full Hedge at Deuce
    logger.info("\n" + "="*80)
    logger.info("SCENARIO 1: Entry Break at 3.20 → Full Hedge at Deuce")
    logger.info("="*80)
    
    # Game starts 0-0
    system.update_score(0, 0, 0, 0)
    system.process_odds(break_odds=3.20, hold_odds=1.22)
    
    # Points played: 15-0
    system.update_score(1, 0, 0, 0)
    system.process_odds(break_odds=3.10, hold_odds=1.25)
    
    # 30-0
    system.update_score(2, 0, 0, 0)
    system.process_odds(break_odds=3.05, hold_odds=1.20)
    
    # 30-15
    system.update_score(2, 1, 0, 0)
    system.process_odds(break_odds=3.15, hold_odds=1.40)
    
    # 30-30
    system.update_score(2, 2, 0, 0)
    system.process_odds(break_odds=3.20, hold_odds=1.50)
    
    # Deuce 40-40
    system.update_score(3, 3, 0, 0)
    system.process_odds(break_odds=3.30, hold_odds=1.90)
    
    # Print status
    logger.info("\n📋 System Status Report:")
    status = system.get_status_report()
    logger.info(json.dumps(status, indent=2, default=str))
    
    # Settle: Hold occurs
    logger.info("\n" + "="*80)
    logger.info("SETTLEMENT: Server holds game")
    logger.info("="*80)
    pnl = system.settle_game("HOLD")
