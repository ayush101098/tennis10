"""
Trading Engine — entry decision logic.
Pure function: state + probability → signal.
"""

from __future__ import annotations
import time
from .models import (
    Action, Side, TradingSignal, ScoreState, ProbabilityState, RiskState,
)


class TradingEngine:
    """
    Stateless decision engine.

    Entry Rules (from spec):
        30-15 / 15-30  → EV > 3%  → ENTER
        30-30          → edge > 4% → ENTER (HIGH PRIORITY)
        DEUCE          → |edge| > 3% → SCALP
        else           → SKIP

    Bet sizing: Kelly fraction capped at max_per_trade_pct.
    """

    ENTRY_STATES = {"30-15", "15-30"}
    VOLATILE_STATE = "30-30"
    DEUCE_STATES = {"DEUCE", "AD-IN", "AD-OUT"}

    @classmethod
    def decide(
        cls,
        score: ScoreState,
        prob: ProbabilityState,
        risk: RiskState,
        current_odds_server: float = 1.80,
        current_odds_receiver: float = 2.10,
    ) -> TradingSignal:
        """Return a trading signal for the current state."""
        state_key = score.game_state_key
        now = time.time()

        # ── Risk gate ─────────────────────────────────────────────────────
        if not risk.is_trading_enabled:
            return TradingSignal(
                action=Action.SKIP,
                reason=f"Trading disabled: {risk.stop_reason or 'risk limit'}",
                timestamp=now,
            )

        ev = prob.ev
        edge = prob.edge_pct
        true_p = prob.true_probability

        # ── State-based rules ─────────────────────────────────────────────
        # 30-15 / 15-30 → EV > 3%
        if state_key in cls.ENTRY_STATES:
            if ev > 0.03:
                side = cls._pick_side(state_key, prob)
                return TradingSignal(
                    action=Action.ENTER,
                    side=side,
                    confidence=min(90, int(ev * 1000)),
                    bet_size_pct=cls._kelly(true_p, current_odds_server, risk),
                    reason=f"ENTRY at {state_key} | EV={ev:.2%}",
                    ev=round(ev, 4),
                    edge=round(edge, 4),
                    timestamp=now,
                )

        # 30-30 → edge > 4%
        if state_key == cls.VOLATILE_STATE:
            if edge > 0.04:
                return TradingSignal(
                    action=Action.ENTER,
                    side=cls._pick_side(state_key, prob),
                    confidence=min(95, int(edge * 1200)),
                    bet_size_pct=cls._kelly(true_p, current_odds_server, risk),
                    reason=f"HIGH-PRIORITY ENTRY at 30-30 | edge={edge:.2%}",
                    ev=round(ev, 4),
                    edge=round(edge, 4),
                    timestamp=now,
                )

        # DEUCE zone → scalp
        if state_key in cls.DEUCE_STATES:
            if abs(edge) > 0.03:
                side = Side.SERVER if edge > 0 else Side.RECEIVER
                return TradingSignal(
                    action=Action.SCALP,
                    side=side,
                    confidence=min(85, int(abs(edge) * 1000)),
                    bet_size_pct=cls._kelly(true_p, current_odds_server, risk) * 0.5,
                    reason=f"SCALP at {state_key} | edge={edge:.2%}",
                    ev=round(ev, 4),
                    edge=round(edge, 4),
                    timestamp=now,
                )

        # Default SKIP
        return TradingSignal(
            action=Action.SKIP,
            reason=f"No signal at {state_key} | EV={ev:.2%}",
            ev=round(ev, 4),
            edge=round(edge, 4),
            timestamp=now,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _pick_side(state: str, prob: ProbabilityState) -> Side:
        if prob.true_probability > prob.market_probability:
            return Side.SERVER
        return Side.RECEIVER

    @staticmethod
    def _kelly(true_p: float, odds: float, risk: RiskState) -> float:
        """Fractional Kelly (25%), capped at max_per_trade_pct."""
        b = odds - 1
        q = 1 - true_p
        if b <= 0:
            return 0.0
        f = (true_p * b - q) / b
        f = max(0.0, f) * 0.25          # 25% Kelly
        return round(min(f, risk.max_per_trade_pct / 100.0) * 100, 2)  # pct
