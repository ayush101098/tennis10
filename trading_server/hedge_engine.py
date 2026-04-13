"""
Hedge Engine — separate service for hedge logic.

Trigger conditions:
  1. Trend Break:  30-15 → 30-30  |  40-30 → Deuce
  2. Adverse Move: odds move > 25% against position
  3. Deuce Loss:   Deuce → AD-OUT

Hedge formula:
  hedge_size = (entry_size × entry_odds) / current_odds

Execution:
  Within 1 tick.  If latency > 1 s → force exit.
"""

from __future__ import annotations
import time
from typing import Optional
from .models import (
    HedgeSignal, HedgeTrigger, PositionState, ScoreState,
    TradeEntry, Action,
)


class HedgeEngine:
    """Stateless hedge calculator."""

    # State transitions that count as "trend break"
    TREND_BREAKS = {
        ("30-15", "30-30"),
        ("15-30", "30-30"),
        ("40-30", "DEUCE"),
        ("30-40", "DEUCE"),
        ("AD-IN", "DEUCE"),
    }

    @classmethod
    def evaluate(
        cls,
        score: ScoreState,
        prev_state_key: Optional[str],
        position: PositionState,
        current_odds_break: float,
        current_odds_hold: float,
        latency_ms: int = 0,
    ) -> HedgeSignal:
        """Return a hedge signal (or no-hedge)."""
        # No position → nothing to hedge
        if not position.open_trades:
            return HedgeSignal()

        state_key = score.game_state_key

        # ── Latency guard ─────────────────────────────────────────────────
        if latency_ms > 1000:
            entry_trade = position.open_trades[0]
            hs = cls._compute_hedge(entry_trade, current_odds_hold)
            return HedgeSignal(
                should_hedge=True,
                hedge_size=hs,
                trigger=HedgeTrigger.MANUAL,
                reason=f"FORCE EXIT — latency {latency_ms}ms > 1000ms",
                urgency="IMMEDIATE",
                expected_neutral_pnl=cls._neutral_pnl(entry_trade, hs, current_odds_hold),
            )

        entry_trade = position.open_trades[0]

        # ── 1. Trend break ────────────────────────────────────────────────
        if prev_state_key and (prev_state_key, state_key) in cls.TREND_BREAKS:
            hs = cls._compute_hedge(entry_trade, current_odds_hold)
            return HedgeSignal(
                should_hedge=True,
                hedge_size=hs,
                trigger=HedgeTrigger.TREND_BREAK,
                reason=f"Trend break: {prev_state_key} → {state_key}",
                urgency="HIGH",
                expected_neutral_pnl=cls._neutral_pnl(entry_trade, hs, current_odds_hold),
            )

        # ── 2. Adverse odds move > 25% ───────────────────────────────────
        if entry_trade.entry_odds > 0:
            move = abs(current_odds_break - entry_trade.entry_odds) / entry_trade.entry_odds
            if move > 0.25 and current_odds_break > entry_trade.entry_odds:
                hs = cls._compute_hedge(entry_trade, current_odds_hold)
                return HedgeSignal(
                    should_hedge=True,
                    hedge_size=hs,
                    trigger=HedgeTrigger.ADVERSE_MOVE,
                    reason=f"Odds moved {move:.0%} against position",
                    urgency="HIGH",
                    expected_neutral_pnl=cls._neutral_pnl(entry_trade, hs, current_odds_hold),
                )

        # ── 3. Deuce loss (→ AD-OUT) ─────────────────────────────────────
        if state_key == "AD-OUT" and prev_state_key == "DEUCE":
            hs = cls._compute_hedge(entry_trade, current_odds_hold)
            return HedgeSignal(
                should_hedge=True,
                hedge_size=hs,
                trigger=HedgeTrigger.DEUCE_LOSS,
                reason="Deuce → AD-OUT — hedge immediately",
                urgency="IMMEDIATE",
                expected_neutral_pnl=cls._neutral_pnl(entry_trade, hs, current_odds_hold),
            )

        return HedgeSignal()

    # ── Hedge formula ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_hedge(entry: TradeEntry, current_odds_hold: float) -> float:
        """hedge_size = (entry_size × entry_odds) / current_odds"""
        if current_odds_hold <= 0:
            return 0.0
        return round((entry.stake * entry.entry_odds) / current_odds_hold, 2)

    @staticmethod
    def _neutral_pnl(entry: TradeEntry, hedge_size: float, hold_odds: float) -> float:
        """Expected PnL if hedge results in neutral."""
        win_pnl = (entry.entry_odds - 1) * entry.stake - hedge_size
        lose_pnl = (hold_odds - 1) * hedge_size - entry.stake
        return round((win_pnl + lose_pnl) / 2, 2)
