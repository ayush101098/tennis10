"""
Risk Manager — per-trade and per-match constraints.

Per Trade:   max 1% bankroll
Per Match:   max 10% exposure
Stop:        3 consecutive losses → stop trading
             Latency spike → disable entries
"""

from __future__ import annotations
from .models import RiskState, RiskLevel, PositionState


class RiskManager:
    """Enforces risk constraints. Mutates RiskState in-place."""

    def __init__(
        self,
        max_per_trade_pct: float = 1.0,
        max_per_match_pct: float = 10.0,
        max_consecutive_losses: int = 3,
    ) -> None:
        self.state = RiskState(
            max_per_trade_pct=max_per_trade_pct,
            max_per_match_pct=max_per_match_pct,
        )
        self.max_consec = max_consecutive_losses

    def update(self, position: PositionState) -> RiskState:
        """Recompute risk level from current position."""
        total_bankroll = position.account_a_balance + position.account_b_balance
        if total_bankroll <= 0:
            self.state.is_trading_enabled = False
            self.state.stop_reason = "Bankrupt"
            self.state.risk_level = RiskLevel.CRITICAL
            return self.state.model_copy()

        exposure = position.combined_exposure
        exposure_pct = (exposure / total_bankroll) * 100

        self.state.current_exposure_pct = round(exposure_pct, 2)

        # Risk level
        if exposure_pct >= self.state.max_per_match_pct:
            self.state.risk_level = RiskLevel.CRITICAL
            self.state.is_trading_enabled = False
            self.state.stop_reason = f"Match exposure {exposure_pct:.1f}% ≥ {self.state.max_per_match_pct}%"
        elif exposure_pct >= self.state.max_per_match_pct * 0.7:
            self.state.risk_level = RiskLevel.HIGH
        elif exposure_pct >= self.state.max_per_match_pct * 0.4:
            self.state.risk_level = RiskLevel.MEDIUM
        else:
            self.state.risk_level = RiskLevel.LOW

        # Consecutive losses
        if self.state.consecutive_losses >= self.max_consec:
            self.state.is_trading_enabled = False
            self.state.stop_reason = f"{self.state.consecutive_losses} consecutive losses"

        return self.state.model_copy()

    def register_loss(self) -> None:
        self.state.consecutive_losses += 1

    def register_win(self) -> None:
        self.state.consecutive_losses = 0

    def register_latency_spike(self) -> None:
        self.state.is_trading_enabled = False
        self.state.stop_reason = "Latency spike detected"

    def reset(self) -> None:
        self.state = RiskState(
            max_per_trade_pct=self.state.max_per_trade_pct,
            max_per_match_pct=self.state.max_per_match_pct,
        )
