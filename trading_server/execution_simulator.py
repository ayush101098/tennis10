"""
Execution Simulator — realistic latency, slippage, and fill probability.

  Random delay:      200–800 ms
  Slippage:          1–3 ticks
  Fill probability:  85–95%

All PnL must use the adjusted price.
"""

from __future__ import annotations
import random
from .models import ExecutionResult


class ExecutionSimulator:
    """Simulates exchange execution."""

    TICK_SIZE = 0.02  # 1 tick = 0.02 odds movement

    def __init__(
        self,
        min_delay_ms: int = 200,
        max_delay_ms: int = 800,
        min_slippage_ticks: int = 1,
        max_slippage_ticks: int = 3,
        min_fill_prob: float = 0.85,
        max_fill_prob: float = 0.95,
    ) -> None:
        self.min_delay = min_delay_ms
        self.max_delay = max_delay_ms
        self.min_slip = min_slippage_ticks
        self.max_slip = max_slippage_ticks
        self.min_fill = min_fill_prob
        self.max_fill = max_fill_prob

    def execute(self, market_price: float, is_back: bool = True) -> ExecutionResult:
        """Simulate an execution at `market_price`."""
        delay = random.randint(self.min_delay, self.max_delay)
        slip_ticks = random.randint(self.min_slip, self.max_slip)
        fill_prob = round(random.uniform(self.min_fill, self.max_fill), 4)
        filled = random.random() < fill_prob

        # Slippage direction: worse for the trader
        if is_back:
            fill_price = market_price - slip_ticks * self.TICK_SIZE  # lower is worse for back
        else:
            fill_price = market_price + slip_ticks * self.TICK_SIZE  # higher is worse for lay

        fill_price = round(max(1.01, fill_price), 2)

        return ExecutionResult(
            requested_price=round(market_price, 2),
            fill_price=fill_price if filled else 0.0,
            slippage_ticks=slip_ticks,
            delay_ms=delay,
            filled=filled,
            fill_probability=fill_prob,
        )
