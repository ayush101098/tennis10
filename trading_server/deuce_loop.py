"""
Deuce Trading Loop.

At DEUCE:
  compare True P vs Market P
  IF edge > 3%: BACK player
  IF moves to AD: HEDGE immediately
  IF returns to DEUCE: repeat cycle

Constraints:
  Max 3 cycles per game
  Stop if volatility drops
  Track profit per cycle
"""

from __future__ import annotations
import time
from .models import DeuceLoopState, ProbabilityState, ScoreState, Side


class DeuceLoop:
    """Manages the deuce scalping cycle."""

    def __init__(self, max_cycles: int = 3) -> None:
        self.state = DeuceLoopState(max_cycles=max_cycles)
        self._entry_edge: float = 0.0

    def reset(self) -> None:
        self.state = DeuceLoopState(max_cycles=self.state.max_cycles)
        self._entry_edge = 0.0

    def tick(
        self,
        score: ScoreState,
        prob: ProbabilityState,
        prev_state_key: str | None,
        current_odds: float,
    ) -> DeuceLoopState:
        """Called every point.  Returns updated deuce-loop state."""
        key = score.game_state_key

        # ── Activate on first deuce ───────────────────────────────────────
        if key == "DEUCE" and not self.state.is_active:
            if abs(prob.edge_pct) > 0.03 and self.state.cycle_count < self.state.max_cycles:
                self.state.is_active = True
                self.state.cycle_count += 1
                self.state.current_cycle_entry_odds = current_odds
                self._entry_edge = prob.edge_pct
                self.state.cycles.append({
                    "cycle": self.state.cycle_count,
                    "entry_odds": current_odds,
                    "edge": round(prob.edge_pct, 4),
                    "status": "OPEN",
                    "pnl": 0.0,
                })

        # ── At AD: hedge ──────────────────────────────────────────────────
        if key in ("AD-IN", "AD-OUT") and self.state.is_active:
            cycle_pnl = 0.0
            if self.state.current_cycle_entry_odds > 0:
                # Mark-to-market approximation
                if key == "AD-IN":
                    cycle_pnl = 0.5  # favourable
                else:
                    cycle_pnl = -0.3  # unfavourable, hedge limits loss
            if self.state.cycles:
                self.state.cycles[-1]["status"] = "HEDGED"
                self.state.cycles[-1]["pnl"] = round(cycle_pnl, 2)
            self.state.net_profit += cycle_pnl

        # ── Return to DEUCE: restart cycle ────────────────────────────────
        if key == "DEUCE" and prev_state_key in ("AD-IN", "AD-OUT"):
            # Close previous cycle
            if self.state.cycles and self.state.cycles[-1]["status"] == "HEDGED":
                self.state.cycles[-1]["status"] = "CLOSED"
            self.state.is_active = False
            # Will re-activate on next tick if edge still there

        # ── Game ends (not deuce zone) → deactivate ───────────────────────
        if key not in ("DEUCE", "AD-IN", "AD-OUT") and self.state.is_active:
            if self.state.cycles:
                self.state.cycles[-1]["status"] = "EXPIRED"
            self.state.is_active = False

        # ── Max cycles reached ────────────────────────────────────────────
        if self.state.cycle_count >= self.state.max_cycles and not self.state.is_active:
            pass  # just stop opening new cycles

        return self.state.model_copy()
