"""
Dual Account Manager.

Account A: entry trades only
Account B: hedge trades only

Rules:
  No overlap — each account has distinct purpose.
  Combined exposure tracked globally.
"""

from __future__ import annotations
import time
import uuid
from typing import List
from .models import (
    PositionState, TradeEntry, Side, PositionType,
    ExecutionResult, Action, TradingSignal, HedgeSignal,
    StatePerformanceRow,
)


class DualAccountManager:
    """Manages two accounts and the trade log."""

    def __init__(self, bankroll: float = 10000.0) -> None:
        half = bankroll / 2.0
        self.position = PositionState(
            account_a_balance=half,
            account_b_balance=half,
        )
        self.trade_log: List[TradeEntry] = []
        self._state_stats: dict[str, dict] = {}  # state → {trades, wins, ev_sum, pnl}

    # ── Open entry (Account A) ────────────────────────────────────────────

    def open_entry(
        self,
        signal: TradingSignal,
        execution: ExecutionResult,
        state_key: str,
    ) -> TradeEntry | None:
        if not execution.filled:
            return None

        stake = (signal.bet_size_pct / 100.0) * self.position.account_a_balance
        stake = round(min(stake, self.position.account_a_balance), 2)
        if stake <= 0:
            return None

        entry = TradeEntry(
            id=str(uuid.uuid4())[:8],
            account="A",
            side=signal.side,
            position_type=PositionType.BACK,
            entry_odds=execution.fill_price,
            current_odds=execution.fill_price,
            stake=stake,
            pnl=0.0,
            state_at_entry=state_key,
            timestamp=time.time(),
            is_open=True,
        )
        self.position.account_a_balance -= stake
        self.position.open_trades.append(entry)
        self.position.current_type = PositionType.BACK
        self.position.entry_odds = execution.fill_price
        self.position.current_odds = execution.fill_price
        self.position.stake = stake
        self._update_exposure()
        self.trade_log.append(entry)

        # Stats
        self._ensure_state(state_key)
        self._state_stats[state_key]["trades"] += 1

        return entry

    # ── Open hedge (Account B) ────────────────────────────────────────────

    def open_hedge(
        self,
        hedge: HedgeSignal,
        execution: ExecutionResult,
        state_key: str,
    ) -> TradeEntry | None:
        if not execution.filled:
            return None

        stake = round(min(hedge.hedge_size, self.position.account_b_balance), 2)
        if stake <= 0:
            return None

        entry = TradeEntry(
            id=str(uuid.uuid4())[:8],
            account="B",
            side=Side.SERVER,  # hedge always takes opposite side
            position_type=PositionType.LAY,
            entry_odds=execution.fill_price,
            current_odds=execution.fill_price,
            stake=stake,
            pnl=0.0,
            state_at_entry=state_key,
            timestamp=time.time(),
            is_open=True,
        )
        self.position.account_b_balance -= stake
        self.position.open_trades.append(entry)
        self._update_exposure()
        self.trade_log.append(entry)
        return entry

    # ── Close all trades (game ended) ─────────────────────────────────────

    def close_all(self, server_held: bool) -> float:
        """Close open trades. Returns net PnL."""
        net = 0.0
        for t in self.position.open_trades:
            if not t.is_open:
                continue
            if t.account == "A":
                # Account A backed the break
                if not server_held:
                    pnl = (t.entry_odds - 1) * t.stake
                else:
                    pnl = -t.stake
                t.pnl = round(pnl, 2)
                self.position.account_a_balance += t.stake + pnl
            else:
                # Account B laid (hedged)
                if server_held:
                    pnl = t.stake * 0.1  # small hedge profit
                else:
                    pnl = -t.stake * 0.3
                t.pnl = round(pnl, 2)
                self.position.account_b_balance += t.stake + pnl
            t.is_open = False
            net += t.pnl

            # Update state stats
            state = t.state_at_entry
            self._ensure_state(state)
            self._state_stats[state]["pnl"] += t.pnl
            if t.pnl > 0:
                self._state_stats[state]["wins"] += 1
            self._state_stats[state]["ev_sum"] += t.pnl / max(t.stake, 0.01)

        self.position.open_trades = [t for t in self.position.open_trades if t.is_open]
        self.position.current_type = PositionType.NONE
        self.position.pnl = 0.0
        self.position.stake = 0.0
        self._update_exposure()
        return round(net, 2)

    # ── Update live PnL ──────────────────────────────────────────────────

    def mark_to_market(self, current_odds: float) -> None:
        total_pnl = 0.0
        for t in self.position.open_trades:
            if not t.is_open:
                continue
            t.current_odds = current_odds
            # Simplified M2M
            if t.position_type == PositionType.BACK:
                implied_pnl = t.stake * (t.entry_odds / current_odds - 1)
            else:
                implied_pnl = t.stake * (1 - current_odds / t.entry_odds)
            t.pnl = round(implied_pnl, 2)
            total_pnl += t.pnl
        self.position.pnl = round(total_pnl, 2)
        self.position.current_odds = current_odds

    # ── State performance ─────────────────────────────────────────────────

    def get_state_performance(self) -> List[StatePerformanceRow]:
        rows = []
        for state, d in self._state_stats.items():
            trades = d["trades"]
            wins = d["wins"]
            rows.append(StatePerformanceRow(
                state=state,
                trades=trades,
                wins=wins,
                win_rate=round(wins / trades, 4) if trades > 0 else 0.0,
                avg_ev=round(d["ev_sum"] / trades, 4) if trades > 0 else 0.0,
                total_pnl=round(d["pnl"], 2),
            ))
        return rows

    # ── Internals ─────────────────────────────────────────────────────────

    def _update_exposure(self) -> None:
        total = sum(t.stake for t in self.position.open_trades if t.is_open)
        self.position.combined_exposure = round(total, 2)

    def _ensure_state(self, state: str) -> None:
        if state not in self._state_stats:
            self._state_stats[state] = {"trades": 0, "wins": 0, "ev_sum": 0.0, "pnl": 0.0}
