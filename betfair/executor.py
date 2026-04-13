"""
Betfair Execution Engine
========================
Bridges the trading server's signals to real Betfair orders.
Replaces the ExecutionSimulator for production use.

Flow:
  TradingEngine.decide() → TradingSignal
  → BetfairExecutor.execute_signal() → places real BACK/LAY on Betfair
  → returns ExecutionResult (same model the rest of the system uses)
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field

from .client import BetfairClient, OrderSide, MarketSnapshot, RunnerOdds, OrderResult

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_server.models import (
    Action, Side, TradingSignal, HedgeSignal, ExecutionResult, TradeEntry,
)

logger = logging.getLogger(__name__)


@dataclass
class MarketMapping:
    """Maps our internal player references to Betfair selection IDs."""
    market_id: str
    player1_selection_id: int
    player2_selection_id: int
    player1_name: str = ""
    player2_name: str = ""


@dataclass
class ExecutionConfig:
    """Risk controls for live execution."""
    max_stake: float = 50.0          # Max single bet in £/€
    min_odds: float = 1.10           # Don't back below this
    max_odds: float = 20.0           # Don't back above this
    min_liquidity: float = 20.0      # Minimum available size at best price
    persistence: str = "LAPSE"       # LAPSE at in-play, or PERSIST
    use_market_price: bool = True    # True = take best available, False = use signal price
    dry_run: bool = True             # Start in paper mode (log only, no real orders)


class BetfairExecutor:
    """
    Live execution engine that translates TradingSignals into Betfair orders.

    Usage:
        client = BetfairClient.from_env()
        client.login()

        executor = BetfairExecutor(client, config=ExecutionConfig(dry_run=False))
        executor.set_market_mapping(MarketMapping(
            market_id="1.234567890",
            player1_selection_id=12345,
            player2_selection_id=67890,
        ))

        # Called by the trading loop:
        result = executor.execute_signal(signal, bankroll=10000.0)
        result = executor.execute_hedge(hedge_signal, stake=50.0)
    """

    def __init__(
        self,
        client: BetfairClient,
        config: Optional[ExecutionConfig] = None,
    ) -> None:
        self.client = client
        self.config = config or ExecutionConfig()
        self.mapping: Optional[MarketMapping] = None
        self._open_bets: Dict[str, dict] = {}  # bet_id → order info
        self._execution_log: list = []

    def set_market_mapping(self, mapping: MarketMapping) -> None:
        self.mapping = mapping

    # ─── Execute Trading Signal ───────────────────────────────────────────

    def execute_signal(
        self,
        signal: TradingSignal,
        bankroll: float,
        current_snapshot: Optional[MarketSnapshot] = None,
    ) -> ExecutionResult:
        """
        Execute a TradingSignal on Betfair.
        Returns an ExecutionResult compatible with the existing system.
        """
        if signal.action not in (Action.ENTER, Action.SCALP):
            return ExecutionResult(filled=False, delay_ms=0, slippage_ticks=0)

        if not self.mapping:
            logger.error("No market mapping set — cannot execute")
            return ExecutionResult(filled=False, delay_ms=0, message="No market mapping")

        # Determine selection and side
        if signal.side == Side.SERVER:
            # Back the server (we think they'll hold)
            selection_id = self.mapping.player1_selection_id
            order_side = OrderSide.BACK
        else:
            # Back the receiver (we think they'll break)
            selection_id = self.mapping.player2_selection_id
            order_side = OrderSide.BACK

        # Calculate stake
        stake = round(bankroll * (signal.bet_size_pct / 100.0), 2)
        stake = min(stake, self.config.max_stake)
        stake = max(stake, 2.0)  # Betfair minimum bet is £2

        # Get best available price
        market_price = self._get_best_price(
            selection_id, order_side, current_snapshot
        )

        if market_price <= 0:
            logger.warning("No price available for selection %d", selection_id)
            return ExecutionResult(
                requested_price=0.0, fill_price=0.0,
                filled=False, delay_ms=0, slippage_ticks=0,
            )

        # Validate price
        if market_price < self.config.min_odds or market_price > self.config.max_odds:
            logger.info("Price %.2f outside limits [%.2f, %.2f]",
                        market_price, self.config.min_odds, self.config.max_odds)
            return ExecutionResult(
                requested_price=market_price, fill_price=0.0,
                filled=False, delay_ms=0, slippage_ticks=0,
            )

        # Check liquidity
        avail = self._get_available_size(selection_id, order_side, current_snapshot)
        if avail < self.config.min_liquidity:
            logger.info("Insufficient liquidity: %.2f < %.2f", avail, self.config.min_liquidity)
            return ExecutionResult(
                requested_price=market_price, fill_price=0.0,
                filled=False, delay_ms=0, slippage_ticks=0,
                fill_probability=0.0,
            )

        t0 = time.time()

        # ── DRY RUN MODE ─────────────────────────────────────────────────
        if self.config.dry_run:
            delay_ms = int((time.time() - t0) * 1000) + 50
            logger.info(
                "[DRY RUN] %s %s %.2f @ %.2f (stake=%.2f)",
                order_side.value, signal.side.value, stake, market_price, stake,
            )
            self._execution_log.append({
                "time": time.time(), "dry_run": True,
                "side": order_side.value, "price": market_price,
                "stake": stake, "signal": signal.reason,
            })
            return ExecutionResult(
                requested_price=market_price,
                fill_price=market_price,
                slippage_ticks=0,
                delay_ms=delay_ms,
                filled=True,
                fill_probability=1.0,
            )

        # ── LIVE ORDER ───────────────────────────────────────────────────
        result = self.client.place_order(
            market_id=self.mapping.market_id,
            selection_id=selection_id,
            side=order_side,
            price=market_price,
            size=stake,
            persistence_type=self.config.persistence,
        )

        delay_ms = int((time.time() - t0) * 1000)

        if result.success:
            self._open_bets[result.bet_id] = {
                "market_id": self.mapping.market_id,
                "selection_id": selection_id,
                "side": order_side.value,
                "price": result.matched_price or market_price,
                "stake": stake,
                "bet_id": result.bet_id,
                "time": time.time(),
            }
            logger.info(
                "ORDER PLACED: %s %s @ %.2f (bet_id=%s, matched=%.2f)",
                order_side.value, signal.side.value,
                result.matched_price or market_price,
                result.bet_id, result.matched_size,
            )

        self._execution_log.append({
            "time": time.time(), "dry_run": False,
            "success": result.success, "bet_id": result.bet_id,
            "side": order_side.value, "price": result.matched_price,
            "stake": stake, "signal": signal.reason,
        })

        fill_price = result.matched_price if result.matched_price else market_price
        slippage = abs(fill_price - market_price) / 0.02 if market_price else 0

        return ExecutionResult(
            requested_price=market_price,
            fill_price=fill_price if result.success else 0.0,
            slippage_ticks=int(slippage),
            delay_ms=delay_ms,
            filled=result.success,
            fill_probability=1.0 if result.success else 0.0,
        )

    # ─── Execute Hedge ────────────────────────────────────────────────────

    def execute_hedge(
        self,
        hedge: HedgeSignal,
        original_selection_id: int,
        stake: float,
        current_snapshot: Optional[MarketSnapshot] = None,
    ) -> ExecutionResult:
        """
        Execute a hedge — LAY the same selection we previously BACKed.
        """
        if not hedge.should_hedge or not self.mapping:
            return ExecutionResult(filled=False, delay_ms=0, slippage_ticks=0)

        order_side = OrderSide.LAY
        market_price = self._get_best_price(
            original_selection_id, order_side, current_snapshot
        )

        if market_price <= 0:
            return ExecutionResult(filled=False, delay_ms=0, slippage_ticks=0)

        hedge_stake = round(min(stake * hedge.hedge_size, self.config.max_stake), 2)
        hedge_stake = max(hedge_stake, 2.0)

        t0 = time.time()

        if self.config.dry_run:
            delay_ms = int((time.time() - t0) * 1000) + 50
            logger.info(
                "[DRY RUN] HEDGE LAY @ %.2f (stake=%.2f)", market_price, hedge_stake
            )
            return ExecutionResult(
                requested_price=market_price, fill_price=market_price,
                slippage_ticks=0, delay_ms=delay_ms,
                filled=True, fill_probability=1.0,
            )

        result = self.client.place_order(
            market_id=self.mapping.market_id,
            selection_id=original_selection_id,
            side=order_side,
            price=market_price,
            size=hedge_stake,
            persistence_type=self.config.persistence,
        )

        delay_ms = int((time.time() - t0) * 1000)

        if result.success:
            self._open_bets[result.bet_id] = {
                "market_id": self.mapping.market_id,
                "selection_id": original_selection_id,
                "side": "LAY",
                "price": result.matched_price or market_price,
                "stake": hedge_stake,
                "bet_id": result.bet_id,
                "time": time.time(),
                "is_hedge": True,
            }
            logger.info("HEDGE PLACED: LAY @ %.2f (bet_id=%s)", result.matched_price, result.bet_id)

        fill_price = result.matched_price if result.matched_price else market_price
        slippage = abs(fill_price - market_price) / 0.02 if market_price else 0

        return ExecutionResult(
            requested_price=market_price,
            fill_price=fill_price if result.success else 0.0,
            slippage_ticks=int(slippage),
            delay_ms=delay_ms,
            filled=result.success,
            fill_probability=1.0 if result.success else 0.0,
        )

    # ─── Cancel All Open Orders ───────────────────────────────────────────

    def cancel_all(self) -> int:
        """Cancel all open orders. Returns count of cancelled."""
        if self.config.dry_run or not self.mapping:
            return 0

        orders = self.client.get_current_orders(self.mapping.market_id)
        cancelled = 0
        for order in orders:
            if order["status"] in ("EXECUTABLE",):
                result = self.client.cancel_order(
                    self.mapping.market_id, order["bet_id"]
                )
                if result.success:
                    cancelled += 1
        return cancelled

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _get_best_price(
        self,
        selection_id: int,
        side: OrderSide,
        snapshot: Optional[MarketSnapshot] = None,
    ) -> float:
        """Get the best available price for a selection."""
        if snapshot and selection_id in snapshot.runners:
            runner = snapshot.runners[selection_id]
            if side == OrderSide.BACK:
                return runner.best_back
            else:
                return runner.best_lay

        # Fall back to REST polling
        if self.mapping:
            snap = self.client.get_market_odds(self.mapping.market_id)
            if snap and selection_id in snap.runners:
                runner = snap.runners[selection_id]
                return runner.best_back if side == OrderSide.BACK else runner.best_lay

        return 0.0

    def _get_available_size(
        self,
        selection_id: int,
        side: OrderSide,
        snapshot: Optional[MarketSnapshot] = None,
    ) -> float:
        """Get available size at best price."""
        if snapshot and selection_id in snapshot.runners:
            runner = snapshot.runners[selection_id]
            if side == OrderSide.BACK:
                return runner.best_back_size
            else:
                return runner.best_lay_size
        return 0.0

    @property
    def execution_log(self) -> list:
        return self._execution_log

    @property
    def open_bets(self) -> Dict[str, dict]:
        return self._open_bets
