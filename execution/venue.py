"""Venue abstraction — the seam between the intelligence layer and execution.

The architecture the terminal is converging on:

    signals / intel  →  Execution Router  →  VenueAdapter (Polymarket | SX Bet | …)

`VenueAdapter` is the contract each venue implements so the signal / Kelly /
hedge / journal layers stay venue-agnostic. Reads (`find_markets`, `best_ask`,
`fair_prob`) need no credentials; execution (`place`, `cancel`) needs the venue's
auth (Polymarket: py-clob-client key; SX Bet: EIP-712 wallet signing) and is
gated by the same paper/live guards as the rest of the system.

Current implementers:
  - execution.polymarket.PolymarketClient  — full read + (guarded) execute
  - execution.sxbet.SXBetClient            — read-only for now (no order signing)

This module intentionally defines the *contract* only; it does not force a
rewrite of the working Polymarket path. New venues should conform to it so the
router can treat them uniformly.
"""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class VenueAdapter(Protocol):
    """Minimal interface a betting venue must expose to the execution router."""

    name: str

    # ── market data (public, no auth) ─────────────────────────────────────────
    def fair_prob(self, player1: str, player2: str) -> Optional[float]:
        """De-vigged / model-free fair probability that player1 wins, if the
        venue prices the fixture; else None. Exchanges (SX Bet, Polymarket)
        derive this from the live order book."""
        ...

    # ── execution (requires venue auth; paper unless guards open) ─────────────
    def place(self, market_id: str, side: str, price: float, size: float,
              dry_run: bool = True) -> dict:
        """Place (or simulate) an order. Returns a status dict with at least
        {"status": "dry_run"|"placed"|"failed"}."""
        ...

    def cancel(self, order_id: str) -> dict:
        """Cancel a resting order. Returns a status dict."""
        ...
