"""
Betfair Exchange Client
=======================
Production Betfair API integration using betfairlightweight.
Handles authentication, market discovery, live odds streaming,
and order placement/cancellation.

Supports:
  - SSO Login (non-interactive cert-based)
  - Tennis market discovery (Match Odds)
  - Real-time price streaming via Betfair Streaming API
  - Back / Lay order placement
  - Order cancellation and replacement
  - Keep-alive session management

Environment variables required:
  BETFAIR_USERNAME
  BETFAIR_PASSWORD
  BETFAIR_APP_KEY
  BETFAIR_CERT_PATH      → path to client-2048.crt
  BETFAIR_KEY_PATH        → path to client-2048.key
"""

from __future__ import annotations

import os
import time
import logging
import threading
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.filters import (
    market_filter,
    price_projection,
)

logger = logging.getLogger(__name__)


# ─── Data Classes ─────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BACK = "BACK"
    LAY = "LAY"


@dataclass
class BetfairCredentials:
    username: str
    password: str
    app_key: str
    cert_path: str  # .crt file
    key_path: str   # .key file

    @classmethod
    def from_env(cls) -> "BetfairCredentials":
        return cls(
            username=os.environ["BETFAIR_USERNAME"],
            password=os.environ["BETFAIR_PASSWORD"],
            app_key=os.environ["BETFAIR_APP_KEY"],
            cert_path=os.environ["BETFAIR_CERT_PATH"],
            key_path=os.environ["BETFAIR_KEY_PATH"],
        )


@dataclass
class RunnerOdds:
    """Live odds for one runner (player)."""
    selection_id: int
    runner_name: str
    back_prices: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    lay_prices: List[Tuple[float, float]] = field(default_factory=list)
    last_traded_price: float = 0.0
    total_matched: float = 0.0

    @property
    def best_back(self) -> float:
        return self.back_prices[0][0] if self.back_prices else 0.0

    @property
    def best_lay(self) -> float:
        return self.lay_prices[0][0] if self.lay_prices else 0.0

    @property
    def best_back_size(self) -> float:
        return self.back_prices[0][1] if self.back_prices else 0.0

    @property
    def best_lay_size(self) -> float:
        return self.lay_prices[0][1] if self.lay_prices else 0.0


@dataclass
class MarketSnapshot:
    """Point-in-time snapshot of a Betfair market."""
    market_id: str
    market_name: str
    event_name: str
    runners: Dict[int, RunnerOdds] = field(default_factory=dict)
    total_matched: float = 0.0
    status: str = "OPEN"
    in_play: bool = False
    timestamp: float = 0.0


@dataclass
class OrderResult:
    """Result of placing / cancelling an order."""
    success: bool
    bet_id: str = ""
    message: str = ""
    matched_price: float = 0.0
    matched_size: float = 0.0


# ─── Main Client ──────────────────────────────────────────────────────────────

class BetfairClient:
    """
    Wrapper around betfairlightweight for tennis trading.

    Usage:
        client = BetfairClient.from_env()
        client.login()
        markets = client.find_tennis_markets("Djokovic")
        client.subscribe_market(market_id, on_update=callback)
        result = client.place_order(market_id, selection_id, OrderSide.BACK, 2.50, 10.0)
        client.logout()
    """

    TENNIS_EVENT_TYPE_ID = "2"  # Betfair event type for Tennis

    def __init__(self, creds: BetfairCredentials) -> None:
        self._creds = creds
        self._client = betfairlightweight.APIClient(
            username=creds.username,
            password=creds.password,
            app_key=creds.app_key,
            certs=os.path.dirname(creds.cert_path),
        )
        self._logged_in = False
        self._last_keepalive = 0.0
        self._stream_listener: Optional[StreamListener] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._market_cache: Dict[str, MarketSnapshot] = {}
        self._update_callbacks: Dict[str, List[Callable]] = {}

    @classmethod
    def from_env(cls) -> "BetfairClient":
        return cls(BetfairCredentials.from_env())

    # ─── Authentication ───────────────────────────────────────────────────

    def login(self) -> bool:
        """Login via certificate-based SSO (non-interactive)."""
        try:
            self._client.login()
            self._logged_in = True
            self._last_keepalive = time.time()
            logger.info("Betfair login successful for %s", self._creds.username)
            return True
        except Exception as e:
            logger.error("Betfair login failed: %s", e)
            return False

    def login_interactive(self) -> bool:
        """Login via interactive (username/password, no certs)."""
        try:
            self._client.login_interactive()
            self._logged_in = True
            self._last_keepalive = time.time()
            logger.info("Betfair interactive login successful")
            return True
        except Exception as e:
            logger.error("Betfair interactive login failed: %s", e)
            return False

    def keep_alive(self) -> bool:
        """Keep the session alive (call every ~15 min)."""
        try:
            self._client.keep_alive()
            self._last_keepalive = time.time()
            return True
        except Exception as e:
            logger.error("Keep-alive failed: %s", e)
            return False

    def _ensure_alive(self) -> None:
        """Auto keep-alive if session is older than 15 minutes."""
        if time.time() - self._last_keepalive > 900:  # 15 min
            self.keep_alive()

    def logout(self) -> None:
        """Logout and close streaming connections."""
        self.stop_stream()
        try:
            self._client.logout()
        except Exception:
            pass
        self._logged_in = False
        logger.info("Betfair logged out")

    @property
    def is_logged_in(self) -> bool:
        return self._logged_in

    # ─── Market Discovery ────────────────────────────────────────────────

    def find_tennis_markets(
        self,
        player_name: str = "",
        competition: str = "",
        in_play_only: bool = False,
        max_results: int = 50,
    ) -> List[MarketSnapshot]:
        """
        Find tennis Match Odds markets.
        Optionally filter by player name or competition.
        """
        self._ensure_alive()

        text_query = player_name if player_name else None

        mf = market_filter(
            event_type_ids=[self.TENNIS_EVENT_TYPE_ID],
            market_type_codes=["MATCH_ODDS"],
            in_play_only=in_play_only,
            text_query=text_query,
        )

        if competition:
            mf["competition_ids"] = [competition]

        try:
            catalogues = self._client.betting.list_market_catalogue(
                filter=mf,
                market_projection=[
                    "RUNNER_DESCRIPTION",
                    "EVENT",
                    "COMPETITION",
                    "MARKET_START_TIME",
                ],
                max_results=max_results,
                sort="FIRST_TO_START",
            )
        except Exception as e:
            logger.error("Market catalogue fetch failed: %s", e)
            return []

        results = []
        for cat in catalogues:
            snap = MarketSnapshot(
                market_id=cat.market_id,
                market_name=cat.market_name,
                event_name=cat.event.name if cat.event else "",
                timestamp=time.time(),
            )
            for runner in cat.runners:
                snap.runners[runner.selection_id] = RunnerOdds(
                    selection_id=runner.selection_id,
                    runner_name=runner.runner_name,
                )
            results.append(snap)

        return results

    def get_market_odds(self, market_id: str) -> Optional[MarketSnapshot]:
        """Fetch current best odds for a market (REST polling)."""
        self._ensure_alive()

        try:
            books = self._client.betting.list_market_book(
                market_ids=[market_id],
                price_projection=price_projection(
                    price_data=["EX_BEST_OFFERS", "EX_TRADED"],
                ),
            )
        except Exception as e:
            logger.error("Market book fetch failed for %s: %s", market_id, e)
            return None

        if not books:
            return None

        book = books[0]

        # Try cache for event name
        cached = self._market_cache.get(market_id)
        event_name = cached.event_name if cached else ""

        snap = MarketSnapshot(
            market_id=market_id,
            market_name="Match Odds",
            event_name=event_name,
            total_matched=book.total_matched or 0.0,
            status=book.status,
            in_play=book.inplay,
            timestamp=time.time(),
        )

        for runner in book.runners:
            ex = runner.ex
            back_prices = [(p.price, p.size) for p in (ex.available_to_back or [])]
            lay_prices = [(p.price, p.size) for p in (ex.available_to_lay or [])]

            snap.runners[runner.selection_id] = RunnerOdds(
                selection_id=runner.selection_id,
                runner_name=cached.runners[runner.selection_id].runner_name
                    if cached and runner.selection_id in cached.runners else f"Runner {runner.selection_id}",
                back_prices=back_prices,
                lay_prices=lay_prices,
                last_traded_price=runner.last_price_traded or 0.0,
                total_matched=runner.total_matched or 0.0,
            )

        self._market_cache[market_id] = snap
        return snap

    # ─── Streaming (real-time odds) ───────────────────────────────────────

    def subscribe_market(
        self,
        market_id: str,
        on_update: Optional[Callable[[MarketSnapshot], None]] = None,
    ) -> None:
        """
        Subscribe to real-time market data via Betfair Streaming API.
        Calls `on_update(snapshot)` on every price change.
        """
        self._ensure_alive()

        if on_update:
            self._update_callbacks.setdefault(market_id, []).append(on_update)

        self._stream_listener = StreamListener(max_latency=0.5)

        stream = self._client.streaming.create_stream(
            listener=self._stream_listener,
        )

        market_filter_stream = {
            "marketIds": [market_id],
        }
        market_data_filter = {
            "fields": ["EX_BEST_OFFERS", "EX_TRADED", "EX_MARKET_DEF"],
            "ladderLevels": 3,
        }

        stream.subscribe_to_markets(
            market_filter=market_filter_stream,
            market_data_filter=market_data_filter,
        )

        def _run():
            stream.start()

        self._stream_thread = threading.Thread(target=_run, daemon=True)
        self._stream_thread.start()
        logger.info("Streaming started for market %s", market_id)

    def poll_stream(self) -> Optional[MarketSnapshot]:
        """Poll the stream listener for latest data (call in your event loop)."""
        if not self._stream_listener:
            return None

        output = self._stream_listener.output_list
        if not output:
            return None

        # Get latest market book from stream
        for market_book in output:
            for runner in market_book.runners:
                market_id = market_book.market_id
                cached = self._market_cache.get(market_id)
                if not cached:
                    cached = MarketSnapshot(
                        market_id=market_id,
                        market_name="Match Odds",
                        event_name="",
                        timestamp=time.time(),
                    )
                    self._market_cache[market_id] = cached

                ex = runner.ex
                if ex:
                    back_prices = [(p.price, p.size) for p in (ex.available_to_back or [])]
                    lay_prices = [(p.price, p.size) for p in (ex.available_to_lay or [])]

                    cached.runners[runner.selection_id] = RunnerOdds(
                        selection_id=runner.selection_id,
                        runner_name=cached.runners.get(runner.selection_id, RunnerOdds(
                            selection_id=runner.selection_id,
                            runner_name=f"Runner {runner.selection_id}",
                        )).runner_name,
                        back_prices=back_prices,
                        lay_prices=lay_prices,
                        last_traded_price=runner.last_price_traded or 0.0,
                    )

                cached.status = market_book.status
                cached.in_play = market_book.inplay
                cached.timestamp = time.time()

                # Fire callbacks
                for cb in self._update_callbacks.get(market_id, []):
                    try:
                        cb(cached)
                    except Exception as e:
                        logger.error("Stream callback error: %s", e)

        return cached if cached else None

    def stop_stream(self) -> None:
        """Stop the streaming connection."""
        if self._stream_listener:
            try:
                self._stream_listener = None
            except Exception:
                pass
        logger.info("Streaming stopped")

    # ─── Order Placement ──────────────────────────────────────────────────

    def place_order(
        self,
        market_id: str,
        selection_id: int,
        side: OrderSide,
        price: float,
        size: float,
        persistence_type: str = "LAPSE",  # LAPSE | PERSIST | MARKET_ON_CLOSE
    ) -> OrderResult:
        """
        Place a limit order on the exchange.

        Args:
            market_id: Betfair market ID
            selection_id: Runner selection ID
            side: BACK or LAY
            price: Desired odds (must be valid Betfair price increment)
            size: Stake in £/€
            persistence_type: What happens if unmatched at in-play transition
        """
        self._ensure_alive()

        limit_order = betfairlightweight.filters.limit_order(
            size=round(size, 2),
            price=self._snap_to_valid_price(price),
            persistence_type=persistence_type,
        )

        instructions = [
            betfairlightweight.filters.place_instruction(
                order_type="LIMIT",
                selection_id=selection_id,
                side=side.value,
                limit_order=limit_order,
            )
        ]

        try:
            resp = self._client.betting.place_orders(
                market_id=market_id,
                instructions=instructions,
            )

            if resp.status == "SUCCESS":
                report = resp.place_instruction_reports[0]
                return OrderResult(
                    success=True,
                    bet_id=report.bet_id,
                    matched_price=report.average_price_matched or 0.0,
                    matched_size=report.size_matched or 0.0,
                    message="Order placed successfully",
                )
            else:
                return OrderResult(
                    success=False,
                    message=f"Order failed: {resp.error_code}",
                )
        except Exception as e:
            logger.error("Place order failed: %s", e)
            return OrderResult(success=False, message=str(e))

    def cancel_order(
        self,
        market_id: str,
        bet_id: str,
        size_reduction: Optional[float] = None,
    ) -> OrderResult:
        """Cancel an unmatched order (or reduce its size)."""
        self._ensure_alive()

        instructions = [
            betfairlightweight.filters.cancel_instruction(
                bet_id=bet_id,
                size_reduction=size_reduction,
            )
        ]

        try:
            resp = self._client.betting.cancel_orders(
                market_id=market_id,
                instructions=instructions,
            )

            if resp.status == "SUCCESS":
                return OrderResult(
                    success=True,
                    bet_id=bet_id,
                    message="Order cancelled",
                )
            else:
                return OrderResult(
                    success=False,
                    bet_id=bet_id,
                    message=f"Cancel failed: {resp.error_code}",
                )
        except Exception as e:
            logger.error("Cancel order failed: %s", e)
            return OrderResult(success=False, message=str(e))

    def replace_order(
        self,
        market_id: str,
        bet_id: str,
        new_price: float,
    ) -> OrderResult:
        """Replace an unmatched order with a new price."""
        self._ensure_alive()

        instructions = [
            betfairlightweight.filters.replace_instruction(
                bet_id=bet_id,
                new_price=self._snap_to_valid_price(new_price),
            )
        ]

        try:
            resp = self._client.betting.replace_orders(
                market_id=market_id,
                instructions=instructions,
            )

            if resp.status == "SUCCESS":
                report = resp.replace_instruction_reports[0]
                return OrderResult(
                    success=True,
                    bet_id=report.place_instruction_report.bet_id,
                    message="Order replaced",
                )
            else:
                return OrderResult(
                    success=False,
                    bet_id=bet_id,
                    message=f"Replace failed: {resp.error_code}",
                )
        except Exception as e:
            logger.error("Replace order failed: %s", e)
            return OrderResult(success=False, message=str(e))

    def get_current_orders(self, market_id: Optional[str] = None) -> List[dict]:
        """Get all current (unmatched/partially matched) orders."""
        self._ensure_alive()
        try:
            orders = self._client.betting.list_current_orders(
                market_ids=[market_id] if market_id else None,
            )
            return [
                {
                    "bet_id": o.bet_id,
                    "market_id": o.market_id,
                    "selection_id": o.selection_id,
                    "side": o.side,
                    "price": o.price_size.price if o.price_size else 0,
                    "size": o.price_size.size if o.price_size else 0,
                    "matched_size": o.size_matched,
                    "status": o.status,
                }
                for o in orders.orders
            ]
        except Exception as e:
            logger.error("List orders failed: %s", e)
            return []

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _snap_to_valid_price(price: float) -> float:
        """
        Snap a price to the nearest valid Betfair price increment.
        Betfair price ladder:
          1.01–2.00  →  0.01 increments
          2.00–3.00  →  0.02
          3.00–4.00  →  0.05
          4.00–6.00  →  0.10
          6.00–10.0  →  0.20
          10.0–20.0  →  0.50
          20.0–30.0  →  1.00
          30.0–50.0  →  2.00
          50.0–100   →  5.00
          100–1000   →  10.00
        """
        if price <= 1.01:
            return 1.01
        if price <= 2.0:
            return round(round(price / 0.01) * 0.01, 2)
        if price <= 3.0:
            return round(round(price / 0.02) * 0.02, 2)
        if price <= 4.0:
            return round(round(price / 0.05) * 0.05, 2)
        if price <= 6.0:
            return round(round(price / 0.10) * 0.10, 2)
        if price <= 10.0:
            return round(round(price / 0.20) * 0.20, 2)
        if price <= 20.0:
            return round(round(price / 0.50) * 0.50, 1)
        if price <= 30.0:
            return round(round(price / 1.00) * 1.00, 0)
        if price <= 50.0:
            return round(round(price / 2.00) * 2.00, 0)
        if price <= 100.0:
            return round(round(price / 5.00) * 5.00, 0)
        return round(round(price / 10.00) * 10.00, 0)

    def get_account_funds(self) -> dict:
        """Get available balance on the Betfair account."""
        self._ensure_alive()
        try:
            funds = self._client.account.get_account_funds()
            return {
                "available": funds.available_to_bet_balance,
                "exposure": funds.exposure,
                "retained_commission": funds.retained_commission,
                "points_balance": funds.points_balance,
            }
        except Exception as e:
            logger.error("Account funds fetch failed: %s", e)
            return {}
