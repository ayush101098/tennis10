"""
Live Betfair Trading Loop
=========================
Connects the trading server's engines to real Betfair markets
AND live match data feeds for fully automated trading.

This is the production entry point that:
  1. Logs into Betfair
  2. Finds the target tennis match market
  3. Attaches to a live score feed (Sofascore/ESPN/api-tennis)
  4. Subscribes to live Betfair odds streaming
  5. On every score change: updates match state → runs ML pipeline → executes trades
  6. On every odds change: re-evaluates edge → adjusts positions

Usage:
  # Dry run (paper trading — no real money):
  python -m betfair.live_loop --player1 "Djokovic" --player2 "Alcaraz" --dry-run

  # Live execution:
  python -m betfair.live_loop --player1 "Djokovic" --player2 "Alcaraz" --live

  # With custom params:
  python -m betfair.live_loop \\
      --player1 "Djokovic" --player2 "Alcaraz" \\
      --p1-serve 68 --p2-serve 65 \\
      --p1-rank 2 --p2-rank 3 \\
      --bankroll 5000 --max-stake 25 \\
      --dry-run

  # Score feed only (no Betfair, just watch live scores):
  python -m betfair.live_loop --player1 "Sinner" --player2 "Alcaraz" --score-only

Environment:
  BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY,
  BETFAIR_CERT_PATH, BETFAIR_KEY_PATH
  API_TENNIS_KEY (optional — enables point-level data)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from betfair.client import BetfairClient, MarketSnapshot
from betfair.executor import BetfairExecutor, ExecutionConfig, MarketMapping
from trading_server.models import (
    Action,
    MatchSetupRequest,
    PointWinner,
    OddsUpdateRequest,
)
from trading_server.main import MatchSession
from trading_server.live_feed import LiveMatchFeed, ScoreChange, DataSource
from trading_server.live_stats import PointStats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("betfair_trading.log"),
    ],
)
logger = logging.getLogger("betfair.live_loop")


class LiveTradingLoop:
    """
    Main trading loop that connects:
      - Live score feed (ESPN/Sofascore/api-tennis) → match state updates
      - Betfair odds stream → market price updates
      - Trading engine → signal generation
      - Betfair executor → order placement
    """

    def __init__(
        self,
        client: Optional[BetfairClient],
        executor: Optional[BetfairExecutor],
        session: MatchSession,
        feed: LiveMatchFeed,
        poll_interval: float = 0.5,
        score_poll_interval: float = 2.0,  # seconds between score polls
    ) -> None:
        self.client = client
        self.executor = executor
        self.session = session
        self.feed = feed
        self.poll_interval = poll_interval
        self.score_poll_interval = score_poll_interval
        self._running = False
        self._tick_count = 0
        self._score_changes = 0
        self._last_odds_server = 0.0
        self._last_odds_receiver = 0.0
        self._last_score_poll = 0.0

    # ─── Score Change Handler ─────────────────────────────────────────────

    def on_score_change(self, change: ScoreChange) -> None:
        """
        Called when the live feed detects a score change.
        Feeds the point into the match state engine and live stats.
        """
        self._score_changes += 1

        # Convert to PointWinner enum
        point_winner = PointWinner.SERVER if change.point_winner == "SERVER" else PointWinner.RECEIVER

        # Build point stats from the change
        pt_stats = PointStats(
            is_ace=change.is_ace,
            is_double_fault=change.is_double_fault,
            is_first_serve_in=change.is_first_serve_in,
            is_break_point=change.is_break_point,
        )

        # Determine who is serving before point registration
        server = self.session.match_engine.info.server
        winner_player = server if point_winner == PointWinner.SERVER else (3 - server)

        # Feed live stats
        self.session.live_stats.register_point(server, winner_player, pt_stats)

        # Update score
        self.session.match_engine.register_point(point_winner)

        # If game ended (points reset), close trades
        score = self.session.match_engine.score
        if score.server_points == 0 and score.receiver_points == 0:
            server_held = point_winner == PointWinner.SERVER
            if server_held:
                self.session.live_stats.register_game_won(server, was_serving=True)
            else:
                other = 3 - server
                self.session.live_stats.register_game_won(other, was_serving=False)

            pnl = self.session.accounts.close_all(server_held)
            if pnl < 0:
                self.session.risk.register_loss()
            elif pnl > 0:
                self.session.risk.register_win()
            self.session.deuce.reset()

        # Log the score change
        new_s = change.new_score
        logger.info(
            "🎾 POINT %d | %s won | %s | Sets: %d-%d | Games: %d-%d | Points: %s | Source: %s",
            self._score_changes,
            change.point_winner,
            "BREAK!" if change.is_break else "hold",
            new_s.sets_p1, new_s.sets_p2,
            new_s.games_p1, new_s.games_p2,
            new_s.point_text,
            change.source.value,
        )

    def on_odds_update(self, snapshot: MarketSnapshot) -> None:
        """
        Called every time odds change in the stream.
        Updates the match session with new odds and runs the full
        trading pipeline.
        """
        if not self.executor.mapping:
            return

        mapping = self.executor.mapping
        p1_runner = snapshot.runners.get(mapping.player1_selection_id)
        p2_runner = snapshot.runners.get(mapping.player2_selection_id)

        if not p1_runner or not p2_runner:
            return

        odds_server = p1_runner.best_back if p1_runner.best_back > 0 else p1_runner.last_traded_price
        odds_receiver = p2_runner.best_back if p2_runner.best_back > 0 else p2_runner.last_traded_price

        if odds_server <= 0 or odds_receiver <= 0:
            return

        # Only process if odds actually changed
        if (abs(odds_server - self._last_odds_server) < 0.005 and
            abs(odds_receiver - self._last_odds_receiver) < 0.005):
            return

        self._last_odds_server = odds_server
        self._last_odds_receiver = odds_receiver
        self._tick_count += 1

        # Update session with live market odds
        self.session.market_odds_server = odds_server
        self.session.market_odds_receiver = odds_receiver

        # Estimate break/hold odds from match odds
        if odds_server > 0:
            implied_hold = 1 / (1 - 1 / odds_server * 0.35) if odds_server > 1 else 1.5
            implied_break = odds_server * 1.5
            self.session.break_odds = min(max(implied_break, 1.5), 10.0)
            self.session.hold_odds = min(max(implied_hold, 1.1), 5.0)

        # Build full frame (runs Markov + ML + trading engine)
        frame = self.session.build_frame()

        # Log state
        logger.info(
            "TICK %d | %s | Odds: %.2f / %.2f | True P: %.3f | Edge: %.2f%% | Signal: %s",
            self._tick_count,
            frame.match.score.point_display,
            odds_server, odds_receiver,
            frame.probability.true_probability,
            frame.probability.edge_pct * 100,
            frame.signal.action.value,
        )

        # Execute signal if actionable
        if frame.signal.action in (Action.ENTER, Action.SCALP):
            result = self.executor.execute_signal(
                frame.signal,
                bankroll=frame.position.account_a_balance,
                current_snapshot=snapshot,
            )
            if result.filled:
                logger.info(
                    "✅ EXECUTED: %s @ %.2f (slippage=%d ticks, delay=%dms)",
                    frame.signal.action.value,
                    result.fill_price,
                    result.slippage_ticks,
                    result.delay_ms,
                )

        # Execute hedge if needed
        if frame.hedge.should_hedge:
            hedge_result = self.executor.execute_hedge(
                frame.hedge,
                original_selection_id=mapping.player1_selection_id,
                stake=frame.position.stake,
                current_snapshot=snapshot,
            )
            if hedge_result.filled:
                logger.info(
                    "🛡️ HEDGED: LAY @ %.2f (reason=%s)",
                    hedge_result.fill_price,
                    frame.hedge.reason,
                )

    async def run(self, market_id: Optional[str] = None) -> None:
        """
        Main async loop — interleaves:
          1) Score feed polling  (every score_poll_interval seconds)
          2) Betfair odds stream (every poll_interval seconds)

        Can run in score-only mode (market_id=None) for testing
        without a Betfair connection.
        """
        self._running = True
        score_only = market_id is None or self.client is None

        if score_only:
            logger.info("🚀 Score-only mode — no Betfair connection")
        else:
            logger.info("🚀 Trading loop started for market %s", market_id)
            self.client.subscribe_market(market_id, on_update=self.on_odds_update)

        keepalive_interval = 600  # 10 min
        last_keepalive = time.time()

        # Attach the live feed
        if self.feed:
            player1 = self.session.match_engine.info.player1_name
            player2 = self.session.match_engine.info.player2_name
            self.feed.on_change(self.on_score_change)
            attached = await asyncio.get_event_loop().run_in_executor(
                None, self.feed.attach, player1, player2,
            )
            if attached:
                logger.info("📡 Live feed attached for %s vs %s", player1, player2)
            else:
                logger.warning("⚠️  Could not attach live feed — will rely on manual point input")

        try:
            while self._running:
                now = time.time()

                # ── Poll score feed ──────────────────────────────────
                if self.feed and (now - self._last_score_poll) >= self.score_poll_interval:
                    try:
                        changes = await asyncio.get_event_loop().run_in_executor(
                            None, self.feed.poll,
                        )
                        # Changes are handled by on_score_change callback
                    except Exception as e:
                        logger.debug("Score poll error: %s", e)
                    self._last_score_poll = now

                # ── Poll Betfair odds ────────────────────────────────
                if not score_only:
                    snapshot = self.client.poll_stream()
                    if not snapshot:
                        snapshot = self.client.get_market_odds(market_id)
                        if snapshot:
                            self.on_odds_update(snapshot)

                    # Keep-alive
                    if now - last_keepalive > keepalive_interval:
                        self.client.keep_alive()
                        last_keepalive = now

                await asyncio.sleep(self.poll_interval)

        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully stop trading."""
        self._running = False

        if self.executor:
            cancelled = self.executor.cancel_all()
            logger.info("Cancelled %d open orders", cancelled)

        if self.client:
            self.client.stop_stream()

        logger.info(
            "Trading loop stopped after %d ticks, %d score changes",
            self._tick_count, self._score_changes,
        )

        # Print summary
        if self.executor:
            log = self.executor.execution_log
            if log:
                logger.info("─── Execution Summary ───")
                logger.info("Total executions: %d", len(log))
                successful = sum(1 for e in log if e.get("success", e.get("dry_run")))
                logger.info("Successful: %d", successful)
                for entry in log[-10:]:
                    logger.info("  %s", entry)


def find_and_map_market(
    client: BetfairClient,
    player1: str,
    player2: str,
) -> Optional[MarketMapping]:
    """Search Betfair for the match and create a mapping."""
    # Try searching by player1 name
    markets = client.find_tennis_markets(player_name=player1, in_play_only=False)

    if not markets:
        # Try player2
        markets = client.find_tennis_markets(player_name=player2, in_play_only=False)

    if not markets:
        logger.error("No markets found for %s vs %s", player1, player2)
        return None

    # Find the match that contains both players
    for market in markets:
        names = [r.runner_name.lower() for r in market.runners.values()]
        event_lower = market.event_name.lower()

        p1_match = any(player1.lower().split()[-1] in n for n in names) or player1.lower().split()[-1] in event_lower
        p2_match = any(player2.lower().split()[-1] in n for n in names) or player2.lower().split()[-1] in event_lower

        if p1_match or p2_match:
            runners = list(market.runners.values())
            if len(runners) >= 2:
                # Map player1 to the first matching runner
                p1_sel = runners[0].selection_id
                p2_sel = runners[1].selection_id

                for r in runners:
                    if player1.lower().split()[-1] in r.runner_name.lower():
                        p1_sel = r.selection_id
                    elif player2.lower().split()[-1] in r.runner_name.lower():
                        p2_sel = r.selection_id

                mapping = MarketMapping(
                    market_id=market.market_id,
                    player1_selection_id=p1_sel,
                    player2_selection_id=p2_sel,
                    player1_name=player1,
                    player2_name=player2,
                )
                logger.info(
                    "Found market: %s (%s) → P1=%d, P2=%d",
                    market.event_name, market.market_id,
                    p1_sel, p2_sel,
                )
                return mapping

    # If no exact match, use the first market
    market = markets[0]
    runners = list(market.runners.values())
    if len(runners) >= 2:
        mapping = MarketMapping(
            market_id=market.market_id,
            player1_selection_id=runners[0].selection_id,
            player2_selection_id=runners[1].selection_id,
            player1_name=player1,
            player2_name=player2,
        )
        logger.warning(
            "Using best-guess market: %s (%s)",
            market.event_name, market.market_id,
        )
        return mapping

    return None


def main():
    parser = argparse.ArgumentParser(description="Live Betfair Trading Loop")
    parser.add_argument("--player1", required=True, help="Player 1 name")
    parser.add_argument("--player2", required=True, help="Player 2 name")
    parser.add_argument("--surface", default="Hard")
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--p1-serve", type=float, default=63.0)
    parser.add_argument("--p2-serve", type=float, default=63.0)
    parser.add_argument("--p1-return", type=float, default=35.0)
    parser.add_argument("--p2-return", type=float, default=35.0)
    parser.add_argument("--p1-rank", type=int, default=50)
    parser.add_argument("--p2-rank", type=int, default=50)
    parser.add_argument("--bankroll", type=float, default=10000.0)
    parser.add_argument("--max-stake", type=float, default=50.0)
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Paper trading mode (no real orders)")
    parser.add_argument("--live", action="store_true",
                        help="Enable real order execution (overrides --dry-run)")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--score-poll-interval", type=float, default=2.0,
                        help="Seconds between live score polls")
    parser.add_argument("--market-id", type=str, default=None,
                        help="Directly specify Betfair market ID")
    parser.add_argument("--score-only", action="store_true",
                        help="Run with live score feed only — no Betfair connection needed")
    parser.add_argument("--score-source", type=str, default=None,
                        choices=["sofascore", "espn", "api-tennis"],
                        help="Preferred score data source")
    parser.add_argument("--api-tennis-key", type=str, default=None,
                        help="API key for api-tennis.com (paid)")

    args = parser.parse_args()

    dry_run = not args.live

    if not args.score_only and not dry_run:
        logger.warning("⚠️  LIVE MODE — Real money will be used!")
        confirm = input("Type 'YES' to confirm live trading: ")
        if confirm != "YES":
            logger.info("Aborted.")
            return

    # ── Set up live score feed ────────────────────────────────────────────
    api_key = args.api_tennis_key or os.environ.get("API_TENNIS_KEY")
    feed = LiveMatchFeed(api_tennis_key=api_key)

    if args.score_source:
        source_map = {
            "sofascore": DataSource.SOFASCORE,
            "espn": DataSource.ESPN,
            "api-tennis": DataSource.API_TENNIS,
        }
        feed.preferred_source = source_map.get(args.score_source)
    logger.info("📡 Live score feed initialized (sources: %s)", 
                ", ".join(a.source.value for a in feed.adapters))

    # ── Set up match session ──────────────────────────────────────────────
    setup_req = MatchSetupRequest(
        player1_name=args.player1,
        player2_name=args.player2,
        surface=args.surface,
        best_of=args.best_of,
        initial_server=1,
        p1_serve_pct=args.p1_serve,
        p2_serve_pct=args.p2_serve,
        p1_return_pct=args.p1_return,
        p2_return_pct=args.p2_return,
        p1_rank=args.p1_rank,
        p2_rank=args.p2_rank,
        bankroll=args.bankroll,
    )
    session = MatchSession(setup_req)

    client = None
    executor = None
    mapping = None

    if not args.score_only:
        # ── Login to Betfair ──────────────────────────────────────────────
        logger.info("Logging into Betfair...")
        client = BetfairClient.from_env()
        if not client.login():
            logger.error("Login failed. Check credentials and certs.")
            sys.exit(1)

        funds = client.get_account_funds()
        logger.info("Account funds: %s", funds)

        # ── Find market ───────────────────────────────────────────────────
        if args.market_id:
            snapshot = client.get_market_odds(args.market_id)
            if not snapshot or len(snapshot.runners) < 2:
                logger.error("Market %s not found or has no runners", args.market_id)
                client.logout()
                sys.exit(1)

            runners = list(snapshot.runners.values())
            mapping = MarketMapping(
                market_id=args.market_id,
                player1_selection_id=runners[0].selection_id,
                player2_selection_id=runners[1].selection_id,
                player1_name=args.player1,
                player2_name=args.player2,
            )
        else:
            mapping = find_and_map_market(client, args.player1, args.player2)

        if not mapping:
            logger.error("Could not find market. Try specifying --market-id directly.")
            client.logout()
            sys.exit(1)

        # ── Set up executor ───────────────────────────────────────────────
        config = ExecutionConfig(
            max_stake=args.max_stake,
            dry_run=dry_run,
        )
        executor = BetfairExecutor(client, config=config)
        executor.set_market_mapping(mapping)

    # ── Run loop ──────────────────────────────────────────────────────────
    loop = LiveTradingLoop(
        client=client,
        executor=executor,
        session=session,
        feed=feed,
        poll_interval=args.poll_interval,
        score_poll_interval=args.score_poll_interval,
    )

    # Graceful shutdown
    def shutdown(sig, frame):
        logger.info("Received signal %s — stopping...", sig)
        loop.stop()
        if client:
            client.logout()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    market_id = mapping.market_id if mapping else None

    try:
        asyncio.run(loop.run(market_id))
    finally:
        if client:
            client.logout()


if __name__ == "__main__":
    main()
