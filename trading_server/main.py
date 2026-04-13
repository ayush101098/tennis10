"""
FastAPI Trading Server
======================
WebSocket + REST for the real-time trading terminal.

  WS  /ws/{match_id}     → pushes TradeBoxFrame every tick
  POST /match/setup       → create match
  POST /match/{id}/point  → register a point
  POST /match/{id}/odds   → update market odds
  GET  /match/{id}/state  → current state snapshot
  GET  /matches           → list active matches
  GET  /health            → server health

Run:
  uvicorn trading_server.main:app --reload --port 8888
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    Action,
    EnsembleData,
    HedgeSignal,
    LiveStatsSnapshot,
    MatchListItem,
    MatchSetupRequest,
    MatchState,
    ModelContributionData,
    NextAction,
    OddsUpdateRequest,
    PlayerStatsSnapshot,
    PointUpdateRequest,
    PositionType,
    TradeBoxFrame,
    TradingSignal,
)
from .match_state_engine import MatchStateEngine
from .true_probability_engine import TrueProbabilityEngine
from .trading_engine import TradingEngine
from .hedge_engine import HedgeEngine
from .deuce_loop import DeuceLoop
from .execution_simulator import ExecutionSimulator
from .risk_manager import RiskManager
from .dual_account import DualAccountManager
from .live_stats import LiveMatchStatsAccumulator, PointStats
from .live_features import LiveFeatureEngine, PlayerProfile
from .ml_predictor import MLPredictor
from .ensemble_probability import EnsembleProbabilityEngine
from .live_feed import LiveMatchFeed, DataSource, MatchInfo


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Tennis Trading Terminal", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Shared ML predictor (loaded once) ────────────────────────────────────────

_ml_predictor = MLPredictor()


# ─── Per-match session ───────────────────────────────────────────────────────

class MatchSession:
    """Holds all engines for one match."""

    def __init__(self, req: MatchSetupRequest) -> None:
        self.match_engine = MatchStateEngine()
        self.match_engine.setup(req)
        self.true_p = TrueProbabilityEngine(
            p1_serve_pct=req.p1_serve_pct,
            p2_serve_pct=req.p2_serve_pct,
            p1_return_pct=req.p1_return_pct,
            p2_return_pct=req.p2_return_pct,
            p1_rank=req.p1_rank,
            p2_rank=req.p2_rank,
        )
        self.executor = ExecutionSimulator()
        self.risk = RiskManager()
        self.accounts = DualAccountManager(bankroll=req.bankroll)
        self.deuce = DeuceLoop()

        # ── Live stats + ML pipeline ──────────────────────────────────────
        self.live_stats = LiveMatchStatsAccumulator()

        p1_profile = PlayerProfile(
            rank=req.p1_rank,
            ranking_points=req.p1_ranking_points,
            serve_pct=req.p1_serve_pct / 100.0,
            return_pct=req.p1_return_pct / 100.0,
            first_serve_pct=req.p1_first_serve_pct / 100.0,
            first_serve_win_pct=req.p1_first_serve_win_pct / 100.0,
            second_serve_win_pct=req.p1_second_serve_win_pct / 100.0,
            bp_save_pct=req.p1_bp_save_pct / 100.0,
            win_rate=req.p1_win_rate / 100.0,
        )
        p2_profile = PlayerProfile(
            rank=req.p2_rank,
            ranking_points=req.p2_ranking_points,
            serve_pct=req.p2_serve_pct / 100.0,
            return_pct=req.p2_return_pct / 100.0,
            first_serve_pct=req.p2_first_serve_pct / 100.0,
            first_serve_win_pct=req.p2_first_serve_win_pct / 100.0,
            second_serve_win_pct=req.p2_second_serve_win_pct / 100.0,
            bp_save_pct=req.p2_bp_save_pct / 100.0,
            win_rate=req.p2_win_rate / 100.0,
        )
        self.feature_engine = LiveFeatureEngine(
            p1_profile, p2_profile,
            surface=req.surface,
            tournament_level=req.tournament_level,
        )
        self.ensemble = EnsembleProbabilityEngine(
            markov=self.true_p,
            feature_engine=self.feature_engine,
            ml_predictor=_ml_predictor,
        )

        # Mutable odds (updated by user or feed)
        self.market_odds_server: float = 1.80
        self.market_odds_receiver: float = 2.10
        self.break_odds: float = 3.00
        self.hold_odds: float = 1.40

    def build_frame(self) -> TradeBoxFrame:
        ms = self.match_engine.snapshot()
        score = ms.score
        server = ms.info.server

        # ── Ensemble probability (Markov + ML) ───────────────────────────
        prob, model_contribs = self.ensemble.evaluate(
            score, server,
            self.market_odds_server,
            self.market_odds_receiver,
            self.live_stats,
        )

        risk = self.risk.update(self.accounts.position)

        signal = TradingEngine.decide(
            score, prob, risk,
            self.market_odds_server,
            self.market_odds_receiver,
        )

        hedge = HedgeEngine.evaluate(
            score,
            self.match_engine.previous_state_key(),
            self.accounts.position,
            self.break_odds,
            self.hold_odds,
        )

        # Deuce loop
        deuce = self.deuce.tick(
            score, prob,
            self.match_engine.previous_state_key(),
            self.break_odds,
        )

        # Mark positions to market
        self.accounts.mark_to_market(self.market_odds_server)

        # Auto-execute signals
        exec_result = None
        has_open_entries = any(t.account == "A" and t.is_open for t in self.accounts.position.open_trades)
        has_open_hedges = any(t.account == "B" and t.is_open for t in self.accounts.position.open_trades)

        if signal.action in (Action.ENTER, Action.SCALP) and not has_open_entries:
            exec_result = self.executor.execute(self.market_odds_server, is_back=True)
            self.accounts.open_entry(signal, exec_result, score.game_state_key)

        if hedge.should_hedge and has_open_entries and not has_open_hedges:
            he = self.executor.execute(self.hold_odds, is_back=False)
            self.accounts.open_hedge(hedge, he, score.game_state_key)

        # Next action
        next_act = NextAction(
            if_point_won="HOLD" if signal.action != Action.HEDGE else "EXIT",
            if_point_lost="HEDGE" if self.accounts.position.open_trades else "SKIP",
            hedge_size_if_lost=hedge.hedge_size if hedge.should_hedge else 0.0,
        )

        # ── Build live stats snapshot ─────────────────────────────────────
        raw = self.live_stats.snapshot()
        stats_snap = LiveStatsSnapshot(
            points_played=raw["points_played"],
            player1=PlayerStatsSnapshot(**raw["player1"]),
            player2=PlayerStatsSnapshot(**raw["player2"]),
        )

        # ── Ensemble data for UI ──────────────────────────────────────────
        blend_w = min(1.0, self.live_stats.points_played / 20.0)
        blended = self.feature_engine.blended_summary(
            self.live_stats.p1, self.live_stats.p2
        )
        ensemble_data = EnsembleData(
            models=[ModelContributionData(**mc) for mc in model_contribs],
            blend_weight=round(blend_w, 4),
            blended_stats=blended,
        )

        return TradeBoxFrame(
            match=ms,
            probability=prob,
            signal=signal,
            hedge=hedge,
            position=self.accounts.position.model_copy(),
            next_action=next_act,
            deuce_loop=deuce,
            risk=risk,
            execution=exec_result,
            state_performance=self.accounts.get_state_performance(),
            trade_log=self.accounts.trade_log[-50:],
            live_stats=stats_snap,
            ensemble=ensemble_data,
            server_ts=time.time(),
        )


# ─── Global state ────────────────────────────────────────────────────────────

sessions: Dict[str, MatchSession] = {}
subscribers: Dict[str, Set[WebSocket]] = {}  # match_id → set of WS


async def broadcast(match_id: str) -> None:
    """Push latest frame to all WS subscribers."""
    if match_id not in sessions:
        return
    frame = sessions[match_id].build_frame()
    payload = frame.model_dump_json()
    dead: List[WebSocket] = []
    for ws in subscribers.get(match_id, set()):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        subscribers[match_id].discard(ws)


# ─── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "active_matches": len(sessions), "ts": time.time()}


@app.post("/match/setup")
async def setup_match(req: MatchSetupRequest):
    sess = MatchSession(req)
    mid = sess.match_engine.info.match_id
    sessions[mid] = sess
    subscribers.setdefault(mid, set())
    return {"match_id": mid, "state": sess.build_frame().model_dump()}


@app.post("/match/{match_id}/point")
async def register_point(match_id: str, req: PointUpdateRequest):
    sess = sessions.get(match_id)
    if not sess:
        return {"error": "Match not found"}

    sess.market_odds_server = req.market_odds_server
    sess.market_odds_receiver = req.market_odds_receiver

    # Determine who is serving before the point is registered
    server = sess.match_engine.info.server
    winner_player = 1 if req.winner.value == "SERVER" else 2
    # Translate SERVER/RECEIVER winner into player number
    point_won_by = server if req.winner.value == "SERVER" else (3 - server)

    # ── Feed live stats ───────────────────────────────────────────────────
    pt_stats = None
    if req.stats:
        pt_stats = PointStats(
            is_ace=req.stats.is_ace,
            is_double_fault=req.stats.is_double_fault,
            is_first_serve_in=req.stats.is_first_serve_in,
            is_winner=req.stats.is_winner,
            is_unforced_error=req.stats.is_unforced_error,
            is_break_point=req.stats.is_break_point,
        )
    else:
        # Auto-infer basic stats when operator doesn't annotate
        pt_stats = PointStats(is_first_serve_in=True)  # assume 1st serve in

    sess.live_stats.register_point(server, point_won_by, pt_stats)

    # ── Update score ──────────────────────────────────────────────────────
    sess.match_engine.register_point(req.winner)

    # If game ended, close trades & track serve games
    if sess.match_engine.score.server_points == 0 and sess.match_engine.score.receiver_points == 0:
        server_held = req.winner.value == "SERVER"
        # Track serve game completion
        if server_held:
            sess.live_stats.register_game_won(server, was_serving=True)
        else:
            other = 3 - server
            sess.live_stats.register_game_won(other, was_serving=False)

        pnl = sess.accounts.close_all(server_held)
        if pnl < 0:
            sess.risk.register_loss()
        elif pnl > 0:
            sess.risk.register_win()
        sess.deuce.reset()

    frame = sess.build_frame()
    await broadcast(match_id)
    return frame.model_dump()


@app.post("/match/{match_id}/odds")
async def update_odds(match_id: str, req: OddsUpdateRequest):
    sess = sessions.get(match_id)
    if not sess:
        return {"error": "Match not found"}
    sess.break_odds = req.break_odds
    sess.hold_odds = req.hold_odds
    frame = sess.build_frame()
    await broadcast(match_id)
    return frame.model_dump()


@app.get("/match/{match_id}/state")
async def get_state(match_id: str):
    sess = sessions.get(match_id)
    if not sess:
        return {"error": "Match not found"}
    return sess.build_frame().model_dump()


@app.get("/matches")
async def list_matches():
    items: List[dict] = []
    for mid, sess in sessions.items():
        ms = sess.match_engine.snapshot()
        items.append(MatchListItem(
            match_id=mid,
            label=f"{ms.info.player1_name} vs {ms.info.player2_name}",
            is_live=ms.is_live,
            score_summary=ms.score.point_display,
        ).model_dump())
    return items


# ─── Live Match Data Feed ────────────────────────────────────────────────────

import os as _os

_api_tennis_key = _os.environ.get("API_TENNIS_KEY")
_live_feed = LiveMatchFeed(api_tennis_key=_api_tennis_key)
_feed_tasks: Dict[str, asyncio.Task] = {}  # match_id → background poll task


@app.get("/live/matches")
async def live_matches():
    """
    Discover live tennis matches from all available data sources.
    Returns matches from Sofascore, ESPN, and api-tennis (if key set).
    Covers ATP, WTA, ITF, and Challenger tours.
    """
    try:
        raw = await asyncio.get_event_loop().run_in_executor(
            None, _live_feed.get_live_matches,
        )
    except Exception:
        raw = []

    all_matches: List[dict] = []
    for m in raw:
        score = m.get("score")
        source = m.get("source", DataSource.ESPN)
        source_val = source.value if isinstance(source, DataSource) else str(source)
        all_matches.append({
            "id": m.get("source_id", ""),
            "player1": m.get("p1_name", ""),
            "player2": m.get("p2_name", ""),
            "tournament": m.get("tournament", ""),
            "round": m.get("round", ""),
            "tour": m.get("tour", "ATP"),
            "source": source_val,
            "score": {
                "sets_p1": score.sets_p1 if score else 0,
                "sets_p2": score.sets_p2 if score else 0,
                "games_p1": score.games_p1 if score else 0,
                "games_p2": score.games_p2 if score else 0,
                "point_text": score.point_text if score else "",
            } if score else None,
            "server": score.server if score else 0,
            "status": "live" if (score and score.is_live) else "pre",
        })

    return {"matches": all_matches, "sources": [a.source.value for a in _live_feed.adapters]}


@app.post("/live/attach/{match_id}")
async def attach_live_feed(
    match_id: str,
    player1: Optional[str] = None,
    player2: Optional[str] = None,
):
    """
    Attach a live score feed to an existing match session.
    The feed will auto-poll and register points into the match.
    """
    sess = sessions.get(match_id)
    if not sess:
        return {"error": "Match not found"}

    p1 = player1 or sess.match_engine.info.player1_name
    p2 = player2 or sess.match_engine.info.player2_name

    # Attach feed
    attached = await asyncio.get_event_loop().run_in_executor(
        None, _live_feed.attach, p1, p2,
    )

    if not attached:
        return {"error": f"Could not find live match for {p1} vs {p2}"}

    # Start background polling task
    if match_id in _feed_tasks:
        _feed_tasks[match_id].cancel()

    async def _poll_loop():
        while True:
            try:
                changes = await asyncio.get_event_loop().run_in_executor(
                    None, _live_feed.poll,
                )
                for change in changes:
                    from .models import PointWinner as PW
                    winner = PW.SERVER if change.point_winner == "SERVER" else PW.RECEIVER
                    server = sess.match_engine.info.server
                    point_won_by = server if winner == PW.SERVER else (3 - server)

                    pt_stats = PointStats(
                        is_ace=change.is_ace,
                        is_double_fault=change.is_double_fault,
                        is_first_serve_in=change.is_first_serve_in,
                        is_break_point=change.is_break_point,
                    )
                    sess.live_stats.register_point(server, point_won_by, pt_stats)
                    sess.match_engine.register_point(winner)

                    # Close trades on game end
                    if (sess.match_engine.score.server_points == 0 and
                            sess.match_engine.score.receiver_points == 0):
                        server_held = winner == PW.SERVER
                        if server_held:
                            sess.live_stats.register_game_won(server, was_serving=True)
                        else:
                            sess.live_stats.register_game_won(3 - server, was_serving=False)
                        pnl = sess.accounts.close_all(server_held)
                        if pnl < 0:
                            sess.risk.register_loss()
                        elif pnl > 0:
                            sess.risk.register_win()
                        sess.deuce.reset()

                    await broadcast(match_id)

            except Exception:
                pass
            await asyncio.sleep(2.0)

    _feed_tasks[match_id] = asyncio.create_task(_poll_loop())

    return {
        "status": "attached",
        "player1": p1,
        "player2": p2,
        "source": _live_feed.current_match.source.value if _live_feed.current_match else "unknown",
    }


@app.post("/live/detach/{match_id}")
async def detach_live_feed(match_id: str):
    """Stop the live score feed for a match."""
    if match_id in _feed_tasks:
        _feed_tasks[match_id].cancel()
        del _feed_tasks[match_id]
        return {"status": "detached"}
    return {"status": "no feed active"}


@app.get("/live/feed/{match_id}/status")
async def feed_status(match_id: str):
    """Check if a live feed is active for a match."""
    active = match_id in _feed_tasks and not _feed_tasks[match_id].done()
    return {
        "match_id": match_id,
        "feed_active": active,
        "current_match": {
            "player1": _live_feed.current_match.player1,
            "player2": _live_feed.current_match.player2,
            "source": _live_feed.current_match.source.value,
        } if _live_feed.current_match else None,
    }


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/{match_id}")
async def ws_endpoint(websocket: WebSocket, match_id: str):
    await websocket.accept()
    subscribers.setdefault(match_id, set())
    subscribers[match_id].add(websocket)

    # Send initial frame
    if match_id in sessions:
        frame = sessions[match_id].build_frame()
        await websocket.send_text(frame.model_dump_json())

    try:
        while True:
            data = await websocket.receive_text()
            # Client can send point updates via WS too
            if data == "ping":
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        subscribers[match_id].discard(websocket)
