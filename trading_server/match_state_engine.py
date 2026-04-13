"""
Match State Engine — point-by-point state machine.
Deterministic score transitions.  No probability logic here.
"""

from __future__ import annotations
import time
from typing import Optional, Tuple
from .models import (
    ScoreState, MatchInfo, MatchState, PointWinner, MatchSetupRequest,
)


class MatchStateEngine:
    """Pure state machine: score tracking only."""

    def __init__(self) -> None:
        self.info = MatchInfo()
        self.score = ScoreState()
        self.last_point_winner: Optional[PointWinner] = None
        self.is_live = False
        self.point_history: list[PointWinner] = []
        self._prev_score: Optional[ScoreState] = None

    # ── Setup ─────────────────────────────────────────────────────────────
    def setup(self, req: MatchSetupRequest) -> MatchState:
        self.info = MatchInfo(
            player1_name=req.player1_name,
            player2_name=req.player2_name,
            surface=req.surface,
            best_of=req.best_of,
            tournament=req.tournament,
            server=req.initial_server,
        )
        self.score = ScoreState()
        self.is_live = True
        self.point_history = []
        return self.snapshot()

    # ── Point Update ──────────────────────────────────────────────────────
    def register_point(self, winner: PointWinner) -> MatchState:
        """Apply one point and return the new state."""
        self._prev_score = self.score.model_copy()
        self.last_point_winner = winner
        self.point_history.append(winner)

        s = self.score
        is_server_point = winner == PointWinner.SERVER

        if s.is_tiebreak:
            self._apply_tiebreak_point(is_server_point)
        else:
            self._apply_regular_point(is_server_point)

        return self.snapshot()

    # ── Regular game logic ────────────────────────────────────────────────
    def _apply_regular_point(self, server_wins: bool) -> None:
        s = self.score
        if server_wins:
            s.server_points += 1
        else:
            s.receiver_points += 1

        # Check game won
        sp, rp = s.server_points, s.receiver_points
        if sp >= 4 and sp - rp >= 2:
            self._server_wins_game()
        elif rp >= 4 and rp - sp >= 2:
            self._receiver_wins_game()

    def _apply_tiebreak_point(self, server_wins: bool) -> None:
        s = self.score
        if server_wins:
            s.server_points += 1
        else:
            s.receiver_points += 1

        sp, rp = s.server_points, s.receiver_points
        if sp >= 7 and sp - rp >= 2:
            self._server_wins_set()
        elif rp >= 7 and rp - sp >= 2:
            self._receiver_wins_set()
        else:
            # Alternate serve every 2 points in tiebreak
            total = sp + rp
            if total > 0 and total % 2 == 0:
                self._switch_server()

    def _server_wins_game(self) -> None:
        s = self.score
        s.server_games += 1
        s.server_points = 0
        s.receiver_points = 0
        self._check_set()
        if self.is_live:
            self._switch_server()

    def _receiver_wins_game(self) -> None:
        s = self.score
        s.receiver_games += 1
        s.server_points = 0
        s.receiver_points = 0
        self._check_set()
        if self.is_live:
            self._switch_server()

    def _check_set(self) -> None:
        s = self.score
        sg, rg = s.server_games, s.receiver_games
        if sg >= 6 and sg - rg >= 2:
            self._server_wins_set()
        elif rg >= 6 and rg - sg >= 2:
            self._receiver_wins_set()
        elif sg == 6 and rg == 6:
            s.is_tiebreak = True

    def _server_wins_set(self) -> None:
        s = self.score
        s.server_sets += 1
        s.server_games = 0
        s.receiver_games = 0
        s.server_points = 0
        s.receiver_points = 0
        s.is_tiebreak = False
        self._check_match()

    def _receiver_wins_set(self) -> None:
        s = self.score
        s.receiver_sets += 1
        s.server_games = 0
        s.receiver_games = 0
        s.server_points = 0
        s.receiver_points = 0
        s.is_tiebreak = False
        self._check_match()

    def _check_match(self) -> None:
        sets_to_win = (self.info.best_of // 2) + 1
        if (self.score.server_sets >= sets_to_win or
                self.score.receiver_sets >= sets_to_win):
            self.is_live = False

    def _switch_server(self) -> None:
        self.info.server = 2 if self.info.server == 1 else 1

    # ── Query helpers ─────────────────────────────────────────────────────
    def previous_state_key(self) -> Optional[str]:
        if self._prev_score is None:
            return None
        return self._prev_score.game_state_key

    def current_state_key(self) -> str:
        return self.score.game_state_key

    def is_deuce(self) -> bool:
        return self.current_state_key() == "DEUCE"

    def is_break_point(self) -> bool:
        k = self.current_state_key()
        return k in ("AD-OUT", "30-40", "15-40", "0-40")

    def server_name(self) -> str:
        return (self.info.player1_name if self.info.server == 1
                else self.info.player2_name)

    def receiver_name(self) -> str:
        return (self.info.player2_name if self.info.server == 1
                else self.info.player1_name)

    def snapshot(self) -> MatchState:
        self.score._recompute()
        return MatchState(
            info=self.info.model_copy(),
            score=self.score.model_copy(),
            last_point_winner=self.last_point_winner,
            is_live=self.is_live,
            timestamp=time.time(),
        )
