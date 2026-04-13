"""
Live Match Stats Accumulator
=============================
Tracks point-by-point statistics for both players during a live match.
Every point input can carry optional stat metadata (was it an ace? DF?
first serve in? winner? UE? break point?). The accumulator maintains
running totals and computes derived percentages in real time.

These stats feed directly into the ML feature engine to produce
data-driven True P estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Raw per-point annotation (sent with each point) ──────────────────────────

@dataclass
class PointStats:
    """Optional metadata the operator attaches to each point."""
    is_ace: bool = False
    is_double_fault: bool = False
    is_first_serve_in: bool = False        # True if 1st serve landed in
    is_first_serve_point_won: bool = False  # True if won the point on 1st serve
    is_second_serve_point_won: bool = False # True if won the point on 2nd serve
    is_winner: bool = False                 # forehand/backhand winner
    is_unforced_error: bool = False
    is_break_point: bool = False            # was this point a break point?
    is_break_point_converted: bool = False  # returner won the break point
    rally_length: int = 0                   # 0 = unknown


# ── Per-player accumulator ───────────────────────────────────────────────────

@dataclass
class PlayerMatchStats:
    """Running totals for one player during a match."""
    # Serve
    serve_points_played: int = 0
    first_serves_in: int = 0
    first_serve_points_won: int = 0
    second_serve_points_won: int = 0
    aces: int = 0
    double_faults: int = 0
    serve_games: int = 0
    # Break points
    break_points_faced: int = 0
    break_points_saved: int = 0
    # Return
    return_points_played: int = 0
    return_points_won: int = 0
    # General
    winners: int = 0
    unforced_errors: int = 0
    total_points_won: int = 0
    total_points_played: int = 0
    games_won: int = 0
    sets_won: int = 0

    # ── Derived percentages (computed, not stored) ────────────────────────

    @property
    def first_serve_pct(self) -> float:
        if self.serve_points_played == 0:
            return 0.0
        return self.first_serves_in / self.serve_points_played

    @property
    def first_serve_win_pct(self) -> float:
        if self.first_serves_in == 0:
            return 0.0
        return self.first_serve_points_won / self.first_serves_in

    @property
    def second_serve_win_pct(self) -> float:
        second_serves = self.serve_points_played - self.first_serves_in
        if second_serves <= 0:
            return 0.0
        return self.second_serve_points_won / second_serves

    @property
    def serve_points_won_pct(self) -> float:
        """WSP — overall serve points won %."""
        if self.serve_points_played == 0:
            return 0.0
        total_won = self.first_serve_points_won + self.second_serve_points_won
        return total_won / self.serve_points_played

    @property
    def return_points_won_pct(self) -> float:
        """WRP — return points won %."""
        if self.return_points_played == 0:
            return 0.0
        return self.return_points_won / self.return_points_played

    @property
    def break_point_save_pct(self) -> float:
        if self.break_points_faced == 0:
            return 0.0
        return self.break_points_saved / self.break_points_faced

    @property
    def aces_per_game(self) -> float:
        if self.serve_games == 0:
            return 0.0
        return self.aces / self.serve_games

    @property
    def df_per_game(self) -> float:
        if self.serve_games == 0:
            return 0.0
        return self.double_faults / self.serve_games

    @property
    def win_rate(self) -> float:
        if self.total_points_played == 0:
            return 0.0
        return self.total_points_won / self.total_points_played

    def to_dict(self) -> dict:
        """Serialise for JSON / WS frame."""
        return {
            # raw
            "serve_points_played": self.serve_points_played,
            "first_serves_in": self.first_serves_in,
            "first_serve_points_won": self.first_serve_points_won,
            "second_serve_points_won": self.second_serve_points_won,
            "aces": self.aces,
            "double_faults": self.double_faults,
            "serve_games": self.serve_games,
            "break_points_faced": self.break_points_faced,
            "break_points_saved": self.break_points_saved,
            "return_points_played": self.return_points_played,
            "return_points_won": self.return_points_won,
            "winners": self.winners,
            "unforced_errors": self.unforced_errors,
            "total_points_won": self.total_points_won,
            "total_points_played": self.total_points_played,
            "games_won": self.games_won,
            "sets_won": self.sets_won,
            # derived
            "first_serve_pct": round(self.first_serve_pct, 4),
            "first_serve_win_pct": round(self.first_serve_win_pct, 4),
            "second_serve_win_pct": round(self.second_serve_win_pct, 4),
            "serve_points_won_pct": round(self.serve_points_won_pct, 4),
            "return_points_won_pct": round(self.return_points_won_pct, 4),
            "break_point_save_pct": round(self.break_point_save_pct, 4),
            "aces_per_game": round(self.aces_per_game, 4),
            "df_per_game": round(self.df_per_game, 4),
            "win_rate": round(self.win_rate, 4),
        }


# ── Match-level accumulator ─────────────────────────────────────────────────

class LiveMatchStatsAccumulator:
    """
    Maintains running stats for both players across a live match.

    Call `register_point()` after every point with the stats metadata.
    The accumulator updates the correct player's counters based on who
    was serving and who won the point.
    """

    def __init__(self) -> None:
        self.p1 = PlayerMatchStats()
        self.p2 = PlayerMatchStats()
        self.points_played: int = 0

    def register_point(
        self,
        server: int,          # 1 or 2
        point_won_by: int,    # 1 or 2  (who won the rally)
        stats: Optional[PointStats] = None,
    ) -> None:
        """Update accumulators after a single point."""
        if stats is None:
            stats = PointStats()

        srv = self.p1 if server == 1 else self.p2
        ret = self.p2 if server == 1 else self.p1
        server_won = (point_won_by == server)

        self.points_played += 1
        srv.total_points_played += 1
        ret.total_points_played += 1

        # ── Serve stats ──────────────────────────────────────────────────
        srv.serve_points_played += 1
        ret.return_points_played += 1

        if stats.is_ace:
            srv.aces += 1

        if stats.is_double_fault:
            srv.double_faults += 1

        if stats.is_first_serve_in:
            srv.first_serves_in += 1
            if server_won:
                srv.first_serve_points_won += 1
        else:
            # second serve situation
            if server_won:
                srv.second_serve_points_won += 1

        # ── Break point tracking ─────────────────────────────────────────
        if stats.is_break_point:
            srv.break_points_faced += 1
            if server_won:
                srv.break_points_saved += 1

        # ── Point winner ─────────────────────────────────────────────────
        if server_won:
            srv.total_points_won += 1
        else:
            ret.total_points_won += 1
            ret.return_points_won += 1

        # ── Shot quality ─────────────────────────────────────────────────
        if stats.is_winner:
            winner_player = srv if server_won else ret
            winner_player.winners += 1

        if stats.is_unforced_error:
            loser_player = ret if server_won else srv
            loser_player.unforced_errors += 1

    def register_game_won(self, player: int, was_serving: bool) -> None:
        """Track that a player won a game."""
        p = self.p1 if player == 1 else self.p2
        p.games_won += 1
        if was_serving:
            p.serve_games += 1
        # Also track the other player's serve games if they were serving
        if not was_serving:
            other = self.p2 if player == 1 else self.p1
            other.serve_games += 1

    def register_set_won(self, player: int) -> None:
        p = self.p1 if player == 1 else self.p2
        p.sets_won += 1

    def snapshot(self) -> dict:
        """Full stats snapshot for both players."""
        return {
            "points_played": self.points_played,
            "player1": self.p1.to_dict(),
            "player2": self.p2.to_dict(),
        }

    def has_meaningful_data(self, min_serve_points: int = 4) -> bool:
        """Do we have enough data for ML features to be meaningful?"""
        return (
            self.p1.serve_points_played >= min_serve_points
            and self.p2.serve_points_played >= min_serve_points
        )
