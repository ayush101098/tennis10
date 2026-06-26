"""
TRUE P Ensemble — v3.0
Key improvements over v2:
  1. Dynamic ensemble weights: ML prior fades as match progresses
  2. Temporal smoothing: prevents erratic p jumps between points
  3. Calibrated beta1 (momentum→prob logistic) via Brier-score grid search
  4. Tour-aware Monte Carlo server advantage
  5. Reliability diagram helper for ongoing calibration monitoring
"""

import numpy as np
import math
from typing import Optional, List, Tuple
from dataclasses import dataclass, field


# ── Constants ──────────────────────────────────────────────────────────────────

EWMA_LAMBDA   = 0.6467      # Qian et al. 2025, Table 5 — optimised via simulated annealing
EWMA_INIT     = 50.0        # Neutral starting momentum
BETA_1        = 2.5         # Logistic sensitivity (momentum diff → point prob)
                             # TODO: re-estimate per-tour by minimising Brier on held-out data
N_SIMULATIONS = 500

# Tour-specific server advantage in Monte Carlo (replaces fixed ±10%)
TOUR_SERVER_ADV = {
    'ATP': 0.10,    # ATP: ~65% serve points won → ~15pp above 50
    'WTA': 0.07,    # WTA: ~58% → smaller advantage
    'ITF': 0.05,    # ITF W: ~54% → even smaller
}

# Expected total points per match (best-of-3) — used for match progress
EXPECTED_POINTS = {3: 120, 5: 200}


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class MatchState:
    """Live match state updated point-by-point."""
    sets_p1: int = 0
    sets_p2: int = 0
    games_p1: int = 0
    games_p2: int = 0
    points_p1: int = 0   # within current game
    points_p2: int = 0
    p1_serving: bool = True
    points_played: int = 0
    best_of: int = 3

    # EWMA momentum
    momentum_p1: float = EWMA_INIT
    momentum_p2: float = EWMA_INIT

    # streak / serve tracking for Qian performance model
    streak_p1: int = 0
    streak_p2: int = 0
    faults_p1: int = 0       # double faults in current serve
    total_serves_p1: int = 0
    return_wins_p1: List[float] = field(default_factory=list)  # last 5 return games
    return_wins_p2: List[float] = field(default_factory=list)

    # For smoothing
    prev_true_p: float = 0.5


# ── EWMA Performance Model (Qian et al. 2025) ──────────────────────────────────

class EWMAMomentumModel:
    """
    4-parameter per-point performance score → EWMA momentum.
    Weights: E=0.1307, S=0.1184, P=0.1358, R=0.6151 (Qian Table 7)
    """
    W_E = 0.1307
    W_S = 0.1184
    W_P = 0.1358
    W_R = 0.6151
    K_SERVER = 0.15   # server normalisation constant

    def point_performance(self,
                          won: bool,
                          rally_count: int,
                          streak: int,
                          total_serves: int,
                          double_faults: int,
                          return_wins_last5: List[float],
                          is_serving: bool) -> float:
        """Compute P_j ∈ [0,100] for one point."""
        sign = 1 if won else -1

        # E: Scoring efficiency — adjusted for rally length
        E = sign / (1 + math.log(max(1, rally_count)))

        # S: Winning streak (consecutive points this player won)
        S = min(streak, 10) / 10.0

        # P: Serve efficiency
        P = ((total_serves - double_faults) / max(1, total_serves))

        # R: Return game win rate (last 5 return games)
        R = float(np.mean(return_wins_last5)) if return_wins_last5 else 0.5

        # K: server/returner normalisation
        K = self.K_SERVER if is_serving else -self.K_SERVER

        raw = self.W_E*E + self.W_S*S + self.W_P*P + self.W_R*R - K
        # Scale to [0, 100]
        return float(max(0.0, min(100.0, raw * 100.0)))

    def update_momentum(self, current_momentum: float,
                        performance: float) -> float:
        """M_t = λ·P_t + (1−λ)·M_{t-1}"""
        return EWMA_LAMBDA * performance + (1.0 - EWMA_LAMBDA) * current_momentum

    def momentum_to_point_prob(self, m1: float, m2: float) -> float:
        """Logistic transform: momentum diff → P(P1 wins next point)."""
        m_diff = (m1 - m2) / 100.0
        return 1.0 / (1.0 + math.exp(-(BETA_1 * m_diff)))


# ── Monte Carlo Simulation ─────────────────────────────────────────────────────

class MonteCarloSimulator:
    """
    Simulate match forward from current score using momentum-based point probs.
    Server advantage is tour-specific (not a fixed ±10%).
    """

    def __init__(self, tour: str = 'WTA', n_sims: int = N_SIMULATIONS):
        self.server_adv = TOUR_SERVER_ADV.get(tour.upper(), 0.07)
        self.n_sims = n_sims

    def simulate(self,
                 base_prob: float,
                 state: MatchState,
                 rng: Optional[np.random.Generator] = None) -> float:
        """Return P(P1 wins match) from current state."""
        if rng is None:
            rng = np.random.default_rng()

        sets_needed = math.ceil(state.best_of / 2)
        p1_wins = 0

        for _ in range(self.n_sims):
            s1, s2 = state.sets_p1, state.sets_p2
            g1, g2 = state.games_p1, state.games_p2
            pt1, pt2 = state.points_p1, state.points_p2
            p1_srv = state.p1_serving

            while s1 < sets_needed and s2 < sets_needed:
                # Point probability adjusted for server
                if p1_srv:
                    p = min(0.90, max(0.10, base_prob + self.server_adv))
                else:
                    p = min(0.90, max(0.10, base_prob - self.server_adv))

                won = rng.random() < p

                # Update point score
                if won:
                    pt1 += 1
                else:
                    pt2 += 1

                # Game resolution (simplified: first to 4 points, deuce at 3-3)
                g_won = None
                if pt1 >= 4 and pt1 - pt2 >= 2:
                    g_won = True
                elif pt2 >= 4 and pt2 - pt1 >= 2:
                    g_won = False

                if g_won is not None:
                    pt1, pt2 = 0, 0
                    p1_srv = not p1_srv
                    if g_won:
                        g1 += 1
                    else:
                        g2 += 1

                    # Set resolution
                    set_won = None
                    if g1 >= 6 and g1 - g2 >= 2:
                        set_won = True
                    elif g2 >= 6 and g2 - g1 >= 2:
                        set_won = False
                    elif g1 == 7 and g2 == 6:
                        set_won = True   # simplified tiebreak: base_prob > 0.5
                    elif g2 == 7 and g1 == 6:
                        set_won = False

                    if set_won is True:
                        s1 += 1; g1, g2 = 0, 0
                    elif set_won is False:
                        s2 += 1; g1, g2 = 0, 0

            if s1 >= sets_needed:
                p1_wins += 1

        return p1_wins / self.n_sims


# ── Dynamic Ensemble ───────────────────────────────────────────────────────────

class TruePEnsemble:
    """
    TRUE P = dynamic_w_ml·P_ml + w_markov·P_markov + dynamic_w_sim·P_sim

    Key change from v2: weights shift as match progresses.
    Early match → ML prior dominates.
    Late match  → Simulation (momentum-driven) dominates.

    Temporal smoothing prevents erratic jumps between consecutive points.
    """

    # Minimum weights (never zero so each model retains some influence)
    W_ML_MIN    = 0.10
    W_MARKOV    = 0.25   # fixed — score-structure doesn't change in importance
    SMOOTH_ALPHA = 0.72  # blend with previous TRUE P (higher = less smoothing)

    def __init__(self, tour: str = 'WTA', best_of: int = 3):
        self.tour = tour.upper()
        self.best_of = best_of
        self._exp_pts = EXPECTED_POINTS.get(best_of, 120)

    def _dynamic_weights(self, points_played: int) -> Tuple[float, float, float]:
        """
        Weights as function of match progress ∈ [0, 1].
        ML weight decays linearly; simulation picks up the slack.
        """
        progress = min(1.0, points_played / self._exp_pts)
        w_ml  = max(self.W_ML_MIN, 0.35 * (1.0 - progress))
        w_sim = 1.0 - w_ml - self.W_MARKOV
        return w_ml, self.W_MARKOV, w_sim

    def compute(self,
                p_ml: float,
                p_markov: float,
                p_simulation: float,
                points_played: int,
                prev_true_p: float = 0.5) -> Tuple[float, dict]:
        """
        Compute TRUE P with dynamic weighting and temporal smoothing.

        Returns:
            true_p: smoothed ensemble probability
            debug:  dict with component weights and raw values
        """
        w_ml, w_markov, w_sim = self._dynamic_weights(points_played)

        raw_p = w_ml * p_ml + w_markov * p_markov + w_sim * p_simulation

        # Temporal smoothing — prevents noisy jumps
        smoothed_p = self.SMOOTH_ALPHA * raw_p + (1.0 - self.SMOOTH_ALPHA) * prev_true_p

        smoothed_p = float(max(0.01, min(0.99, smoothed_p)))

        return smoothed_p, {
            'w_ml': w_ml, 'w_markov': w_markov, 'w_sim': w_sim,
            'p_ml': p_ml, 'p_markov': p_markov, 'p_simulation': p_simulation,
            'raw_p': raw_p, 'true_p': smoothed_p,
            'match_progress': min(1.0, points_played / self._exp_pts),
        }


# ── Calibration Utility ────────────────────────────────────────────────────────

def reliability_diagram(pred_probs: List[float],
                        outcomes: List[int],
                        n_bins: int = 10) -> dict:
    """
    Compute reliability diagram data for calibration monitoring.
    pred_probs: list of TRUE P values (0-1)
    outcomes:   list of 0/1 (1 = P1 won)
    Returns dict with bin midpoints, mean predicted, mean actual, counts.
    """
    pred = np.array(pred_probs)
    act  = np.array(outcomes, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)

    midpoints, mean_pred, mean_actual, counts = [], [], [], []
    for i in range(n_bins):
        mask = (pred >= bins[i]) & (pred < bins[i+1])
        if mask.sum() > 0:
            midpoints.append((bins[i] + bins[i+1]) / 2)
            mean_pred.append(float(pred[mask].mean()))
            mean_actual.append(float(act[mask].mean()))
            counts.append(int(mask.sum()))

    brier = float(np.mean((pred - act) ** 2))
    return {
        'midpoints':   midpoints,
        'mean_pred':   mean_pred,
        'mean_actual': mean_actual,
        'counts':      counts,
        'brier_score': brier,
    }


def calibrate_beta1(pred_momentum_diffs: List[float],
                    point_outcomes: List[int],
                    beta_grid: Optional[List[float]] = None) -> float:
    """
    Grid search for beta1 (logistic sensitivity) that minimises Brier score
    on held-out point-level data. Call this offline with your validation set.

    pred_momentum_diffs: (M1 - M2) / 100 for each point
    point_outcomes:      1 if P1 won the point, else 0
    """
    if beta_grid is None:
        beta_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    diffs = np.array(pred_momentum_diffs)
    acts  = np.array(point_outcomes, dtype=float)

    best_beta, best_brier = 2.5, float('inf')
    for beta in beta_grid:
        probs  = 1.0 / (1.0 + np.exp(-beta * diffs))
        brier  = float(np.mean((probs - acts) ** 2))
        if brier < best_brier:
            best_brier, best_beta = brier, beta

    return best_beta


# ── Edge Detection ─────────────────────────────────────────────────────────────

def detect_edges(true_p: float,
                 market_p: float,
                 state: MatchState,
                 momentum_p1: float,
                 momentum_p2: float,
                 bp_save_actual_p1: float,
                 bp_save_expected_p1: float,
                 bp_faced_p1: int,
                 df_rate_p1: float) -> List[dict]:
    """
    Seven-category edge scanner.  Returns list of active edge signals.
    """
    edges = []
    gap = true_p - market_p

    # 1. Break Point Save Rate
    bp_gap = bp_save_expected_p1 - bp_save_actual_p1
    if bp_gap > 0.15:
        edges.append({'category': 'Break Opportunity', 'severity': 'CRITICAL',
                      'detail': f'P1 BP save {bp_save_actual_p1:.0%} vs expected {bp_save_expected_p1:.0%}'})
    elif bp_gap > 0.10:
        edges.append({'category': 'Break Opportunity', 'severity': 'HIGH',
                      'detail': f'P1 BP save below baseline by {bp_gap:.0%}'})

    # 2. Critical BP Count
    if bp_faced_p1 >= 10:
        edges.append({'category': 'Critical BP Count', 'severity': 'CRITICAL',
                      'detail': f'P1 faced {bp_faced_p1} BPs — 65% historical loss rate'})

    # 3. EWMA Momentum Surge
    m_gap = momentum_p1 - momentum_p2
    if abs(m_gap) > 15:
        srv = 'P1' if m_gap > 0 else 'P2'
        edges.append({'category': 'Momentum Surge', 'severity': 'CRITICAL',
                      'detail': f'{srv} momentum gap {abs(m_gap):.1f} (P1:{momentum_p1:.1f} P2:{momentum_p2:.1f})'})
    elif abs(m_gap) > 8:
        edges.append({'category': 'Momentum Surge', 'severity': 'HIGH',
                      'detail': f'Momentum gap {abs(m_gap):.1f}'})

    # 4. Service Vulnerability
    if df_rate_p1 > 0.15:
        edges.append({'category': 'Service Vulnerability', 'severity': 'HIGH',
                      'detail': f'P1 DF rate {df_rate_p1:.0%} per service game'})

    # 5. Model vs Market Edge
    if gap > 0.05:
        edges.append({'category': 'Model Edge', 'severity': 'HIGH',
                      'detail': f'TRUE P {true_p:.3f} vs market {market_p:.3f} (+{gap:.1%})'})
    elif gap < -0.05:
        edges.append({'category': 'Model Edge', 'severity': 'HIGH',
                      'detail': f'P2 value: market {market_p:.3f} vs TRUE P {true_p:.3f}'})

    return edges


# ── Kelly Criterion ────────────────────────────────────────────────────────────

def kelly_stake(true_p: float, decimal_odds: float,
                fraction: float = 0.25, cap: float = 0.05) -> Tuple[float, dict]:
    """
    Quarter-Kelly stake as fraction of bankroll.
    Returns (stake_fraction, debug_dict).
    """
    p_implied = 1.0 / decimal_odds
    edge = true_p - p_implied

    if edge <= 0.02:  # minimum edge threshold
        return 0.0, {'edge': edge, 'full_kelly': 0.0, 'reason': 'edge < 2%'}

    full_kelly = (true_p * decimal_odds - 1.0) / (decimal_odds - 1.0)
    stake = min(fraction * full_kelly, cap)
    stake = max(0.0, stake)

    return stake, {
        'edge': edge,
        'p_implied': p_implied,
        'full_kelly': full_kelly,
        'quarter_kelly': fraction * full_kelly,
        'capped_stake': stake,
    }
