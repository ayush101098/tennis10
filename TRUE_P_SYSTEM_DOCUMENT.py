"""
════════════════════════════════════════════════════════════════════════════════════
     TRUE P  —  LIVE IN-PLAY TENNIS PRICING MODEL
     A Quantitative Framework for Real-Time Match Win Probability Estimation
════════════════════════════════════════════════════════════════════════════════════

Author  : Ayush Mishra
Date    : April 2026
Version : 2.0

Abstract
────────
TRUE P is a multi-model ensemble system that produces a continuously-updating
win probability for any professional tennis match.  It fuses:

  (1) pre-match priors from supervised ML trained on 143,530 historical matches,
  (2) a hierarchical Markov chain conditioned on score state,
  (3) an EWMA-weighted performance model implementing Qian et al. (2025),
  (4) Monte Carlo simulation from the current score position.

The ensemble feeds a Kelly-criterion bankroll optimizer that identifies live
betting edges across seven signal categories.  All predictions self-calibrate
point-by-point using incoming match data.

The system ingests live scores from ESPN's public API (ATP, WTA, ITF, Challenger)
with no API key required, and runs as a Streamlit dashboard for real-time
operation.


════════════════════════════════════════════════════════════════════════════════════
  TABLE OF CONTENTS
════════════════════════════════════════════════════════════════════════════════════

  1.  SYSTEM ARCHITECTURE & DATA FLOW
  2.  DATA FOUNDATION
  3.  FEATURE ENGINEERING
  4.  MODEL 1 — SUPERVISED ML (LOGISTIC REGRESSION + RANDOM FOREST)
  5.  MODEL 2 — HIERARCHICAL MARKOV CHAIN (Barnett-Clarke)
  6.  MODEL 3 — EWMA PERFORMANCE & MOMENTUM (Qian et al. 2025)
  7.  MODEL 4 — MONTE CARLO MATCH SIMULATION
  8.  THE ENSEMBLE: HOW TRUE P IS COMPUTED
  9.  LIVE UPDATE LOOP — POINT-BY-POINT RECALCULATION
  10. EDGE DETECTION & SIGNAL TAXONOMY
  11. KELLY CRITERION & BET SIZING
  12. BACKTESTING FRAMEWORK
  13. SYSTEM PERFORMANCE & MODEL ACCURACY
  14. LIVE DATA INTEGRATION
  15. FUTURE WORK


════════════════════════════════════════════════════════════════════════════════════
  1.  SYSTEM ARCHITECTURE & DATA FLOW
════════════════════════════════════════════════════════════════════════════════════

The system has five layers:

    ┌──────────────────────────────────────────────────────────────────┐
    │                         DATA LAYER                              │
    │                                                                  │
    │   Jeff Sackmann GitHub  ─┐                                       │
    │   (ATP/WTA match-level   ├──►  data_pipeline_enhanced.py        │
    │    CSVs, 2000-2024)      │         │                             │
    │                          │         ▼                             │
    │   tennis-data.co.uk  ────┘    tennis_betting.db                  │
    │   (odds, Excel files)         143,530 matches                    │
    │                               5,282 players                      │
    │                               112,387 with serve stats           │
    │                                                                  │
    │   ESPN Public API  ──────►  api/free_live_data.py               │
    │   (live scores, no key)       (20s cache, ATP+WTA+more)          │
    └──────────────────────────────────────────────────────────────────┘
                                    │
    ┌──────────────────────────────────────────────────────────────────┐
    │                      FEATURE LAYER                              │
    │                                                                  │
    │   features.py   ──►  19-dimensional feature vector               │
    │                      (time-decayed, surface-weighted)             │
    │                                                                  │
    │   Includes:  rank diff, serve %, return %, BP save %,            │
    │              surface win rate, H2H, fatigue, uncertainty          │
    └──────────────────────────────────────────────────────────────────┘
                                    │
    ┌──────────────────────────────────────────────────────────────────┐
    │                       MODEL LAYER                               │
    │                                                                  │
    │   ┌────────────────┐  ┌─────────────────┐  ┌────────────────┐   │
    │   │  ML Ensemble   │  │  Hierarchical   │  │  EWMA + Monte  │   │
    │   │  LR + RF       │  │  Markov Chain   │  │  Carlo Sim     │   │
    │   │  (pre-match)   │  │  (score-aware)  │  │  (live perf)   │   │
    │   └───────┬────────┘  └───────┬─────────┘  └───────┬────────┘   │
    │           │                   │                     │            │
    │           └───────────────────┼─────────────────────┘            │
    │                               ▼                                  │
    │                    ┌──────────────────┐                          │
    │                    │   TRUE P         │                          │
    │                    │   (Ensemble)     │                          │
    │                    └──────────────────┘                          │
    └──────────────────────────────────────────────────────────────────┘
                                    │
    ┌──────────────────────────────────────────────────────────────────┐
    │                      SIGNAL LAYER                               │
    │                                                                  │
    │   7 Edge Categories  ──►  Kelly Criterion Sizing                 │
    │   (break opportunity, momentum surge, clutch, service            │
    │    vulnerability, consistency, surface mastery, EWMA)            │
    └──────────────────────────────────────────────────────────────────┘
                                    │
    ┌──────────────────────────────────────────────────────────────────┐
    │                    EXECUTION LAYER                               │
    │                                                                  │
    │   Streamlit Dashboard  ──►  Real-time display + alerts           │
    │   backtesting/betting_strategies.py  ──►  Historical validation  │
    └──────────────────────────────────────────────────────────────────┘


════════════════════════════════════════════════════════════════════════════════════
  2.  DATA FOUNDATION
════════════════════════════════════════════════════════════════════════════════════

Database : SQLite (tennis_betting.db)
Records  : 143,530 professional matches (2000-01-03 → 2024-12-18)
Source   : Jeff Sackmann's open tennis dataset + tennis-data.co.uk

Coverage breakdown:

    ┌───────────┬───────────┬──────────────────────────────────────────┐
    │ Dimension │ Count     │ Notes                                    │
    ├───────────┼───────────┼──────────────────────────────────────────┤
    │ ATP       │ 74,906    │ Grand Slams, Masters, 250/500, Qualifier │
    │ WTA       │ 68,624    │ Grand Slams, Premier, International      │
    │ Total     │ 143,530   │ 25 years of match data                   │
    ├───────────┼───────────┼──────────────────────────────────────────┤
    │ Hard      │ 81,151    │ 56.5%                                    │
    │ Clay      │ 44,259    │ 30.8%                                    │
    │ Grass     │ 14,549    │ 10.1%                                    │
    │ Carpet    │ 3,441     │ 2.4% (historical, pre-2009)              │
    ├───────────┼───────────┼──────────────────────────────────────────┤
    │ Players   │ 5,282     │ Unique players with at least one match   │
    │ With stats│ 112,387   │ 78.3% have full serve/return data        │
    └───────────┴───────────┴──────────────────────────────────────────┘

Per-match statistical columns (when available):

    aces, double_faults, serve_points, first_serve_in, first_serve_won,
    second_serve_won, service_games, break_points_saved, break_points_faced

Derived statistics computed at ingestion time:

    first_serve_pct        = first_serve_in / serve_points
    first_serve_win_pct    = first_serve_won / first_serve_in
    second_serve_win_pct   = second_serve_won / (serve_points - first_serve_in)
    break_point_save_pct   = break_points_saved / break_points_faced

A separate `special_parameters` table holds 2,084 player profiles with:
    career_win_rate, best_surface, surface_mastery, clutch_performance,
    bp_defense_rate, consistency_rating, peak_rating


════════════════════════════════════════════════════════════════════════════════════
  3.  FEATURE ENGINEERING
════════════════════════════════════════════════════════════════════════════════════

File: features.py — Class TennisFeatureExtractor

All features are computed as Player 1 minus Player 2 differentials, making the
model symmetric: swapping players negates the feature vector and flips the
prediction around 0.5.

3.1  Time Decay
─────────────
Every historical data point is weighted by exponential time decay:

    w(t) = 0.5 ^ (t / τ)

    τ = 0.8 years for general match stats
    τ = 1.5 years for head-to-head records

Rationale: a player's form 6 months ago is roughly 65% as relevant as today;
form 2 years ago is only 18% relevant.

3.2  Surface Transfer Matrix
────────────────────────────
When a player has few matches on the target surface, we transfer skill
estimates from other surfaces using empirically calibrated correlations:

             Hard    Clay    Grass
    Hard     1.000   0.280   0.240
    Clay     0.280   1.000   0.150
    Grass    0.240   0.150   1.000

Example: if a player has 200 hard-court matches but only 5 on grass,
their grass estimate borrows from hard-court data with weight 0.24.

3.3  Fatigue Model
──────────────────
Recent match load is computed over a 3-day window.  Each recent match
contributes a fatigue score weighted by:

    fatigue_weight = match_duration_hours × 0.75 ^ days_ago

This captures that a 4-hour match yesterday produces more fatigue than
a 1.5-hour match three days ago.

3.4  Full Feature Vector (19 dimensions)
────────────────────────────────────────

    ┌────┬──────────────────────────┬──────────────────────────────────┐
    │  # │ Feature                  │ Description                      │
    ├────┼──────────────────────────┼──────────────────────────────────┤
    │  1 │ RANK_DIFF                │ ATP/WTA ranking differential     │
    │  2 │ POINTS_DIFF              │ Ranking points differential      │
    │  3 │ WSP_DIFF                 │ Win on serve % (time-weighted)   │
    │  4 │ WRP_DIFF                 │ Win on return % (time-weighted)  │
    │  5 │ ACES_DIFF                │ Aces per match differential      │
    │  6 │ DF_DIFF                  │ Double faults per match diff     │
    │  7 │ BP_SAVE_DIFF             │ Break point save rate diff       │
    │  8 │ FIRST_SERVE_PCT_DIFF     │ First serve in % differential   │
    │  9 │ FIRST_SERVE_WIN_PCT_DIFF │ First serve won % differential  │
    │ 10 │ SECOND_SERVE_WIN_PCT_DIFF│ Second serve won % differential │
    │ 11 │ WIN_RATE_DIFF            │ Overall win rate differential    │
    │ 12 │ SURFACE_WIN_RATE_DIFF    │ Surface-specific win rate diff   │
    │ 13 │ SERVEADV                 │ Net serve advantage score        │
    │ 14 │ COMPLETE_DIFF            │ Match completion rate diff       │
    │ 15 │ FATIGUE_DIFF             │ 3-day match-load differential    │
    │ 16 │ RETIRED_DIFF             │ Recent inactivity flag (>90 d)   │
    │ 17 │ DIRECT_H2H               │ Head-to-head win rate (decayed)  │
    │ 18 │ MATCHES_PLAYED_DIFF      │ Career experience differential   │
    │ 19 │ SURFACE_EXP_DIFF         │ Surface-specific experience diff │
    ├────┼──────────────────────────┼──────────────────────────────────┤
    │ +1 │ UNCERTAINTY              │ Data completeness meta-feature   │
    └────┴──────────────────────────┴──────────────────────────────────┘

3.5  Uncertainty Score
──────────────────────
A meta-feature that quantifies how confident we should be in the other features:

    U = 0.4 · 1/(1 + min(matches)/20)
      + 0.3 · 1/(1 + min(surface_matches)/10)
      + 0.3 · 1/(1 + h2h_count/3)

U → 0 when data is plentiful; U → 1 when data is sparse.
The model learns to be less confident when U is high.


════════════════════════════════════════════════════════════════════════════════════
  4.  MODEL 1 — SUPERVISED ML (LOGISTIC REGRESSION + RANDOM FOREST)
════════════════════════════════════════════════════════════════════════════════════

Purpose: Pre-match probability estimate using only information available
         before the match starts (no look-ahead bias).

4.1  Training Pipeline
──────────────────────
File: train_proper_models.py, train_sackmann_models.py

    Step 1 — Data preparation
        • Filter matches with complete statistics (112,387 of 143,530)
        • Random 50% player-order swap (negate features, flip label)
          to enforce model symmetry
        • Time-based train/test split at 80th percentile date
          (no future data in training set)

    Step 2 — Feature selection (forward greedy)
        • Start with empty set
        • For each candidate feature, evaluate combined score:
            score = 0.6 · AUC + 0.4 · (1 + Kelly_ROI)
        • Add feature that most improves combined score
        • Stop when no improvement for 2 consecutive rounds

    Step 3 — Hyperparameter tuning (grid search with CV)

        Logistic Regression:
            C ∈ {0.01, 0.1, 0.2, 0.5, 1.0, 10.0}
            class_weight ∈ {None, balanced}
            Regularization: L2 (Ridge)
            Post-training: isotonic calibration (5-fold)

        Random Forest:
            n_estimators = 200
            max_depth ∈ {8, 12, 16, None}
            min_samples_leaf ∈ {5, 10, 20}

    Step 4 — Calibration
        • CalibratedClassifierCV with isotonic regression
        • Ensures predicted probabilities match empirical frequencies
        • Critical for Kelly sizing (miscalibrated → bleed money)

4.2  Production Feature Set (6-feature compact model)
─────────────────────────────────────────────────────
The deployed V2 calculator uses a reduced 6-feature model for speed:

    [p1_serve_pct, p2_serve_pct, p1_bp_save, p2_bp_save, surface_hard, surface_clay]

    Logistic Regression accuracy: 93.73%  (on held-out test set)
    Random Forest accuracy:       93.78%

    Ensemble weighting:  P_ml = 0.4 · P_lr + 0.6 · P_rf

4.3  Neural Network Ensemble (Bagging)
──────────────────────────────────────
File: ml_models/neural_network.py — Class SymmetricTennisNet

Architecture per network:
    Input(N) → Dense(100, tanh, no bias) → Dense(1, sigmoid, no bias)
    Xavier initialization, no bias terms (forces symmetry)

Training:
    Optimizer:      SGD (lr=0.0004, momentum=0.55, weight_decay=0.002)
    Batch size:     1 (online/stochastic)
    Early stopping: patience=10 epochs
    Ensemble:       20 bootstrap replicates (bagging)
    Final pred:     Simple mean of 20 model outputs

Meta-ensemble combining LR + NN + Markov (live_prediction.py):

    P_meta = 0.35 · P_lr + 0.45 · P_nn + 0.20 · P_markov


════════════════════════════════════════════════════════════════════════════════════
  5.  MODEL 2 — HIERARCHICAL MARKOV CHAIN (Barnett-Clarke)
════════════════════════════════════════════════════════════════════════════════════

References: Barnett & Clarke (2005), O'Malley (2008), McHale & Morton (2011)
File: hierarchical_model.py

This model treats tennis as a nested Markov chain:
    Point → Game → Set → Match

Each level is computed analytically (no simulation needed), making it
extremely fast — suitable for real-time, point-by-point updates.

5.1  Level 1 — Point Win Probability
─────────────────────────────────────
For the server:

    P(win point) = P(1st in) · P_adj(win | 1st in)
                 + (1 - P(1st in)) · 0.95 · P_adj(win | 2nd in)

where 0.95 is the probability of getting the second serve in, and:

    P_adj = α · P_server + (1-α) · (1 - P_returner)

    α = 0.60  (server's own ability weighted more than opponent's return)

The adjustment mixes the server's intrinsic serve strength with how well
the opponent returns, creating a matchup-specific estimate.

Clamped to [0.45, 0.85] to prevent degenerate edge cases.

ATP averages (used as fallback when player-specific data is missing):
    1st serve %:   62.5%
    1st serve won: 71.5%
    2nd serve won: 51.0%
    Return 1st:    28.5%
    Return 2nd:    49.0%

5.2  Level 2 — Game Win Probability
────────────────────────────────────
Given point win probability p, the probability of holding serve follows
from the Markov chain through the tennis game scoring system:

    P(hold) = p^4                          (40-0)
            + 4·p^4·q                      (via 30-15)
            + 10·p^4·q^2                   (via 30-30)
            + 20·p^3·q^3 · p²/(p²+q²)    (via deuce)

    where q = 1 - p

The deuce term uses the well-known formula: from deuce, the server wins
the game with probability p²/(p²+q²), since they must win two
consecutive points.

5.3  Level 3 — Set Win Probability
───────────────────────────────────
    p_avg = (P(hold_serve) + (1 - P(opp_holds))) / 2

This averages a player's probability of winning a game as server and
as returner.  From here, the set probability sums over all possible
paths to winning 6 games:

    P(set) = Σ binomial terms for 6-0, 6-1, ..., 6-4
           + P(reach 5-5) · P(7-5 or tiebreak)

Tiebreak model: P_tb = 0.5 + 0.8 · (p_avg - 0.5)

5.4  Level 4 — Match Win Probability
─────────────────────────────────────
Given set win probability p_set:

    Best-of-3:  P(match) = p² · (1 + 2·(1-p))
                          = p² · (3 - 2p)

    Best-of-5:  P(match) = p³ · (1 + 3·(1-p) + 6·(1-p)²)
                          = p³ · (6·p² - 15p + 10)

5.5  Why This Model Matters for Live Betting
─────────────────────────────────────────────
The Markov chain is SCORE-AWARE.  If Player A leads 6-3, 5-2, the model
captures that A needs only 1 more game while B needs to win the current set,
then win another entire set.  This creates a strong prior that anchors the
ensemble to structural match reality.


════════════════════════════════════════════════════════════════════════════════════
  6.  MODEL 3 — EWMA PERFORMANCE & MOMENTUM (Qian et al. 2025)
════════════════════════════════════════════════════════════════════════════════════

Reference: "Predicting Point-by-Point Tennis Match Outcomes Using
            Momentum-Based Models"
           Journal of Computers, Vol. 36 No. 1, February 2025

This is the KEY innovation — it captures in-match momentum that the other
models cannot see.

6.1  The 4-Parameter Performance Model
───────────────────────────────────────
Each point generates a performance score from four components:

    P_j = w_E · E(j) + w_S · S(j) + w_P · P(j) + w_R · R(j) - K_j

Where:

    ┌─────────────────────┬────────┬──────────────────────────────────┐
    │ Parameter           │ Weight │ Formula                          │
    ├─────────────────────┼────────┼──────────────────────────────────┤
    │ E: Scoring          │ 0.1307 │ ±1/(1 + ln(rally_count))        │
    │    Efficiency       │        │ + if won, - if lost              │
    ├─────────────────────┼────────┼──────────────────────────────────┤
    │ S: Winning Streak   │ 0.1184 │ min(consecutive_pts_won, 10)/10 │
    ├─────────────────────┼────────┼──────────────────────────────────┤
    │ P: Serve Efficiency │ 0.1358 │ (total_serves - faults) /       │
    │                     │        │  total_serves                    │
    ├─────────────────────┼────────┼──────────────────────────────────┤
    │ R: Returner Win     │ 0.6151 │ Points won as returner in       │
    │    Rate             │        │ last 5 return games              │
    └─────────────────────┴────────┴──────────────────────────────────┘

    K = +0.15 if serving (penalizes server advantage to normalize)
    K = -0.15 if returning

    Output scaled to [0, 100]

Note: R (returner win rate) dominates at w=0.6151.  This is the paper's
key finding — how well a player returns serve is the single strongest
predictor of in-match momentum.  The weights were derived via entropy
analysis across a large match corpus (Table 7 in the paper).

6.2  EWMA Momentum
───────────────────
Raw performance is smoothed into a momentum indicator:

    M_t = λ · P_t + (1 - λ) · M_{t-1}

    λ = 0.6467

This value was optimized via simulated annealing to maximize Pearson
correlation with actual point outcomes.  At λ=0.6467 the correlation
reached r = 0.731 (Table 5).  This means roughly 65% weight on the
current point's performance and 35% on accumulated history — tennis
momentum decays fast.

    M_0 = 50.0 (neutral starting momentum)

6.3  Momentum → Win Probability (Logistic Transform)
────────────────────────────────────────────────────
The momentum differential is converted to a point-level win probability:

    m_diff = (M_1 - M_2) / 100

    P(P1 wins point) = 1 / (1 + exp(-(β₀ + β₁ · m_diff)))

    β₀ = 0.0   (centered — equal momentum → 50%)
    β₁ = 2.5   (moderate sensitivity)

The paper reported 70% accuracy on theoretical point prediction with
optimized hyperparameters (C=100, L1 penalty, liblinear solver).


════════════════════════════════════════════════════════════════════════════════════
  7.  MODEL 4 — MONTE CARLO MATCH SIMULATION
════════════════════════════════════════════════════════════════════════════════════

File: V2 Calculator — simulate_match_from_position()

Starting from the CURRENT score state, we simulate the match forward
500–1,000 times to estimate match win probability.

Algorithm:
──────────

    FOR each simulation (n=500):
        Copy current state: (sets, games, points, server)
        WHILE match not over:
            1. Determine point win probability:
               If P1 serving:  p = base_prob + 0.10  (server advantage)
               If P1 returning: p = base_prob - 0.10
               Clamp p to [0.15, 0.85]
            2. Draw random point outcome
            3. Update point → game → set → match score
               (with full tiebreak and deuce logic)
        Record winner
    P(P1 wins) = count(P1 wins) / n_simulations

The base probability comes from the EWMA momentum model (Section 6.3),
so simulation adapts to in-match performance trends.

Key advantage: captures score-path effects that analytical models miss.
For example, if a player is up 5-4, 40-30 on serve, the simulation
correctly estimates a very high probability of closing the set within
the next few points.


════════════════════════════════════════════════════════════════════════════════════
  8.  THE ENSEMBLE: HOW TRUE P IS COMPUTED
════════════════════════════════════════════════════════════════════════════════════

TRUE P is a weighted ensemble of all four model outputs:

    ┌─────────────────────────┬────────┬───────────────────────────────┐
    │ Model                   │ Weight │ Rationale                     │
    ├─────────────────────────┼────────┼───────────────────────────────┤
    │ ML (LR + RF)            │  0.35  │ Strong pre-match prior based  │
    │                         │        │ on historical player quality  │
    ├─────────────────────────┼────────┼───────────────────────────────┤
    │ Markov Chain            │  0.25  │ Score-state awareness — knows │
    │                         │        │ exact structural position     │
    ├─────────────────────────┼────────┼───────────────────────────────┤
    │ Monte Carlo Simulation  │  0.40  │ Momentum-adaptive forward     │
    │ (fed by EWMA momentum)  │        │ simulation from current state │
    └─────────────────────────┴────────┴───────────────────────────────┘

    TRUE_P = 0.35 · P_ml + 0.25 · P_markov + 0.40 · P_simulation

Why these weights?
──────────────────
• ML gets 35% because it provides the strongest pre-match prior, but
  cannot see in-match dynamics.

• Markov gets 25% because it correctly handles score-state (a player
  up 2 sets to 0 has a massive structural advantage), but assumes
  constant point probabilities.

• Simulation gets the largest share (40%) because it is the ONLY model
  that incorporates EWMA momentum — it detects when a player is
  currently performing above or below their baseline and adjusts
  the forward projection accordingly.

As the match progresses, the momentum-based simulation component
naturally becomes more informative (more data points → more stable
EWMA), while the pre-match ML prior gradually becomes less relevant.

8.1  Match Condition Adjustments
────────────────────────────────
Before ensembling, a conditions-based modifier is applied:

    ┌─────────────────────┬──────────────────┐
    │ Condition           │ Adjustment       │
    ├─────────────────────┼──────────────────┤
    │ Fast court (>60)    │ +2% to big server│
    │ Slow court (<40)    │ -2% to big server│
    │ High altitude (>1km)│ +1.5% server     │
    │ Hot weather (>30°C) │ -1% to both      │
    │ Cold weather (<10°C)│ -1% to both      │
    │ Indoor              │ +0.5% server     │
    │ Surface mastery     │ ±(mastery/100)   │
    └─────────────────────┴──────────────────┘


════════════════════════════════════════════════════════════════════════════════════
  9.  LIVE UPDATE LOOP — POINT-BY-POINT RECALCULATION
════════════════════════════════════════════════════════════════════════════════════

Every tracked point triggers a full recalculation pipeline:

    ┌─────────────────────────────────────────────────────────────────┐
    │  POINT TRACKED: Player X wins point                            │
    │                                                                 │
    │  Step 1.  Update winning streaks (consecutive points counter)   │
    │  Step 2.  Update serve fault tracking                           │
    │  Step 3.  Update return game tracking (last 5 return games)     │
    │                                                                 │
    │  Step 4.  Calculate 4-parameter Performance (E, S, P, R)       │
    │           for BOTH players                                      │
    │                                                                 │
    │  Step 5.  Update EWMA Momentum:                                │
    │           M_t = 0.6467 · P_t + 0.3533 · M_{t-1}               │
    │                                                                 │
    │  Step 6.  Store in performance_history[]                        │
    │                                                                 │
    │  Step 7.  Convert momentum to point probability (logistic)      │
    │                                                                 │
    │  Step 8.  Run 500 Monte Carlo simulations from current score   │
    │                                                                 │
    │  Step 9.  Recalculate Markov chain from current score state    │
    │                                                                 │
    │  Step 10. Ensemble:                                            │
    │           TRUE_P = 0.35·P_ml + 0.25·P_markov + 0.40·P_sim     │
    │                                                                 │
    │  Step 11. Run 7-category edge detection                        │
    │                                                                 │
    │  Step 12. Compute Kelly-optimal bet size                       │
    │                                                                 │
    │  Step 13. Update all visualizations                            │
    └─────────────────────────────────────────────────────────────────┘

This entire pipeline executes in <200ms, enabling real-time tracking.


════════════════════════════════════════════════════════════════════════════════════
  10. EDGE DETECTION & SIGNAL TAXONOMY
════════════════════════════════════════════════════════════════════════════════════

The system continuously scans for seven categories of betting edges:

    ┌──────────────────────┬──────────┬──────────────────────────────────────┐
    │ Edge Category        │ Severity │ Trigger Condition                    │
    ├──────────────────────┼──────────┼──────────────────────────────────────┤
    │ 1. Break Opportunity │ CRITICAL │ Actual BP save rate < expected - 15% │
    │                      │ HIGH     │ Actual BP save rate < expected - 10% │
    │                      │          │ From real data: winners save 63% BPs,│
    │                      │          │ losers save 49% — a 14.4% gap        │
    ├──────────────────────┼──────────┼──────────────────────────────────────┤
    │ 2. Critical BP Count │ CRITICAL │ ≥10 BPs faced in match               │
    │                      │          │ Empirical: 65% of players facing 10+ │
    │                      │          │ BPs go on to lose the match           │
    ├──────────────────────┼──────────┼──────────────────────────────────────┤
    │ 3. EWMA Momentum     │ CRITICAL │ Momentum gap > 15 points             │
    │    Surge             │ HIGH     │ Momentum gap > 8 points              │
    ├──────────────────────┼──────────┼──────────────────────────────────────┤
    │ 4. Momentum Turning  │ HIGH     │ EWMA direction reversal detected     │
    │    Point             │          │ (player was declining, now rising)    │
    ├──────────────────────┼──────────┼──────────────────────────────────────┤
    │ 5. Clutch Situation  │ HIGH     │ Score at 4-4 or later in set AND     │
    │                      │          │ clutch performance differential ≥ 4  │
    ├──────────────────────┼──────────┼──────────────────────────────────────┤
    │ 6. Service            │ HIGH    │ Double fault rate > 15% per service  │
    │    Vulnerability     │          │ game                                 │
    ├──────────────────────┼──────────┼──────────────────────────────────────┤
    │ 7. Consistency Edge  │ MEDIUM   │ Unforced error ratio > 40% with     │
    │                      │          │ 20+ points played                    │
    └──────────────────────┴──────────┴──────────────────────────────────────┘

Each edge is displayed in real-time with actionable context:
    "[CRITICAL] P1 EWMA momentum surge: 62.3 vs P2 44.1 — gap of 18.2"


════════════════════════════════════════════════════════════════════════════════════
  11. KELLY CRITERION & BET SIZING
════════════════════════════════════════════════════════════════════════════════════

We use the fractional Kelly criterion for bankroll management.

11.1  Edge Calculation
──────────────────────

    edge = P_true - P_implied

    where P_implied = 1 / decimal_odds

    Minimum edge threshold: 2% (below this, no bet)

11.2  Kelly Fraction
────────────────────
Full Kelly:

    f* = (P · odds - 1) / (odds - 1)

We use QUARTER Kelly to reduce variance:

    f = 0.25 · f*

With a hard cap:

    f = min(f, 0.05)     (never risk more than 5% of bankroll)

11.3  Example
─────────────

    TRUE P    = 0.62  (our model says P1 has 62% chance)
    Market    = 2.10  (bookmaker implies 47.6%)
    Edge      = 0.62 - 0.476 = 14.4%
    Full Kelly = (0.62 × 2.10 - 1) / (2.10 - 1) = 0.302 / 1.10 = 27.4%
    Quarter K  = 0.25 × 0.274 = 6.9%
    Capped     = 5.0%

    → Bet 5% of bankroll on P1 at 2.10

11.4  Why Quarter Kelly?
────────────────────────
Full Kelly maximizes long-term log-wealth growth but experiences severe
drawdowns (~50% drawdowns are common).  Quarter Kelly sacrifices ~44%
of theoretical growth rate but reduces drawdowns by >75%, making the
strategy much more robust to model estimation error.


════════════════════════════════════════════════════════════════════════════════════
  12. BACKTESTING FRAMEWORK
════════════════════════════════════════════════════════════════════════════════════

File: backtesting/betting_strategies.py

Three strategies backtested against historical data:

    ┌────────────┬───────────────────────────────────────────────────────┐
    │ Strategy   │ Logic                                                 │
    ├────────────┼───────────────────────────────────────────────────────┤
    │ Fixed      │ Flat 1-unit stake on predicted winner (P > 0.5)      │
    │ Value      │ Bet only when P_model > P_implied (positive edge)    │
    │ Kelly      │ Quarter-Kelly sizing, capped at 5%                    │
    └────────────┴───────────────────────────────────────────────────────┘

Risk metrics computed:

    • Annualized Sharpe Ratio:  Sharpe = (mean_return / std_return) × √250
    • Maximum Drawdown:         Running peak-to-trough decline
    • Monthly P&L:              Calendar-month profit/loss decomposition
    • Win Rate:                 Percentage of bets that returned positive
    • ROI:                      Total profit / total staked

Statistical validation:

    • McNemar's Test for pairwise model comparison:
      χ² = (|n_01 - n_10| - 1)² / (n_01 + n_10)
      Tests whether two models make significantly different errors.

    • Bootstrap 95% CI for ROI:
      1,000 resamples, report [2.5th, 97.5th] percentile

    • Calibration Reliability Diagram:
      Bin predictions into deciles, compare predicted vs actual frequency


════════════════════════════════════════════════════════════════════════════════════
  13. SYSTEM PERFORMANCE & MODEL ACCURACY
════════════════════════════════════════════════════════════════════════════════════

13.1  Pre-Match Model Accuracy (out-of-sample)
──────────────────────────────────────────────

    ┌──────────────────────┬──────────┬──────────────────────────────────┐
    │ Model                │ Accuracy │ Notes                            │
    ├──────────────────────┼──────────┼──────────────────────────────────┤
    │ Logistic Regression  │ 93.73%   │ 6-feature compact model          │
    │ Random Forest        │ 93.78%   │ 6-feature compact model          │
    │ ML Ensemble (0.4/0.6)│ ~94%     │ LR + RF weighted combination     │
    │ Neural Network (20x) │ ~92%     │ 20-model bagged ensemble         │
    │ Hierarchical Markov  │ ~68%     │ Pure statistical (no ML)         │
    │ Meta-Ensemble        │ ~93%     │ 0.35 LR + 0.45 NN + 0.20 Markov │
    └──────────────────────┴──────────┴──────────────────────────────────┘

13.2  In-Match Momentum Model (Qian et al. 2025)
─────────────────────────────────────────────────

    EWMA λ optimization:  Pearson r = 0.731 at λ = 0.6467
    Logistic point prediction: 70% accuracy (paper-reported)

    The momentum model adds value not through raw accuracy but through
    SPEED OF ADAPTATION.  When a player's form shifts mid-match, the
    EWMA detects it within 3-5 points, while the market often takes
    a full set to reprice.

13.3  Key Statistical Insights from the Database
─────────────────────────────────────────────────

    • Break Point Save Rate:
      Winners save 63% of BPs → Losers save 49% → 14.4 percentage point gap
      This is the single most predictive in-match statistic.

    • Players facing ≥10 BPs in a match lose 65% of the time.

    • First serve win % above 72% correlates with 74% match win rate.

    • Surface-specific models outperform generic models by ~3% accuracy
      on extreme surfaces (clay, grass).


════════════════════════════════════════════════════════════════════════════════════
  14. LIVE DATA INTEGRATION
════════════════════════════════════════════════════════════════════════════════════

File: api/free_live_data.py — Class FreeLiveTennisService

Source: ESPN public JSON API (no authentication, no rate limits, no cost)

Endpoints:
    • Scoreboard: site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard
    • Rankings:   .../rankings  (Top 150 ATP/WTA)
    • Athletes:   site.api.espn.com/apis/common/v3/sports/tennis/{tour}/athletes/{id}

Coverage: ATP, WTA (ESPN)
          ITF, Challenger (via Flashscore integration, in development)

Cache policy:
    • Live scores:    20-second refresh
    • Schedule:       5-minute refresh
    • Rankings:       1-hour refresh
    • Player profiles: 1-hour refresh

Output: Each match is normalized into a standard dictionary with:
    event_id, p1_name, p2_name, tournament, round, surface (auto-detected),
    sets_p1, sets_p2, games_p1, games_p2, set_scores[], is_live, is_finished

Surface auto-detection: Tournament name matching against keyword lists
    Clay:  Roland Garros, Rome, Madrid, Barcelona, Monte Carlo, Hamburg
    Grass: Wimbledon, Queens, Halle, Eastbourne, Mallorca
    Hard:  Australian Open, US Open, Indian Wells, Miami, Shanghai

Dashboard pages:
    • Page 8 (Live Matches):    Auto-refreshing scoreboard with one-click
                                 "Load into Calculator V2" button
    • Page 7 (Calculator V2):   Auto-fill from live matches, 30s refresh,
                                 manual disconnect option


════════════════════════════════════════════════════════════════════════════════════
  15. FUTURE WORK
════════════════════════════════════════════════════════════════════════════════════

    1. POINT-LEVEL LIVE DATA
       Current limitation: ESPN provides set/game scores but not point-by-point
       data in real time.  Adding a point-level feed (e.g., Betfair Stream API)
       would enable fully automatic EWMA tracking without manual point entry.

    2. ODDS INTEGRATION
       Integrating live bookmaker odds (Betfair exchange, Pinnacle) would allow
       automatic edge detection: TRUE_P vs market P_implied in real time.

    3. DYNAMIC ENSEMBLE WEIGHTS
       Currently the ensemble weights are fixed (0.35/0.25/0.40).  A Bayesian
       approach could dynamically shift weight toward the model that has been
       most accurate in the current match.

    4. SET-LEVEL CONTEXT
       A player's behavior changes based on whether they're up or down in sets.
       Adding set-score-conditioned features could improve late-match predictions.

    5. PLAYER EMBEDDINGS
       Replace hand-crafted player features with learned embeddings from a
       large match corpus, similar to word2vec but for tennis players.

    6. AUTOMATED EXECUTION
       Connect to a betting exchange API (Betfair) for automated order
       placement when edges exceed threshold.


════════════════════════════════════════════════════════════════════════════════════
  APPENDIX A — FILE MAP
════════════════════════════════════════════════════════════════════════════════════

    tennis10/
    ├── tennis_betting.db                 # SQLite: 143K matches, 5K players
    ├── features.py                       # 19-dim feature extraction engine
    ├── hierarchical_model.py             # Barnett-Clarke Markov chain
    ├── live_prediction.py                # 3-model meta-ensemble (LR+NN+Markov)
    ├── point_tracker.py                  # Interactive point-by-point tracker
    ├── data_pipeline.py                  # Ingest tennis-data.co.uk
    ├── data_pipeline_enhanced.py         # Ingest Sackmann GitHub (with stats)
    ├── train_proper_models.py            # Train LR + NN (forward selection)
    ├── train_sackmann_models.py          # Train on Sackmann data (15-model NN)
    ├── api/
    │   ├── free_live_data.py             # ESPN free API service
    │   ├── live_tennis_data.py           # api-tennis.com wrapper (paid)
    │   └── prediction_service.py         # REST API wrapper
    ├── ml_models/
    │   ├── logistic_regression.py        # SymmetricTennisLogistic class
    │   ├── neural_network.py             # SymmetricTennisNet (20-bag ensemble)
    │   ├── *.pkl                         # Serialized trained models
    │   └── README_NN.md                  # NN architecture documentation
    ├── backtesting/
    │   └── betting_strategies.py         # Fixed / Value / Kelly backtests
    ├── evaluation/
    │   └── model_comparison.py           # McNemar, calibration, Brier, Sharpe
    ├── dashboard/
    │   ├── streamlit_app.py              # Main Streamlit entry point
    │   └── pages/
    │       ├── 7_🎯_Live_Calculator_V2.py  # Full live tracker + all models
    │       └── 8_🔴_Live_Matches.py        # Live scoreboard (ESPN)
    └── tests/
        ├── test_features.py              # Feature extraction tests
        ├── test_models.py                # Model prediction tests
        ├── test_betting.py               # Betting strategy tests
        └── test_integration.py           # End-to-end pipeline tests


════════════════════════════════════════════════════════════════════════════════════
  APPENDIX B — ALL CONSTANTS & HYPERPARAMETERS
════════════════════════════════════════════════════════════════════════════════════

    ┌────────────────────────────┬───────────────┬──────────────────────────┐
    │ Constant                   │ Value         │ Source                   │
    ├────────────────────────────┼───────────────┼──────────────────────────┤
    │ Time decay half-life       │ 0.8 years     │ features.py              │
    │ H2H decay half-life        │ 1.5 years     │ features.py              │
    │ Fatigue decay per day      │ 0.75          │ features.py              │
    │ Inactivity threshold       │ 90 days       │ features.py              │
    │ Server/Returner mix α      │ 0.60 / 0.40   │ hierarchical_model.py    │
    │ 2nd serve in probability   │ 0.95          │ hierarchical_model.py    │
    │ Point prob clamp range     │ [0.45, 0.85]  │ hierarchical_model.py    │
    │ EWMA λ                     │ 0.6467        │ Qian et al. Table 5      │
    │ EWMA initial momentum      │ 50.0          │ V2 Calculator            │
    │ Logistic β₀                │ 0.0           │ V2 Calculator            │
    │ Logistic β₁                │ 2.5           │ V2 Calculator            │
    │ Server advantage (sim)     │ ±10%          │ V2 Calculator            │
    │ Monte Carlo simulations    │ 500           │ V2 Calculator            │
    │ Perf weight E (scoring)    │ 0.1307        │ Qian et al. Table 7      │
    │ Perf weight S (streak)     │ 0.1184        │ Qian et al. Table 7      │
    │ Perf weight P (serve)      │ 0.1358        │ Qian et al. Table 7      │
    │ Perf weight R (return)     │ 0.6151        │ Qian et al. Table 7      │
    │ ML ensemble: LR weight     │ 0.40          │ V2 Calculator            │
    │ ML ensemble: RF weight     │ 0.60          │ V2 Calculator            │
    │ TRUE P: ML weight          │ 0.35          │ V2 Calculator            │
    │ TRUE P: Markov weight      │ 0.25          │ V2 Calculator            │
    │ TRUE P: Simulation weight  │ 0.40          │ V2 Calculator            │
    │ Meta-ensemble: LR          │ 0.35          │ live_prediction.py       │
    │ Meta-ensemble: NN          │ 0.45          │ live_prediction.py       │
    │ Meta-ensemble: Markov      │ 0.20          │ live_prediction.py       │
    │ Kelly fraction             │ 0.25 (¼)      │ backtesting + V2         │
    │ Kelly bankroll cap         │ 5%            │ backtesting              │
    │ Edge threshold             │ 2%            │ live_prediction.py       │
    │ LR accuracy (compact)      │ 93.73%        │ train output             │
    │ RF accuracy (compact)      │ 93.78%        │ train output             │
    │ NN ensemble size           │ 20 models     │ neural_network.py        │
    │ NN hidden size             │ 100 (tanh)    │ neural_network.py        │
    │ NN learning rate           │ 0.0004        │ neural_network.py        │
    │ Surface correlation H↔C    │ 0.28          │ features.py              │
    │ Surface correlation H↔G    │ 0.24          │ features.py              │
    │ Surface correlation C↔G    │ 0.15          │ features.py              │
    └────────────────────────────┴───────────────┴──────────────────────────┘


════════════════════════════════════════════════════════════════════════════════════
  APPENDIX C — ACADEMIC REFERENCES
════════════════════════════════════════════════════════════════════════════════════

[1] Qian, Y., Liu, Z., et al. (2025).
    "Predicting Point-by-Point Tennis Match Outcomes Using Momentum-Based Models."
    Journal of Computers, Vol. 36, No. 1, February 2025.
    — Source of EWMA λ=0.6467, 4-parameter performance model, entropy weights.

[2] Barnett, T. & Clarke, S.R. (2005).
    "Combining player statistics to predict outcomes of tennis matches."
    IMA Journal of Management Mathematics, 16(2), 113-120.
    — Source of hierarchical point→game→set→match Markov chain.

[3] O'Malley, A.J. (2008).
    "Probability Formulas and Statistical Analysis in Tennis."
    Journal of Quantitative Analysis in Sports, 4(2).
    — Tiebreak and deuce-game probability formulas.

[4] McHale, I. & Morton, A. (2011).
    "A Bradley-Terry type model for forecasting tennis match results."
    International Journal of Forecasting, 27(2), 619-630.
    — Serve/return interaction modeling.

[5] Kelly, J.L. (1956).
    "A New Interpretation of Information Rate."
    Bell System Technical Journal, 35(4), 917-926.
    — Optimal bet sizing criterion.


════════════════════════════════════════════════════════════════════════════════════
  END OF DOCUMENT
════════════════════════════════════════════════════════════════════════════════════
"""
