# TRUE P — Live In-Play Tennis Pricing Model

### A Quantitative Framework for Real-Time Match Win Probability Estimation

**Author:** Ayush Mishra  
**Date:** April 2026  
**Version:** 2.0

---

## Executive Summary

TRUE P is a multi-model ensemble system that produces a continuously-updating win probability for any professional tennis match. It fuses:

1. **Pre-match priors** from supervised ML trained on **143,530 historical matches**
2. **A hierarchical Markov chain** conditioned on score state (Barnett-Clarke)
3. **An EWMA-weighted performance model** implementing Qian et al. (2025)
4. **Monte Carlo simulation** from the current score position

The ensemble feeds a **Kelly-criterion bankroll optimizer** that identifies live betting edges across **seven signal categories**. All predictions self-calibrate point-by-point using incoming match data.

The system ingests live scores from ESPN's public API (ATP, WTA) with no API key, and runs as a **Streamlit dashboard** for real-time operation.

---

## Table of Contents

1. [System Architecture & Data Flow](#1-system-architecture--data-flow)
2. [Data Foundation](#2-data-foundation)
3. [Feature Engineering](#3-feature-engineering)
4. [Model 1 — Supervised ML](#4-model-1--supervised-ml-logistic-regression--random-forest)
5. [Model 2 — Hierarchical Markov Chain](#5-model-2--hierarchical-markov-chain-barnett-clarke)
6. [Model 3 — EWMA Performance & Momentum](#6-model-3--ewma-performance--momentum-qian-et-al-2025)
7. [Model 4 — Monte Carlo Simulation](#7-model-4--monte-carlo-match-simulation)
8. [The Ensemble: How TRUE P Is Computed](#8-the-ensemble-how-true-p-is-computed)
9. [Live Update Loop](#9-live-update-loop--point-by-point-recalculation)
10. [Edge Detection & Signal Taxonomy](#10-edge-detection--signal-taxonomy)
11. [Kelly Criterion & Bet Sizing](#11-kelly-criterion--bet-sizing)
12. [Backtesting Framework](#12-backtesting-framework)
13. [System Performance](#13-system-performance--model-accuracy)
14. [Live Data Integration](#14-live-data-integration)
15. [Future Work](#15-future-work)

---

## 1. System Architecture & Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│                                                                      │
│   Jeff Sackmann GitHub  ─┐                                           │
│   (ATP/WTA match-level   ├──►  data_pipeline_enhanced.py             │
│    CSVs, 2000-2024)      │         │                                 │
│                          │         ▼                                 │
│   tennis-data.co.uk  ────┘    tennis_betting.db                      │
│   (odds, Excel files)         143,530 matches | 5,282 players        │
│                               112,387 with full serve stats          │
│                                                                      │
│   ESPN Public API  ──────►  api/free_live_data.py                    │
│   (live scores, no key)       (20s cache, ATP + WTA)                 │
└──────────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────────┐
│                      FEATURE LAYER                                   │
│                                                                      │
│   features.py  ──►  19-dimensional feature vector                    │
│                      (time-decayed, surface-weighted, symmetric)      │
└──────────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────────┐
│                       MODEL LAYER                                    │
│                                                                      │
│   ┌────────────┐   ┌──────────────┐   ┌──────────────────────────┐  │
│   │  ML (35%)  │   │ Markov (25%) │   │  EWMA + Monte Carlo (40%)│  │
│   │  LR + RF   │   │ Score-aware  │   │  Momentum-adaptive       │  │
│   └─────┬──────┘   └──────┬───────┘   └────────────┬─────────────┘  │
│         └─────────────────┼────────────────────────┘                 │
│                           ▼                                          │
│                  ┌────────────────┐                                   │
│                  │    TRUE P      │                                   │
│                  │   (Ensemble)   │                                   │
│                  └────────────────┘                                   │
└──────────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────────┐
│                 SIGNAL & EXECUTION LAYER                             │
│                                                                      │
│   7 Edge Categories  ──►  Kelly Criterion  ──►  Dashboard Display    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Foundation

**Database:** SQLite (`tennis_betting.db`)  
**Records:** 143,530 professional matches (2000-01-03 → 2024-12-18)  
**Sources:** Jeff Sackmann open tennis dataset + tennis-data.co.uk

| Dimension | Count | Coverage |
|-----------|-------|----------|
| ATP matches | 74,906 | Grand Slams, Masters, 250/500, Qualifiers |
| WTA matches | 68,624 | Grand Slams, Premier, International |
| **Total** | **143,530** | **25 years of match data** |

| Surface | Count | Share |
|---------|-------|-------|
| Hard | 81,151 | 56.5% |
| Clay | 44,259 | 30.8% |
| Grass | 14,549 | 10.1% |
| Carpet | 3,441 | 2.4% |

| Data Quality | Count | Share |
|--------------|-------|-------|
| With full serve/return stats | 112,387 | **78.3%** |
| Unique players | 5,282 | — |
| Special parameter profiles | 2,084 | — |

### Per-Match Statistical Columns

```
aces, double_faults, serve_points, first_serve_in, first_serve_won,
second_serve_won, service_games, break_points_saved, break_points_faced
```

### Derived Statistics

| Statistic | Formula |
|-----------|---------|
| `first_serve_pct` | `first_serve_in / serve_points` |
| `first_serve_win_pct` | `first_serve_won / first_serve_in` |
| `second_serve_win_pct` | `second_serve_won / (serve_points - first_serve_in)` |
| `break_point_save_pct` | `break_points_saved / break_points_faced` |

---

## 3. Feature Engineering

**File:** `features.py` — Class `TennisFeatureExtractor`

All features are **Player 1 minus Player 2 differentials**, making the model **symmetric**: swapping players negates the feature vector and flips the prediction around 0.5.

### 3.1 Time Decay

Every historical data point is weighted by exponential time decay:

$$w(t) = 0.5^{t/\tau}$$

| Parameter | Half-Life (τ) | Rationale |
|-----------|---------------|-----------|
| General match stats | 0.8 years | Form 6 months ago ≈ 65% relevant |
| Head-to-head records | 1.5 years | H2H patterns are more persistent |
| Fatigue | 0.75/day | Per-day decay for 3-day match load |

### 3.2 Surface Transfer Matrix

When a player has sparse data on the target surface, skill estimates are transferred from other surfaces:

|  | Hard | Clay | Grass |
|--|------|------|-------|
| **Hard** | 1.000 | 0.280 | 0.240 |
| **Clay** | 0.280 | 1.000 | 0.150 |
| **Grass** | 0.240 | 0.150 | 1.000 |

> **Example:** A player with 200 hard-court matches but only 5 on grass — their grass estimate borrows from hard-court data with weight 0.24.

### 3.3 Fatigue Model

Recent match load over a 3-day window:

$$\text{fatigue} = \sum_{\text{recent matches}} \text{duration\_hours} \times 0.75^{\text{days\_ago}}$$

### 3.4 Full Feature Vector (19 dimensions + 1 meta-feature)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `RANK_DIFF` | ATP/WTA ranking differential |
| 2 | `POINTS_DIFF` | Ranking points differential |
| 3 | `WSP_DIFF` | Win on serve % (time-weighted) |
| 4 | `WRP_DIFF` | Win on return % (time-weighted) |
| 5 | `ACES_DIFF` | Aces per match differential |
| 6 | `DF_DIFF` | Double faults per match differential |
| 7 | `BP_SAVE_DIFF` | Break point save rate differential |
| 8 | `FIRST_SERVE_PCT_DIFF` | First serve in % differential |
| 9 | `FIRST_SERVE_WIN_PCT_DIFF` | First serve won % differential |
| 10 | `SECOND_SERVE_WIN_PCT_DIFF` | Second serve won % differential |
| 11 | `WIN_RATE_DIFF` | Overall win rate differential |
| 12 | `SURFACE_WIN_RATE_DIFF` | Surface-specific win rate differential |
| 13 | `SERVEADV` | Net serve advantage composite score |
| 14 | `COMPLETE_DIFF` | Match completion rate differential |
| 15 | `FATIGUE_DIFF` | 3-day match-load differential |
| 16 | `RETIRED_DIFF` | Recent inactivity flag (>90 days) |
| 17 | `DIRECT_H2H` | Head-to-head win rate (time-decayed) |
| 18 | `MATCHES_PLAYED_DIFF` | Career experience differential |
| 19 | `SURFACE_EXP_DIFF` | Surface-specific experience differential |
| +1 | `UNCERTAINTY` | Data completeness meta-feature |

### 3.5 Uncertainty Score

$$U = 0.4 \cdot \frac{1}{1 + \min(\text{matches})/20} + 0.3 \cdot \frac{1}{1 + \min(\text{surface\_matches})/10} + 0.3 \cdot \frac{1}{1 + \text{h2h\_count}/3}$$

$U \to 0$ when data is plentiful; $U \to 1$ when data is sparse. The model learns to be less confident when $U$ is high.

---

## 4. Model 1 — Supervised ML (Logistic Regression + Random Forest)

**Purpose:** Pre-match probability estimate using only information available before the match starts (no look-ahead bias).

### 4.1 Training Pipeline

**Files:** `train_proper_models.py`, `train_sackmann_models.py`

| Step | Action | Detail |
|------|--------|--------|
| 1 | Data preparation | Filter to 112,387 matches with complete stats |
| 2 | Symmetry enforcement | Random 50% player-order swap (negate features, flip label) |
| 3 | Temporal split | 80/20 by date (no future data in training set) |
| 4 | Feature selection | Forward greedy: score = 0.6·AUC + 0.4·(1 + Kelly_ROI) |
| 5 | Hyperparameter tuning | GridSearchCV (details below) |
| 6 | Calibration | Isotonic regression (5-fold CalibratedClassifierCV) |

### 4.2 Logistic Regression

```
Regularization: L2 (Ridge)
C ∈ {0.01, 0.1, 0.2, 0.5, 1.0, 10.0}
class_weight ∈ {None, balanced}
Post-calibration: Isotonic regression
```

### 4.3 Random Forest

```
n_estimators = 200
max_depth ∈ {8, 12, 16, None}
min_samples_leaf ∈ {5, 10, 20}
```

### 4.4 Production Model (6-feature compact)

The deployed V2 calculator uses a reduced 6-feature model for speed:

```python
features = [p1_serve_pct, p2_serve_pct, p1_bp_save, p2_bp_save, surface_hard, surface_clay]
```

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **93.73%** |
| Random Forest | **93.78%** |

**ML Ensemble:** $P_{ml} = 0.4 \cdot P_{lr} + 0.6 \cdot P_{rf}$

### 4.5 Neural Network Ensemble (Bagging)

**File:** `ml_models/neural_network.py` — Class `SymmetricTennisNet`

| Parameter | Value |
|-----------|-------|
| Architecture | Input(N) → Dense(100, tanh, no bias) → Dense(1, sigmoid, no bias) |
| Initialization | Xavier (symmetric, no bias terms) |
| Optimizer | SGD (lr=0.0004, momentum=0.55, weight_decay=0.002) |
| Batch size | 1 (online/stochastic gradient descent) |
| Early stopping | patience=10 epochs |
| Ensemble size | **20 bootstrap replicates** (bagging) |
| Final prediction | Simple mean of 20 model outputs |

**Meta-ensemble** (from `live_prediction.py`):

$$P_{\text{meta}} = 0.35 \cdot P_{lr} + 0.45 \cdot P_{nn} + 0.20 \cdot P_{\text{markov}}$$

---

## 5. Model 2 — Hierarchical Markov Chain (Barnett-Clarke)

**References:** Barnett & Clarke (2005), O'Malley (2008)  
**File:** `hierarchical_model.py`

Tennis is modeled as a **nested Markov chain**: **Point → Game → Set → Match**

Each level is computed **analytically** (no simulation), making it fast enough for real-time, point-by-point updates.

### 5.1 Level 1 — Point Win Probability

For the server:

$$P(\text{win point}) = P(\text{1st in}) \cdot P_{\text{adj}}(\text{win} | \text{1st in}) + (1 - P(\text{1st in})) \cdot 0.95 \cdot P_{\text{adj}}(\text{win} | \text{2nd in})$$

where $0.95$ is the probability of getting the second serve in, and:

$$P_{\text{adj}} = \alpha \cdot P_{\text{server}} + (1-\alpha) \cdot (1 - P_{\text{returner}})$$

| Parameter | Value | Meaning |
|-----------|-------|---------|
| α | 0.60 | Server's own ability weighted higher |
| 1-α | 0.40 | Opponent's return ability |
| Clamp range | [0.45, 0.85] | Prevent degenerate edge cases |

**ATP averages** (used as fallback):

| Stat | ATP Average |
|------|-------------|
| 1st serve in % | 62.5% |
| 1st serve won % | 71.5% |
| 2nd serve won % | 51.0% |
| Return 1st won % | 28.5% |
| Return 2nd won % | 49.0% |

### 5.2 Level 2 — Game Win Probability

Given point win probability $p$ (and $q = 1-p$), the probability of holding serve:

$$P(\text{hold}) = p^4 + 4p^4q + 10p^4q^2 + 20p^3q^3 \cdot \frac{p^2}{p^2 + q^2}$$

The deuce term: from deuce, the server wins the game with probability:

$$P(\text{win from deuce}) = \frac{p^2}{p^2 + q^2}$$

### 5.3 Level 3 — Set Win Probability

$$p_{\text{avg}} = \frac{P(\text{hold serve}) + (1 - P(\text{opp holds}))}{2}$$

Set probability sums over all paths to 6 games, with tiebreak:

$$P(\text{tiebreak}) = 0.5 + 0.8 \cdot (p_{\text{avg}} - 0.5)$$

### 5.4 Level 4 — Match Win Probability

**Best-of-3:**
$$P(\text{match}) = p_s^2 \cdot (3 - 2p_s)$$

**Best-of-5:**
$$P(\text{match}) = p_s^3 \cdot (6p_s^2 - 15p_s + 10)$$

### 5.5 Why This Model Matters for Live Betting

The Markov chain is **score-aware**. If Player A leads 6-3, 5-2, the model knows A needs only 1 more game while B needs to win the current set AND another entire set. This structural awareness **anchors the ensemble** to match reality.

---

## 6. Model 3 — EWMA Performance & Momentum (Qian et al. 2025)

**Reference:** "Predicting Point-by-Point Tennis Match Outcomes Using Momentum-Based Models," *Journal of Computers*, Vol. 36 No. 1, February 2025.

This is the **key innovation** — it captures **in-match momentum** that the other models cannot see.

### 6.1 The 4-Parameter Performance Model

Each point generates a performance score from four components:

$$P_j = w_E \cdot E(j) + w_S \cdot S(j) + w_P \cdot P(j) + w_R \cdot R(j) - K_j$$

| Parameter | Weight | Formula | Meaning |
|-----------|--------|---------|---------|
| **E:** Scoring Efficiency | **0.1307** | $\pm 1/(1 + \ln(\text{rally\_count}))$ | Efficient winners weighted higher |
| **S:** Winning Streak | **0.1184** | $\min(\text{consecutive\_pts}, 10)/10$ | Momentum from consecutive wins |
| **P:** Serve Efficiency | **0.1358** | $(\text{total\_serves} - \text{faults}) / \text{total\_serves}$ | Service game quality |
| **R:** Returner Win Rate | **0.6151** | Return points won in last 5 return games | **Strongest predictor** |

$K_j = +0.15$ if serving (penalizes server advantage), $-0.15$ if returning.

> **Key Insight:** $R$ (returner win rate) dominates at $w=0.6151$. This is the paper's central finding — **how well a player returns serve is the single strongest predictor of in-match momentum**. Weights were derived via entropy analysis across a large match corpus (Table 7).

### 6.2 EWMA Momentum

Raw performance is smoothed into a momentum indicator:

$$M_t = \lambda \cdot P_t + (1 - \lambda) \cdot M_{t-1}$$

| Parameter | Value | Source |
|-----------|-------|--------|
| $\lambda$ | **0.6467** | Optimized via simulated annealing |
| Pearson $r$ | **0.731** | Correlation with actual point outcomes |
| $M_0$ | 50.0 | Neutral starting momentum |

> At $\lambda=0.6467$: roughly **65% weight on current point, 35% on history** — tennis momentum decays fast.

### 6.3 Momentum → Win Probability (Logistic Transform)

$$m_{\text{diff}} = \frac{M_1 - M_2}{100}$$

$$P(P_1 \text{ wins point}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \cdot m_{\text{diff}})}}$$

| Parameter | Value |
|-----------|-------|
| $\beta_0$ | 0.0 (centered — equal momentum → 50%) |
| $\beta_1$ | 2.5 (moderate sensitivity) |

Paper-reported accuracy: **70%** on point-level prediction.

---

## 7. Model 4 — Monte Carlo Match Simulation

**File:** V2 Calculator — `simulate_match_from_position()`

Starting from the **CURRENT score state**, we simulate the match forward **500–1,000 times**.

### Algorithm

```
FOR each simulation (n=500):
    Copy current state: (sets, games, points, server)
    WHILE match not over:
        1. Determine point win probability:
           If P1 serving:   p = base_prob + 0.10  (server advantage)
           If P1 returning: p = base_prob - 0.10
           Clamp p to [0.15, 0.85]
        2. Draw random point outcome ~ Bernoulli(p)
        3. Update point → game → set → match score
           (with full tiebreak and deuce logic)
    Record winner

P(P1 wins) = count(P1 wins) / n_simulations
```

The `base_prob` comes from the **EWMA momentum model** (Section 6.3), so simulation adapts to in-match performance trends.

> **Key advantage:** Captures score-path effects that analytical models miss. If a player is up 5-4, 40-30 on serve, the simulation correctly estimates a very high probability of closing the set within the next few points.

---

## 8. The Ensemble: How TRUE P Is Computed

$$\boxed{\text{TRUE\_P} = 0.35 \cdot P_{ml} + 0.25 \cdot P_{\text{markov}} + 0.40 \cdot P_{\text{simulation}}}$$

| Model | Weight | Rationale |
|-------|--------|-----------|
| **ML** (LR + RF) | **0.35** | Strong pre-match prior from historical player quality |
| **Markov Chain** | **0.25** | Score-state awareness — exact structural position |
| **Monte Carlo** (EWMA-fed) | **0.40** | Momentum-adaptive forward simulation |

### Why These Weights?

- **ML (35%):** Strongest pre-match prior, but blind to in-match dynamics
- **Markov (25%):** Correctly handles score-state (2 sets to 0 = massive structural advantage), but assumes constant point probabilities
- **Simulation (40%):** Largest share because it's the ONLY model incorporating **EWMA momentum** — detects form shifts within 3-5 points, while the market often takes a full set to reprice

As the match progresses, momentum-based components naturally become more informative (more data → more stable EWMA), while the pre-match ML prior becomes less relevant.

### Match Condition Adjustments

Applied before ensembling:

| Condition | Adjustment |
|-----------|------------|
| Fast court (speed > 60) | +2% to big server |
| Slow court (speed < 40) | -2% to big server |
| High altitude (> 1 km) | +1.5% to server |
| Hot weather (> 30°C) | -1% to both (fatigue) |
| Cold weather (< 10°C) | -1% to both |
| Indoor | +0.5% to server |
| Surface mastery | ±(mastery/100) |

---

## 9. Live Update Loop — Point-by-Point Recalculation

Every tracked point triggers a **full recalculation pipeline** (<200ms):

```
POINT TRACKED: Player X wins point
    │
    ├─ Step 1.  Update winning streaks (consecutive points counter)
    ├─ Step 2.  Update serve fault tracking
    ├─ Step 3.  Update return game tracking (last 5 return games)
    │
    ├─ Step 4.  Calculate 4-parameter Performance (E, S, P, R) for BOTH players
    ├─ Step 5.  Update EWMA:  M_t = 0.6467 · P_t + 0.3533 · M_{t-1}
    ├─ Step 6.  Store in performance_history[]
    │
    ├─ Step 7.  Convert momentum → point probability (logistic transform)
    ├─ Step 8.  Run 500 Monte Carlo simulations from current score
    ├─ Step 9.  Recalculate Markov chain from current score state
    │
    ├─ Step 10. Ensemble: TRUE_P = 0.35·P_ml + 0.25·P_markov + 0.40·P_sim
    │
    ├─ Step 11. Run 7-category edge detection
    ├─ Step 12. Compute Kelly-optimal bet size
    └─ Step 13. Update all visualizations
```

---

## 10. Edge Detection & Signal Taxonomy

Seven categories of betting edges are continuously scanned:

| # | Edge Category | Severity | Trigger Condition |
|---|---------------|----------|-------------------|
| 1 | **Break Opportunity** | CRITICAL / HIGH | BP save rate < expected by 15% / 10% |
| 2 | **Critical BP Count** | CRITICAL | ≥10 BPs faced in match (65% lose) |
| 3 | **EWMA Momentum Surge** | CRITICAL / HIGH | Momentum gap > 15 / > 8 points |
| 4 | **Momentum Turning Point** | HIGH | EWMA direction reversal detected |
| 5 | **Clutch Situation** | HIGH | Score at 4-4+ AND clutch differential ≥ 4 |
| 6 | **Service Vulnerability** | HIGH | Double fault rate > 15% per service game |
| 7 | **Consistency Edge** | MEDIUM | Unforced error ratio > 40% with 20+ pts |

**Key empirical insights driving edge detection:**
- Winners save **63%** of BPs → Losers save **49%** → **14.4pp gap**
- Players facing **≥10 BPs** in a match lose **65%** of the time
- First serve win % above **72%** correlates with **74%** match win rate

---

## 11. Kelly Criterion & Bet Sizing

### Edge Calculation

$$\text{edge} = P_{\text{true}} - P_{\text{implied}}$$

where $P_{\text{implied}} = 1 / \text{decimal\_odds}$

**Minimum edge threshold: 2%** (below this, no bet)

### Kelly Fraction

Full Kelly:
$$f^* = \frac{P \cdot \text{odds} - 1}{\text{odds} - 1}$$

We use **quarter-Kelly** to reduce variance:

$$f = 0.25 \cdot f^*$$

With hard cap: $f = \min(f, \, 0.05)$ — never risk more than 5% of bankroll.

### Worked Example

| Input | Value |
|-------|-------|
| TRUE P | 0.62 (our model: 62% chance) |
| Market odds | 2.10 (bookmaker implies 47.6%) |
| Edge | 0.62 - 0.476 = **14.4%** |
| Full Kelly | (0.62 × 2.10 - 1) / (2.10 - 1) = **27.4%** |
| Quarter Kelly | 0.25 × 0.274 = **6.9%** |
| Capped | **5.0%** |

→ **Bet 5% of bankroll on P1 at 2.10**

### Why Quarter Kelly?

Full Kelly maximizes long-term log-wealth growth but experiences severe drawdowns (~50% drawdowns are common). Quarter Kelly sacrifices ~44% of theoretical growth rate but **reduces drawdowns by >75%**, making the strategy robust to model estimation error.

---

## 12. Backtesting Framework

**File:** `backtesting/betting_strategies.py`

### Three Strategies

| Strategy | Logic |
|----------|-------|
| **Fixed** | Flat 1-unit stake on predicted winner ($P > 0.5$) |
| **Value** | Bet only when $P_{\text{model}} > P_{\text{implied}}$ (positive edge) |
| **Kelly** | Quarter-Kelly sizing, capped at 5% |

### Risk Metrics

| Metric | Formula |
|--------|---------|
| Annualized Sharpe | $\text{Sharpe} = \frac{\bar{r}}{\sigma_r} \times \sqrt{250}$ |
| Maximum Drawdown | Running peak-to-trough decline |
| Monthly P&L | Calendar-month decomposition |
| ROI | Total profit / total staked |

### Statistical Validation

| Test | Purpose |
|------|---------|
| **McNemar's Test** | $\chi^2 = \frac{(\|n_{01} - n_{10}\| - 1)^2}{n_{01} + n_{10}}$ — pairwise model comparison |
| **Bootstrap 95% CI** | 1,000 resamples for ROI confidence interval |
| **Calibration Diagram** | Binned predictions vs actual frequency |

---

## 13. System Performance & Model Accuracy

### Pre-Match Model Accuracy (out-of-sample)

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | **93.73%** | 6-feature compact model |
| Random Forest | **93.78%** | 6-feature compact model |
| ML Ensemble (0.4/0.6) | **~94%** | LR + RF weighted |
| Neural Network (20×) | **~92%** | 20-model bagged ensemble |
| Hierarchical Markov | **~68%** | Pure statistical, no ML |
| Meta-Ensemble | **~93%** | 0.35 LR + 0.45 NN + 0.20 Markov |

### EWMA Momentum Model Performance

| Metric | Value |
|--------|-------|
| EWMA $\lambda$ optimization | Pearson $r$ = **0.731** at $\lambda = 0.6467$ |
| Logistic point prediction | **70% accuracy** (paper-reported) |

> The momentum model adds value not through raw accuracy but through **speed of adaptation**. When a player's form shifts mid-match, the EWMA detects it within 3-5 points, while the market often takes a full set to reprice.

---

## 14. Live Data Integration

**File:** `api/free_live_data.py` — Class `FreeLiveTennisService`

| Property | Detail |
|----------|--------|
| **Source** | ESPN public JSON API |
| **Cost** | Free (no API key, no rate limits) |
| **Coverage** | ATP, WTA |
| **Scoreboard** | `site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard` |
| **Rankings** | Top 150 ATP + 150 WTA |

### Cache Policy

| Data Type | Refresh Rate |
|-----------|-------------|
| Live scores | 20 seconds |
| Schedule | 5 minutes |
| Rankings | 1 hour |
| Player profiles | 1 hour |

### Surface Auto-Detection

Tournament name matching against keyword lists:
- **Clay:** Roland Garros, Rome, Madrid, Barcelona, Monte Carlo, Hamburg
- **Grass:** Wimbledon, Queens, Halle, Eastbourne, Mallorca
- **Hard:** Australian Open, US Open, Indian Wells, Miami, Shanghai

### Dashboard Integration

| Page | Function |
|------|----------|
| **Page 8** (Live Matches) | Auto-refreshing scoreboard with "Load into Calculator" button |
| **Page 7** (Calculator V2) | Auto-fill from live matches, 30s refresh, manual override |

---

## 15. Future Work

| Priority | Enhancement | Impact |
|----------|-------------|--------|
| 🔴 High | **Point-level live data** (e.g., Betfair Stream) | Full auto EWMA — no manual point entry |
| 🔴 High | **Live odds integration** (Betfair exchange, Pinnacle) | Automatic TRUE P vs market edge detection |
| 🟡 Medium | **Dynamic ensemble weights** (Bayesian) | Shift toward best-performing model per match |
| 🟡 Medium | **Set-level context features** | Player behavior changes when up/down in sets |
| 🟢 Low | **Player embeddings** (word2vec-style) | Replace hand-crafted features |
| 🟢 Low | **Automated execution** (Betfair API) | Auto-place orders when edge > threshold |

---

## Appendix A — Complete File Map

```
tennis10/
├── tennis_betting.db                     # SQLite: 143K matches, 5K players
├── features.py                           # 19-dim feature extraction engine
├── hierarchical_model.py                 # Barnett-Clarke Markov chain
├── live_prediction.py                    # 3-model meta-ensemble
├── point_tracker.py                      # Point-by-point Markov tracker
├── data_pipeline.py                      # Ingest tennis-data.co.uk
├── data_pipeline_enhanced.py             # Ingest Jeff Sackmann GitHub
├── train_proper_models.py                # Train LR + NN (forward selection)
├── train_sackmann_models.py              # Train on Sackmann data (15 models)
├── api/
│   ├── free_live_data.py                 # ESPN free API service
│   └── prediction_service.py             # REST API wrapper
├── ml_models/
│   ├── logistic_regression.py            # SymmetricTennisLogistic
│   ├── neural_network.py                 # SymmetricTennisNet (20-bag)
│   └── *.pkl                             # Serialized trained models
├── backtesting/
│   └── betting_strategies.py             # Fixed / Value / Kelly backtests
├── evaluation/
│   └── model_comparison.py               # McNemar, calibration, Brier
├── dashboard/
│   ├── streamlit_app.py                  # Streamlit entry point
│   └── pages/
│       ├── 7_🎯_Live_Calculator_V2.py    # Full live tracker + all models
│       └── 8_🔴_Live_Matches.py          # Live ESPN scoreboard
└── tests/
    ├── test_features.py                  # Feature extraction tests
    ├── test_models.py                    # Model prediction tests
    ├── test_betting.py                   # Betting strategy tests
    └── test_integration.py               # End-to-end pipeline tests
```

---

## Appendix B — All Constants & Hyperparameters

| Constant | Value | Source |
|----------|-------|--------|
| Time decay half-life | 0.8 years | `features.py` |
| H2H decay half-life | 1.5 years | `features.py` |
| Fatigue decay per day | 0.75 | `features.py` |
| Inactivity threshold | 90 days | `features.py` |
| Server/Returner mix α | 0.60 / 0.40 | `hierarchical_model.py` |
| 2nd serve in probability | 0.95 | `hierarchical_model.py` |
| Point prob clamp range | [0.45, 0.85] | `hierarchical_model.py` |
| EWMA λ | **0.6467** | Qian et al. Table 5 |
| EWMA initial momentum | 50.0 | V2 Calculator |
| Logistic β₀ | 0.0 | V2 Calculator |
| Logistic β₁ | 2.5 | V2 Calculator |
| Server advantage (sim) | ±10% | V2 Calculator |
| Monte Carlo simulations | 500 | V2 Calculator |
| Perf weight E (scoring) | 0.1307 | Qian et al. Table 7 |
| Perf weight S (streak) | 0.1184 | Qian et al. Table 7 |
| Perf weight P (serve) | 0.1358 | Qian et al. Table 7 |
| Perf weight R (return) | **0.6151** | Qian et al. Table 7 |
| ML ensemble: LR / RF | 0.40 / 0.60 | V2 Calculator |
| TRUE P: ML / Markov / Sim | 0.35 / 0.25 / 0.40 | V2 Calculator |
| Meta-ensemble: LR / NN / Markov | 0.35 / 0.45 / 0.20 | `live_prediction.py` |
| Kelly fraction | 0.25 (quarter) | backtesting |
| Kelly bankroll cap | 5% | backtesting |
| Edge threshold | 2% | `live_prediction.py` |
| LR accuracy | 93.73% | training output |
| RF accuracy | 93.78% | training output |
| NN ensemble size | 20 models | `neural_network.py` |
| NN hidden size | 100 (tanh) | `neural_network.py` |
| NN learning rate | 0.0004 | `neural_network.py` |
| Surface correlation H↔C | 0.28 | `features.py` |
| Surface correlation H↔G | 0.24 | `features.py` |
| Surface correlation C↔G | 0.15 | `features.py` |

---

## Appendix C — Academic References

1. **Qian, Y., Liu, Z., et al. (2025).** "Predicting Point-by-Point Tennis Match Outcomes Using Momentum-Based Models." *Journal of Computers*, Vol. 36, No. 1. — *Source of EWMA λ=0.6467, 4-parameter performance model, entropy weights.*

2. **Barnett, T. & Clarke, S.R. (2005).** "Combining player statistics to predict outcomes of tennis matches." *IMA Journal of Management Mathematics*, 16(2), 113-120. — *Source of hierarchical Markov chain.*

3. **O'Malley, A.J. (2008).** "Probability Formulas and Statistical Analysis in Tennis." *Journal of Quantitative Analysis in Sports*, 4(2). — *Tiebreak and deuce formulas.*

4. **McHale, I. & Morton, A. (2011).** "A Bradley-Terry type model for forecasting tennis match results." *International Journal of Forecasting*, 27(2), 619-630. — *Serve/return interaction modeling.*

5. **Kelly, J.L. (1956).** "A New Interpretation of Information Rate." *Bell System Technical Journal*, 35(4), 917-926. — *Optimal bet sizing criterion.*

---

*End of Document*
