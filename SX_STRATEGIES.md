# SX Bet — three-tier execution (game / set / match)

Connects the live intelligence engine to [sx.bet](https://sx.bet), an on-chain
betting **exchange**. Reads are public; orders are EIP-712-signed taker fills
against the order book.

```bash
python -m execution.sx_strategies --once                  # one dry-run scan
python -m execution.sx_strategies --interval 30           # loop (dry-run)
python -m execution.sx_strategies --strategies match      # subset
python -m execution.sx_strategies --interval 30 --arm     # REAL (needs key + funds)

python -m execution.sx_backtest                           # walk-forward backtest
```

Default tiers are `match,set`. **GAME is disabled** — pre-match it backtests worse
than a constant; it is an in-play-only market (§8-§9). Nothing posts without
`--arm` plus a funded, KYC'd wallet (§5-§6).

---

## 1. The markets SX actually lists

Verified live against `GET /markets/active?sportIds=6` — SX tennis carries far
more than the moneyline the old adapter used:

| Type | Market | Tier | Example outcomes |
|---|---|---|---|
| 52 | Match winner (moneyline) | **MATCH** | `Taylor Fritz` / `Alex Michelsen` |
| 202 / 203 / 204 | 1st / 2nd / 3rd set winner | **SET** | `Zakharova (1st Period)` |
| 866 | Set spread (−1.5 sets) | **SET** | `Zakharova −1.5 (sets)` |
| 165 | Total sets O/U 2.5 | **SET** | `Over 2.5 (sets)` |
| 201 | Game spread (handicap) | **GAME** | `Jodar −2.5` / `Musetti +2.5` |
| 166 | Total games O/U | **GAME** | `Over 22.5` / `Under 22.5` |

Type 201 and 166 are the *deepest* books (46 and 32 live markets in one snapshot
versus 4 moneylines). Tempting — but depth of book is not evidence of a signal:
the backtest in §8 shows pre-match GAME pricing is worse than a constant (root
cause in §9), so that tier ships **disabled**. Liquidity you cannot price is not
opportunity.

## 2. How each tier is priced

**MATCH** uses the engine's live True P directly (`execution.inplay`), because
that number carries the momentum and serve-regression adjustments the simulator
doesn't model. Edge = True P − SX taker price.

**SET and GAME** are priced by a new Monte-Carlo pricer in `sx_strategies.py`:

1. `hold_prob(p)` — closed-form P(hold) from serve point-win probability, deuce
   solved exactly. Validated: 0.60 → 73.6% hold, 0.65 → 83.0% (textbook values).
2. `simulate()` — from the live score, plays out every remaining game
   `SX_SIM_N` times (default 4,000): alternating serve, analytic holds,
   point-by-point tiebreaks. Yields distributions over set winners, total games,
   game differential, total sets, and 2-0 sweeps.
3. The sim's match probability is **anchored to the engine's True P** (a shift
   term) so derivative prices inherit the same view of who is better, momentum
   included.

Sanity checks that ship with the module:

| Scenario | Result |
|---|---|
| Even match, 0-0, serve 0.62 each | p1 = 0.494, three sets 48.2%, P(>22.5 games) 58.0% |
| Favourite 1-0, 3-1 up, serve 0.66 v 0.58 | p1 = 0.985, wins set 2 93.5%, P(>20.5 games) 15.9% |

## 3. The venue drives the scan (not the live feed)

SX books mostly **pre-match** main-tour fixtures. A snapshot measured 14 SX
fixtures with open books and **zero** overlap with Sofascore's 30 live matches —
so a scanner that iterates live fixtures (as `sx_breakbot` does) can find
nothing to trade for long stretches.

`sx_fixtures()` therefore enumerates SX's own market index, and `true_p_for()`
prices each fixture with the live engine when the match is in progress, falling
back to the **same Markov engine run from 0-0** when it is not. Same serve
inputs, same model — only the scoreboard differs.

## 4. Risk controls

Every fill passes the same gauntlet, per tier:

- **Edge floors** — match 4%, set 6%, game 6%. Derivative markets get higher
  floors because simulation error compounds on top of model error.
- **Suspect ceiling** — any edge above `SX_MAX_EDGE` (20%) is *skipped and
  logged*, not bet. A 25% edge against a sharp exchange means our inputs are
  broken, not that we found free money. This is the lesson the tennis journal
  taught at −26.4% ROI in the 40%+ edge bucket.
- **¼ Kelly**, capped at 5% of bankroll and `$25` absolute per fill.
- **One position per (fixture, tier, outcome)** — deduped against the journal.
- **Per-fixture exception isolation** — one bad fixture can't kill the loop.

Every fill (simulated or real) is journaled to `tennis_betting.db` via
`trade_log` with `venue="sxbet"` and `market_type` in `match|set|game`, so P&L
by tier is queryable from day one.

## 5. Testnet: verified, and it found two real bugs

The Toronto testnet (`SXBET_API=https://api.toronto.sx.bet`) **lists no tennis
markets** — 0 across every snapshot, though basketball/soccer/football carry
~100 each. So a tennis bet cannot be rehearsed there. Of 514 testnet markets
sampled, exactly **one** had a two-sided book.

Posting a real signed order to that market (throwaway unfunded key, testnet
endpoint only) exposed two defects that would have failed on **mainnet** too:

1. `takerSig must be a valid hex string of length 65 bytes` — `eth_account`
   ≥0.13 returns bare hex from `.signature.hex()`; SX requires the `0x` prefix.
2. `market must be a valid hex string of length 32 bytes` — the request body was
   sending the literal `"N/A"` placeholder instead of the market hash. (`"N/A"`
   is correct *inside* the signed Details struct, not in the POST body.)

Both are fixed in `sxbet.py`. After the fix the same POST passes schema
validation and signature checks, and stops at `INSUFFICIENT_KYC` — an
account-level requirement, not a code path issue. **The order plumbing is
verified as far as an unfunded, un-KYC'd wallet can reach.**

To go further you need a real SX account: complete KYC, fund the SX-Network
wallet with USDC, and set `SXBET_PRIVATE_KEY` / `SXBET_WALLET_ADDRESS`.

## 6. Safety contract

Dry-run is the default and it is **real**: the code builds the fill, and if
`SXBET_PRIVATE_KEY` is present it EIP-712 **signs** it — then throws it away
without posting. That proves the whole execution path without risking a cent.

A live post requires **all four**:

1. `--arm` on the command line
2. `TRADING_DRY_RUN=false`
3. `SXBET_PRIVATE_KEY` set
4. a funded SX-Network wallet

Rehearse on the testnet first: `SXBET_API=https://api.toronto.sx.bet` (chain id
and contracts auto-detect from `GET /metadata`).

## 7. ⚠ Measured on first run: DO NOT ARM YET

A production-floor scan against the real live SX book (2026-08-02) produced:

| | |
|---|---|
| Fills that would fire | **35** |
| Additional skipped as suspect (>20%) | 20 |
| Edge distribution | min 5.1% · **median 10.6%** · max 19.6% |
| By bucket | 4-6%: 2 · 6-10%: 13 · 10-15%: 14 · 15-20%: 6 |

**A median 10.6% edge across 35 simultaneous positions against a sharp exchange
is not alpha — it is a calibration failure.** Real edges on a liquid two-sided
book are rare and small; finding 55 of them at once means our number is wrong,
not that SX is asleep. This is precisely the shape the tennis journal already
documented on Polymarket: ROI −11.0% in the 10-20% claimed-edge bucket and
−26.4% at 40%+, i.e. the *bigger* the edge shown, the *more* it lost.

The plumbing is verified and correct. The **signal is not ready to trade.**

## 8. Backtest results — which tiers can actually price

`python -m execution.sx_backtest` runs a **walk-forward, leakage-safe** backtest
over `tennis_data.db`: every serve input comes only from a player's *prior*
matches, and the tracker updates only after the match is scored. 6,000 best-of-3
matches, 400 sims each.

| Tier | Brier | Baseline | Verdict |
|---|---|---|---|
| **MATCH** — P(wins match) | **0.2308** | 0.2500 coin-flip | ✅ better by 0.0192, accuracy 61.5% |
| **SET** — P(wins set 1) | **0.2387** | 0.2500 coin-flip | ✅ better by 0.0113, accuracy 58.7% |
| **GAME** — P(games > 22.5) | **0.2680** | 0.2466 base rate | ❌ **WORSE by 0.0214** |

**SET is the best-calibrated tier**, tracking actual frequencies within ~2pp all
the way to the 70-80% bucket. **MATCH is sound but overconfident at the top** —
it says 74.5% and wins 68.1%, says 84.6% and wins 77.5%, the same ~6-7pp
overconfidence the engine-calibration finding already flagged.

**GAME is broken and is now OFF by default.** It is worse than always guessing
the base rate, and worse than a coin flip on accuracy (45.6%). The bias is
systematic and huge — in the two buckets holding 5,464 of 6,000 samples:

| Predicted P(over) | n | Actual | Gap |
|---|---|---|---|
| 50-60% | 3,090 | 44.4% | **+11.9pp** |
| 60-70% | 2,374 | 44.0% | **+18.7pp** |

The simulator thinks matches run long far more often than they do. That also
explains the live scan firing so many "Over" bets at fat edges — it was betting
its own bias. **I had this backwards earlier:** I suggested `game` as the tier
most worth studying because it has the deepest books. The data says the
opposite; depth of book is not evidence of a signal.

## 9. Root cause of the GAME failure — diagnosed

The GAME tier didn't fail because the simulator is wrong. It fails because
**pre-match it is fed a feature with no signal in it.**

**The simulator is sound.** Fed each match's *actual* in-match serve
percentages (an oracle it can't have pre-match), the identical code scores:

| GAME totals, oracle serve inputs | Brier |
|---|---|
| Simulator | **0.1749** |
| Base-rate constant | 0.2491 |

That is a large, genuine win. The machinery works.

**The input is the problem.** Correlation with actual total games:

| Feature | r |
|---|---|
| **Actual in-match serve gap** | **−0.6714** |
| Actual in-match serve level | +0.2073 |
| **Career-average serve gap** (what it was fed) | **+0.0202** |
| Career-average serve level | +0.0973 |

The in-match serve gap dominates — a big gap means a blowout, so fewer games.
Career-average serve gap correlates **+0.02 with the outcome: nothing.** Career
averages regress every player toward ~0.64 because they average over all
opponents, compressing the real gap 3.8× (mean |gap| 0.030 vs 0.113 observed).

Two fixes were tested and rejected on evidence:

- **Opponent adjustment** (Barnett-Clarke serve−return): moved compression
  3.78× → 3.69×. Negligible.
- **Spread restoration** (scale the gap by k): no value of k made GAME beat the
  baseline, and k>1 actively degraded MATCH and SET. You cannot amplify a
  zero-signal feature — scaling noise gives you louder noise.

**The real bug, now fixed:** `scan_fixture` passed raw `ip._serve_win()` to the
simulator, while `live_true_p` internally blends in live in-match serve stats.
So even for in-progress matches the simulator was getting the useless career
number and never the informative live one. `serve_inputs()` now mirrors the
engine's blending.

**Conclusion — GAME is an in-play-only market.** The signal it needs is only
observable once the match is under way, so the tier is now gated on live serve
data being present, and stays off the default tier list until the in-play
version is validated on its own journal. Pre-match totals are not a hard
modelling problem to solve; they are unpredictable from what we know pre-match.

## 10. Before you arm this

1. **GAME is now gated to in-play only** (§9). Validate the in-play version on its
   own journal before enabling it; the pre-match version is provably signal-free
   and no amount of recalibration fixes a feature with r=0.02.
2. **MATCH needs shrinking at the top.** Pull `true_p` toward the market above
   ~70%, then re-run the backtest and confirm the top buckets close.
3. **SET is the tier to study first** — best calibrated, and sets settle in
   hours so live evidence accrues fast.
4. **Run dry and let it journal**, then settle and score per tier against the
   journal before arming anything, at $1 stakes.

Remember what calibration does and does not prove: it is a *precondition* for an
edge, not an edge. Beating a coin flip does not mean beating SX's book.

---

*Verified end-to-end on 2026-08-02: market discovery across all six types,
de-vigged taker pricing, Monte-Carlo distributions, a signed dry-run fill on a
real live book (San Marino Challenger ATP moneyline, back1 0.685 / overround
2.6%), and all three tiers firing + journaling (game 28, match 7, set 11 in one
low-floor scan). See §7-§8 before arming anything.*
