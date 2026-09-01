# Live Market Engine

Implementation of the Live Market Engine PRD. Point in → repriced, compared to
market, gated, broadcast out.

```
provider → normalize → sequence check → state → model → market
         → signal gate → room broadcast → browser
```

## Run it

```bash
python -m execution.live doctor                      # what is configured
python -m execution.live replay tapes/sample.jsonl   # whole pipeline, no credentials
python -m execution.live serve                       # gateway on :8080
```

`replay` exists because the live point feed is a paid tier. The entire pipeline
runs off a recorded tape on a laptop with no API key, which is also how the
failure modes get tested.

## Modules

| File | PRD | Responsibility |
|---|---|---|
| `events.py` | §5, §6 | Canonical `LiveEvent`; derives GAME_END / BREAK / set boundaries most feeds omit |
| `feed.py` | §7, §21, §22 | Sequence integrity, freshness bands, latency breakdown |
| `provider.py` | §4 | `TennisDataProvider` interface + `ReplayProvider` |
| `providers/livetennis.py` | §4 | Live Tennis API adapter |
| `providers/failover.py` | §23 | Feed health and failover decisions |
| `state.py` | §8 | Match state, game tape, store interface (in-memory / Redis) |
| `odds.py` | §9, §10, §11 | Bookmaker vs exchange, de-vig, consensus |
| `engine.py` | §12, §13, §14 | Bridge to the existing model; tiering; match/set/game ladder |
| `signals.py` | §15, §16, §24 | Hysteresis state machine and the publish gate |
| `gateway.py` | §17-§20 | Match rooms, fan-out, dynamic subscription |
| `runtime.py` | — | The order in which all of it happens |
| `calibration.py` | — | Recorder with a fixed orientation; reliability, Platt, isotonic |
| `edge_publisher.py` | §25 | Pushes updates to the Cloudflare edge |
| `providers/polymarket_odds.py` | §9, §10 | Exchange prices with real depth |
| `../../edge/` | §25 | Worker + Durable Object match rooms |

## What this package does NOT do

**It does not compute probabilities.** The model already exists in this repo and
is tuned against settled bets:

- `execution/inplay.py` — score-aware Markov `true_p`
- `execution/momentum.py` — live momentum
- `execution/edgescore.py` — uncertainty gate and confidence sizing

A second live engine here would be the **third** implementation of the same
idea (there is already a TypeScript one under `trading-terminal/src/lib`), and
three engines that disagree in production is a worse problem than the
duplication that already exists. Anything in this package that starts computing
a probability is a bug.

## Decisions that differ from the PRD

**Sequence gaps get a reorder window (§7).** "101 → 103 means resync" fires on
ordinary network jitter; load-balanced WebSocket fan-out reorders routinely. A
hole has 750 ms to fill before it counts as a gap. A resync storm is itself an
outage.

**Failover is silence + debounce, not silence alone (§23).** A tennis point can
be 30 s apart, so a 20 s silence threshold alone would fail over mid-game.
Switching costs the momentum tape, so the bar is a sustained outage.

**Breaks are never inferred without a known server (§6).** Momentum and the
signal engine weight breaks heavily; a fabricated break is worse than a missing
one. A two-game jump likewise produces no game winner.

**The entry gate is not a fixed edge threshold (§16).** The PRD proposes
`edge > 6%`. Against this model that fires on nearly everything: the measured
model-vs-market gap is ~13 points per leg and ~14 points high on favourites, and
`execution/calibrate.py` records the same finding independently ("predicts
80%+, wins ~60%"). `edgescore.score_edge` divides the edge by an uncertainty
that *includes* that measured error, so an edge only grades green when it is
large relative to how wrong we know we are. Both that and a raw-edge floor must
pass.

**Absent market ≠ stale market.** A match with no odds attached reports a LIVE
scoreboard and no signal, rather than OFFLINE. Found by running the pipeline: a
working feed was being diagnosed as an outage because no odds source existed.

**Health transitions always broadcast.** Update suppression inspects only the
model probability, so a feed going DEGRADED while the probability sat still was
silently dropped and the client kept rendering stale state as live — the exact
failure §22 exists to prevent, arriving through the broadcast layer.

## Cost note

Fan-out stops upstream cost scaling with **users**. It does not reduce the
requests needed to track **matches** — that is set by endpoint shape:

```
20 matches, polled individually every 30s   57,600 req/day   fits no tier
20 matches via one bulk live endpoint          1,440 req/day   fits Basic
```

`poll_live()` therefore fetches the whole live slate in one call. Sub-second
updates need the WebSocket tier regardless; a polling feed cannot beat its own
interval.

## Calibration

`calibration.py` holds the recorder, the diagnostics (reliability diagram,
Brier, log-loss, ECE) and two correctors (Platt, isotonic).

**There is still no calibration, and that is a data problem, not a code one.**
Fitting one on `tennis_betting.db:trade_log` was attempted and abandoned on the
evidence:

```
corr(true_p, market_price)     +0.15    should be ~+0.8 for the same side
p in 0.0-0.1                   88% won
p in 0.9-1.0                   64% won
```

A prediction of 0.05 winning 88% of the time is a mislabelled column, not a
miscalibrated model. All four orientation conventions were tested across match,
set1 and set2: set markets correlate −0.62 as stored and +0.38 under a
different flip. The log mixes conventions across market types and code
versions. Fitting on it would produce an authoritative-looking number that
means nothing — and it would then be wired into the live signal gate.

So the fix is to start recording properly. `CalibrationRecorder` enforces one
orientation: `p_model` is always the probability of `selection`, and settlement
names the WINNER rather than asserting per-row correctness, which removes the
last place a caller could invert it. The runtime records one observation per
match per tier and `settle_match()` closes the loop.

The machinery is verified against synthetic data with a known injected bias
(19 tests): the diagnostics detect the bias in the right direction and
magnitude, Platt and isotonic both remove it, and a correctly-calibrated model
is left alone. It is ready for the day the data exists.

```bash
python -m execution.live calibrate      # diagram first, fit second
```

## Status

Built and tested: everything above. **111 live tests**, and the full repository
suite is green (196 passed).

Not done, and why:

- **The provider adapter has never spoken to the real endpoint.** It is written
  against the documented shape and exercised through an injected transport;
  a live key is needed. `python -m execution.live smoke` prints raw frames
  beside their normalized form so the first five minutes with a key are a
  field-name diff rather than a debugging session.
- **The edge is not deployed.** `edge/` holds the Worker and Durable Object,
  syntax-checked but never pushed — that needs a Cloudflare account.
  `RoomRegistry` does the same fan-out in-process and works today.
- **No calibration dataset yet.** See above. Weeks of recording, not hours of
  coding.
