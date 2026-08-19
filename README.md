# tennis10 🎾

A live tennis **trading** system: historical data pipeline → probability models →
live in-play intelligence → market execution on Polymarket / SX Bet → a hosted
trading terminal.

What started as an ATP data pipeline is now four layers deep:

| Layer | What it does | Where |
|---|---|---|
| **Data** | Historical matches + odds, Match Charting rally data, live Sofascore feed | [data_pipeline.py](data_pipeline.py), [fetch_charting_data.py](fetch_charting_data.py), [sofa_proxy.py](sofa_proxy.py) |
| **Model** | Feature engineering, hierarchical Markov, neural net, score-aware in-play engine | [features.py](features.py), [hierarchical_model.py](hierarchical_model.py), [neural_network.py](neural_network.py), [execution/inplay.py](execution/inplay.py) |
| **Intelligence** | Live momentum, rally profiles, smart-money, cross-market fair odds | [execution/momentum.py](execution/momentum.py), [execution/rally.py](execution/rally.py), [execution/smartmoney.py](execution/smartmoney.py), [execution/sxbet.py](execution/sxbet.py) |
| **Execution** | Signal → sized order → journal → settlement → calibration | [execution/pipeline.py](execution/pipeline.py), [execution/agent.py](execution/agent.py), [execution/trade_log.py](execution/trade_log.py) |

Plus a **Next.js trading terminal** ([trading-terminal/](trading-terminal/)) deployed at
`tennisalpha.in`, a **table tennis** predictor ([tabletennis/](tabletennis/)), and a
**computer-vision** prototype ([tennis_vision/](tennis_vision/)).

---

## ⚡ Quick start

```bash
# Python deps
pip install pandas openpyxl matplotlib seaborn requests jupyter tls_client

# 1. Historical database
python data_pipeline.py

# 2. Local Sofascore proxy (required by everything live)
python sofa_proxy.py                  # port 3001

# 3. Live intelligence table for every open fixture
python -m execution.intel

# 4. Trading terminal (web UI)
cd trading-terminal && npm install && npm run start:all   # proxy + next dev on :3000
```

---

## 🗄️ Historical data pipeline

[data_pipeline.py](data_pipeline.py) fetches ATP match data from
[tennis-data.co.uk](http://www.tennis-data.co.uk/) and builds a validated SQLite
database.

**Schema** — `players`, `matches`, `statistics`, `odds`.

**Coverage as built:**

- 11,794 matches · 517 unique players · 35,265 odds records (99.9% coverage)
- Surfaces: Hard (58.6%), Clay, Grass · Grand Slam → ATP 250
- Bookmakers: Pinnacle, Bet365, Max — 3–5% average overround
- Upset rate ~30% (lower-ranked player wins)

```bash
python data_pipeline.py                 # rebuild (~20s)
jupyter notebook data_exploration.ipynb # 14-section EDA
```

> The free tennis-data.co.uk files carry **no** serve/return splits. The
> `statistics` table exists for that data; it is filled from the Match Charting
> Project instead (below).

### Rally intelligence (Tier 1)

[fetch_charting_data.py](fetch_charting_data.py) ingests Jeff Sackmann's **Match
Charting Project** into `rally_stats`, giving each player a first-strike vs
grinding profile (rally-length win-rate curves). Validated out-of-sample for lift.

```bash
python fetch_charting_data.py
python validate_rally.py
```

---

## 🧮 Models

**[features.py](features.py) — feature engineering v3.0**
Surface correlations at academic values (H↔G 0.10, C↔G 0.08), tour-aware serve
baselines (ATP / WTA / ITF separately), strict leakage controls, recent form,
head-to-head, ranking momentum, tournament tier, implied-probability features.

**[hierarchical_model.py](hierarchical_model.py) — hierarchical Markov v3.0**
Barnett & Clarke (2005) point→game→set→match chain, tour-aware serve defaults,
score-conditioned win probability, clean API for ensembling.

**[neural_network.py](neural_network.py) — symmetric NN v3.0**
Bias terms restored with symmetry enforced by data augmentation rather than
architecture; 100/128-unit hidden layer, dropout regularisation.

**[execution/inplay.py](execution/inplay.py) — live score-aware engine**
The real engine. Takes the live scoreboard (sets/games + current server from
point-by-point) plus each player's career serve-points-won %, and runs
`win_prob_from_score` to produce a `true_p` that moves with the match.
Sanity-checked: level-serve start 0.58, up-a-set 0.80, down-a-set 0.30,
strong-server-up-a-set 0.99. Toggle with `TRADING_INPLAY`.

**[execution/momentum.py](execution/momentum.py) — live momentum (Tier 2)**
Derives momentum and serve-regression signals from Sofascore point-by-point and
nudges the Markov engine. Flag: `TRADING_MOMENTUM`. Validate with
`python validate_momentum.py`.

---

## 📡 Live data feed

Sofascore blocks programmatic requests two different ways, and they need two
different answers:

1. **TLS fingerprint** — a stock python/curl handshake is 403'd (`"Forbidden"`).
   `tls_client` impersonates Chrome and gets past it.
2. **IP reputation** — once an address has pulled enough API traffic, Sofascore
   challenges *that address* on `/api/v1/*` (`403 {"reason":"challenge"}`) while
   plain HTML pages still return 200. **No handshake trick fixes this.** The
   request has to leave from somewhere else, or come from another provider.

So the feed runs **from a local machine**, with an egress pool and a fallback:

```
                    ┌─ lane 0 ─ residential proxy ─┐
sofa_proxy.py ──────┼─ lane 1 ─ residential proxy ─┼──▶ Sofascore API
   (localhost:3001) └─ lane N ─ direct ────────────┘
      │  all lanes challenged?  ──▶ execution/flashscore.py ──▶ Flashscore
      ▼                                (translated to Sofascore's schema)
push_sofa.py  ──POST /api/sofa/_push──▶  Netlify blob cache  ──▶  terminal
```

- [sofa_proxy.py](sofa_proxy.py) — local proxy. Each **lane** is one Chrome TLS
  session pinned to one egress; a challenged lane is benched on an escalating
  cooldown (1→60 min) and traffic moves to the others, so one burned IP degrades
  throughput instead of killing the feed. 3s fresh / 30s stale-while-revalidate
  cache, schedule endpoints pre-warmed.

  ```bash
  python sofa_proxy.py --check     # probe every egress, print exit IP + status
  curl localhost:3001/_health      # which lanes are alive, which are cooling down
  ```

  Configure egress in `.env` — with nothing set it runs one direct lane:

  ```bash
  SOFA_EGRESS=http://user:pass@gate.provider.com:7000,socks5://user:pass@host:1080
  SOFA_EGRESS_DIRECT=1     # also keep an un-proxied lane
  SOFA_LANES=2             # sessions per egress
  SOFA_FALLBACK=1          # Flashscore fallback (default on)
  ```

  > Residential or mobile egress only. Datacenter IPs (AWS/GCP/Netlify) are
  > challenged on sight — that's why the deployed proxy never worked.
- [push_sofa.py](push_sofa.py) — uploads live scores, per-category schedules
  (ATP 3, WTA 6, Challenger 72, ITF-M 785, ITF-W 213), bulk odds, and per-event
  odds/point-by-point/statistics. Each path refreshes at the rate it actually
  changes, with exponential backoff (5→60 min) when challenged.

Requires in `.env`: `TT_SITE_URL`, `TT_PUSH_TOKEN` (the same token must be set in
the Netlify site env, or pushes 401).

```bash
python push_sofa.py --once    # one push, prints every path
python push_sofa.py           # 45s loop (leave running)
```

### Flashscore fallback — [execution/flashscore.py](execution/flashscore.py)

When every egress is challenged, an empty board is the worst possible answer.
Flashscore covers the same tours from separate infrastructure and answers
normally from addresses Sofascore has burned, so its feed is translated into
Sofascore's event schema and served in place. **The client URL contract is
unchanged** — the terminal, `push_sofa` and the execution pipeline can't tell
the difference beyond a `"source": "flashscore"` marker.

| | Sofascore | Flashscore |
|---|---|---|
| Live scores, per-set games, tiebreaks | ✅ | ✅ |
| Current game point score (0/15/30/40/A) | ✅ | ✅ |
| Serve/return splits, aces, break points | ✅ | ✅ |
| Scheduled draws (ATP/WTA/Ch/ITF M/ITF W) | ✅ | ✅ |
| **Point-by-point history** | ✅ | ❌ *derived* |
| **Odds** | ✅ | ❌ |
| **Current server** | ✅ | ❌ |

Flashscore publishes **no point-by-point feed** — probing `df_pbp_1_<id>` across
every live match returns empty, every time. Since the momentum engine needs the
point *sequence*, `PointStream` reconstructs it by polling the current-game score
and recording each transition:

```bash
python -m execution.flashscore --live              # live matches + point scores
python -m execution.flashscore --stream            # reconstructed point stream
python -m execution.flashscore --stats <FS_ID>     # serve/return splits
```

That yields points from the moment you start watching — not the match's earlier
history, which isn't recoverable from this source. Two points landing inside one
polling interval are missed, so poll every 5–10s. Odds and the current server
have no equivalent at all and are reported as absent rather than guessed.

> ESPN contributes nothing usable — it returns tournaments but zero individual
> matches, and 403s browsers.

---

## 🎯 Execution

Full detail in [EXECUTION_PIPELINE.md](EXECUTION_PIPELINE.md) and
[SX_STRATEGIES.md](SX_STRATEGIES.md).

### Venues

- **Polymarket** ([execution/polymarket.py](execution/polymarket.py)) — market
  discovery, live pricing, order placement. Match-winner and set-winner only.
- **SX Bet** ([execution/sxbet.py](execution/sxbet.py)) — read-only public API
  (`sportId=6`, `type=52`). Being an *exchange*, its two-sided book de-vigs to a
  near-margin-free live fair probability: the sharpest signal available. Order
  placement (EIP-712 signing) is deliberately not implemented.
- [execution/venue.py](execution/venue.py) — the router contract both sit behind.

### The agent

```bash
python -m execution.agent on|off|status   # paper mode by default
python -m execution.intel                 # live intelligence table
python -m execution.watch                 # live edges, auto-bet on threshold
python -m execution.webapp --port 8899    # local dashboard
python -m execution.healthcheck           # end-to-end stage verification
```

Auto-generates signals, sizes orders, and hedges on full-lock.

### Journal, settlement, calibration

```bash
python -m execution.signals_gen           # signals for open fixtures
python -m execution.settle                # resolve bets, book PnL
python -m execution.report                # performance report
python -m execution.agent calibration     # predicted vs actual
python backtest_bets.py                   # base-engine backtest on settled bets
```

Every intended or placed trade is one row in [execution/trade_log.py](execution/trade_log.py).

> ⚠️ **Calibration finding — read before trusting an edge.** Over settled bets the
> base engine is **over-confident** (~0.15 in the 60–90% buckets: predicts 75–90%,
> wins 50–62%), **Brier 0.35** (worse than a 0.25 coinflip), **ROI −3.7%**.
> Recalibrate ([execution/calibrate.py](execution/calibrate.py)) before sizing up.
> The gate `TRADING_INPLAY_MIN_DISAGREE` only emits a signal when it disagrees
> with SX's sharp line by a set margin.

### SX strategies

[execution/sx_strategies.py](execution/sx_strategies.py) runs three tiers —
GAME / SET / MATCH markets — off the live intelligence engine.
[execution/sx_breakbot.py](execution/sx_breakbot.py) backs the returner when break
probability is >70% and servers otherwise, then hedges (testnet/dry-run).
Walk-forward pricing backtest: [execution/sx_backtest.py](execution/sx_backtest.py).

### Smart money

[execution/smartmoney.py](execution/smartmoney.py) scores Polymarket wallets by
historical performance so you can see whether sharp money is on your side.

---

## 🖥️ Trading terminal

Next.js app in [trading-terminal/](trading-terminal/), deployed to `tennisalpha.in`.

```bash
cd trading-terminal
npm run start:all     # sofa proxy + dev server on :3000
npm test              # vitest
npm run build
```

**Routes** — `/terminal` (live board), `/calculator` (edge/bookmaker calculator),
`/tt` (table tennis board), `/manual`, `/resources`, `/admin` (analytics).

**Client engines** — [breakHoldEngine.ts](trading-terminal/src/lib/breakHoldEngine.ts),
[momentumEngine.ts](trading-terminal/src/lib/momentumEngine.ts),
[nnModel.ts](trading-terminal/src/lib/nnModel.ts),
[pmTrading.ts](trading-terminal/src/lib/pmTrading.ts),
[portfolio.ts](trading-terminal/src/lib/portfolio.ts),
[scheduleService.ts](trading-terminal/src/lib/scheduleService.ts).

**Netlify functions** — Sofascore blob-cache proxy, Google auth, entitlements and
plans, Stripe / Razorpay / PayPal checkout + webhooks, email capture, presence,
analytics tracking.

> `NETLIFY_API_TOKEN` must be set in the site env — blob context is never
> injected, and that token is what keeps all the functions alive.

---

## 🏓 Table tennis

[tabletennis/](tabletennis/) is a parallel predictor: Sofascore ingest through the
same proxy, walk-forward Elo + form GBDT, residual model, and its own dashboard.

```bash
python -m tabletennis.pipeline
python -m tabletennis.live       # live daemon
python -m tabletennis.push       # push to site
```

Live/push/refresh daemons run on the Mac alongside `sofa_proxy.py`.

---

## 👁️ Tennis vision

[tennis_vision/](tennis_vision/) — a runnable three-layer CV prototype
(Perception → Modeling → Application) that turns rally video into an annotated
overlay, an HTML dashboard, and a JSON report, every output confidence-tagged.

```bash
python -m tennis_vision.pipeline                   # synthetic rally + ground truth
python -m tennis_vision.pipeline --video clip.mp4  # real footage
```

Latest synthetic run (seed 7): ball position error 0.18 px, detection rate 1.00,
hit recall 0.833, court corner error 3.9 px.

---

## 📁 Repo layout

```
tennis10/
├── data_pipeline.py            # historical ATP pipeline → tennis_data.db
├── features.py                 # feature engineering v3
├── hierarchical_model.py       # Barnett & Clarke Markov chain
├── neural_network.py           # symmetric NN
├── sofa_proxy.py               # local TLS-impersonating Sofascore proxy
├── push_sofa.py                # push live feed to deployed site
├── fetch_charting_data.py      # Match Charting Project → rally_stats
├── backtest_bets.py            # base-engine backtest
├── validate_rally.py / validate_momentum.py
├── execution/                  # signals, venues, agent, journal, calibration
├── trading-terminal/           # Next.js terminal + Netlify functions
├── tabletennis/                # table tennis predictor
├── tennis_vision/              # CV prototype
├── backtesting/ evaluation/ validation/ tests/
├── ml_models/ notebooks/ dashboard/ api/
└── betfair/ trading_server/ deploy/
```

---

## ⚙️ Configuration

`.env` at the repo root:

```bash
# Trading
TRADING_DRY_RUN=true            # paper mode
TRADING_MAX_STAKE=...
TRADING_BANKROLL=...
TRADING_LIVE_ONLY=...
TRADING_MARKET_FALLBACK=...
TRADING_INPLAY=1                # score-aware engine (default on)
TRADING_MOMENTUM=1              # Tier 2 momentum
TRADING_INPLAY_MIN_DISAGREE=0   # 0 = gate off

# Live feed push
TT_SITE_URL=https://tennisalpha.in
TT_PUSH_TOKEN=...               # must match the Netlify site env

# Servers
TRADING_SERVER_PORT=... / API_SERVER_PORT=... / NEXT_PUBLIC_API_URL=...

# Betfair (legacy)
BETFAIR_USERNAME / BETFAIR_PASSWORD / BETFAIR_APP_KEY
BETFAIR_CERT_PATH / BETFAIR_KEY_PATH
```

---

## 🛠️ Troubleshooting

**Live board empty / stale.** Check the proxy first —
`curl http://127.0.0.1:3001/sport/tennis/events/live`. A
`403 {"reason":"challenge"}` means Sofascore is challenging your IP; `push_sofa.py`
will back off (5→60 min) and resume on its own. The terminal keeps serving the
last good blob cache meanwhile.

**Pushes return 401.** `TT_PUSH_TOKEN` differs between `.env` and the Netlify env.

**CERTIFICATE_VERIFY_FAILED on push.** A stock python.org macOS build ships no
root certs — `pip install certifi`.

**Terminal functions dead.** `NETLIFY_API_TOKEN` missing from the site env.

**Database empty.** `rm tennis_data.db && python data_pipeline.py`.

**Module not found.** `pip install pandas openpyxl matplotlib seaborn requests tls_client`

---

## 📚 Documentation

| Doc | Covers |
|---|---|
| [EXECUTION_PIPELINE.md](EXECUTION_PIPELINE.md) | Signal → order → journal, calibration, in-play engine |
| [SX_STRATEGIES.md](SX_STRATEGIES.md) | SX Bet three-tier GAME/SET/MATCH strategies |
| [TRUE_P_ARCHITECTURE.md](TRUE_P_ARCHITECTURE.md) | True-probability system design |
| [TRADING_EXECUTION_MANUAL.md](TRADING_EXECUTION_MANUAL.md) | Operating the agent |
| [ENTRY_EXIT_TIMING_GUIDE.md](ENTRY_EXIT_TIMING_GUIDE.md) | Trade timing |
| [LIVE_ODDS_EDGE_CALCULATOR_GUIDE.md](LIVE_ODDS_EDGE_CALCULATOR_GUIDE.md) | Edge calculator |
| [BOOKMAKER_CALCULATOR_GUIDE.md](BOOKMAKER_CALCULATOR_GUIDE.md) · [V2_CALCULATOR_GUIDE.md](V2_CALCULATOR_GUIDE.md) | Calculator surfaces |
| [MATCH_DATA_GUIDE.md](MATCH_DATA_GUIDE.md) | Data sources and shapes |
| [FEATURES_README.md](FEATURES_README.md) | Feature engineering detail |
| [DEPLOYMENT_GUIDE_GCP.md](DEPLOYMENT_GUIDE_GCP.md) · [deployment_guide.md](deployment_guide.md) | Deployment |
| [TESTING_GUIDE.md](TESTING_GUIDE.md) | Test strategy |
| [SYSTEM_STATUS.md](SYSTEM_STATUS.md) | Current system state |

---

## ⚠️ Disclaimer

Research and educational software. Betting carries real financial risk, the
calibration report above shows the engine is **not currently profitable**, and
nothing here is financial advice. Default to `TRADING_DRY_RUN=true`.
