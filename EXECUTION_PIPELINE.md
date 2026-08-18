# Polymarket Execution Pipeline

Wires a model signal → live Polymarket market → sized order → trade journal.
Trades **match-winner** and **set-winner** markets only.

## Calibration report

```bash
python -m execution.agent calibration      # or: python -m execution.calibration --source inplay
```

Over settled bets, bins predicted `true_p` (backed side) vs actual win rate, with
Brier score, per-bucket ROI, and a per-source breakdown (inplay/sxbet/model/…).
This is the arbiter of whether a signal is *sharp* or just *confident*. Current
read (legacy bets, mostly pre-tagging): **Brier 0.35 (worse than 0.25 coinflip),
ROI −3.7%**, and the engine is **over-confident in the 60–90% buckets** (predicts
~75–90%, wins ~50–62%) — that's where the losses concentrate. Per-source rows
populate as newly-tagged inplay/sxbet bets settle.

**Betting gate (`TRADING_INPLAY_MIN_DISAGREE`, default 0 = off):** only emit an
in-play signal when it disagrees with SX's sharp line by ≥ this (edge the market
hasn't priced). Combine with the calibration report: only lean in on
buckets/sources the report shows calibrated + ROI-positive.

## In-play engine (live score-aware true_p)

`execution.inplay.InPlayModel` is the "real engine": it takes the **live
scoreboard** (Sofascore sets/games + current server via point-by-point) plus each
player's **serve strength** (career serve-points-won %, model DB; 0.62 baseline
for unknown ITF players) and runs the repo's Markov
`HierarchicalTennisModel.win_prob_from_score` to produce a **score-aware live
`true_p`** that updates as the match moves. Sanity-checked: level-serve start
0.58, up-a-set 0.80, down-a-set 0.30, strong-server-up-a-set 0.99.

It is the **top-priority signal** in both `/intel` (the `live%` column) and
`signals_gen` (source `inplay`), ahead of SX / Sofascore / model. `TRADING_INPLAY`
(default on) toggles it. ⚠ Defensible ≠ proven: the engine is principled and
independent, but whether it *beats* the market is empirical — track it via the
settle/report tools before trusting its edges with real money.

## SX Bet (live exchange odds, read-only)

`execution.sxbet.SXBetClient` reads SX Bet's public API (`https://api.sx.bet`,
no key) — tennis `sportId=6`, match-winner `type=52`. Because SX is an *exchange*,
the two-sided order book de-vigs to a **near-margin-free, live in-play fair
probability** (`quote()` / `fair_prob()`), the sharpest signal available. Its edge
vs the Polymarket price is a genuine cross-market gap. **Reads only** — order
placement would need EIP-712 wallet signing + a funded SX-Network wallet and is
deliberately not implemented. `execution.venue.VenueAdapter` documents the shared
router contract (Polymarket and SX Bet are adapters behind it).

Signal priority in `/intel`: **SX Bet (live exchange) > Sofascore (pre-match) >
model**. Note SX/Polymarket live-fixture overlap is partial, so SX prices only a
subset of rows; where both price a match they tend to agree (thin real edges).

### SX Bet execution (taker fill) — real money, guarded

`SXBetClient.place_bet()` places a **taker fill** via `POST /orders/fill/v2`,
signing the order with **EIP-712** (`eth-account`) per SX's `Details`/`FillObject`
schema (domain `SX Bet` v6.0). It is **paper by default**: dry-run builds and (if a
key is present) signs the order but never posts. A live post needs
`SXBET_PRIVATE_KEY` + a funded SX-Network wallet + `TRADING_DRY_RUN=false`.
**Rehearse on the SX testnet first** — set `SXBET_API=https://api.toronto.sx.bet`
(+ `SXBET_CHAIN_ID` from `GET /metadata`) to place real signed orders with no real
money. The `/intel` board has a **⚡ SX bet** button on SX-priced rows that fires
`POST /api/sxbet/place` (paper unless guards open) and journals it under
`venue=sxbet`. Odds/units: USDC 6dp `stakeWei`; `desiredOdds` = implied×10²⁰;
`oddsSlippage` % tolerance. Network config (chainId / fill-hasher / USDC) is
auto-detected from `GET /metadata`, so **testnet is one env var**:
`SXBET_API=https://api.toronto.sx.bet` (testnet chainId 79479957).

### Auto-fire (TRADING_SX_AUTOBET) — OFF by default

`execution.sx_autobet.autobet_pass` runs in the watch loop when
`TRADING_SX_AUTOBET=true`: for each signal it fires an SX taker fill when
`true_p − SX_taker_price ≥ TRADING_SX_MIN_EDGE` (default 0.08), Kelly-sized,
deduped, journaled `side=sx_auto`, paper unless SX live guards are open.
⚠ **-EV caveat:** SX is a sharp de-vigged live exchange; our signals are weaker,
so a large edge usually means our number is wrong, not that SX is mispriced.
This is plumbing for when the signal is genuinely sharper than SX — not a
money-printer. Ships off.

## Live-odds edge (Sofascore)

`execution.live_odds.SofascoreOdds` pulls each in-play match's "Full time" odds
from Sofascore (via `tls_client`), converts fractional→decimal, and **de-vigs**
them into a fair win probability — a real signal independent of Polymarket.
`signals_gen` uses it as the preferred source for live fixtures the model can't
price (unknown/ITF players), falling back to the market-favorite only if no odds
match. Toggle with `TRADING_LIVE_ODDS` (default on). Fixture matching requires
both surnames in the same Sofascore event, else it declines (no bad-match bets).

## Betting-intelligence dashboard (/intel)

`http://127.0.0.1:8899/intel` — a read-only decision board for **manual** bets
across **every in-play singles fixture (ITF included)**. Columns: Polymarket
price per side, model %, Sofascore fair %, and a suggested side + edge (signal −
price), best edges first. Each row also shows the **live score** (sets/games/point),
a **Kelly stake**, and **copy buttons** (player name + full bet string) for fast
manual placement. Computed by `execution.intel.compute_intel` on a ~30s background
refresh. It does not place bets — the agent does that; this is for your own calls.

⚠ **Sofascore's free odds are the PRE-MATCH line (`isLive=false`), not live**, so
`fair%` and its edge are only a hint — the real-time truth is the live score +
Polymarket price. When the scoreboard contradicts a suggestion (backed side is
trailing) it's tagged **⚠ stale** (`_score_contradicts`) and should be ignored. A
genuine live edge needs a live-updating odds feed (paid api-tennis.com
`get_live_odds`). Suggestions are gated to sane prices (0.03–0.97).

## Local dashboard

```bash
python -m execution.webapp            # -> http://127.0.0.1:8899  (localhost only)
```

A single-page dashboard (FastAPI, bound to `127.0.0.1` — not reachable off the
machine). Shows the agent on/off state, live journal (trades as the agent takes
them, flashing new rows) and the full log, plus summary tiles (exposure,
realized PnL, W–L, **unrealized PnL**). The **Open positions** table carries a
live **current price** and **uPnL** column, marked to the CLOB best bid by a
background refresher (every ~20s, off the request path so the page stays snappy).
From it you can **Sell** any
position (liquidates at the live CLOB bid via `close_trade`, books realized PnL)
or **Cancel** it (voids the bet via `cancel_trade`). Header buttons turn the
agent ON/OFF and run a settle pass. Polls every 3s.

## Agent (toggle on/off)

The `execution.agent` wraps everything into a single switch. When ON it
(1) generates fresh model signals for the currently-open fixtures, (2)
paper-trades match + set markets that clear the Kelly/edge gate, and
(3) **auto-hedges** any open position whose price drops against it.

```bash
python -m execution.agent on          # generate signals + start the watch+hedge loop (background)
python -m execution.agent on --fg     # run in the foreground instead
python -m execution.agent status      # ON/OFF + journal summary
python -m execution.agent off         # stop it
python -m execution.agent regen       # just refresh signals.auto.json
python -m execution.agent settle      # book PnL for finished bets (resolution via CLOB)
python -m execution.agent report      # performance report: exposure, edges, Kelly, PnL, calibration
```

- **Settle pass** (`execution.settle`): looks up each unsettled bet's market
  resolution by `condition_id` on the CLOB (`/markets/{condition_id}`, works even
  after the event closes) and marks it win/loss so realized PnL lands in the
  journal. The watch loop also auto-settles every ~10 cycles. Manual:
  `python -m execution.settle [--dry] [--user NAME]`.
- **Report** (`execution.report`): exposure by market, average edge / Kelly
  fraction / stake, hedging summary, and — once settled — realized PnL, ROI,
  win rate, and a model-vs-actual calibration check.

- **Live-only** (`TRADING_LIVE_ONLY=true`, default): the agent only *enters* on
  matches that are in-play, detected via the market's `gameStartTime`. Pre-match
  fixtures are held (`pre-match (waiting for live)`) until they start. Flatten all
  open bets with `python -m execution.agent cancel-open`.
- **Market-price fallback** (`TRADING_MARKET_FALLBACK`): most live tennis on
  Polymarket is ITF players the model can't price. With this on, those live
  fixtures are still bet by backing the market **favorite** (match market, price
  ≥ `TRADING_FALLBACK_MIN_FAV`) with a fixed `TRADING_FALLBACK_EDGE`. ⚠️ **No real
  predictive edge** — this is price/momentum trading and is -EV after the spread;
  it exists to keep the agent active across the live board, not because it wins.
- The watch loop **re-generates signals every `TRADING_REGEN_EVERY` cycles** so
  newly-started matches and refreshed favorites are picked up automatically.
- **Signals** come from `execution.signals_gen`, which prices each open fixture
  with the trained Sackmann logistic model via `execution.model_predict`
  (career averages from `tennis_data.db`). Fixtures whose players aren't in the
  DB are skipped. Match probability is converted to per-set probability with a
  Markov-consistent best-of-3/5 inversion.
- **Auto-hedge** (`_hedge_positions` in `execution.watch`): when our side's live
  price falls by `TRADING_HEDGE_DROP` (default 0.15), it buys enough of the
  opposite outcome to equalize payout across results — a **full lock** that
  fixes a capped loss. Dedupe: once the opposite token is held it won't re-hedge.
- Everything stays **paper** unless the three live guards are open (below); the
  agent never passes `--live`.

> ⚠️ Model calibration: the logistic model is uncalibrated and can saturate to
> extreme probabilities (e.g. 0.98–1.00). Kelly + `TRADING_MAX_STAKE` cap the
> per-bet exposure, but treat the size of any single edge with skepticism.

### Health check

```bash
python -m execution.healthcheck   # verifies config, discovery, pricing, model, signals, journal, hedge
```

### Endpoint / chain overrides

`POLYMARKET_GAMMA_URL`, `POLYMARKET_CLOB_URL`, `POLYMARKET_CHAIN_ID` (see
`.env.example`) let you point the client at an alternate host without code
changes. Defaults are Polymarket production (Polygon mainnet, chain 137).
**Polymarket has no public testnet with populated markets** — a testnet host
returns no tennis events/order books, so the agent would have nothing to trade.
The high-fidelity safe sandbox is the default **paper/dry-run mode**, which runs
on real production prices with simulated fills.

## Run

```bash
# dry-run (default, safe): maps signals to live markets, sizes bets, journals them
python -m execution.pipeline --signals signals.sample.json

# view the bet journal (also mirrored to trades_log.csv)
python -m execution.pipeline --log

# per-user journals: stamp trades / filter the log by user
python -m execution.pipeline --signals my_signals.json --user alice
python -m execution.pipeline --log --user alice
# (or set TRADING_USER=alice in .env; default user is "local")

# settle a finished bet so PnL shows in the journal
python -m execution.pipeline --settle 1 win     # or: loss

# real money (requires setup below)
python -m execution.pipeline --signals my_signals.json --live
```

## Signal format

One JSON entry per bet decision. `true_p` comes from your TRUE-P ensemble
(`true_p_ensemble.py`) — it is the model probability that `side` wins that
market.

```json
[
  {"player1": "Marta Kostyuk", "player2": "Jasmine Paolini",
   "market": "match", "side": "player1", "true_p": 0.78},
  {"player1": "Marta Kostyuk", "player2": "Jasmine Paolini",
   "market": "set2",  "side": "player1", "true_p": 0.60}
]
```

`market`: `match`, `set1`, `set2`, or `set3`.

## What the pipeline does per signal

1. **Map** — finds the fixture among open Polymarket tennis events (Gamma API,
   surname matching, accents handled) and picks the requested market type.
   O/U, total-sets, and "completed match" markets are ignored.
2. **Price** — pulls the live best ask from the CLOB order book (falls back to
   the Gamma snapshot if the book is empty).
3. **Size** — quarter-Kelly via `true_p_ensemble.kelly_stake`, minimum edge 2%,
   capped at 5% of `TRADING_BANKROLL` and at `TRADING_MAX_STAKE` (from `.env`).
4. **Dedupe** — skips if the journal already holds an open bet on that token.
5. **Execute** — limit BUY at the ask (simulated in dry-run).
6. **Log** — every bet goes to the `trade_log` table in `tennis_betting.db`
   and is mirrored to `trades_log.csv`.

## Going live

1. `pip install py-clob-client`
2. In `.env`: set `POLYMARKET_PRIVATE_KEY`, `POLYMARKET_PROXY_ADDRESS`
   (see `.env.example`), and `TRADING_DRY_RUN=false`.
3. Fund your Polymarket account with USDC and run with `--live`.

All three guards must open (`--live` flag, `TRADING_DRY_RUN=false`, key
present) or the run stays in dry-run.
