# Polymarket Execution Pipeline

Wires a model signal → live Polymarket market → sized order → trade journal.
Trades **match-winner** and **set-winner** markets only.

## Run

```bash
# dry-run (default, safe): maps signals to live markets, sizes bets, journals them
python -m execution.pipeline --signals signals.sample.json

# view the bet journal (also mirrored to trades_log.csv)
python -m execution.pipeline --log

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
