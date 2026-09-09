# Running the live model for yourself

One machine, live data, no cloud, no quota. This is not the deployed product —
it is the same engine reading the proxy that is already running on your Mac.

## Start it

```bash
cd ~/tennis10
./run-local.sh            # web UI    -> http://localhost:3000
./run-local.sh board      # terminal board (EdgeScore, market reaction)
```

The script checks `sofa_proxy` first and tells you how to start it if it is
down — an unexplained empty board is the least diagnosable failure, so it is
not allowed to happen quietly. Ctrl-C stops either.

**The web UI needs `SOFA_PROXY_URL` set**, which the script does for you.
Without it the page loads but reads the cloud blob cache and looks stale.
Confirm with:

```bash
curl -sD- -o /dev/null http://localhost:3000/api/sofa/sport/tennis/events/live | grep x-sofa-source
# x-sofa-source: upstream   <- reading your local proxy, no cloud hop
```

```
14:07:59   matches 6   priced 5   events 11

MATCH                                  SCORE            SRV    MODEL  MARKET    EDGE  SCORE
--------------------------------------------------------------------------------------------
Lina Gjorcheska vs Aurora Zantedeschi  0-0  0-0 15/0    p1    57.7%   45.5%  +12.2%   1.09

  5 live match(es) without a tradeable edge or price
```

`SCORE` on the right is **EdgeScore** — the edge divided by the model's own
uncertainty. It is the column to read, not the edge: a big edge on a match the
model is unsure about is worth less than a small one it is confident in.

## Why this is fast

The deployed path is `push_sofa → Netlify blob → CDN → browser poll`, and every
hop adds staleness (~3s + up to 8s + up to 10s). That chain exists to serve many
people cheaply. Running for yourself, none of it applies:

```
sofa_proxy (already running, localhost)  →  engine  →  your terminal
```

The lag is the poll interval and nothing else. Default 3s, which is roughly
bookmaker cadence — going lower buys nothing because SofaScore's own feed does
not update faster.

## You will not hit a limit

There is no billing quota on this path. The only real constraint is SofaScore
challenging your IP, which happened once before at roughly **262 requests per
minute** across 131 paths per cycle.

This polls **one** bulk endpoint:

| | requests/min to SofaScore |
|---|---|
| The ban that happened | ~262 |
| `push_sofa` (already running) | ~4 |
| This board at 3s | ~20 |
| **Total** | **~24** |

An order of magnitude under the line. Polymarket prices are on a separate,
slower beat (10s) because market prices move slower than the scoreboard and
each one costs a CLOB request per token.

## Tuning

```bash
BOARD_POLL_S=2    python -m execution.live board   # faster scoreboard
BOARD_PRICE_S=5   python -m execution.live board   # faster prices (more CLOB calls)
BOARD_MIN_EDGE=0  python -m execution.live board   # show everything priced
```

Wider coverage — by default the board tracks ATP, WTA, Challenger and WTA-125:

```python
SofaProxyProvider(categories=frozenset({"atp", "wta", "challenger",
                                        "wta-125", "itf-men", "itf-women"}))
```

ITF is excluded by default because those draws are mostly players outside the
top-500 rankings file, so the model has no prior, the gate correctly stays
silent, and the rows are padding.

## If the board is empty

```bash
python -m execution.live doctor            # what is configured
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3001/sport/tennis/events/live
```

`board` fails loudly if the proxy is not answering and tells you to run
`python sofa_proxy.py` — an unexplained empty board is the least diagnosable
failure, so it is not allowed to happen quietly.

An empty board with the proxy up usually just means no tour-level matches are
in play. `matches N` in the header tells you which it is.

## The server and the point tape are RECONSTRUCTED

SofaScore is currently challenging this IP — every path 403s and `sofa_proxy`
falls back to Flashscore for everything. Flashscore carries the score but not
the server, and has no point-by-point feed at all. Both are rebuilt from
endpoints that do answer (`execution/live/pointtape.py`):

**Server**, from `statistics`. "Service Points Won" reports `(won/served)`; the
denominator is points that player has served. Between two polls, whoever's
denominator grew is on serve. One anchor is enough — serve alternates every
completed game — so this costs a handful of extra requests, not one per match
per poll, and re-anchors every six games to correct drift.

Measured live: 7 of 8 matches anchored within a minute, the eighth being one
where no point was played in the window (correctly left unknown rather than
guessed).

**Point tape**, from the point score. Each transition is a point, and whoever
advanced won it. Advantage lost is scored as the opponent winning, a game
boundary is not a point, and tiebreaks are counted numerically because they
score 1,2,3 rather than 15/30/40.

Two limits, both reported rather than hidden:

- The tape starts when you start watching. The match's earlier history is not
  recoverable from these sources.
- Two points inside one poll interval collapse into one observation. Those are
  counted as `gaps` rather than invented — a sampled tape admits what it missed.

## What does not work locally, and why

Two SofaScore endpoints are **refused outright** — verified direct against the
local proxy, so it is their block and not your setup:

| Endpoint | Status |
|---|---|
| `event/<id>/statistics` | works |
| `event/<id>/point-by-point` | **403** |
| `event/<id>/odds/1/all` | **403** |

Consequences worth knowing:

- **Live momentum is unavailable in the web UI.** It is computed from
  point-by-point, which is blocked. The terminal board is unaffected: it builds
  its own point tape by polling.
- **Per-match SofaScore odds are unavailable**, which is why prices come from
  Polymarket.

The proxy now remembers a refusal for five minutes instead of re-asking every
cycle, so these show up as a handful of log lines rather than hundreds.

## What the numbers do and do not mean

Read this once.

- **The model is not calibrated.** It sits roughly 13 points from the market on
  average and about 14 points high on favourites. `edgescore` already accounts
  for that — which is why the Gjorcheska row above shows a +12.2% edge but an
  EdgeScore of only 1.09, i.e. *inside the noise*. **Trust the EdgeScore
  column, not the edge column.**
- **The `SRV` column shows nothing when the proxy is on its Flashscore
  fallback.** Flashscore renders the server as an icon, not a feed field, so it
  is genuinely unavailable — and unknown is shown as `?` rather than guessed,
  because a wrong server inverts the game market. Check the source with:
  `curl -s http://127.0.0.1:3001/sport/tennis/events/live | grep -o '"source":"[a-z]*"' | head -1`
- **The match model ignores the server.** `win_prob_from_score` declares
  `p1_serving` and never reads it (see `execution/live/README.md`). The `SRV`
  column is real and the set/game ladder uses it correctly, but the match
  probability does not.
- **Missed points are possible.** This polls; two points inside one interval
  collapse into a score jump. The engine refuses to attribute a game winner to
  a jump, so nothing is fabricated — but the point tape has a hole.
- **Nothing here places a bet.** `TRADING_DRY_RUN=true` governs the execution
  pipeline, which is a different program.
