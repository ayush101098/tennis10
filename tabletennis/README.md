# TT Predictor — table tennis match probabilities

A separate, self-contained project (independent of the tennis trading terminal):
scraping-free data pipeline → SQLite → walk-forward features → sklearn models →
static `predictions.json` → standalone dashboard. The plan's architecture on
zero-cost infra.

## Run

```bash
python -m tabletennis.pipeline --serve          # ingest → train → predict → :3222
# or piecewise:
python -m tabletennis.ingest --days 45          # backfill history
python -m tabletennis.train                     # time-split validation + model.pkl
python -m tabletennis.predict                   # → site/predictions.json
python -m http.server 3222 -d tabletennis/site  # dashboard
```

Dashboard: **http://localhost:3222** · Live in-play: **http://localhost:3222/live.html**

## In-play intelligence layer

```bash
python -m tabletennis.analytic     # sanity-check the exact win-prob recursion
python -m tabletennis.traits       # character traits (actual − expected, causal)
python -m tabletennis.residual     # train bounded residual; ships only if it wins
python -m tabletennis.live         # 8s poller → live_predictions.json + state log
```

`live win prob = analytic baseline (exact race-to-11/win-by-2 recursion,
anchored to the pre-match model by inversion) + tanh-bounded character residual
(±15pp cap)`. The residual shipped on merit: held-out log-loss 0.5522 → 0.5501
over 159k historical game states. Traits computable from game-level history:
clutch, deuce composure, comeback, front-running, fatigue. Momentum & serve-
pressure traits need point sequences — `live.py` logs every state transition to
`live_states` precisely to accrue that training set. Sofascore exposes no
server identity, so game probabilities average both first-server assumptions
(TT serve edge ≈0.03 → sub-point effect).

## Key deviations from the original plan (deliberate)

- **No Flashscore/Playwright scraping.** The repo's `sofa_proxy.py` (port 3001)
  already defeats Sofascore's TLS fingerprinting for the tennis terminal, and
  Sofascore carries every league the plan names (Czech Liga Pro, Setka Cup,
  TT Cup, Liga Pro RUS, WTT). One request per category-day returns *all* of that
  day's matches with per-game scores — a 45-day backfill is ~225 requests total.
  **The proxy must be running** for ingest/predict.
- **SQLite over Postgres** — same schema (players/matches/games), zero infra.
- **sklearn `HistGradientBoosting` instead of LightGBM** (broken libomp on this
  machine; same algorithm family) and LogisticRegression as the sanity baseline.
  The plan's PyTorch player-embedding model is the natural next step once the
  baseline plateaus — slot it into `train.py` beside the other two.

## Honesty notes (read before trusting the numbers)

- Validation is **time-split walk-forward** (never random), and features are
  computed strictly from prior matches — leakage-safe by construction.
- With a short backfill the Elo pool is cold-started; accuracy grows with
  history depth. Check `site/metrics.json` (also shown in the dashboard's
  transparency panel) for current held-out accuracy / log-loss / calibration —
  the dashboard shows what the model *is*, not what we wish it were.
- These leagues run deliberately competitive matchups (that's their product),
  so headline accuracy will sit well below tennis-style favourites leagues.
  Log-loss vs the 0.6931 coin-flip line is the more meaningful number.
- No odds ingestion yet → no market-edge claims. The plan's OddsPortal/Betfair
  step is the right way to test "does this beat the market" — until then this
  is a probability model, not a betting edge.
