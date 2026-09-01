# Tennis Data Sources — Audit & Integration Priorities

Written 2026-08-31. Grounded in what the code actually consumes today, verified
against live endpoints from this machine.

> Supersedes the impact estimates in `TENNIS_DATA_RESEARCH.md` /
> `TENNIS_APIS_INTEGRATION_GUIDE.md`. Those documents assert accuracy gains
> ("+3-5% model accuracy") that were never measured against this model. Nothing
> in this file claims a lift that has not been backtested.

---

## 1. What the model consumes today

| Source | Endpoint / file | Status | Feeds |
|---|---|---|---|
| SofaScore schedule | `/api/sofa/category/{3,6,72,785,213}/scheduled-events/<date>` | **working** | fixtures, surface, round, live score |
| SofaScore point-by-point | `/api/sofa/event/<id>/point-by-point` | **working** | momentum, serve regression |
| SofaScore statistics | `/api/sofa/event/<id>/statistics` | **working** | aces, DF, 1st-serve %, BP conversion |
| SofaScore odds (per event) | `/api/sofa/event/<id>/odds/1/all` | **403 Forbidden** | — |
| SofaScore odds (daily bulk) | `/api/sofa/sport/tennis/odds/1/<date>` | **returns unusable data** | — |
| SofaScore generic schedule | `/api/sofa/sport/tennis/scheduled-events/<date>` | **403 Forbidden** | — |
| Polymarket | `gamma-api.polymarket.com` | **working** — 386 tennis fixtures, US Open covered | market price |
| Sackmann rankings | `public/rankings.json` (ATP/WTA top 500, dated 2026-07-02) | working, **2 months stale** | Elo prior |
| NN model | `public/nn_model.json` (41,750 matches, OOS Brier 0.2204) | working | pre-match True P |
| Match Charting Project | `tennis_data.db` → `rally_stats` | offline only | rally features |

### The odds finding (blocking)

**There are currently no usable bookmaker odds.** Two independent failures:

1. Per-event odds return `403 Forbidden` for every US Open match tested.
2. The daily bulk feed answers, but its event-id space has **zero overlap**
   with the schedule's:

```
2026-08-31: odds ids 16,939,109..16,966,189   (444 events)
            US Open event ids 1,074,555,726..2,145,021,510  (60 events)
            intersection = 0
```

It also returns the **same 444 ids for every date requested** — it is stale,
foreign data, not today's card. The `dailyOdds.get(evtId)` join in
`fetchSofaScheduled` therefore matches nothing, silently.

Consequence: before this change, every US Open match reached
`attachIntelligence` with no odds and was discarded. **Polymarket is the only
working price of record**, which is consistent with the pipeline targeting it
for execution anyway.

---

## 2. Data that would genuinely improve live probability

Ordered by (impact × feasibility), given what is already plumbed.

### Tier 1 — reachable now, no new vendor

1. **Polymarket order book depth** (`CLOB /book?token_id=`, already wired in
   `fetchQuote`). The gamma snapshot price is a midpoint; best bid/ask is the
   price you actually trade. Using the mid as "market probability" overstates
   edge by half the spread on every single row. This is a correctness fix to
   the edge calculation, not a new feature.
2. **Fresh rankings.** `rankings.json` is dated 2026-07-02 — two months and a
   full hard-court swing stale, feeding the Elo prior on every match. It is a
   file regeneration (`generate_rankings.py`), not an integration.
3. **Serve-hold priors per player** rather than tour averages. The Markov
   engine is tour-aware but player-agnostic on serve; SofaScore
   `/statistics` already returns the inputs, and `rally_stats` holds the
   historical side. This is the single largest live-probability lever
   available without a new data source.
4. **Retirement / medical-timeout signal.** SofaScore's live event carries
   status changes; a mid-match retirement is the largest possible probability
   move and the engine currently cannot see it coming.

### Tier 2 — new source, real signal

5. **Betfair Exchange in-play prices.** Tennis is the third-largest Betfair
   trading market and functions as the sharp reference; positive closing-line
   value against it is the standard evidence of a real edge. A `betfair/`
   directory already exists in this repo. Best use is not as a betting venue
   but as **a calibration target** — see §3.
6. **Match Charting Project refresh** (`JeffSackmann/tennis_MatchChartingProject`,
   shot-by-shot, CC BY-NC-SA — note the **NonCommercial** clause, which matters
   for a paid product). Rally-length and first-strike features are already
   validated OOS in this project.
7. **Court speed index.** Tennis Abstract publishes ace-adjusted surface-speed
   ratings per tournament per year; Ultimate Tennis Statistics computes a Court
   Speed Index and is Apache-2.0 open source. Both are free and directly
   condition serve-hold probability — the Markov engine's core input.

### Tier 3 — commercial, only if the above is exhausted

8. **Sportradar Tennis API** — official ATP partnership, point-by-point across
   4,000+ competitions, includes win probabilities. Paid.
9. **api-tennis.com / tennis-api.com** — cheaper point-by-point and odds feeds
   with WebSocket push. Would replace the broken SofaScore odds path if a paid
   feed is acceptable.

### Deliberately not recommended

- **ESPN** — already proven to 403 browser-shaped requests and contributes no
  individual matches; the fallback path in `scheduleService` is dead weight.
- **Weather feeds** — the Tennis Abstract surface-speed metric already absorbs
  temperature, humidity and wind indirectly. A separate weather integration
  duplicates it at higher cost.

---

## 3. The blocker that outranks every source above

Adding data will not help while the model disagrees with the market this much.
Measured on tonight's US Open card (76 matches priced against Polymarket):

```
model-vs-market gap    mean 12.9pp   median 9.4pp
gap > 10pp             37 / 76
gap > 20pp             13 / 76   (quarantined as suspect)
favourites (trueP≥80%) model sits 14.4pp ABOVE the market  (n=19)
```

A calibrated model disagrees with an efficient market by a couple of points and
in both directions. A **26-point average absolute disagreement**, with a
one-directional +14.4pp bias on favourites, is model error, not edge. It
reproduces the finding already recorded in memory (base True P overconfident by
~0.15, Brier worse than baseline, −5.7% ROI).

Concretely: the model prices Alcaraz over Safiullin at 97% where the market says
81%, and Rublev at 99%. Those are not opportunities.

**Recommendation: recalibrate against Betfair/Polymarket closing prices before
this board is presented to anyone as betting advice.** Isotonic or Platt
recalibration on closing-line targets is the direct fix, and the closing line is
free from both venues. Until then the US Open board is best read as a
disagreement monitor — where the model and the market differ, and why.

---

## Sources

- [Tennis Abstract — 2026 ATP Surface Speed Ratings](https://tennisabstract.com/reports/atp_surface_speed.html)
- [Ultimate Tennis Statistics — Glossary (Court Speed Index)](https://www.ultimatetennisstatistics.com/glossary)
- [JeffSackmann/tennis_MatchChartingProject](https://github.com/JeffSackmann/tennis_MatchChartingProject)
- [Sportradar Tennis API](https://marketplace.sportradar.com/products/6501e20f236aba44b550bdae)
- [Tennis Point-by-Point API (tennis-api.com)](https://tennis-api.com/tennis-point-by-point-api/)
- [Live Tennis API](https://livetennisapi.com/)
- [Betfair Exchange — in-play tennis](https://www.betfair.com/exchange/inplay/tennis)
- [Betfair tennis trading — market mechanics](https://caanberry.com/tennis-trading-strategies/)
