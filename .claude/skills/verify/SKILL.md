---
name: verify
description: Build, run, and drive the tennis10 trading terminal + Python execution pipeline to verify changes end-to-end.
---

# Verifying tennis10 changes

## Web app (trading-terminal/, Next.js 14, static-export on Netlify)

Build + typecheck:
```bash
cd trading-terminal && npx tsc --noEmit && npx next build
```
Only two routes exist: `/` (landing) and `/terminal`. Local dev keeps the
`/api/sofa` proxy route (stripped on Netlify, replaced by a function).

Run + drive headlessly (no playwright dep in the repo; use playwright-core
against the cached Chromium at
`~/Library/Caches/ms-playwright/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium`):
```bash
cd trading-terminal && npx next dev --port 3111   # background
```
- Sign in by writing localStorage `tt_session_v1` =
  `{"email":"x@y.com","tier":"pro","isAdmin":false,"since":<now>}` then load
  `/terminal`. Auth/tiers are 100% client-side (src/lib/auth.tsx).
- Schedule takes 30–120 s to load live ESPN/SofaScore data; wait for
  "Loading ESPN data…" to disappear.
- Value Board tab → rows have ⚡ TRADE CTAs → TradeTicket modal (Polymarket
  buy / paper trade). Paper trade needs no wallet. Bet journal is per-email
  localStorage `tt_bets_v2_<email>`.
- Wait ~2.5 s after any reload before clicking — hydration; clicks before
  that silently no-op.
- Beware ambiguous text selectors: "DONE" also matches the ✓ DONE filter
  pill; scope modal clicks to `div.fixed`.
- Polymarket data comes live from gamma-api/clob.polymarket.com (open CORS);
  a fixture may legitimately have no PM market → ticket falls back to
  book-odds paper mode.

## Python execution pipeline (execution/)

```bash
.venv/bin/python -m pytest tests/ -q          # note: test_features.py has
                                              # pre-existing failures (10) —
                                              # unrelated to execution/
.venv/bin/python -m execution.pipeline --signals signals.sample.json  # dry-run
.venv/bin/python -m execution.pipeline --log --user <name>
```
Dry-run is the default and safe: no keys → simulated orders, journaled to
tennis_betting.db (trade_log) + trades_log.csv.
