# Operations — what the admin must supply and keep running

Audited against the code on 2026-09-02, not from memory. Every variable listed
here is one the code actually reads; every daemon is one that is actually
loaded.

---

## 0. Right now, the system is down

```
https://tennispredictions.netlify.app/api/account    503  usage_exceeded
https://tennispredictions.netlify.app/api/sofa/...   503  usage_exceeded
https://tennispredictions.netlify.app/api/presence   503  usage_exceeded
```

**Netlify is over its plan quota.** Everything server-side is dead: the live
board, sign-in, payments, entitlement checks and the trial grant. The Mac side
is healthy and still pushing data into a site that cannot serve it.

Nothing else in this document matters until that is resolved. Either upgrade
the plan or reduce function invocations — the `s-maxage` CDN caching on
`sofa-proxy` and the new `pm-proxy` is the lever that already exists for the
latter.

---

## 1. Secrets and configuration

### A. Netlify site environment — the production backend

Set at **Site configuration → Environment variables**. Without these the
matching function returns an error; nothing falls back to a default.

| Variable | Used by | What breaks without it |
|---|---|---|
| `NETLIFY_API_TOKEN` | `_blobs.js`, `_store.js` | **Everything.** Blob context is not injected automatically; this token is what keeps all storage-backed functions alive — accounts, leads, payments, the SofaScore cache. |
| `TT_PUSH_TOKEN` | `sofa-proxy.js` | The Mac cannot push tennis data. Board goes empty. Must be **identical** to the value in the local `.env`. Name is historical — it predates the table-tennis removal and is now the tennis push secret. |
| `LEADS_ADMIN_TOKEN` | `account.js`, `subscribe.js` | No admin access; manual grants (`action:"grant"`) return 401. This is the token needed to comp an account through the API. |
| `SOFA_PROXY_URL` | `sofa-proxy.js` | Optional upstream. Absent, the function serves only what the Mac has pushed — which is the normal operating mode. |
| `STRIPE_SECRET_KEY` | checkout, confirm, webhook | Card payments fail. |
| `STRIPE_WEBHOOK_SECRET` | `stripe-webhook.js` | Webhooks unverified and rejected. |
| `PAYPAL_CLIENT_ID` / `PAYPAL_CLIENT_SECRET` / `PAYPAL_ENV` | `paypal.js` | PayPal checkout fails. `PAYPAL_ENV` = `sandbox` or `live`. |
| `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` / `RAZORPAY_WEBHOOK_SECRET` | `razorpay.js` | India payments fail. |
| `RAZORPAY_AMOUNT_INR`, `USD_INR` | razorpay | Wrong INR pricing. |
| `GOOGLE_CLIENT_ID` + `NEXT_PUBLIC_GOOGLE_CLIENT_ID` | `google-auth.js` | Google sign-in fails. **Both** are needed — one verifies server-side, one is bundled for the button. |
| `SHEETS_WEBHOOK_URL` / `SHEETS_WEBHOOK_TOKEN` | `subscribe.js` | Leads still stored in Blobs; only the Sheet mirror stops. |
| `GOOGLE_SHEET_ID` / `GOOGLE_SHEET_TAB` / `GOOGLE_SERVICE_ACCOUNT_EMAIL` / `GOOGLE_PRIVATE_KEY` | `_sheets.js` | Same — mirror only. |
| `URL` / `SITE_URL` | stripe, paypal | Payment return URLs break. `URL` is injected by Netlify; `SITE_URL` is the override. |
| `BLOB_READ_WRITE_TOKEN` | `_store.js` | Vercel-only fallback; unused on Netlify. |

### B. Local `.env` on the Mac — the data pipeline

Currently set: `TT_PUSH_TOKEN`, `TT_SITE_URL`, `TRADING_*`, `BETFAIR_*`,
`NEXT_PUBLIC_API_URL`, ports.

| Variable | Required? | Purpose |
|---|---|---|
| `TT_PUSH_TOKEN` | **Yes** | Must match the Netlify value or every push 401s. |
| `TT_SITE_URL` | **Yes** | Where `push_sofa.py` pushes. |
| `TRADING_DRY_RUN` | **Yes — keep `true`** | `false` places real money orders. |
| `TRADING_BANKROLL`, `TRADING_MAX_STAKE`, `TRADING_KELLY_CAP` | Yes | Position sizing limits. |
| `POLYMARKET_PRIVATE_KEY` | Only for live trading | A wallet key. Not set today, which is correct while `TRADING_DRY_RUN=true`. |
| `SXBET_API`, `SXBET_CHAIN_ID` | Only for SX bot | `execution/sx_breakbot.py`. |
| `BETFAIR_*` (5 vars) | Only for Betfair | Set, but Betfair needs a £299 activation to be useful. |
| `LIVETENNIS_API_KEY` | **Not set** | The live engine's tennis feed. Without it `python -m execution.live` has no data. See §4. |
| `EDGE_BASE_URL`, `EDGE_PUSH_TOKEN` | **Not set** | Cloudflare edge fan-out. Not deployed yet. |

`TRADING_*` has ~25 further tuning flags (`execution/`), all with sane
defaults. They are knobs, not requirements.

---

## 2. Processes that must keep running (the Mac)

These are launchd agents in `~/Library/LaunchAgents/`. **All four are healthy
as of this audit.**

| Agent | Status | What it does | If it stops |
|---|---|---|---|
| `in.tennisalpha.sofa-proxy` | running (pid 15143) | Defeats SofaScore's TLS fingerprinting on port 3001 | Every other job loses its data source |
| `in.tennisalpha.push-sofa` | running (pid 15640) | Pushes schedule/scores/odds to the site's blob cache | **The live board goes empty.** Nothing in the cloud can fetch SofaScore |
| `in.tennisalpha.pointstore` | running (pid 37482) | Persists point-by-point data | Point corpus stops growing; board unaffected |
| `in.tennisalpha.archive` | loaded, last exit 0 | Commits the daily match archive to git | SSR homepage serves stale matches to crawlers |

**This machine is the single point of failure for the whole product.** Nothing
feeds tennis from the cloud — SofaScore blocks servers, ESPN
403s them. If the Mac is off, the board is empty regardless of Netlify.

Check them with:

```bash
launchctl list | grep tennisalpha
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3001/sport/tennis/events/live
```

---

## 3. Routine admin tasks

| Task | How | Frequency |
|---|---|---|
| Comp an account | `POST /api/account` `{email, action:"grant", days, reason, adminToken}` — needs `LEADS_ADMIN_TOKEN` | As needed |
| Comp without the API | Add to `TIME_GRANTS` in `src/lib/auth.tsx` (epoch ms expiry) and deploy | When the API is down |
| Turn trials on/off | `TRIALS_ENABLED` in **both** `netlify/functions/account.js` (authority) and `src/lib/auth.tsx` (copy) | Rare |
| Refresh rankings | `python generate_rankings.py` → `trading-terminal/public/rankings.json` | **Monthly** — currently dated 2026-07-02, two months stale, and it feeds the Elo prior on every match |
| Check the board | `/admin` page, or `/health` on the live engine | Daily |

---

## 4. What is NOT configured, and what that costs

| Missing | Consequence | To fix |
|---|---|---|
| `LIVETENNIS_API_KEY` | The live market engine has no feed. Everything else works; the engine serves an empty board and says so. | Ultra tier (~$100/mo) for WebSocket point data; lower tiers cannot beat their own poll interval |
| Cloudflare account | No edge fan-out. Every viewer holds a socket to one Python process. | `edge/README.md` |
| Bookmaker odds | SofaScore's odds endpoints 403 and its bulk feed returns a disjoint id space. Polymarket is the only working price. | A licensed odds feed |
| Calibration data | No calibration exists. The model runs ~13pp from the market. | Weeks of recording — see `execution/live/README.md` |

---

## 5. Single points of failure, ranked

1. **The Mac.** Off or asleep → no tennis data anywhere. No cloud fallback exists.
2. **Netlify quota.** Currently exceeded → entire backend 503. *Live now.*
3. **`NETLIFY_API_TOKEN`.** Expires or is rotated → all ten storage functions fail at once, silently.
4. **`TT_PUSH_TOKEN` drift.** Changed on one side only → pushes 401, board goes stale while everything looks healthy.
5. **`rankings.json` staleness.** Degrades quietly — no error, just worse priors.

---

## 6. Minimum viable configuration

To serve a working public board with sign-in and no payments:

```
Netlify:  NETLIFY_API_TOKEN, TT_PUSH_TOKEN, LEADS_ADMIN_TOKEN,
          GOOGLE_CLIENT_ID, NEXT_PUBLIC_GOOGLE_CLIENT_ID
Mac:      TT_PUSH_TOKEN (matching), TT_SITE_URL, TRADING_DRY_RUN=true
Running:  sofa-proxy + push-sofa
```

Payments add Stripe / PayPal / Razorpay. The live engine adds
`LIVETENNIS_API_KEY`. Everything else is optional.
