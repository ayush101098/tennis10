# Deploying on Vercel

The app already supports three hosts; `next.config.js` picks the output mode
from the environment:

```
Netlify   output: "export"      static site + netlify/functions/*
Vercel    output: undefined     a normal Next app — the API ROUTES are the backend
Docker    output: "standalone"
```

On Vercel there are no Netlify functions. `src/app/api/**` is the backend, and
every route runs the **same handler** as the corresponding Netlify function via
`src/lib/netlifyAdapter.ts` — one implementation, so the two hosts cannot
drift.

---

## Why move

Production was serving code from before 2026-09-02 — ten commits behind — while
the Netlify build itself was fine (verified: exit 0, pages emitted). The deploy
simply never ran. The legacy `tennispredictions.netlify.app` host is also
returning `503 usage_exceeded`.

Vercel builds on push, which removes the failure mode where the fix is on main
and the site is not.

---

## Steps

```bash
cd trading-terminal
npx vercel link          # once, to attach the project
npx vercel --prod        # or just push to main once the Git integration is on
```

**Root directory must be `trading-terminal`.** The repo root is the Python
project; pointing Vercel at it will build nothing.

### Environment variables

Set the same values documented in `/OPERATIONS.md` §1A. The ones that are not
optional:

| Variable | Why |
|---|---|
| `NETLIFY_API_TOKEN` | Still required — the Blobs store is the database, whichever host serves the app. Without it every storage-backed route fails. |
| `TT_PUSH_TOKEN` | Must match the Mac's `.env`, or `push_sofa.py` pushes 401 and the board empties. |
| `LEADS_ADMIN_TOKEN` | Admin grants. |
| `GOOGLE_CLIENT_ID` + `NEXT_PUBLIC_GOOGLE_CLIENT_ID` | Sign-in needs both. |
| `STRIPE_*`, `PAYPAL_*`, `RAZORPAY_*` | Payments. |

`URL` is injected by Netlify and is **not** set on Vercel — set `SITE_URL`
explicitly or Stripe and PayPal return URLs break.

### Point the pipeline at the new host

```bash
# .env on the Mac
TT_SITE_URL=https://<your-vercel-domain>
```

`push_sofa.py` reads this. Until it is changed the Mac keeps pushing to the old
host and the new one has no data.

---

## Two things that had to be fixed for this path

Both were latent: they only bite when the Next routes are the backend rather
than dev conveniences.

**`/api/pm/*` was a standalone implementation** sending `no-cache, no-store`.
On Netlify that was harmless — the redirect sends the path to the function and
the build strips the route. On Vercel it IS the backend, so every Polymarket
request would have been a serverless invocation with no CDN caching,
reintroducing the exact per-visitor upstream cost the proxy exists to remove.
It now shares `pm-proxy.js` and inherits its per-host cache lifetimes.

**`/api/razorpay/webhook` did not exist as a route.** On Netlify it is its own
redirect. On Vercel it would have returned 404 — Razorpay would accept the
payment and we would silently drop the confirmation, so the customer is charged
and never entitled. That is the worst available failure mode for a payment
webhook, and it would not have surfaced until real money moved.

---

## Verify after deploying

```bash
# 1. the new code is actually live (these strings do not exist in the old build)
curl -s https://<domain>/ | grep -c "Majors\|Parlay"

# 2. the feed is flowing and fresh
curl -sD- -o /dev/null https://<domain>/api/sofa/sport/tennis/events/live \
  | grep -iE "x-sofa-age-ms|cache-control"

# 3. pre-match True P is populated — this is what the stale deploy was hiding
#    (the ranking name fix landed in 5267026 and was never shipped)
```

Expect `x-sofa-age-ms` in the low thousands and `s-maxage=8` on the live path.
