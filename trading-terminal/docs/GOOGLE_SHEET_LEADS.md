# Waitlist → Google Sheet

Captured emails are mirrored into the sheet

<https://docs.google.com/spreadsheets/d/1CDJls5iS71bsWzb3rEMgCQrhRWjoQh64mzFDwJi9d-4/edit>

Netlify Blobs remains the system of record; the sheet is a mirror. If Google is
slow or the script is misdeployed the signup still succeeds — `/api/subscribe`
returns a `sheet` field saying what happened, and nothing is lost, because the
lead is already in Blobs and can be backfilled.

## Two ways to connect it

| | Service account (wired, default) | Apps Script Web App |
|---|---|---|
| Setup | paste two env vars, share the sheet | deploy a script from the sheet's UI |
| Secret | a private key in the Netlify env | a token you invent |
| Runs as | a robot account you create | you |

Both are supported. **If the service-account vars are set they win**; otherwise
the webhook is used; if neither, `/api/subscribe` reports `"not configured"` and
the signup still succeeds (Blobs is the system of record either way).

## Setup A — service account (recommended)

1. <https://console.cloud.google.com> → create or pick a project.
2. **APIs & Services → Library → Google Sheets API → Enable.**
3. **APIs & Services → Credentials → Create credentials → Service account.**
   Name it anything; no roles are needed — access comes from sharing the sheet.
4. Open the new service account → **Keys → Add key → Create new key → JSON**.
   Download it. It contains `client_email` and `private_key`.
5. **Share the sheet with the robot**: open the sheet → Share → paste the
   `client_email` value (`…@….iam.gserviceaccount.com`) → give it **Editor** →
   untick "Notify people". Without this step every append returns 403; the key
   alone grants nothing.
6. Netlify → Site configuration → Environment variables:
   - `GOOGLE_SERVICE_ACCOUNT_EMAIL` = the `client_email`
   - `GOOGLE_PRIVATE_KEY` = the `private_key`, **including** the
     `-----BEGIN PRIVATE KEY-----` / `-----END PRIVATE KEY-----` lines.
     Paste it as-is; escaped `\n` newlines are handled either way.
   - `GOOGLE_SHEET_ID` (optional) = `1CDJls5iS71bsWzb3rEMgCQrhRWjoQh64mzFDwJi9d-4`
   - `GOOGLE_SHEET_TAB` (optional) = a tab name, if not the first one
7. Redeploy — env vars only bind at build.

The private key is a real credential: it can edit every sheet the robot has been
shared into. Keep it to that one sheet, and rotate it in the console if it leaks.

## Setup B — Apps Script Web App

Use this if you would rather not put a key in the environment.

1. Open the sheet → **Extensions → Apps Script**.
2. Replace `Code.gs` with the script below.
3. Set `TOKEN` to any random string you invent.
4. **Deploy → New deployment → Web app**, *Execute as* **Me**,
   *Who has access* **Anyone** (Netlify calls it unauthenticated; `TOKEN` is
   what protects it).
5. Set `SHEETS_WEBHOOK_URL` (the /exec URL) and `SHEETS_WEBHOOK_TOKEN` in
   Netlify, and redeploy.

## What lands in the sheet

Two columns, one row per person:

| Column | Meaning |
|---|---|
| `email` | the address, lowercased — also the dedup key |
| `joinedAt` | ISO timestamp of when they joined the waitlist |

Every capture point feeds it: the hero form, the form at the foot of the
landing page, and the email taken before the PayPal link is released. A
returning visitor re-submitting the same address does not create a second row.

## Backfilling addresses captured before the sheet was connected

Those signups are in Blobs, not the sheet. Replay them once the env vars are live:

```bash
LEADS_ADMIN_TOKEN=... node trading-terminal/scripts/backfill-leads-to-sheet.js
# -> 128 captured · 126 appended · 2 already present
```

The work happens server-side (`POST /api/subscribe {action:"resync"}`), so it
uses whichever transport is configured and needs no Google credentials on your
machine. Addresses already in the sheet are skipped — running it twice is
harmless.
