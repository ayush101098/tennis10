# Leads → Google Sheet

Captured emails are mirrored into the sheet

<https://docs.google.com/spreadsheets/d/1CDJls5iS71bsWzb3rEMgCQrhRWjoQh64mzFDwJi9d-4/edit>

Netlify Blobs remains the system of record; the sheet is a mirror. If Google is
slow or the script is misdeployed the signup still succeeds — `/api/subscribe`
returns a `sheet` field saying what happened, and nothing is lost, because the
lead is already in Blobs and can be backfilled.

## Why a Web App and not the Sheets API

The Sheets API needs a service-account private key living in the Netlify env and
the sheet shared with that robot account. An Apps Script Web App bound to the
sheet runs **as you**, needs no key, and is a single URL. Fewer secrets, fewer
moving parts.

## Setup (once, ~3 minutes)

1. Open the sheet → **Extensions → Apps Script**.
2. Replace the contents of `Code.gs` with the script below.
3. Set `TOKEN` to any random string you invent (it stops strangers writing rows).
4. **Deploy → New deployment → type: Web app**
   - *Execute as*: **Me**
   - *Who has access*: **Anyone**  ← required; Netlify calls it unauthenticated.
     The `TOKEN` check is what actually protects it.
5. Copy the **/exec** URL it gives you.
6. In Netlify → Site configuration → Environment variables, add:
   - `SHEETS_WEBHOOK_URL` = that /exec URL
   - `SHEETS_WEBHOOK_TOKEN` = the same string you put in `TOKEN`
7. Redeploy (env vars only bind at build/deploy).

Verify with a real signup, or:

```bash
curl -s -X POST https://tennispredictions.netlify.app/api/subscribe \
  -H 'Content-Type: application/json' \
  -d '{"email":"you+test@example.com","source":"manual-test"}'
# -> {"ok":true,"sheet":"ok"}      "not configured" means the env var is missing
```

## The script

```javascript
const TOKEN = 'PUT-A-RANDOM-STRING-HERE';   // must match SHEETS_WEBHOOK_TOKEN
const HEADERS = ['capturedAt', 'email', 'source', 'lastEvent', 'lastEventAt', 'paid', 'amount', 'txHash'];

function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents || '{}');
    if (TOKEN && body.token !== TOKEN) {
      return out({ ok: false, reason: 'unauthorized' });
    }
    const email = String(body.email || '').toLowerCase().trim();
    if (!email) return out({ ok: false, reason: 'email required' });

    const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
    if (sheet.getLastRow() === 0) sheet.appendRow(HEADERS);

    // Upsert, not append-or-skip. Someone who signed up as a lead weeks ago
    // and is now about to pay must not be silently dropped — their existing
    // row gets the payment-intent stamp instead of a duplicate being created.
    const rows = sheet.getLastRow() - 1;
    const emails = rows > 0
      ? sheet.getRange(2, 2, rows, 1).getValues().map(function (r) { return String(r[0]).toLowerCase().trim(); })
      : [];
    const idx = emails.indexOf(email);

    if (idx === -1) {
      sheet.appendRow([
        body.capturedAt || new Date().toISOString(),
        email,
        body.source || '',
        body.event || '',
        body.eventAt || '',
        body.paid ? 'yes' : '',
        body.amount || '',
        body.txHash || '',
      ]);
      return out({ ok: true, created: true });
    }

    const row = idx + 2;                       // +1 header, +1 to 1-index
    if (body.event) {
      sheet.getRange(row, 4).setValue(body.event);
      sheet.getRange(row, 5).setValue(body.eventAt || new Date().toISOString());
    }
    if (body.paid) sheet.getRange(row, 6).setValue('yes');
    if (body.amount) sheet.getRange(row, 7).setValue(body.amount);
    if (body.txHash) sheet.getRange(row, 8).setValue(body.txHash);
    return out({ ok: true, updated: true });
  } catch (err) {
    return out({ ok: false, reason: String(err) });
  }
}

function out(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
```

## What lands in the sheet, and when

| Column | Filled by |
|---|---|
| `capturedAt` | first time the address is seen, ISO timestamp |
| `email` | the address (also the upsert key) |
| `source` | which form — `landing-hero`, `landing-cta`, `paypal-intent`… |
| `lastEvent` / `lastEventAt` | `paypal_intent` is stamped **before** the PayPal link is handed over |
| `paid` / `amount` / `txHash` | on a verified crypto payment |

The PayPal.me link in the pricing modal is withheld until the email has been
recorded here — a PayPal transfer arrives carrying only a display name, so an
address banked before the money moves is the only reliable way to match a
payment to an account.

## Backfilling the leads captured before this existed

Once the env vars are live:

```bash
node trading-terminal/scripts/backfill-leads-to-sheet.js
```

It reads the existing leads from `/api/subscribe` (needs `LEADS_ADMIN_TOKEN`)
and posts each one at the webhook. The script's duplicate check makes it safe to
run more than once.
