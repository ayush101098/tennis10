# Waitlist → Google Sheet

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
curl -s -X POST https://tennisalpha.in/api/subscribe \
  -H 'Content-Type: application/json' \
  -d '{"email":"you+test@example.com","source":"manual-test"}'
# -> {"ok":true,"sheet":"ok"}      "not configured" means the env var is missing
```

## The script

```javascript
const TOKEN = 'PUT-A-RANDOM-STRING-HERE';   // must match SHEETS_WEBHOOK_TOKEN

function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents || '{}');
    if (TOKEN && body.token !== TOKEN) return out({ ok: false, reason: 'unauthorized' });

    const email = String(body.email || '').toLowerCase().trim();
    if (!email) return out({ ok: false, reason: 'email required' });

    const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
    if (sheet.getLastRow() === 0) sheet.appendRow(['email', 'joinedAt']);

    // One row per person. Netlify already dedups, but a retry or a re-run of
    // the backfill must not be able to double up either.
    const rows = sheet.getLastRow() - 1;
    const existing = rows > 0
      ? sheet.getRange(2, 1, rows, 1).getValues().map(function (r) { return String(r[0]).toLowerCase().trim(); })
      : [];
    if (existing.indexOf(email) !== -1) return out({ ok: true, duplicate: true });

    sheet.appendRow([email, body.joinedAt || new Date().toISOString()]);
    return out({ ok: true });
  } catch (err) {
    return out({ ok: false, reason: String(err) });
  }
}

function out(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
```

## What lands in the sheet

Two columns, one row per person:

| Column | Meaning |
|---|---|
| `email` | the address, lowercased — also the dedup key |
| `joinedAt` | ISO timestamp of when they joined the waitlist |

Every capture point feeds it: the hero form, the form at the foot of the
landing page, and the email taken before the PayPal link is released. A
returning visitor re-submitting the same address does not create a second row.

## Backfilling the leads captured before this existed

Once the env vars are live:

```bash
node trading-terminal/scripts/backfill-leads-to-sheet.js
```

It reads the existing leads from `/api/subscribe` (needs `LEADS_ADMIN_TOKEN`)
and posts each one at the webhook. The script's duplicate check makes it safe to
run more than once.
