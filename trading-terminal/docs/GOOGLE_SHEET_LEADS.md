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
const HEADERS = ['capturedAt', 'email', 'source', 'paid', 'amount', 'txHash'];

function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents || '{}');
    if (TOKEN && body.token !== TOKEN) {
      return ContentService.createTextOutput(JSON.stringify({ ok: false, reason: 'unauthorized' }))
        .setMimeType(ContentService.MimeType.JSON);
    }
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];

    // Write the header row once, so a fresh sheet is self-describing.
    if (sheet.getLastRow() === 0) sheet.appendRow(HEADERS);

    // Idempotent: the same address never gets a second row. Netlify already
    // dedups, but a retry or a manual backfill must not double up either.
    const emails = sheet.getLastRow() > 1
      ? sheet.getRange(2, 2, sheet.getLastRow() - 1, 1).getValues().map(r => String(r[0]).toLowerCase())
      : [];
    const email = String(body.email || '').toLowerCase();
    if (!email) throw new Error('email required');
    if (emails.indexOf(email) !== -1) {
      return ContentService.createTextOutput(JSON.stringify({ ok: true, duplicate: true }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    sheet.appendRow([
      body.capturedAt || new Date().toISOString(),
      email,
      body.source || '',
      body.paid ? 'yes' : '',
      body.amount || '',
      body.txHash || '',
    ]);
    return ContentService.createTextOutput(JSON.stringify({ ok: true }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({ ok: false, reason: String(err) }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}
```

## Backfilling the leads captured before this existed

Once the env vars are live:

```bash
node trading-terminal/scripts/backfill-leads-to-sheet.js
```

It reads the existing leads from `/api/subscribe` (needs `LEADS_ADMIN_TOKEN`)
and posts each one at the webhook. The script's duplicate check makes it safe to
run more than once.
