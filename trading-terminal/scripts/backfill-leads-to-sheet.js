#!/usr/bin/env node
/**
 * Push already-captured leads into the Google Sheet.
 *
 * Leads collected before the sheet mirror existed live only in Netlify Blobs.
 * This reads them back through the admin readout and replays each one at the
 * Apps Script webhook. The script on the sheet side skips addresses it already
 * has, so running this twice is harmless.
 *
 *   SITE_URL=https://tennispredictions.netlify.app \
 *   LEADS_ADMIN_TOKEN=... \
 *   SHEETS_WEBHOOK_URL=... SHEETS_WEBHOOK_TOKEN=... \
 *   node trading-terminal/scripts/backfill-leads-to-sheet.js
 */

const SITE = (process.env.SITE_URL || "https://tennispredictions.netlify.app").replace(/\/$/, "");
const ADMIN = process.env.LEADS_ADMIN_TOKEN;
const HOOK = process.env.SHEETS_WEBHOOK_URL;
const TOKEN = process.env.SHEETS_WEBHOOK_TOKEN || "";

async function main() {
  if (!ADMIN) throw new Error("LEADS_ADMIN_TOKEN is required (it gates the lead readout)");
  if (!HOOK) throw new Error("SHEETS_WEBHOOK_URL is required");

  const res = await fetch(`${SITE}/api/subscribe`, { headers: { "x-admin-token": ADMIN } });
  if (!res.ok) throw new Error(`lead readout failed: HTTP ${res.status}`);
  const { leads = [] } = await res.json();
  console.log(`${leads.length} leads to replay`);

  let ok = 0, dup = 0, fail = 0;
  for (const l of leads) {
    const r = await fetch(HOOK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: l.email,
        source: l.source || "backfill",
        paid: !!l.paid,
        capturedAt: new Date(l.ts || Date.now()).toISOString(),
        token: TOKEN,
      }),
      redirect: "follow",
    }).catch((e) => ({ ok: false, statusText: String(e) }));

    let body = {};
    try { body = await r.json(); } catch { /* Apps Script can return HTML on error */ }
    if (r.ok && body.duplicate) dup++;
    else if (r.ok && body.ok) ok++;
    else { fail++; console.warn(`  failed: ${l.email} — ${body.reason || r.statusText || "?"}`); }
  }
  console.log(`appended ${ok} · already present ${dup} · failed ${fail}`);
}

main().catch((e) => { console.error(String(e.message || e)); process.exit(1); });
