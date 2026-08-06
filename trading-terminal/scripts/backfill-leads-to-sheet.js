#!/usr/bin/env node
/**
 * Replay every already-captured address into the waitlist sheet.
 *
 * Signups collected before the sheet was connected live only in Netlify Blobs.
 * The work happens server-side — this just calls the admin resync, so it uses
 * whichever transport the site is configured with (service account or Apps
 * Script) and needs no Google credentials locally.
 *
 * Addresses already in the sheet are skipped, so running it twice is harmless.
 *
 *   LEADS_ADMIN_TOKEN=... node trading-terminal/scripts/backfill-leads-to-sheet.js
 *   SITE_URL=https://staging.example.com LEADS_ADMIN_TOKEN=... node ...   # elsewhere
 */

const SITE = (process.env.SITE_URL || "https://tennisalpha.in").replace(/\/$/, "");
const ADMIN = process.env.LEADS_ADMIN_TOKEN;

async function main() {
  if (!ADMIN) throw new Error("LEADS_ADMIN_TOKEN is required — it is what authorises the resync");

  console.log(`resyncing the waitlist into the sheet via ${SITE}`);
  const res = await fetch(`${SITE}/api/subscribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "resync", adminToken: ADMIN }),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.ok) {
    throw new Error(`resync failed: HTTP ${res.status} ${data.reason || ""}`.trim());
  }
  console.log(
    `  ${data.total} captured · ${data.added} appended · ${data.skipped} already present`
    + (data.failed ? ` · ${data.failed} failed` : ""),
  );
}

main().catch((e) => { console.error(String(e.message || e)); process.exit(1); });
