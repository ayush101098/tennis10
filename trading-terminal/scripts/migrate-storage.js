#!/usr/bin/env node
/**
 * Copy application state from Netlify Blobs to Vercel Blob.
 *
 * Read-only on the source: nothing is deleted or altered on Netlify, so this
 * can be run repeatedly and the old storage stays a working rollback until
 * someone deliberately removes it.
 *
 * Every document is verified after writing by reading it back and comparing
 * the serialised form. A migration that reports success without re-reading is
 * just a hope.
 *
 *   NETLIFY_SITE_ID=... NETLIFY_API_TOKEN=... BLOB_READ_WRITE_TOKEN=... \
 *     node trading-terminal/scripts/migrate-storage.js [--dry-run]
 */

const path = require("path");
const NETLIFY = require(path.join(__dirname, "..", "netlify", "functions", "_blobs.js"));

const DRY = process.argv.includes("--dry-run");

/**
 * Every (store, key) the application uses. Enumerating beats discovery: the
 * Netlify API has no cheap "list every store", and a missed key is a silent
 * data loss that only shows up as a customer with no account weeks later.
 */
const DOCUMENTS = [
  ["leads", "list"],          // captured emails
  ["leads", "payments"],      // payment records linked to emails
  ["accounts", "byEmail"],    // THE account database: access, grants, devices
  ["entitlements", "byEmail"],// legacy store, still backfilled from
  ["analytics", "events"],    // pageview ring
  ["presence", "seen"],       // live-visitor counter
  ["tt", "predictions"],      // table-tennis feed (unwired, kept for restore)
  ["tt", "live"],
  ["tt", "metrics"],
  ["tt", "meta"],
];

async function main() {
  if (!process.env.BLOB_READ_WRITE_TOKEN) throw new Error("BLOB_READ_WRITE_TOKEN is required (the destination)");
  if (!process.env.NETLIFY_API_TOKEN) throw new Error("NETLIFY_API_TOKEN is required (the source)");

  // Force the destination adapter to Vercel regardless of ambient env.
  const { store: vercelStore } = require(path.join(__dirname, "..", "netlify", "functions", "_store.js"));

  let copied = 0, empty = 0, failed = 0;
  const report = [];

  for (const [storeName, key] of DOCUMENTS) {
    let value = null;
    try {
      const src = NETLIFY.store(storeName);
      if (!src) throw new Error("source store unavailable");
      value = await src.get(key, { type: "json" });
    } catch (e) {
      report.push([storeName, key, `SOURCE FAILED: ${String(e.message).slice(0, 60)}`]);
      failed++;
      continue;
    }

    if (value === null || value === undefined) {
      report.push([storeName, key, "empty at source — nothing to copy"]);
      empty++;
      continue;
    }

    const size = JSON.stringify(value).length;
    const count = Array.isArray(value) ? `${value.length} items`
      : typeof value === "object" ? `${Object.keys(value).length} keys` : "scalar";

    if (DRY) {
      report.push([storeName, key, `would copy ${count}, ${size}B`]);
      copied++;
      continue;
    }

    try {
      const dst = vercelStore(storeName);
      await dst.setJSON(key, value);
      // Verify by reading back — a write that is not read back is unproven.
      const back = await dst.get(key, { type: "json" });
      const ok = JSON.stringify(back) === JSON.stringify(value);
      report.push([storeName, key, ok ? `copied ${count}, ${size}B — verified` : "MISMATCH after write"]);
      ok ? copied++ : failed++;
    } catch (e) {
      report.push([storeName, key, `WRITE FAILED: ${String(e.message).slice(0, 60)}`]);
      failed++;
    }
  }

  console.log(DRY ? "DRY RUN — nothing written\n" : "");
  for (const [s, k, msg] of report) console.log(`  ${(s + "/" + k).padEnd(26)} ${msg}`);
  console.log(`\n${copied} copied · ${empty} empty · ${failed} failed`);
  if (failed) process.exit(1);
}

main().catch(e => { console.error(String(e.message || e)); process.exit(1); });
