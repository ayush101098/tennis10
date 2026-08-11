#!/usr/bin/env node
/**
 * Export every captured email to a CSV file.
 *
 * Reads the live leads store (Netlify Blobs) through the admin endpoint and
 * writes leads.csv next to wherever you run it. This is the whole list since
 * the site went up — footer captures, PayPal intents and, since 2026-08-11,
 * sign-ups too.
 *
 *   LEADS_ADMIN_TOKEN=... node trading-terminal/scripts/export-leads.js
 *   LEADS_ADMIN_TOKEN=... node trading-terminal/scripts/export-leads.js --out ~/Desktop/leads.csv
 *
 * The token is the one set as LEADS_ADMIN_TOKEN in the Netlify site env — it is
 * what gates the readout, so the list is never public.
 */

const fs = require("fs");

const SITE = (process.env.SITE_URL || "https://tennisalpha.in").replace(/\/$/, "");
const TOKEN = process.env.LEADS_ADMIN_TOKEN;
const outArg = process.argv.indexOf("--out");
const OUT = outArg > -1 ? process.argv[outArg + 1] : "leads.csv";

/** RFC4180-ish: quote everything and double any embedded quote. */
const cell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;

async function main() {
  if (!TOKEN) {
    throw new Error(
      "LEADS_ADMIN_TOKEN is required.\n" +
      "Find it in Netlify -> Site configuration -> Environment variables.",
    );
  }

  const res = await fetch(`${SITE}/api/subscribe`, { headers: { "x-admin-token": TOKEN } });
  if (res.status === 401) throw new Error("Unauthorized — that token does not match the one in the Netlify env.");
  if (!res.ok) throw new Error(`readout failed: HTTP ${res.status}`);

  const { leads = [], payments = [] } = await res.json();
  // Newest first: the useful end of a waitlist is the recent end.
  leads.sort((a, b) => (b.ts || 0) - (a.ts || 0));

  const paidBy = new Set(payments.map((p) => String(p.email || "").toLowerCase()));
  const iso = (ms) => (ms ? new Date(ms).toISOString() : "");

  const rows = [
    ["email", "capturedAt", "lastSeen", "source", "paid"].map(cell).join(","),
    ...leads.map((l) => [
      l.email,
      iso(l.ts),
      iso(l.lastSeen),
      l.source || "",
      l.paid || paidBy.has(String(l.email || "").toLowerCase()) ? "yes" : "",
    ].map(cell).join(",")),
  ];

  fs.writeFileSync(OUT, rows.join("\n") + "\n");

  const bySource = leads.reduce((m, l) => ((m[l.source || "unknown"] = (m[l.source || "unknown"] || 0) + 1), m), {});
  console.log(`${leads.length} emails -> ${OUT}`);
  console.log("by source:", Object.entries(bySource).map(([k, v]) => `${k}=${v}`).join(" "));
  if (leads.length) {
    console.log(`oldest: ${iso(leads[leads.length - 1].ts)}`);
    console.log(`newest: ${iso(leads[0].ts)}`);
  }
}

main().catch((e) => { console.error(String(e.message || e)); process.exit(1); });
