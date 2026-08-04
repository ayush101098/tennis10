/**
 * Netlify serverless function — email capture + payment ledger.
 *
 * The static export strips the Next.js /api/subscribe route, so this function
 * serves it in production (see the redirect in netlify.toml). State persists in
 * Netlify Blobs (same pattern as presence.js), with a per-container memory
 * fallback if Blobs is unavailable.
 *
 * POST /api/subscribe
 *   { email, source? }                      -> capture a lead
 *   { email, txHash, amount?, from? }       -> record a verified payment AND
 *                                              link it to the email (this is how
 *                                              "who paid" becomes answerable —
 *                                              the on-chain tx alone has no email)
 *
 * GET /api/subscribe   (header  x-admin-token: <LEADS_ADMIN_TOKEN>)
 *   -> { leads:[...], payments:[...], counts:{...} }   admin readout
 *
 * Set LEADS_ADMIN_TOKEN in the Netlify env to enable the GET readout; without it
 * the GET is disabled (never expose the list publicly).
 */

const { store: sharedStore } = require("./_blobs");

const STORE = "leads";
const LEADS_KEY = "list";
const PAY_KEY = "payments";
const ANALYTICS_STORE = "analytics";
const ANALYTICS_KEY = "events";

/** Aggregate raw pageview events into the shape the admin dashboard renders. */
function aggregateTraffic(events) {
  const byPath = {}, byRef = {}, byDay = {}, vids = new Set();
  for (const e of events) {
    vids.add(e.vid || "?");
    byPath[e.path] = (byPath[e.path] || 0) + 1;
    const ref = e.ref ? new URL(e.ref, "http://x").hostname || "direct" : "direct";
    byRef[ref] = (byRef[ref] || 0) + 1;
    const day = new Date(e.ts).toISOString().slice(0, 10);
    byDay[day] = (byDay[day] || 0) + 1;
  }
  const top = (obj) => Object.entries(obj).map(([k, v]) => ({ k, v }))
    .sort((a, b) => b.v - a.v).slice(0, 10);
  const days = [];
  for (let i = 13; i >= 0; i--) {
    const d = new Date(Date.now() - i * 86400000).toISOString().slice(0, 10);
    days.push({ day: d, count: byDay[d] || 0 });
  }
  return {
    views: events.length,
    uniques: vids.size,
    byPath: top(byPath),
    byRef: top(byRef),
    byDay: days,
    recent: events.slice(-25).reverse(),
  };
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

// per-container fallback (best-effort; Blobs is the real store)
const mem = { leads: [], payments: [] };

async function getStoreSafe() {
  try {
        return sharedStore(STORE);
  } catch {
    return null;
  }
}

const reply = (statusCode, obj) => ({
  statusCode,
  headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  body: JSON.stringify(obj),
});

async function readList(store, key) {
  if (store) {
    try { return (await store.get(key, { type: "json" })) || []; } catch { /* fall */ }
  }
  return mem[key === LEADS_KEY ? "leads" : "payments"];
}

async function writeList(store, key, list) {
  if (store) {
    try { await store.setJSON(key, list); return; } catch { /* fall */ }
  }
  mem[key === LEADS_KEY ? "leads" : "payments"] = list;
}

exports.handler = async (event) => {
  const store = await getStoreSafe();

  // ── admin readout ──
  if (event.httpMethod === "GET") {
    const token = process.env.LEADS_ADMIN_TOKEN;
    const given = event.headers["x-admin-token"] || event.headers["X-Admin-Token"];
    if (!token || given !== token) {
      return reply(401, { error: "unauthorized" });
    }
    const leads = await readList(store, LEADS_KEY);
    const payments = await readList(store, PAY_KEY);
    let events = [];
    try {
            events = (await sharedStore(ANALYTICS_STORE).get(ANALYTICS_KEY, { type: "json" })) || [];
    } catch { /* analytics optional */ }
    return reply(200, {
      counts: { leads: leads.length, payments: payments.length, views: events.length },
      leads, payments, traffic: aggregateTraffic(events),
    });
  }

  if (event.httpMethod !== "POST") {
    return reply(405, { error: "method not allowed" });
  }

  let body = {};
  try { body = JSON.parse(event.body || "{}"); } catch { /* ignore */ }
  const email = String(body.email || "").trim().toLowerCase();
  if (!EMAIL_RE.test(email)) {
    return reply(400, { error: "Enter a valid email address." });
  }
  const now = Date.now();

  // ── payment record (email + txHash) ──
  if (body.txHash) {
    const txHash = String(body.txHash).trim();
    const payments = await readList(store, PAY_KEY);
    if (!payments.some((p) => p.txHash === txHash)) {
      payments.push({
        email, txHash,
        amount: body.amount != null ? String(body.amount) : null,
        from: body.from ? String(body.from) : null,
        ts: now,
      });
      await writeList(store, PAY_KEY, payments);
    }
  }

  // ── lead capture (dedup by email) ──
  const leads = await readList(store, LEADS_KEY);
  const existing = leads.find((l) => l.email === email);
  if (existing) {
    existing.lastSeen = now;
    if (body.txHash) existing.paid = true;
  } else {
    leads.push({
      email, ts: now, lastSeen: now,
      source: body.source ? String(body.source).slice(0, 40) : "cta",
      paid: !!body.txHash,
    });
  }
  await writeList(store, LEADS_KEY, leads);

  return reply(200, { ok: true });
};
