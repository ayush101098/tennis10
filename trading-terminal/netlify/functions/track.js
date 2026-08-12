/**
 * Netlify serverless function — first-party pageview tracking.
 *
 * Every page load POSTs a lightweight, anonymous beacon here; we append it to a
 * capped ring in Netlify Blobs. The admin analytics page reads the aggregate via
 * /api/subscribe (authed). No cookies, no PII — just an anonymous per-browser
 * visitor id so we can count unique visitors.
 *
 * POST /api/track  { path, ref, vid }
 */

const { store: sharedStore } = require("./_store");

const STORE = "analytics";
const KEY = "events";
const CAP = 5000; // keep only the most recent N events

const mem = [];

async function getStoreSafe() {
  try {
        return sharedStore(STORE);
  } catch {
    return null;
  }
}

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, body: "method not allowed" };
  }
  let b = {};
  try { b = JSON.parse(event.body || "{}"); } catch { /* ignore */ }

  const rec = {
    ts: Date.now(),
    path: String(b.path || "/").slice(0, 120),
    ref: String(b.ref || "").slice(0, 200),
    vid: String(b.vid || "").slice(0, 40),
  };

  const store = await getStoreSafe();
  if (store) {
    try {
      let events = (await store.get(KEY, { type: "json" })) || [];
      events.push(rec);
      if (events.length > CAP) events = events.slice(-CAP);
      await store.setJSON(KEY, events);
    } catch { mem.push(rec); }
  } else {
    mem.push(rec);
    if (mem.length > CAP) mem.splice(0, mem.length - CAP);
  }

  return {
    statusCode: 200,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    body: JSON.stringify({ ok: true }),
  };
};
