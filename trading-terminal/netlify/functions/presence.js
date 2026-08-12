/**
 * Netlify serverless function — live-presence counter.
 *
 * The static export strips the Next.js /api/presence route, so this function
 * serves it instead (see the redirect in netlify.toml).
 *
 * Each open terminal tab POSTs a heartbeat { id } every ~15s. We keep a map of
 * id -> last-seen timestamp and report how many were seen within the freshness
 * window. State is shared across function invocations via Netlify Blobs; if
 * Blobs isn't available we fall back to per-container memory (best-effort).
 */

const { store: sharedStore } = require("./_store");

const WINDOW_MS = 30_000; // an id is "online" if seen within the last 30s
const STORE_NAME = "presence";
const KEY = "seen";

// Per-container fallback state (used only if Blobs is unavailable)
const memSeen = new Map();

function prune(seen, now) {
  for (const id of Object.keys(seen)) {
    if (now - seen[id] > WINDOW_MS) delete seen[id];
  }
  return seen;
}

async function getStoreSafe() {
  try {
        return sharedStore(STORE_NAME);
  } catch {
    return null;
  }
}

const json = (count) => ({
  statusCode: 200,
  headers: { "Content-Type": "application/json", "Cache-Control": "public, s-maxage=20, stale-while-revalidate=60" },
  body: JSON.stringify({ count }),
});

exports.handler = async (event) => {
  const now = Date.now();

  let id = "";
  if (event.httpMethod === "POST" && event.body) {
    try {
      const b = JSON.parse(event.body);
      if (typeof b.id === "string") id = b.id;
    } catch {
      /* ignore malformed body */
    }
  }

  const store = await getStoreSafe();

  // ── Blobs path (shared across invocations) ──
  if (store) {
    try {
      let seen = (await store.get(KEY, { type: "json" })) || {};
      if (id) seen[id] = now;
      seen = prune(seen, now);
      await store.setJSON(KEY, seen);
      return json(Object.keys(seen).length);
    } catch {
      /* fall through to memory */
    }
  }

  // ── In-memory fallback ──
  if (id) memSeen.set(id, now);
  for (const [k, ts] of memSeen) if (now - ts > WINDOW_MS) memSeen.delete(k);
  return json(memSeen.size);
};
