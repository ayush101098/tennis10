/**
 * Table-tennis intelligence feed for production (twin of src/app/api/tt/route.ts).
 *
 * WHY THIS IS A PUSH STORE, NOT A PROXY
 *   The TT model runs locally: ingest/predict/live all talk to sofa_proxy.py,
 *   which exists only on the operator's machine. Netlify can neither run the
 *   Python pipeline nor read its JSON files, and the static export deletes
 *   src/app/api at build time — so /api/tt simply did not exist in production
 *   and the TT tab reported "feed unreachable".
 *
 *   So the local pipeline PUSHES its artifacts here (see tabletennis/push.py)
 *   and this function serves whatever was last pushed, out of Netlify Blobs.
 *
 *   GET                          -> { predictions, live, metrics, pushedAt }
 *   POST { kind, payload }       -> store one artifact (needs x-tt-token)
 *        kind ∈ predictions | live | metrics
 *
 * Auth: TT_PUSH_TOKEN must be set in the Netlify env and sent by the pusher.
 * Reads stay public — the terminal itself gates access client-side.
 */

const { store: sharedStore } = require("./_blobs");

const STORE = "tt";
const KINDS = new Set(["predictions", "live", "metrics"]);

const reply = (statusCode, obj) => ({
  statusCode,
  headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  body: JSON.stringify(obj),
});

async function store() {
  try {
        return sharedStore(STORE);
  } catch {
    return null;
  }
}

exports.handler = async (event) => {
  const s = await store();

  if (event.httpMethod === "GET") {
    if (!s) {
      return reply(200, {
        predictions: null, live: null, metrics: null, pushedAt: null,
        error: "blob store unavailable",
      });
    }
    const read = async (k) => {
      try { return (await s.get(k, { type: "json" })) || null; } catch { return null; }
    };
    const [predictions, live, metrics, meta] = await Promise.all([
      read("predictions"), read("live"), read("metrics"), read("meta"),
    ]);
    return reply(200, { predictions, live, metrics, pushedAt: (meta || {}).pushedAt || null });
  }

  if (event.httpMethod !== "POST") return reply(405, { error: "method not allowed" });

  const token = process.env.TT_PUSH_TOKEN;
  const sent = event.headers["x-tt-token"] || event.headers["X-Tt-Token"];
  if (!token || sent !== token) return reply(401, { ok: false, reason: "unauthorized" });
  if (!s) return reply(503, { ok: false, reason: "blob store unavailable" });

  let body = {};
  try { body = JSON.parse(event.body || "{}"); } catch {
    return reply(400, { ok: false, reason: "invalid JSON" });
  }
  const kind = String(body.kind || "");
  if (!KINDS.has(kind)) return reply(400, { ok: false, reason: `kind must be one of ${[...KINDS].join(", ")}` });
  if (body.payload == null) return reply(400, { ok: false, reason: "payload required" });

  try {
    await s.setJSON(kind, body.payload);
    const meta = (await s.get("meta", { type: "json" })) || {};
    meta.pushedAt = { ...(meta.pushedAt || {}), [kind]: Date.now() };
    await s.setJSON("meta", meta);
  } catch (e) {
    return reply(500, { ok: false, reason: String(e).slice(0, 200) });
  }
  return reply(200, { ok: true, kind });
};
