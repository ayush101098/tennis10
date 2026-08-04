/**
 * Netlify serverless function — SofaScore API proxy, with a push-fed cache.
 *
 * When the frontend is deployed as a static export on Netlify, Next.js API
 * routes don't exist. This function handles /api/sofa/* requests instead.
 *
 * SofaScore blocks programmatic HTTP requests via TLS fingerprinting: its CDN
 * returns 403 {"error":{"code":403,"reason":"challenge"}} to Node fetch, curl,
 * etc. sofa_proxy.py defeats that with Chrome TLS impersonation — but only from
 * a host SofaScore hasn't challenged. In practice the deployed proxy gets
 * challenged too, so production had NO working tennis source (ESPN returns
 * tournaments but zero individual matches), and the board rendered empty while
 * ATP/WTA matches were live.
 *
 * So this function now has two sources, in order:
 *   1. UPSTREAM  — SOFA_PROXY_URL (a deployed sofa_proxy.py), when it works.
 *                  A successful response is written to the cache.
 *   2. CACHE     — Netlify Blobs, populated either by (1) or by pushes from a
 *                  local machine that CAN reach SofaScore (tennis/push_sofa.py).
 *
 * The URL contract is unchanged, so no client code has to know any of this.
 *
 *   GET  /api/sofa/<sofa path>      -> JSON (upstream, else cache)
 *   POST /api/sofa/_push            -> { path, payload }, needs x-tt-token
 *
 * Env: SOFA_PROXY_URL (optional upstream), TT_PUSH_TOKEN (required to push).
 */

const STORE = "sofa";
const MAX_CACHE_AGE_MS = 30 * 60 * 1000;   // served with a warning past this

const { store: sharedStore, blobStatus } = require("./_blobs");

const blobs = () => sharedStore(STORE);

// Blob keys can't contain "/" — flatten the SofaScore path.
const keyFor = (p) => "p_" + String(p).replace(/^\/+|\/+$/g, "").replace(/[^A-Za-z0-9._-]/g, "_");

const json = (statusCode, obj, extra) => ({
  statusCode,
  headers: {
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
    "Access-Control-Allow-Origin": "*",
    ...(extra || {}),
  },
  body: typeof obj === "string" ? obj : JSON.stringify(obj),
});

function pathFrom(event) {
  const fnPrefix = "/.netlify/functions/sofa-proxy/";
  let p = event.path || "";
  if (p.startsWith(fnPrefix)) p = p.slice(fnPrefix.length);
  else if (p.startsWith("/api/sofa/")) p = p.slice("/api/sofa/".length);
  return p;
}

exports.handler = async (event) => {
  const sofaPath = pathFrom(event);
  const store = blobs();

  // Diagnostic: is push auth actually configured in this runtime? Reports only
  // whether the var is present and its length — never the value — so a 401 can
  // be told apart from a missing env var (the usual cause is a Netlify variable
  // scoped to Builds only, so Functions never receive it).
  if (sofaPath === "_authcheck") {
    const t = process.env.TT_PUSH_TOKEN;
    const st = blobStatus();
    return json(200, {
      tokenConfigured: !!t,
      tokenLength: t ? t.length : 0,
      blobStore: !!store,
      blobError: st.lastError,
      hasSiteId: st.hasSiteId,
      hasApiToken: st.hasApiToken,
      upstreamConfigured: !!process.env.SOFA_PROXY_URL,
    });
  }

  // ── push from a machine that can actually reach SofaScore ──
  if (event.httpMethod === "POST") {
    const token = process.env.TT_PUSH_TOKEN;
    const sent = event.headers["x-tt-token"] || event.headers["X-Tt-Token"];
    if (!token || sent !== token) return json(401, { ok: false, reason: "unauthorized" });
    if (!store) return json(503, { ok: false, reason: "blob store unavailable" });

    let body = {};
    try { body = JSON.parse(event.body || "{}"); } catch {
      return json(400, { ok: false, reason: "invalid JSON" });
    }
    const p = String(body.path || "").replace(/^\/+/, "");
    if (!p) return json(400, { ok: false, reason: "path required" });
    if (body.payload == null) return json(400, { ok: false, reason: "payload required" });
    try {
      await store.setJSON(keyFor(p), { at: Date.now(), payload: body.payload });
    } catch (e) {
      return json(500, { ok: false, reason: String(e).slice(0, 200) });
    }
    return json(200, { ok: true, path: p });
  }

  if (event.httpMethod !== "GET") return json(405, { error: "method not allowed" });

  // ── 1. upstream ──
  const upstream = process.env.SOFA_PROXY_URL;
  if (upstream) {
    try {
      const res = await fetch(`${upstream.replace(/\/$/, "")}/${sofaPath}`, {
        headers: { Accept: "application/json" },
      });
      const text = await res.text();
      // SofaScore answers a challenge with 403 and a JSON error body; treat any
      // non-2xx as a miss so we fall through to the cache rather than handing
      // the client an error page it will render as "no matches".
      if (res.ok) {
        if (store) {
          try {
            await store.setJSON(keyFor(sofaPath), { at: Date.now(), payload: JSON.parse(text) });
          } catch { /* caching is best-effort */ }
        }
        return json(200, text, {
          "Cache-Control": "public, s-maxage=3, stale-while-revalidate=5",
          "x-sofa-source": "upstream",
        });
      }
      console.warn(`[sofa-proxy] upstream ${res.status} for ${sofaPath}`);
    } catch (err) {
      console.warn("[sofa-proxy] upstream unreachable:", String(err).slice(0, 120));
    }
  }

  // ── 2. cache ──
  if (store) {
    try {
      const hit = await store.get(keyFor(sofaPath), { type: "json" });
      if (hit && hit.payload != null) {
        const age = Date.now() - (hit.at || 0);
        return json(200, hit.payload, {
          "x-sofa-source": "cache",
          "x-sofa-age-ms": String(age),
          ...(age > MAX_CACHE_AGE_MS ? { "x-sofa-stale": "true" } : {}),
        });
      }
    } catch (e) {
      console.error("[sofa-proxy] cache read failed:", String(e).slice(0, 120));
    }
  }

  return json(503, {
    error: "No SofaScore data available.",
    detail: upstream
      ? "Upstream proxy was challenged/unreachable and nothing is cached for this path. "
        + "Run `python -m tennis_push` (tennis/push_sofa.py) locally to populate the cache."
      : "SOFA_PROXY_URL is not set and nothing is cached for this path.",
    path: sofaPath,
  });
};
