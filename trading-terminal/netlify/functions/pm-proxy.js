/**
 * Netlify serverless function — Polymarket proxy (production twin of
 * src/app/api/pm/[...path]/route.ts).
 *
 * WHY THIS EXISTS
 *   The dev route above has existed for a while, but the Netlify build strips
 *   `src/app/api` (see netlify.toml) and no function replaced it — so in
 *   production `/api/pm/*` simply did not exist, and lib/polymarket.ts fell
 *   back to calling gamma-api.polymarket.com straight from the browser.
 *
 *   That meant one upstream connection PER VISITOR: every open tab refetched
 *   the whole tennis event list every 60s, and every open trade ticket polled
 *   the CLOB order book every 10s. Polymarket saw N consumers, not one, and
 *   the cost of visitor number 1,000 was identical to visitor number 1.
 *
 *   With this function the browser talks only to us, and we talk to Polymarket
 *   once per cache window no matter how many people are watching.
 *
 * THE CACHING IS THE POINT, NOT AN OPTIMISATION
 *   `Cache-Control: s-maxage` lets Netlify's CDN answer almost every request
 *   without invoking the function at all. The dev route sends
 *   `no-cache, no-store` — copying that here would have made every client
 *   request a function invocation, which is precisely the mistake sofa-proxy.js
 *   documents having already made once (~40 invocations per visitor per minute).
 *
 *   TTLs are per host class, and deliberately not uniform:
 *     gamma  30s  — the fixture index; the client only refreshes every 60s
 *     clob    3s  — order book. This is a PRICE, and a stale price rendered as
 *                   live is the most expensive bug this product can ship, so
 *                   it gets the shortest window that still collapses concurrent
 *                   readers of the same token onto one upstream call.
 *     data   60s  — wallet history, changes slowly
 *     lb    300s  — leaderboard, changes very slowly
 *
 *   GET /api/pm/<host>/<path>?<query>   host ∈ gamma | clob | data | lb
 */

const HOSTS = {
  data: "https://data-api.polymarket.com",
  lb: "https://lb-api.polymarket.com",
  clob: "https://clob.polymarket.com",
  gamma: "https://gamma-api.polymarket.com",
};

/** Seconds of shared (CDN) cache per host class. See the note above. */
const TTL = { gamma: 30, clob: 3, data: 60, lb: 300 };

const UPSTREAM_TIMEOUT_MS = 10_000;

const json = (statusCode, obj, ttl) => ({
  statusCode,
  headers: {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": ttl
      ? `public, s-maxage=${ttl}, stale-while-revalidate=${ttl * 4}`
      : "no-store",
  },
  body: JSON.stringify(obj),
});

function pathFrom(event) {
  const fnPrefix = "/.netlify/functions/pm-proxy/";
  let p = event.path || "";
  if (p.startsWith(fnPrefix)) p = p.slice(fnPrefix.length);
  else if (p.startsWith("/api/pm/")) p = p.slice("/api/pm/".length);
  return p.replace(/^\/+/, "");
}

exports.handler = async (event) => {
  if (event.httpMethod === "OPTIONS") {
    return {
      statusCode: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      },
      body: "",
    };
  }
  if (event.httpMethod !== "GET") {
    return json(405, { error: "method not allowed" });
  }

  const raw = pathFrom(event);
  const [hostKey, ...rest] = raw.split("/");
  const base = HOSTS[hostKey];

  // Allowlist, not passthrough: without it this is an open proxy that anyone
  // can point at any host on our bandwidth.
  if (!base) {
    return json(400, {
      error: `unknown polymarket host '${hostKey}'; use one of ${Object.keys(HOSTS).join(", ")}`,
    });
  }

  const qs = event.rawQuery ? `?${event.rawQuery}` : "";
  const url = `${base}/${rest.join("/")}${qs}`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      headers: { "User-Agent": "tennis10-terminal/2.0", Accept: "application/json" },
    });
    if (!res.ok) {
      // Do NOT cache an upstream failure at the CDN — a cached 502 would
      // outlive the outage that caused it.
      return json(res.status, { error: `polymarket ${hostKey} returned ${res.status}`, url });
    }
    return json(200, await res.json(), TTL[hostKey]);
  } catch (err) {
    const aborted = err && err.name === "AbortError";
    console.error("[pm-proxy]", aborted ? `timeout ${url}` : err);
    return json(502, { error: aborted ? "polymarket timed out" : "polymarket proxy unavailable", url });
  } finally {
    clearTimeout(timer);
  }
};
