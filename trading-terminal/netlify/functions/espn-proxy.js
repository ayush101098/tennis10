/**
 * Netlify serverless function — ESPN scoreboard proxy.
 *
 * WHY THIS EXISTS
 *   scheduleService called site.api.espn.com straight from the browser, on the
 *   grounds that ESPN is "CORS-enabled, no backend". It still sends
 *   access-control-allow-origin: * — but only on a 200. It now answers
 *   browser-shaped requests (browser User-Agent, Sec-Fetch-Site: cross-site,
 *   an Origin header) with 403, and a 403 carries no CORS headers, so Chrome
 *   reports it as a CORS failure and the fetch rejects. The same URL fetched
 *   server-side, without those headers, returns 200 with the full scoreboard.
 *
 *   The result was an empty match board in production for EVERYONE — anonymous
 *   visitors and paying subscribers alike — with no error surfaced, because the
 *   caller treats a failed tour fetch as "no matches for this tour".
 *
 *   So the request is made from here instead: no browser headers, no CORS
 *   involved, and the response is handed back to our own origin.
 *
 *   GET /api/espn/<espn path>   ->  JSON
 *   e.g. /api/espn/atp/scoreboard?dates=20260807
 *
 * A short cache keeps a live board (polled every 45s by every open tab) from
 * turning into one upstream request per viewer per cycle.
 */

const ESPN = "https://site.api.espn.com/apis/site/v2/sports/tennis";
const CACHE_MS = 20 * 1000;

// Per-container memory cache. Deliberately not Blobs: this is a hot path with
// a 20s life, and a blob round-trip would cost more than the upstream call.
const cache = new Map();

const json = (statusCode, obj, extra) => ({
  statusCode,
  headers: {
    "Content-Type": "application/json",
    "Cache-Control": "public, s-maxage=120, stale-while-revalidate=600",
    "Access-Control-Allow-Origin": "*",
    ...(extra || {}),
  },
  body: JSON.stringify(obj),
});

exports.handler = async (event) => {
  // /.netlify/functions/espn-proxy/atp/scoreboard  ->  atp/scoreboard
  const path = (event.path || "")
    .replace(/^.*\/espn-proxy\/?/, "")
    .replace(/^.*\/api\/espn\/?/, "")
    .replace(/^\/+/, "");
  if (!path) return json(400, { error: "no ESPN path given" });

  // Only the scoreboard endpoints are proxied — this must not become an open
  // relay that will fetch anything on the internet on request.
  if (!/^(atp|wta)\/scoreboard$/.test(path.split("?")[0])) {
    return json(400, { error: "unsupported path", path });
  }

  const qs = event.rawQuery || "";
  const url = `${ESPN}/${path}${qs ? `?${qs}` : ""}`;

  const hit = cache.get(url);
  if (hit && Date.now() - hit.ts < CACHE_MS) {
    return json(200, hit.data, { "X-Espn-Cache": "hit" });
  }

  try {
    const res = await fetch(url, {
      // No User-Agent, Origin or Sec-Fetch-* — those are what earn the 403.
      headers: { Accept: "application/json" },
    });
    if (!res.ok) {
      return json(res.status === 404 ? 404 : 502, {
        error: `espn ${res.status}`,
        // surfaced so a future block is diagnosable rather than silent
        hint: res.status === 403 ? "ESPN rejected the request shape" : undefined,
      });
    }
    const data = await res.json();
    cache.set(url, { ts: Date.now(), data });
    return json(200, data, { "X-Espn-Cache": "miss" });
  } catch (e) {
    return json(502, { error: String((e && e.message) || e).slice(0, 160) });
  }
};
