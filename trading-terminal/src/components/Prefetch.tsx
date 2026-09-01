/**
 * Warm the data requests before the app exists.
 *
 * Measured on the deployed site: the bundle downloaded and React hydrated for
 * ~1.2s before scheduleService issued its first request, so the board could not
 * appear sooner than that no matter how fast the API answered. This script runs
 * from the document head — before the framework is parsed — and starts exactly
 * the requests the terminal is about to make. The service then adopts the
 * in-flight promise instead of issuing a second one.
 *
 * ESPN is not warmed: it returns no individual matches, so the client stopped
 * calling it.
 *
 * (`__ttPrefetch` / `__ttFeedAge` are "trading terminal", not table tennis —
 * they warm the tennis feed and are load-bearing.)
 *
 * It must stay a plain inline string: a React component would arrive with the
 * very bundle whose latency this exists to hide.
 *
 * The URLs MUST match scheduleService character for character, or the
 * prefetch is dead weight and the request is made twice. They are asserted in
 * the same file that builds them (see PREFETCH_KEYS in scheduleService).
 */

// Category ids: ATP, WTA, Challenger, ITF Men, ITF Women — mirrors SOFA_CAT_URLS.
const CATEGORY_IDS = [3, 6, 72, 785, 213];

const SCRIPT = `
(function () {
  try {
    var d = new Date();
    var day = d.getFullYear() + "-" +
      String(d.getMonth() + 1).padStart(2, "0") + "-" +
      String(d.getDate()).padStart(2, "0");
    var urls = ${JSON.stringify(CATEGORY_IDS)}.map(function (c) {
      return "/api/sofa/category/" + c + "/scheduled-events/" + day;
    });
    urls.push("/api/sofa/sport/tennis/odds/1/" + day);
    var store = {};
    urls.forEach(function (u) {
      // Kept as a promise, not a value: the consumer awaits whatever state this
      // is in when it arrives, whether that is pending or already settled.
      store[u] = fetch(u, { cache: "no-store" })
        .then(function (r) {
          // Record how stale the cache behind this response is, so a warmed
          // request still reports it. Without this the prefetch path silently
          // dropped x-sofa-age-ms and the staleness warning never fired.
          var a = Number(r.headers.get("x-sofa-age-ms"));
          if (isFinite(a) && a > 0) {
            window.__ttFeedAge = Math.max(window.__ttFeedAge || 0, a);
          }
          return r.ok ? r.json() : null;
        })
        .catch(function () { return null; });
    });
    window.__ttPrefetch = store;
  } catch (e) {
    // A failed warm-up must never break the page — the app just fetches normally.
    window.__ttPrefetch = {};
  }
})();
`;

export default function Prefetch() {
  return <script dangerouslySetInnerHTML={{ __html: SCRIPT }} />;
}
