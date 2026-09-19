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

// Category ids: ATP, WTA, Challenger, ITF Men, ITF Women, WTA 125, Davis Cup —
// mirrors SOFA_CAT_URLS. 76 (Davis Cup) and 871 (WTA 125) were added there
// 2026-09-20 — Davis Cup is SofaScore's own top-level category, entirely
// separate from ATP (3), which is why every Davis Cup tie was invisible until
// then regardless of live status. Kept in sync here too: falling behind
// doesn't break anything (scheduleService fetches them either way), it just
// means those two categories load a beat slower than the rest.
const CATEGORY_IDS = [3, 6, 72, 785, 213, 76, 871];

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
    var oddsUrl = "/api/sofa/sport/tennis/odds/1/" + day;
    urls.push(oddsUrl);
    var store = {};
    urls.forEach(function (u) {
      // Kept as a promise, not a value: the consumer awaits whatever state this
      // is in when it arrives, whether that is pending or already settled.
      store[u] = fetch(u, { cache: "no-store" })
        .then(function (r) {
          // Record how stale the cache behind this response is, so a warmed
          // request still reports it. Without this the prefetch path silently
          // dropped x-sofa-age-ms and the staleness warning never fired.
          //
          // SCORES AND ODDS ARE COUNTED SEPARATELY, and must be. SofaScore
          // 403s the odds endpoints but still serves scheduled-events, so the
          // odds file is hours old while every schedule is current. Folding
          // both into one number made the odds age speak for the whole feed:
          // the board announced "Challenger and ITF are 20 h old" over fixtures
          // fetched seconds earlier, and — because that flag also means "don't
          // trust anything it calls live" — the home page counted 0 live
          // matches while 8 were in play.
          var a = Number(r.headers.get("x-sofa-age-ms"));
          if (isFinite(a) && a > 0) {
            var key = u === oddsUrl ? "__ttOddsAge" : "__ttFeedAge";
            window[key] = Math.max(window[key] || 0, a);
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
