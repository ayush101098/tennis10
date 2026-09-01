import type { Page } from "@playwright/test";

/**
 * Determinism for visual tests.
 *
 * This app polls a live schedule, a live market and a presence counter, and
 * blinks a live indicator. Screenshot every one of those and the baseline
 * changes minute to minute for reasons that have nothing to do with the UI —
 * the suite would fail constantly and be ignored, which is worse than not
 * having it.
 *
 * So every network source is stubbed to a fixed response, and the page is
 * exercised in the states we actually want to hold still: empty and populated.
 */

/**
 * Two Polymarket fixtures, matched to the stub schedule below.
 *
 * These use REAL ranked players on purpose. The board refuses to price a match
 * whose players it cannot rank (no prior means no opinion — see
 * attachIntelligence), so a fixture of invented names renders an empty board
 * and the populated path silently goes untested.
 */
const PM_MARKETS = [
  {
    slug: "usopen-sinner-alcaraz",
    title: "US Open: Jannik Sinner vs Carlos Alcaraz",
    active: true, closed: false,
    markets: [{
      question: "US Open: Jannik Sinner vs Carlos Alcaraz",
      conditionId: "0x01", outcomes: '["Jannik Sinner","Carlos Alcaraz"]',
      outcomePrices: '["0.55","0.45"]', clobTokenIds: '["1","2"]',
      negRisk: false, orderPriceMinTickSize: 0.01, closed: false, active: true,
    }],
  },
  {
    slug: "usopen-sabalenka-swiatek",
    title: "US Open: Aryna Sabalenka vs Iga Swiatek",
    active: true, closed: false,
    markets: [{
      question: "US Open: Aryna Sabalenka vs Iga Swiatek",
      conditionId: "0x02", outcomes: '["Aryna Sabalenka","Iga Swiatek"]',
      outcomePrices: '["0.40","0.60"]', clobTokenIds: '["3","4"]',
      negRisk: false, orderPriceMinTickSize: 0.01, closed: false, active: true,
    }],
  },
];

/** A SofaScore-shaped event, in the feed's own abbreviated "Surname F." form —
 *  the format the ranking lookup has to cope with in production. */
function sofaEvent(id: number, home: string, away: string, startTs: number,
                   tour: "atp" | "wta" = "atp", tournament = "US Open") {
  return {
    id,
    tournament: {
      name: tournament, category: { slug: tour, name: tour.toUpperCase() },
      uniqueTournament: { name: tournament, id: 2480 },
    },
    homeTeam: { name: home, type: 1, id: id * 10 },
    awayTeam: { name: away, type: 1, id: id * 10 + 1 },
    status: { code: 0, description: "Not started", type: "notstarted" },
    startTimestamp: startTs,
    groundType: "Hardcourt outdoor",
    homeScore: {}, awayScore: {},
    roundInfo: { name: "1st round" },
  };
}

export interface StubOptions {
  /** "empty" exercises the empty states; "populated" exercises a live board. */
  mode?: "empty" | "populated";
}

export async function stubNetwork(page: Page, { mode = "populated" }: StubOptions = {}) {
  // A fixed clock keeps relative times and the date-keyed schedule stable.
  await page.clock.setFixedTime(new Date("2026-09-01T12:00:00Z"));

  const events = mode === "populated"
    ? [
        sofaEvent(9001, "Sinner J.", "Alcaraz C.", 1788264000),
        sofaEvent(9002, "Sabalenka A.", "Swiatek I.", 1788270000, "wta", "Cincinnati"),
      ]
    : [];

  await page.route("**/api/sofa/**", route => {
    const url = route.request().url();
    if (url.includes("/odds/")) return route.fulfill({ json: { odds: {} } });
    if (url.includes("/events/live")) return route.fulfill({ json: { events: [] } });
    return route.fulfill({ json: { events } });
  });

  // The browser now talks to our proxy, never to Polymarket directly — stub
  // the proxy path. A request to gamma-api.polymarket.com from the page would
  // be a regression, so it is stubbed to fail loudly rather than succeed.
  await page.route("**/api/pm/gamma/**", route =>
    route.fulfill({ json: mode === "populated" ? PM_MARKETS : [] }));

  await page.route("**/api/pm/clob/**", route =>
    route.fulfill({ json: { bids: [], asks: [] } }));

  await page.route("**/*.polymarket.com/**", route =>
    route.abort("blockedbyclient"));

  // Presence and analytics must never reach the network from a test run.
  await page.route("**/api/presence**", route => route.fulfill({ json: { count: 12 } }));
  await page.route("**/api/track**", route => route.fulfill({ json: { ok: true } }));
  await page.route("**/api/account**", route => route.fulfill({ status: 401, json: {} }));
}

/**
 * Hide anything that legitimately varies between runs but is not what the
 * screenshot is asserting. Returns locators for `mask`.
 */
export const volatile = (page: Page) => [
  page.locator("[data-live-count]"),
  page.locator("iframe"),          // the YouTube embed loads at its own pace
];

/** Wait for the client board to settle after hydration. */
export async function settle(page: Page) {
  await page.waitForLoadState("domcontentloaded");
  // The schedule renders after the first fetch resolves; the stub is instant,
  // so a short settle is enough and keeps the suite fast.
  await page.waitForTimeout(1200);

  // Force every below-the-fold section to lay out BEFORE the capture. Without
  // this a fullPage screenshot on a phone viewport measured a different page
  // height between runs, and the baseline was unreproducible — a flaky visual
  // test gets muted, which is worse than having none.
  await page.evaluate(async () => {
    const step = window.innerHeight;
    for (let y = 0; y < document.body.scrollHeight; y += step) {
      window.scrollTo(0, y);
      await new Promise(r => requestAnimationFrame(() => r(null)));
    }
    window.scrollTo(0, 0);
  });
  await page.waitForTimeout(400);
  // Fonts must be ready or text reflows mid-capture.
  await page.evaluate(() => (document as any).fonts?.ready);
}
