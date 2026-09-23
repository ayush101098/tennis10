import type { MarketProvider } from "./base";
import type { CanonicalEvent, NormalizedMarket } from "../types";

/**
 * Pinnacle — NOT currently wired to real data. This is the shape a real
 * integration takes, not a working feed.
 *
 * Pinnacle is the one venue on the requested list that's actually worth
 * prioritizing for arbitrage: it's the industry's reference "sharp" book
 * (tightest margins, least public-money skew), which is exactly the kind of
 * price worth comparing Polymarket against. But it needs a real account and
 * API key (Basic Auth over their v1/v3 odds API) — there is nothing to call
 * without one.
 *
 * WHEN A REAL KEY EXISTS:
 *   1. Add PINNACLE_API_KEY / PINNACLE_USERNAME to the server environment
 *      only (.env.local, Vercel project env) — NEVER to a NEXT_PUBLIC_* var,
 *      which ships to the browser bundle.
 *   2. Add a proxy route (netlify/functions/pinnacle-proxy.js + the matching
 *      src/app/api/pinnacle/[...path]/route.ts adapt() wrapper) following the
 *      exact pattern netlify/functions/pm-proxy.js already uses for
 *      Polymarket — same reasoning: the browser must never hold the key or
 *      call Pinnacle directly, and per-client polling of a metered odds API
 *      is the cost blowout this whole proxy pattern exists to prevent.
 *   3. Flip PINNACLE_CONFIGURED to true below and implement the fetch against
 *      the proxy path, mapping Pinnacle's market/period/price fields into
 *      NormalizedMarket exactly as polymarketAdapter.ts does.
 */
export const PINNACLE_CONFIGURED = false;

export const pinnacleAdapter: MarketProvider = {
  id: "pinnacle",

  async fetchMatchWinnerMarkets(_events: CanonicalEvent[]): Promise<NormalizedMarket[]> {
    if (!PINNACLE_CONFIGURED) return [];
    // Real implementation goes here once a key + proxy route exist. Left
    // unimplemented rather than stubbed with fake data — see types.ts.
    return [];
  },
};
