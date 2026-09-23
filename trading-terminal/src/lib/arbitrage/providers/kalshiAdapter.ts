import type { MarketProvider } from "./base";
import type { CanonicalEvent, NormalizedMarket } from "../types";

/**
 * Kalshi — NOT currently wired to real data. This is the shape a real
 * integration takes, not a working feed.
 *
 * Confirmed directly (2026-09-22): `GET trading-api.kalshi.com/trade-api/v2/series`
 * returns 401 Unauthorized with no credentials — Kalshi's market data is not
 * a public feed. It needs a funded, KYC'd Kalshi account and an API key
 * (RSA-signed request headers, not a simple bearer token).
 *
 * WHEN A REAL KEY EXISTS:
 *   1. Server-side only — KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY in the
 *      deployment env, never a NEXT_PUBLIC_* var.
 *   2. A proxy route (netlify/functions/kalshi-proxy.js + the matching Next
 *      API route) that signs each request server-side, same pattern as
 *      pm-proxy.js — Kalshi's request signing in particular cannot happen in
 *      the browser without exposing the private key.
 *   3. Flip KALSHI_CONFIGURED to true and implement the fetch against the
 *      proxy path. Kalshi's tennis markets (when listed) are typically
 *      single binary "will X win" contracts — map straightforwardly onto
 *      NormalizedMarket the way polymarketAdapter.ts maps Polymarket's.
 */
export const KALSHI_CONFIGURED = false;

export const kalshiAdapter: MarketProvider = {
  id: "kalshi",

  async fetchMatchWinnerMarkets(_events: CanonicalEvent[]): Promise<NormalizedMarket[]> {
    if (!KALSHI_CONFIGURED) return [];
    return [];
  },
};
