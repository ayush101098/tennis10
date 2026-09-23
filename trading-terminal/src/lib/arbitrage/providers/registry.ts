import type { MarketProvider } from "./base";
import { pinnacleAdapter } from "./pinnacleAdapter";
import { kalshiAdapter } from "./kalshiAdapter";

/**
 * Every venue on the requested comparison list, honestly labeled by whether
 * this app can actually reach it.
 *
 * WHY MOST OF THESE HAVE NO ADAPTER FILE
 *   Checked directly (2026-09-22): Kalshi's own market-data endpoint returns
 *   401 without an API key — it is not a public feed. Pinnacle's odds API is
 *   a real, documented thing but is a partner/API-key arrangement, not open.
 *   The eleven consumer crypto sportsbooks (Stake, BC.Game, DEXSPORT, Roobet,
 *   Duel, CSGOEmpire, Shuffle, Reels, Gamdom, BookmakerXYZ, Bet105) and the
 *   two smaller prediction markets (Limitless, PredictFun) publish no odds
 *   API at all — the only way to read their prices programmatically is
 *   scraping a logged-in session, which needs a real account on each one and,
 *   for several, would run against their own terms of service. Writing
 *   "adapter" files for those that quietly return [] forever would look like
 *   coverage that does not exist — see types.ts's whole reason for existing.
 *
 *   What's here instead: real status per venue, and two real (if currently
 *   unconfigured) adapters — Pinnacle and Kalshi — with the exact auth shape
 *   documented and ready, so plugging in a real key later is a config change,
 *   not new code. This is the "provider-agnostic core" the spec asked for:
 *   the screener and classifier never change when a venue moves from
 *   "not configured" to "live".
 */

export type ProviderStatus = "live" | "configured_no_key" | "no_public_api";

export interface ProviderEntry {
  id: string;
  displayName: string;
  category: "sportsbook" | "prediction_market";
  status: ProviderStatus;
  /** How a real integration would authenticate, for whoever adds the key. */
  authNote: string;
  adapter?: MarketProvider;
}

export const PROVIDER_REGISTRY: ProviderEntry[] = [
  {
    id: "polymarket", displayName: "Polymarket", category: "prediction_market",
    status: "live", authNote: "No auth needed for read-only price data — already wired via /api/pm.",
  },
  {
    id: "pinnacle", displayName: "Pinnacle", category: "sportsbook",
    status: "configured_no_key",
    authNote: "Pinnacle API requires an account + API key (Basic Auth). Set PINNACLE_API_KEY / PINNACLE_USERNAME server-side and add a proxy route under /api/pinnacle (never call it from the browser — see pmValue.ts's proxy pattern for why).",
    adapter: pinnacleAdapter,
  },
  {
    id: "kalshi", displayName: "Kalshi", category: "prediction_market",
    status: "configured_no_key",
    authNote: "Confirmed live 2026-09-22: /trade-api/v2/series returns 401 without a key. Needs a funded, KYC'd Kalshi account and API key, proxied server-side the same way.",
    adapter: kalshiAdapter,
  },
  { id: "stake", displayName: "Stake", category: "sportsbook", status: "no_public_api", authNote: "No published odds API. Reading prices means scraping an authenticated session — needs a real account, and may violate Stake's ToS." },
  { id: "bcgame", displayName: "BC.Game", category: "sportsbook", status: "no_public_api", authNote: "No published odds API — same constraint as Stake." },
  { id: "dexsport", displayName: "DEXSPORT", category: "sportsbook", status: "no_public_api", authNote: "No published odds API." },
  { id: "roobet", displayName: "Roobet", category: "sportsbook", status: "no_public_api", authNote: "No published odds API." },
  { id: "duel", displayName: "Duel", category: "sportsbook", status: "no_public_api", authNote: "No published odds API." },
  { id: "csgoempire", displayName: "CSGOEmpire", category: "sportsbook", status: "no_public_api", authNote: "No published odds API." },
  { id: "shuffle", displayName: "Shuffle", category: "sportsbook", status: "no_public_api", authNote: "No published odds API." },
  { id: "reels", displayName: "Reels", category: "sportsbook", status: "no_public_api", authNote: "No published odds API." },
  { id: "gamdom", displayName: "Gamdom", category: "sportsbook", status: "no_public_api", authNote: "No published odds API." },
  { id: "bookmakerxyz", displayName: "BookmakerXYZ", category: "sportsbook", status: "no_public_api", authNote: "No published odds API." },
  { id: "bet105", displayName: "Bet105", category: "sportsbook", status: "no_public_api", authNote: "No published odds API." },
  { id: "limitless", displayName: "Limitless", category: "prediction_market", status: "no_public_api", authNote: "No documented public API found." },
  { id: "predictfun", displayName: "PredictFun", category: "prediction_market", status: "no_public_api", authNote: "No documented public API found." },
  {
    id: "tennisalpha_model", displayName: "TennisAlpha Model", category: "prediction_market",
    status: "live", authNote: "Not a market — the model's own True P, used only for model_positive_ev comparisons.",
  },
];
