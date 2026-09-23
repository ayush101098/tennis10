import type { CanonicalEvent, NormalizedMarket } from "../types";

/**
 * Provider adapter interface. A real second bookmaker/exchange becomes a new
 * file implementing this — the matching engine and classifier never import a
 * provider module directly, only this shape.
 */
export interface MarketProvider {
  readonly id: string;
  /** Fetch current match-winner markets for the given events. Implementations
   *  return [] rather than throwing when a specific event has no price —
   *  "no market" and "provider down" are different things and callers must
   *  not conflate them into a crash. */
  fetchMatchWinnerMarkets(events: CanonicalEvent[]): Promise<NormalizedMarket[]>;
}
