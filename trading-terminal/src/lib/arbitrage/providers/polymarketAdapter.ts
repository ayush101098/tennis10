import type { MarketProvider } from "./base";
import type { CanonicalEvent, NormalizedMarket } from "../types";
import { getPolymarketIndex, outcomeIndex, fixtureKey, fetchQuote } from "@/lib/polymarket";

/**
 * Polymarket as a NormalizedMarket source.
 *
 * Uses the live CLOB best-ask per outcome token — the actual price a taker
 * would pay to buy that side right now — not the Gamma snapshot price used
 * elsewhere for the Value Board's model-vs-market edge. Arbitrage math needs
 * an executable price; a snapshot that can be minutes old is not one.
 *
 * Liquidity is deliberately left null: the shared fetchQuote() helper (reused
 * as-is rather than duplicated — see /lib/polymarket.ts) returns best price
 * only, not resting size. Reporting a made-up size would violate the "never
 * fabricate liquidity" rule this whole module is built around; classify.ts
 * treats null liquidity as "unknown" and downgrades accordingly.
 */
export const POLYMARKET_SETTLEMENT_RULE = "polymarket_binary_v1";

export const polymarketAdapter: MarketProvider = {
  id: "polymarket",

  async fetchMatchWinnerMarkets(events: CanonicalEvent[]): Promise<NormalizedMarket[]> {
    const index = await getPolymarketIndex();

    // Pass 1: pure in-memory lookups (fast) to find which events actually
    // have a Polymarket fixture. Only THESE need a network round trip.
    const matched: Array<{ event: CanonicalEvent; conditionId: string; i1: number; token1: string; i2: number; token2: string }> = [];
    for (const event of events) {
      if (event.status === "finished" || event.status === "cancelled") continue;
      const fixture = index.get(fixtureKey(event.player1, event.player2));
      const market = fixture?.match;
      if (!market || market.tokenIds.length !== 2) continue;

      const i1 = outcomeIndex(market, event.player1);
      const i2 = outcomeIndex(market, event.player2);
      if (i1 < 0 || i2 < 0 || i1 === i2) continue;

      matched.push({ event, conditionId: market.conditionId, i1, token1: market.tokenIds[i1], i2, token2: market.tokenIds[i2] });
    }

    // Pass 2: every CLOB order-book fetch in parallel. This was previously a
    // for-of loop awaiting each event's pair of requests one after another —
    // for dozens of matched fixtures that meant 30-60+ sequential round trips
    // before the screener showed anything at all ("feed isn't loading" was
    // this, not a missing-data problem: Polymarket genuinely has these
    // matches). One Promise.all across the whole matched set instead.
    const quotes = await Promise.all(
      matched.map(m => Promise.all([
        fetchQuote(m.token1).catch(() => ({ bestBid: null, bestAsk: null })),
        fetchQuote(m.token2).catch(() => ({ bestBid: null, bestAsk: null })),
      ])),
    );

    const now = new Date().toISOString();
    const legFor = (
      event: CanonicalEvent,
      conditionId: string,
      selection: "player1" | "player2",
      ask: number | null,
    ): NormalizedMarket | null => {
      if (ask == null || ask <= 0 || ask >= 1) return null;
      return {
        marketId: `pm_${conditionId}_${selection}`,
        eventId: event.matchId,
        providerId: "polymarket",
        marketType: "match_winner",
        period: "full_match",
        selection,
        oddsDecimal: 1 / ask,
        currency: "USD",
        marketStatus: "open",
        availableLiquidity: null,
        maxStake: null,
        commissionRate: 0, // Polymarket charges no separate taker fee today
        providerTimestamp: now,
        receivedTimestamp: now,
        mappingConfidence: 1.0, // same-provider legs — no cross-venue name matching involved
        settlementRuleId: POLYMARKET_SETTLEMENT_RULE,
        isModel: false,
      };
    };

    const out: NormalizedMarket[] = [];
    matched.forEach((m, idx) => {
      const [q1, q2] = quotes[idx];
      const leg1 = legFor(m.event, m.conditionId, "player1", q1.bestAsk);
      const leg2 = legFor(m.event, m.conditionId, "player2", q2.bestAsk);
      if (leg1) out.push(leg1);
      if (leg2) out.push(leg2);
    });

    return out;
  },
};
