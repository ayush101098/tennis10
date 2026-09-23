/**
 * Canonical types for the arbitrage screener.
 *
 * SCOPE, HONESTLY STATED
 *   The full spec this module was built against describes a multi-bookmaker
 *   ingestion platform (Postgres, Redis, a provider per exchange, a WebSocket
 *   gateway, 500-user load testing). This app has one deployment target
 *   (Next.js on Vercel/Netlify, no separate backend service) and exactly two
 *   real data sources already wired up: SofaScore (scores, feeds the model)
 *   and Polymarket (the only tradeable market with its own two-sided price).
 *   Rather than fabricate bookmaker adapters with no real credentials behind
 *   them, this module implements the real taxonomy/matching/math/classification
 *   machinery against those two sources, architected so a genuine second
 *   market provider is a new file in providers/, not a rewrite.
 *
 *   Consequence: "verified mathematical arbitrage" here means Polymarket's own
 *   two outcome tokens (player A ask, player B ask) are inconsistently priced
 *   against each other — a real, executable pattern on a single order book.
 *   The model's True P is a SEPARATE synthetic "provider" used only to surface
 *   category C (model-based EV), and the classifier refuses to ever call a
 *   model leg "arbitrage" (see classify.ts) — that distinction is the one
 *   piece of this spec that is non-negotiable to get right.
 */

/** Subset of the spec's taxonomy this data actually supports today. Adding a
 *  market type here does not by itself make it tradeable — a provider adapter
 *  has to actually emit legs for it. */
export type MarketType = "match_winner" | "set_winner";
export type MarketPeriod = "full_match" | "set1" | "set2" | "set3" | "set4" | "set5";

/** "polymarket" and "tennisalpha_model" are the only ones any adapter emits
 *  legs under today. The rest are registered (see providers/registry.ts) so
 *  the type is ready the moment one of them gets real credentials — adding a
 *  venue should never require widening this union at integration time. */
export type ProviderId =
  | "polymarket" | "tennisalpha_model"
  | "pinnacle" | "kalshi"
  | "stake" | "bcgame" | "dexsport" | "roobet" | "duel" | "csgoempire"
  | "shuffle" | "reels" | "gamdom" | "bookmakerxyz" | "bet105"
  | "limitless" | "predictfun";

export interface CanonicalEvent {
  /** Internal match id — reuses ScheduledMatch.id so this module never
   *  maintains a second identity space for the same match. */
  matchId: string;
  player1: string;
  player2: string;
  tournament: string;
  tour: string;
  round: string;
  surface: string;
  bestOf: number;
  startTimestamp: number;
  status: "scheduled" | "live" | "finished" | "cancelled";
}

export interface NormalizedMarket {
  marketId: string;
  eventId: string;              // CanonicalEvent.matchId
  providerId: ProviderId;
  marketType: MarketType;
  period: MarketPeriod;
  selection: "player1" | "player2";
  oddsDecimal: number;
  currency: "USD";
  marketStatus: "open" | "suspended" | "closed" | "unknown";
  /** Top-of-book size in USDC for Polymarket; null — never fabricated — when
   *  the source doesn't expose it (true of every leg from a snapshot price). */
  availableLiquidity: number | null;
  maxStake: number | null;
  /** Fraction taken on net winnings. 0 for Polymarket (no trading commission
   *  beyond the spread itself); left explicit per-leg for future venues. */
  commissionRate: number;
  providerTimestamp: string;    // ISO-8601 — when the source says this price was live
  receivedTimestamp: string;    // ISO-8601 — when we fetched it
  /** How sure the event/outcome mapping is. 1.0 for same-provider legs (no
   *  cross-venue name matching was needed); lower once a second real venue
   *  requires the fuzzy matcher in eventMatcher.ts. */
  mappingConfidence: number;
  settlementRuleId: string;
  /** True only for the model's synthetic leg. The classifier uses this as the
   *  hard gate that keeps model output out of the "arbitrage" categories. */
  isModel: boolean;
}

export type OpportunityClassification =
  | "verified_arbitrage"
  | "conditional_arbitrage"
  | "model_positive_ev"
  | "invalid";

export interface ArbLeg {
  provider: ProviderId;
  selection: "player1" | "player2";
  playerName: string;
  odds: number;
  recommendedStake: number;
  currency: "USD";
}

export interface ArbitrageOpportunity {
  opportunityId: string;
  matchId: string;
  event: CanonicalEvent;
  marketType: MarketType;
  period: MarketPeriod;
  classification: OpportunityClassification;
  legs: ArbLeg[];
  theoreticalRoiPct: number;
  /** Null exactly when a required execution input (liquidity, commission,
   *  settlement) is unknown — per spec, never silently treated as zero cost. */
  netRoiPct: number | null;
  theoreticalProfit: number;
  netProfit: number | null;
  totalStake: number;
  quoteAgeMs: number | null;
  liquidityStatus: "known" | "unknown" | "insufficient";
  settlementStatus: "compatible" | "unverified" | "incompatible";
  riskFlags: string[];
  detectedAt: string;
}
