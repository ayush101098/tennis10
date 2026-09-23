import type { ScheduledMatch } from "@/lib/scheduleService";
import type { NormalizedMarket } from "../types";

/**
 * TennisAlpha's own True P as a synthetic "provider".
 *
 * Deliberately NOT implementing the MarketProvider interface (base.ts): a
 * real provider fetches a price from outside; this one reads a probability
 * the existing engine has already computed and attached to the match
 * (attachIntelligence in scheduleService.ts) — no network call belongs here.
 *
 * Every leg this emits carries isModel: true. classify.ts hard-gates on that
 * flag and will never return "verified_arbitrage" or "conditional_arbitrage"
 * for a candidate that includes one — a probability estimate is not a price,
 * no matter how good the model is. See types.ts for why this distinction is
 * the one part of the spec that isn't negotiable.
 *
 * Same "real prior" gate as pmValue.ts's polymarketValue(): an unranked,
 * unseeded field (most ITF/M15/M25 matches) makes the ranking model fall back
 * to a bare 0.5 coin flip, not an opinion. Skipping this gate was a real bug
 * caught live: the screener filled up with "170% ROI" rows that were nothing
 * but 1/0.5=2.00 model odds against a real market price — a manufactured
 * edge out of the model having nothing to say, not a signal.
 */
function hasRealPrior(m: ScheduledMatch): boolean {
  return m.prob_method !== "unknown" &&
    ((m.p1_rank > 0 && m.p2_rank > 0) || (m.p1_seed > 0 && m.p2_seed > 0));
}

export function modelMarketsFor(match: ScheduledMatch): NormalizedMarket[] {
  if (!hasRealPrior(match)) return [];
  const p1 = match.status === "live" && match.liveScore?.trueProbabilities?.p1MatchProb != null
    ? match.liveScore.trueProbabilities.p1MatchProb
    : match.p1_win_prob;
  if (p1 == null || p1 <= 0 || p1 >= 1) return [];

  const now = new Date().toISOString();
  const base = {
    eventId: match.id,
    providerId: "tennisalpha_model" as const,
    marketType: "match_winner" as const,
    period: "full_match" as const,
    currency: "USD" as const,
    marketStatus: "open" as const,
    availableLiquidity: null,
    maxStake: null,
    commissionRate: 0,
    providerTimestamp: now,
    receivedTimestamp: now,
    mappingConfidence: 1.0,
    settlementRuleId: "tennisalpha_model_v1",
    isModel: true,
  };

  return [
    { ...base, marketId: `model_${match.id}_player1`, selection: "player1", oddsDecimal: 1 / p1 },
    { ...base, marketId: `model_${match.id}_player2`, selection: "player2", oddsDecimal: 1 / (1 - p1) },
  ];
}
