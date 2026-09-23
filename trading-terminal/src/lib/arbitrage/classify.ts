import type { CanonicalEvent, NormalizedMarket, ArbitrageOpportunity, ArbLeg, OpportunityClassification } from "./types";
import { impliedProbSum, arbPercent, allocateStakes } from "./math";

/** Past this, a quote is not "live" for arbitrage purposes — same order of
 *  magnitude as the terminal's own ODDS_STALE_MS for in-play prices, but
 *  scoped separately: an arb candidate needs both legs fresh AT THE SAME
 *  TIME, which is a stricter bar than "the board isn't reporting an outage." */
const MAX_QUOTE_AGE_LIVE_MS = 15_000;
const MAX_QUOTE_AGE_PREMATCH_MS = 5 * 60 * 1000;

/**
 * Deterministic, not time-based — the SAME real-world comparison (this
 * match, this market, this pair of legs) must keep the same id across scan
 * cycles, or nothing can track how it moved between detection and now. See
 * hedgeSignal.ts, which is the reason this needed to change.
 */
function stableId(
  event: CanonicalEvent, marketType: string, period: string,
  legA: NormalizedMarket, legB: NormalizedMarket,
): string {
  return `arb_${event.matchId}_${marketType}_${period}_${legA.providerId}-${legA.selection}_${legB.providerId}-${legB.selection}`;
}

/**
 * Classify a candidate built from exactly two legs covering the same
 * event + market + period (player1 vs player2). Returns null when the legs
 * don't even form a comparable pair (wrong event, wrong market, wrong period)
 * — that's a matching-layer failure, not a D/"invalid" classification, so it
 * doesn't produce an opportunity record at all.
 */
export function classifyTwoWay(
  event: CanonicalEvent,
  marketType: NormalizedMarket["marketType"],
  period: NormalizedMarket["period"],
  legA: NormalizedMarket,
  legB: NormalizedMarket,
  totalStake: number,
  nowMs: number = Date.now(),
): ArbitrageOpportunity | null {
  if (legA.eventId !== event.matchId || legB.eventId !== event.matchId) return null;
  if (legA.marketType !== marketType || legB.marketType !== marketType) return null;
  if (legA.period !== period || legB.period !== period) return null;
  if (legA.selection === legB.selection) return null; // not mutually exclusive coverage

  const opportunityId = stableId(event, marketType, period, legA, legB);
  const riskFlags: string[] = [];
  const bothModel = legA.isModel && legB.isModel;
  const eitherModel = legA.isModel || legB.isModel;

  const s = impliedProbSum([legA.oddsDecimal, legB.oddsDecimal]);
  const theoreticalRoiPct = arbPercent(s);
  const alloc = allocateStakes([legA.oddsDecimal, legB.oddsDecimal], totalStake);

  const legs: ArbLeg[] = [legA, legB].map((leg, i) => ({
    provider: leg.providerId,
    selection: leg.selection,
    playerName: leg.selection === "player1" ? event.player1 : event.player2,
    odds: leg.oddsDecimal,
    recommendedStake: alloc.stakes[i],
    currency: "USD",
  }));

  // A model leg is never a market price — the spec is explicit that a
  // positive-EV signal must not be presented as arbitrage, so this check
  // runs BEFORE the profitability check and short-circuits the classification
  // regardless of what the math says.
  if (eitherModel) {
    if (!bothModel) riskFlags.push("one_leg_is_model_not_market");
    // An edge this large is far more likely a name-matching or stale-price
    // fault than a real one — same bar as the Value Board's SUSPECT_EDGE.
    // Flagged, not hidden: the row still shows, but not as a clean signal.
    if (theoreticalRoiPct > 50) riskFlags.push("suspect_edge_magnitude");
    return build("model_positive_ev", theoreticalRoiPct, alloc, riskFlags, {
      opportunityId, event, marketType, period, legs, totalStake,
      netRoiPct: null, netProfit: null,
      quoteAgeMs: null, liquidityStatus: "unknown", settlementStatus: "unverified",
    });
  }

  if (theoreticalRoiPct <= 0) {
    return build("invalid", theoreticalRoiPct, alloc, ["no_positive_spread"], {
      opportunityId, event, marketType, period, legs, totalStake,
      netRoiPct: null, netProfit: null,
      quoteAgeMs: null, liquidityStatus: "unknown", settlementStatus: "unverified",
    });
  }

  // ── Freshness ──
  const maxAge = event.status === "live" ? MAX_QUOTE_AGE_LIVE_MS : MAX_QUOTE_AGE_PREMATCH_MS;
  const ageA = nowMs - Date.parse(legA.providerTimestamp);
  const ageB = nowMs - Date.parse(legB.providerTimestamp);
  const quoteAgeMs = Math.max(ageA, ageB);
  const stale = Number.isFinite(quoteAgeMs) && quoteAgeMs > maxAge;
  if (stale) riskFlags.push("stale_quote");

  // ── Liquidity ──
  const liqA = legA.availableLiquidity;
  const liqB = legB.availableLiquidity;
  const liquidityStatus: ArbitrageOpportunity["liquidityStatus"] =
    liqA == null || liqB == null ? "unknown"
      : Math.min(liqA, liqB) < Math.max(alloc.stakes[0], alloc.stakes[1]) ? "insufficient"
      : "known";
  if (liquidityStatus !== "known") riskFlags.push(`liquidity_${liquidityStatus}`);

  // ── Settlement compatibility ──
  const settlementStatus: ArbitrageOpportunity["settlementStatus"] =
    legA.settlementRuleId && legA.settlementRuleId === legB.settlementRuleId
      ? "compatible" : "unverified";
  if (settlementStatus !== "compatible") riskFlags.push("settlement_review_required");

  // ── Market status ──
  if (legA.marketStatus !== "open" || legB.marketStatus !== "open") {
    riskFlags.push("market_suspended");
  }

  const verified =
    !stale &&
    liquidityStatus === "known" &&
    settlementStatus === "compatible" &&
    legA.marketStatus === "open" && legB.marketStatus === "open" &&
    legA.mappingConfidence >= 0.99 && legB.mappingConfidence >= 0.99;

  const classification: OpportunityClassification = verified ? "verified_arbitrage" : "conditional_arbitrage";

  // Net figures require every cost to be known. Polymarket carries no
  // separate commission today (commissionRate is 0, not "unknown" — it's a
  // fact about the venue), so net == theoretical once the legs are otherwise
  // clean; a future venue with a real fee schedule feeds it through here.
  const commissionKnown = legA.commissionRate != null && legB.commissionRate != null;
  const netRoiPct = commissionKnown && classification === "verified_arbitrage" ? theoreticalRoiPct : null;
  const netProfit = commissionKnown && classification === "verified_arbitrage" ? alloc.grossProfit : null;

  return build(classification, theoreticalRoiPct, alloc, riskFlags, {
    opportunityId, event, marketType, period, legs, totalStake,
    netRoiPct, netProfit, quoteAgeMs, liquidityStatus, settlementStatus,
  });
}

function build(
  classification: OpportunityClassification,
  theoreticalRoiPct: number,
  alloc: { grossProfit: number },
  riskFlags: string[],
  fields: {
    opportunityId: string; event: CanonicalEvent; marketType: NormalizedMarket["marketType"]; period: NormalizedMarket["period"];
    legs: ArbLeg[]; totalStake: number; netRoiPct: number | null; netProfit: number | null;
    quoteAgeMs: number | null; liquidityStatus: ArbitrageOpportunity["liquidityStatus"];
    settlementStatus: ArbitrageOpportunity["settlementStatus"];
  },
): ArbitrageOpportunity {
  return {
    opportunityId: fields.opportunityId,
    matchId: fields.event.matchId,
    event: fields.event,
    marketType: fields.marketType,
    period: fields.period,
    classification,
    legs: fields.legs,
    theoreticalRoiPct,
    netRoiPct: fields.netRoiPct,
    theoreticalProfit: alloc.grossProfit,
    netProfit: fields.netProfit,
    totalStake: fields.totalStake,
    quoteAgeMs: fields.quoteAgeMs,
    liquidityStatus: fields.liquidityStatus,
    settlementStatus: fields.settlementStatus,
    riskFlags,
    detectedAt: new Date().toISOString(),
  };
}
