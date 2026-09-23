import { describe, it, expect } from "vitest";
import { classifyTwoWay } from "../classify";
import type { CanonicalEvent, NormalizedMarket } from "../types";

const event: CanonicalEvent = {
  matchId: "m1",
  player1: "Alcaraz C.",
  player2: "Sinner J.",
  tournament: "US Open",
  tour: "ATP",
  round: "Final",
  surface: "Hard",
  bestOf: 5,
  startTimestamp: Math.floor(Date.now() / 1000),
  status: "scheduled",
};

function leg(over: Partial<NormalizedMarket>): NormalizedMarket {
  return {
    marketId: "leg",
    eventId: "m1",
    providerId: "polymarket",
    marketType: "match_winner",
    period: "full_match",
    selection: "player1",
    oddsDecimal: 2.0,
    currency: "USD",
    marketStatus: "open",
    availableLiquidity: null,
    maxStake: null,
    commissionRate: 0,
    providerTimestamp: new Date().toISOString(),
    receivedTimestamp: new Date().toISOString(),
    mappingConfidence: 1.0,
    settlementRuleId: "polymarket_binary_v1",
    isModel: false,
    ...over,
  };
}

describe("classifyTwoWay — verified arbitrage", () => {
  it("clean, fresh, liquid, matching-settlement legs with S < 1 verify", () => {
    const legA = leg({ selection: "player1", oddsDecimal: 2.10, availableLiquidity: 1000 });
    const legB = leg({ selection: "player2", oddsDecimal: 2.05, availableLiquidity: 1000 });
    const opp = classifyTwoWay(event, "match_winner", "full_match", legA, legB, 1000)!;
    expect(opp.classification).toBe("verified_arbitrage");
    expect(opp.theoreticalRoiPct).toBeGreaterThan(0);
    expect(opp.netRoiPct).not.toBeNull();
    expect(opp.netProfit).not.toBeNull();
    expect(opp.riskFlags).toHaveLength(0);
  });

  it("every leg nets strictly positive stake*odds under its own outcome", () => {
    const legA = leg({ selection: "player1", oddsDecimal: 2.10, availableLiquidity: 1000 });
    const legB = leg({ selection: "player2", oddsDecimal: 2.05, availableLiquidity: 1000 });
    const opp = classifyTwoWay(event, "match_winner", "full_match", legA, legB, 1000)!;
    for (const l of opp.legs) {
      expect(l.recommendedStake * l.odds).toBeGreaterThan(opp.totalStake);
    }
  });
});

describe("classifyTwoWay — conditional arbitrage", () => {
  it("unknown liquidity downgrades to conditional", () => {
    const legA = leg({ selection: "player1", oddsDecimal: 2.10, availableLiquidity: null });
    const legB = leg({ selection: "player2", oddsDecimal: 2.05, availableLiquidity: null });
    const opp = classifyTwoWay(event, "match_winner", "full_match", legA, legB, 1000)!;
    expect(opp.classification).toBe("conditional_arbitrage");
    expect(opp.liquidityStatus).toBe("unknown");
    expect(opp.netRoiPct).toBeNull();
    expect(opp.riskFlags).toContain("liquidity_unknown");
  });

  it("insufficient liquidity is flagged and downgraded, not silently ignored", () => {
    const legA = leg({ selection: "player1", oddsDecimal: 2.10, availableLiquidity: 10 });
    const legB = leg({ selection: "player2", oddsDecimal: 2.05, availableLiquidity: 1000 });
    const opp = classifyTwoWay(event, "match_winner", "full_match", legA, legB, 1000)!;
    expect(opp.classification).toBe("conditional_arbitrage");
    expect(opp.liquidityStatus).toBe("insufficient");
  });

  it("a stale quote is flagged and downgraded", () => {
    const stale = new Date(Date.now() - 60_000).toISOString();
    const legA = leg({ selection: "player1", oddsDecimal: 2.10, availableLiquidity: 1000, providerTimestamp: stale });
    const legB = leg({ selection: "player2", oddsDecimal: 2.05, availableLiquidity: 1000 });
    const liveEvent = { ...event, status: "live" as const };
    const opp = classifyTwoWay(liveEvent, "match_winner", "full_match", legA, legB, 1000)!;
    expect(opp.classification).toBe("conditional_arbitrage");
    expect(opp.riskFlags).toContain("stale_quote");
  });

  it("mismatched settlement rules require review rather than being assumed compatible", () => {
    const legA = leg({ selection: "player1", oddsDecimal: 2.10, availableLiquidity: 1000, settlementRuleId: "rule_a" });
    const legB = leg({ selection: "player2", oddsDecimal: 2.05, availableLiquidity: 1000, settlementRuleId: "rule_b" });
    const opp = classifyTwoWay(event, "match_winner", "full_match", legA, legB, 1000)!;
    expect(opp.classification).toBe("conditional_arbitrage");
    expect(opp.settlementStatus).toBe("unverified");
    expect(opp.riskFlags).toContain("settlement_review_required");
  });

  it("a suspended leg is flagged and never verified", () => {
    const legA = leg({ selection: "player1", oddsDecimal: 2.10, availableLiquidity: 1000, marketStatus: "suspended" });
    const legB = leg({ selection: "player2", oddsDecimal: 2.05, availableLiquidity: 1000 });
    const opp = classifyTwoWay(event, "match_winner", "full_match", legA, legB, 1000)!;
    expect(opp.classification).not.toBe("verified_arbitrage");
    expect(opp.riskFlags).toContain("market_suspended");
  });
});

describe("classifyTwoWay — model legs never count as arbitrage", () => {
  it("one model leg + one market leg is model_positive_ev, never arbitrage, even with S < 1", () => {
    const modelLeg = leg({ selection: "player1", oddsDecimal: 2.50, isModel: true, providerId: "tennisalpha_model" });
    const marketLeg = leg({ selection: "player2", oddsDecimal: 2.05, availableLiquidity: 1000 });
    const opp = classifyTwoWay(event, "match_winner", "full_match", modelLeg, marketLeg, 1000)!;
    expect(opp.classification).toBe("model_positive_ev");
    expect(opp.classification).not.toBe("verified_arbitrage");
    expect(opp.classification).not.toBe("conditional_arbitrage");
    expect(opp.netRoiPct).toBeNull();
  });

  it("two model legs are also model_positive_ev", () => {
    const m1 = leg({ selection: "player1", oddsDecimal: 2.2, isModel: true, providerId: "tennisalpha_model" });
    const m2 = leg({ selection: "player2", oddsDecimal: 2.2, isModel: true, providerId: "tennisalpha_model" });
    const opp = classifyTwoWay(event, "match_winner", "full_match", m1, m2, 1000)!;
    expect(opp.classification).toBe("model_positive_ev");
  });

  it("an implausibly large model-vs-market edge is flagged as suspect, not presented clean", () => {
    // Reproduces a real case caught live: an unranked ITF match's model prior
    // defaulted to treating one player as a ~99% underdog (odds 87.52) while
    // the market priced the other side at 1.64 — a ~61% "edge" that is almost
    // certainly a missing-prior artifact, not a real signal.
    const modelLeg = leg({ selection: "player2", oddsDecimal: 87.52, isModel: true, providerId: "tennisalpha_model" });
    const marketLeg = leg({ selection: "player1", oddsDecimal: 1.64, availableLiquidity: 1000 });
    const opp = classifyTwoWay(event, "match_winner", "full_match", modelLeg, marketLeg, 1000)!;
    expect(opp.classification).toBe("model_positive_ev");
    expect(opp.theoreticalRoiPct).toBeGreaterThan(50);
    expect(opp.riskFlags).toContain("suspect_edge_magnitude");
  });
});

describe("classifyTwoWay — invalid", () => {
  it("S >= 1 (no positive spread) is invalid, not conditional", () => {
    const legA = leg({ selection: "player1", oddsDecimal: 1.80, availableLiquidity: 1000 });
    const legB = leg({ selection: "player2", oddsDecimal: 1.80, availableLiquidity: 1000 });
    const opp = classifyTwoWay(event, "match_winner", "full_match", legA, legB, 1000)!;
    expect(opp.classification).toBe("invalid");
  });
});

describe("classifyTwoWay — matching-layer guards", () => {
  it("returns null for legs on different events", () => {
    const legA = leg({ selection: "player1", eventId: "m1" });
    const legB = leg({ selection: "player2", eventId: "m2" });
    expect(classifyTwoWay(event, "match_winner", "full_match", legA, legB, 1000)).toBeNull();
  });

  it("returns null for legs on the same selection (not mutually exclusive coverage)", () => {
    const legA = leg({ selection: "player1" });
    const legB = leg({ selection: "player1" });
    expect(classifyTwoWay(event, "match_winner", "full_match", legA, legB, 1000)).toBeNull();
  });

  it("returns null when a leg's period doesn't match the requested period", () => {
    const legA = leg({ selection: "player1", period: "set1" });
    const legB = leg({ selection: "player2" });
    expect(classifyTwoWay(event, "match_winner", "full_match", legA, legB, 1000)).toBeNull();
  });
});
