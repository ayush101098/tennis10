import { describe, it, expect } from "vitest";
import { impliedProbSum, arbPercent, allocateStakes, effectiveOddsWithCommission, netPayoffVector, roundToIncrement } from "../math";

describe("impliedProbSum / arbPercent", () => {
  it("matches the spec's worked example (2.10 / 2.05 -> ~3.72%)", () => {
    const s = impliedProbSum([2.10, 2.05]);
    expect(s).toBeCloseTo(0.9642, 3);
    expect(arbPercent(s)).toBeCloseTo(3.72, 1);
  });

  it("S >= 1 means no arbitrage", () => {
    const s = impliedProbSum([1.90, 1.90]);
    expect(s).toBeGreaterThan(1);
    expect(arbPercent(s)).toBeLessThan(0);
  });

  it("three-outcome market", () => {
    const s = impliedProbSum([3.2, 3.4, 3.6]);
    expect(s).toBeCloseTo(1 / 3.2 + 1 / 3.4 + 1 / 3.6, 6);
  });
});

describe("allocateStakes", () => {
  // The spec's own worked dollar figures ($492.80 / $507.20) don't actually
  // satisfy its own stated property (equal payout across both legs) —
  // 492.80*2.10=1034.88 vs 507.20*2.05=1039.76, a $4.88 gap. Rather than
  // reproduce that arithmetic error, this asserts the formula's actual
  // defining property (equal payout, ROI/profit matching the spec's own
  // ~3.72%/~$37.20 figures, which DO check out) and treats that as correct.
  it("matches the spec's stated ROI/profit (~3.72%, ~$37.20) for 2.10/2.05", () => {
    const alloc = allocateStakes([2.10, 2.05], 1000);
    expect(alloc.roiPct).toBeCloseTo(3.72, 1);
    expect(alloc.grossProfit).toBeCloseTo(37.2, 0);
    expect(alloc.stakes[0] + alloc.stakes[1]).toBeCloseTo(1000, 6);
  });

  it("equalizes payout across every outcome", () => {
    const alloc = allocateStakes([2.10, 2.05], 1000);
    const payout0 = alloc.stakes[0] * 2.10;
    const payout1 = alloc.stakes[1] * 2.05;
    expect(payout0).toBeCloseTo(payout1, 6);
  });

  it("a losing combination still allocates but with negative ROI", () => {
    const alloc = allocateStakes([1.80, 1.80], 1000);
    expect(alloc.roiPct).toBeLessThan(0);
    expect(alloc.grossProfit).toBeLessThan(0);
  });
});

describe("effectiveOddsWithCommission", () => {
  it("2.00 odds at 2% commission reduces net return correctly", () => {
    // Win $1 profit per $1 stake at 2.00; 2% commission takes $0.02 off that
    // profit, so effective decimal odds = 1 + 0.98 = 1.98.
    expect(effectiveOddsWithCommission(2.00, 0.02)).toBeCloseTo(1.98, 6);
  });

  it("0% commission is a no-op", () => {
    expect(effectiveOddsWithCommission(3.5, 0)).toBeCloseTo(3.5, 6);
  });

  it("commission never changes the stake return, only the profit portion", () => {
    const eff = effectiveOddsWithCommission(2.5, 0.1);
    // profit portion is (2.5-1)=1.5; 10% commission -> 1.35 profit + 1 stake = 2.35
    expect(eff).toBeCloseTo(2.35, 6);
  });
});

describe("a candidate that looks profitable gross but is negative after costs", () => {
  it("commission can erase a thin arbitrage margin", () => {
    const odds: [number, number] = [2.05, 2.00];
    const grossS = impliedProbSum(odds);
    expect(arbPercent(grossS)).toBeGreaterThan(0); // gross: still an "arbitrage"

    const effOdds: [number, number] = [
      effectiveOddsWithCommission(odds[0], 0.05),
      effectiveOddsWithCommission(odds[1], 0.05),
    ];
    const netS = impliedProbSum(effOdds);
    expect(arbPercent(netS)).toBeLessThan(0); // net: commission wiped it out
  });
});

describe("netPayoffVector", () => {
  it("a true two-outcome arbitrage nets strictly positive under every outcome", () => {
    const alloc = allocateStakes([2.10, 2.05], 1000);
    // payoff[j][k]: position j's decimal return if outcome k occurs
    const payoff = [
      [2.10, 0], // position 0 (backing player A) pays 2.10x if A wins, 0 if B wins
      [0, 2.05], // position 1 (backing player B) pays 2.05x if B wins, 0 if A wins
    ];
    const net = netPayoffVector(alloc.stakes, payoff);
    expect(net[0]).toBeGreaterThan(0);
    expect(net[1]).toBeGreaterThan(0);
    expect(net[0]).toBeCloseTo(net[1], 6); // equal-payout allocation
  });

  it("an unknown cost, if passed as a real number, is subtracted from every outcome", () => {
    const stakes = [500, 500];
    const payoff = [[2, 0], [0, 2]];
    const net = netPayoffVector(stakes, payoff, [10, 0]);
    expect(net[0]).toBeCloseTo(1000 - 10, 6);
    expect(net[1]).toBeCloseTo(1000 - 10, 6);
  });
});

describe("roundToIncrement", () => {
  it("rounds to the nearest dollar", () => {
    expect(roundToIncrement(492.8, 1)).toBe(493);
  });
  it("increment 0 is a no-op (unknown increment, don't fabricate one)", () => {
    expect(roundToIncrement(492.837, 0)).toBeCloseTo(492.837, 6);
  });
});
