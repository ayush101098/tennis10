import { describe, it, expect } from "vitest";
import {
  qualifies, quarterKellyStake, kellyFraction, EDGE_FLOOR, MAX_BANKROLL_FRACTION,
} from "../scheduleService";

/**
 * The homepage promises "¼ Kelly, 5% cap, 2% edge floor — no edge, no bet".
 * Nothing enforced that where a trade is actually recommended: the CTA tested
 * `kelly > 0`, which is true of ANY positive edge, so a 0.3% edge rendered a
 * green "RECOMMENDED / TAKE THIS TRADE". These lock the stated discipline.
 */
describe("staking discipline", () => {
  describe("edge floor", () => {
    it("rejects an edge under the floor even when Kelly is positive", () => {
      // 3% true edge on evens would pass; 0.3% must not.
      const kelly = kellyFraction(0.503, 2.0);
      expect(kelly).toBeGreaterThan(0);          // Kelly alone would say "bet"
      expect(qualifies(0.003, kelly)).toBe(false);
    });

    it("rejects an edge exactly under the floor", () => {
      expect(qualifies(EDGE_FLOOR - 0.0001, 0.1)).toBe(false);
    });

    it("accepts an edge exactly at the floor", () => {
      expect(qualifies(EDGE_FLOOR, 0.1)).toBe(true);
    });

    it("rejects a qualifying edge with no stake behind it", () => {
      expect(qualifies(0.10, 0)).toBe(false);
    });

    it("rejects negative edges", () => {
      expect(qualifies(-0.05, 0.1)).toBe(false);
    });
  });

  /**
   * The QA case asked for: when BOTH sides are under the floor, neither side
   * may qualify — the panel must render its "no trade" state.
   */
  it("both sides under the floor => no side qualifies", () => {
    const cases: [number, number][] = [
      [0.019, 0.005], [0.0, 0.0], [-0.03, 0.01], [0.0199, 0.0199],
    ];
    for (const [e1, e2] of cases) {
      const k1 = kellyFraction(0.5 + e1, 2.0);
      const k2 = kellyFraction(0.5 + e2, 2.0);
      expect(qualifies(e1, k1) || qualifies(e2, k2)).toBe(false);
    }
  });

  it("a side over the floor does qualify", () => {
    const k = kellyFraction(0.56, 2.0);
    expect(qualifies(0.06, k)).toBe(true);
  });

  describe("5% bankroll cap", () => {
    it("caps a huge Kelly at 5% of bankroll", () => {
      // 80% Kelly on a $1000 bankroll is $200 at ¼ Kelly — the cap is $50.
      expect(quarterKellyStake(1000, 0.8)).toBe(1000 * MAX_BANKROLL_FRACTION);
    });

    it("leaves a small stake untouched", () => {
      // ¼ of 4% of $1000 = $10, well under the $50 cap
      expect(quarterKellyStake(1000, 0.04)).toBe(10);
    });

    it("never exceeds the cap for any Kelly fraction", () => {
      for (let k = 0; k <= 1; k += 0.05) {
        expect(quarterKellyStake(500, k)).toBeLessThanOrEqual(500 * MAX_BANKROLL_FRACTION);
      }
    });
  });
});
