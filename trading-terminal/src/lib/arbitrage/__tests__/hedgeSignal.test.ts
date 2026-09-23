import { describe, it, expect } from "vitest";
import { evaluateArbHedge, vanishedHedgeSignal } from "../hedgeSignal";

const arb = { classification: "verified_arbitrage" as const };
const model = { classification: "model_positive_ev" as const };

describe("evaluateArbHedge", () => {
  it("HOLDs when ROI is steady", () => {
    const sig = evaluateArbHedge(arb, 3.7, 3.6);
    expect(sig.action).toBe("HOLD");
  });

  it("HOLDs when ROI improved", () => {
    const sig = evaluateArbHedge(arb, 3.0, 4.5);
    expect(sig.action).toBe("HOLD");
    expect(sig.driftPct).toBeGreaterThan(0);
  });

  it("HEDGEs on a material (>40%) relative ROI drop that's still positive", () => {
    const sig = evaluateArbHedge(arb, 10, 5); // 50% drop
    expect(sig.action).toBe("HEDGE");
  });

  it("STOPs when ROI has gone to zero or negative", () => {
    const sig = evaluateArbHedge(arb, 5, -1);
    expect(sig.action).toBe("STOP");
  });

  it("STOPs exactly at zero, not just below it", () => {
    const sig = evaluateArbHedge(arb, 5, 0);
    expect(sig.action).toBe("STOP");
  });

  it("a small drop within the HEDGE threshold still HOLDs", () => {
    const sig = evaluateArbHedge(arb, 10, 7); // 30% drop, under the 40% bar
    expect(sig.action).toBe("HOLD");
  });

  it("uses softer, non-guarantee language for model_positive_ev", () => {
    const sig = evaluateArbHedge(model, 10, 0);
    expect(sig.action).toBe("STOP");
    expect(sig.reason.toLowerCase()).not.toContain("arbitrage");
  });
});

describe("vanishedHedgeSignal", () => {
  it("is always STOP, with the drift computed against the last known ROI", () => {
    const sig = vanishedHedgeSignal(6.5);
    expect(sig.action).toBe("STOP");
    expect(sig.currentRoiPct).toBe(0);
    expect(sig.driftPct).toBeCloseTo(-6.5, 6);
  });
});
