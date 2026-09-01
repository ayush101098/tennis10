import { describe, it, expect } from "vitest";
import {
  priceParlay, stressParlay, stressTolerance, correlationWarnings,
  MAX_STRESS_POINTS, type ParlayLeg,
} from "../parlay";

const leg = (over: Partial<ParlayLeg> = {}): ParlayLeg => ({
  matchId: "m1",
  player: "A",
  opponent: "B",
  tournament: "US Open",
  trueP: 0.6,
  marketP: 0.5,
  odds: 2,
  live: false,
  ...over,
});

describe("priceParlay", () => {
  it("returns null for an empty ticket rather than a 100% one", () => {
    // Multiplying nothing is 1.0; surfacing that as a certain winner would be
    // the worst possible default.
    expect(priceParlay([])).toBeNull();
  });

  it("multiplies probabilities and odds across legs", () => {
    const p = priceParlay([leg(), leg({ matchId: "m2" })])!;
    expect(p.trueP).toBeCloseTo(0.36, 6);   // 0.6 * 0.6
    expect(p.marketP).toBeCloseTo(0.25, 6); // 0.5 * 0.5
    expect(p.odds).toBeCloseTo(4, 6);
    expect(p.edge).toBeCloseTo(0.11, 6);
  });

  it("compounds a per-leg edge into a larger combined edge", () => {
    const one = priceParlay([leg()])!;
    const three = priceParlay([leg(), leg({ matchId: "m2" }), leg({ matchId: "m3" })])!;
    // 10pp per leg becomes 21.6% - 12.5% = 9.1pp of a much smaller number:
    // the RATIO of model to market grows, which is what Kelly reads.
    expect(three.trueP / three.marketP).toBeGreaterThan(one.trueP / one.marketP);
  });

  it("flags a ticket containing a live leg", () => {
    expect(priceParlay([leg(), leg({ matchId: "m2", live: true })])!.hasLive).toBe(true);
    expect(priceParlay([leg()])!.hasLive).toBe(false);
  });
});

describe("stressParlay", () => {
  it("is a no-op at zero points", () => {
    expect(stressParlay([leg()], 0)!.trueP).toBeCloseTo(0.6, 6);
  });

  it("knocks each leg down by the given points", () => {
    const s = stressParlay([leg(), leg({ matchId: "m2" })], 0.1)!;
    expect(s.trueP).toBeCloseTo(0.25, 6);   // 0.5 * 0.5
    expect(s.marketP).toBeCloseTo(0.25, 6); // market is untouched
    expect(s.edge).toBeCloseTo(0, 6);
  });

  it("destroys a multi-leg edge faster than a single-leg one", () => {
    // The whole warning the builder exists to deliver: the same per-leg error
    // costs a longer ticket proportionally more of its edge.
    const one = [leg()];
    const four = [leg(), leg({ matchId: "m2" }), leg({ matchId: "m3" }), leg({ matchId: "m4" })];
    const keep = (legs: ParlayLeg[]) =>
      stressParlay(legs, 0.05)!.edge / priceParlay(legs)!.edge;
    expect(keep(four)).toBeLessThan(keep(one));
  });

  it("never drives a leg to an impossible probability", () => {
    const s = stressParlay([leg({ trueP: 0.05 })], 0.5)!;
    expect(s.trueP).toBeGreaterThan(0);
    expect(s.trueP).toBeLessThan(1);
  });

  it("treats negative points as zero rather than inflating the model", () => {
    expect(stressParlay([leg()], -0.2)!.trueP).toBeCloseTo(0.6, 6);
  });
});

describe("stressTolerance", () => {
  it("reports how many points of per-leg error the ticket absorbs", () => {
    const legs = [leg(), leg({ matchId: "m2" })];
    const tol = stressTolerance(legs)!;
    expect(tol).toBeGreaterThan(0);
    expect(tol).toBeLessThan(MAX_STRESS_POINTS);
    // Just inside is +EV, just outside is not — that is what "tolerance" means.
    expect(stressParlay(legs, tol - 0.005)!.edge).toBeGreaterThan(0);
    expect(stressParlay(legs, tol + 0.005)!.edge).toBeLessThanOrEqual(0);
  });

  it("equals the common per-leg gap, INDEPENDENT of ticket length", () => {
    // Not the intuition one might expect, and worth pinning down: if every leg
    // is g points above its market price, then knocking g points off each leg
    // reproduces the market product exactly, whatever the leg count. So
    // tolerance measures per-leg model error, not ticket length — the
    // length risk shows up in `stressParlay`'s proportional edge loss and in
    // the collapsing win probability, not here.
    const g = 0.1;   // leg() is trueP 0.6 vs marketP 0.5
    for (const n of [1, 2, 3, 5]) {
      const legs = Array.from({ length: n }, (_, i) => leg({ matchId: `m${i}` }));
      expect(stressTolerance(legs)!).toBeCloseTo(g, 3);
    }
  });

  it("is dragged down by the thinnest leg on the ticket", () => {
    const strong = [leg({ trueP: 0.7, marketP: 0.5 })];
    const withWeak = [...strong, leg({ matchId: "m2", trueP: 0.52, marketP: 0.5 })];
    expect(stressTolerance(withWeak)!).toBeLessThan(stressTolerance(strong)!);
  });

  it("is 0 when the ticket has no edge to lose", () => {
    expect(stressTolerance([leg({ trueP: 0.4, marketP: 0.5 })])).toBe(0);
  });

  it("returns null for an empty ticket", () => {
    expect(stressTolerance([])).toBeNull();
  });
});

describe("correlationWarnings", () => {
  it("rejects two legs from the same match", () => {
    const w = correlationWarnings([leg(), leg()]);
    expect(w.some(s => /same match/i.test(s))).toBe(true);
  });

  it("warns on multiple legs from one draw", () => {
    const w = correlationWarnings([leg(), leg({ matchId: "m2" })]);
    expect(w.some(s => /later round/i.test(s))).toBe(true);
  });

  it("warns that a live leg moves while you fill the others", () => {
    const w = correlationWarnings([leg({ live: true })]);
    expect(w.some(s => /every point/i.test(s))).toBe(true);
  });

  it("is silent for independent legs in different events", () => {
    expect(correlationWarnings([
      leg({ matchId: "m1", tournament: "US Open" }),
      leg({ matchId: "m2", tournament: "Challenger Como" }),
    ])).toEqual([]);
  });
});
