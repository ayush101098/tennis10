import { describe, it, expect } from "vitest";
import { gameTree, pointIndices, nextStates, marketEdge } from "../gameTree";

/**
 * PRD §37 acceptance criteria for the probability layer, plus the agreements
 * that stop this engine drifting from the two that already exist.
 */

const P_GRID = [0.5, 0.55, 0.6, 0.62, 0.65, 0.7, 0.75, 0.8];
/** Every legal game score, as index pairs. */
const STATES: [number, number][] = [];
for (let s = 0; s <= 3; s++) for (let r = 0; r <= 3; r++) STATES.push([s, r]);
STATES.push([4, 3], [3, 4]);   // advantage server / advantage returner

describe("§15 joint probability — hard invariants", () => {
  it("the four outcomes sum to exactly 1 from every state", () => {
    for (const p of P_GRID) {
      for (const [s, r] of STATES) {
        const t = gameTree(p, s, r);
        expect(t.aNoD + t.aD + t.bNoD + t.bD).toBeCloseTo(1, 12);
      }
    }
  });

  it("P(A) = P(A,D) + P(A,¬D) and P(D) = P(A,D) + P(B,D)", () => {
    for (const p of P_GRID) {
      for (const [s, r] of STATES) {
        const t = gameTree(p, s, r);
        expect(t.pServer).toBeCloseTo(t.aD + t.aNoD, 12);
        expect(t.pDeuce).toBeCloseTo(t.aD + t.bD, 12);
        expect(t.pServer + t.pReturner).toBeCloseTo(1, 12);
      }
    }
  });

  it("no outcome is ever negative", () => {
    for (const p of P_GRID) {
      for (const [s, r] of STATES) {
        const t = gameTree(p, s, r);
        for (const v of [t.aNoD, t.aD, t.bNoD, t.bD]) expect(v).toBeGreaterThanOrEqual(0);
      }
    }
  });
});

describe("§13–§14 Markov game model — known closed forms", () => {
  it("matches Barnett–Clarke from 0-0", () => {
    // pre = p⁴ + 4p⁴q + 10p⁴q², deuce = 20p³q³, winDeuce = p²/(p²+q²)
    for (const p of P_GRID) {
      const q = 1 - p;
      const pre = p ** 4 + 4 * p ** 4 * q + 10 * p ** 4 * q * q;
      const deuce = 20 * p ** 3 * q ** 3;
      const winDeuce = (p * p) / (p * p + q * q);
      const t = gameTree(p, 0, 0);
      expect(t.pServer).toBeCloseTo(pre + deuce * winDeuce, 12);
      expect(t.pDeuce).toBeCloseTo(deuce, 12);
      // The split momentumEngine.gameWinProb computes and then discards:
      expect(t.aNoD).toBeCloseTo(pre, 12);
      expect(t.aD).toBeCloseTo(deuce * winDeuce, 12);
    }
  });

  it("a game standing at deuce has already reached deuce", () => {
    const t = gameTree(0.65, 3, 3);
    expect(t.pDeuce).toBe(1);
    expect(t.aNoD).toBe(0);
    expect(t.bNoD).toBe(0);
    expect(t.pServer).toBeCloseTo((0.65 * 0.65) / (0.65 ** 2 + 0.35 ** 2), 12);
  });

  it("advantage states are past deuce, so P(deuce) stays 1", () => {
    // This is the settlement rule a "will it reach deuce" market uses: once
    // 40-40 has appeared the market is YES, whatever happens afterwards.
    expect(gameTree(0.65, 4, 3).pDeuce).toBe(1);
    expect(gameTree(0.65, 3, 4).pDeuce).toBe(1);
  });

  it("a game won from 40-0 never touched deuce", () => {
    const t = gameTree(0.65, 3, 0);
    expect(t.aD).toBeLessThan(t.aNoD);
    // 40-0 -> the only route to deuce is losing three straight points.
    expect(t.pDeuce).toBeCloseTo((1 - 0.65) ** 3, 12);
  });

  it("from 40-30 the game reaches deuce exactly when the returner wins the point", () => {
    for (const p of P_GRID) expect(gameTree(p, 3, 2).pDeuce).toBeCloseTo(1 - p, 12);
  });

  it("a decided game is fully resolved", () => {
    expect(gameTree(0.6, 4, 0).pServer).toBe(1);
    expect(gameTree(0.6, 4, 0).pDeuce).toBe(0);
    expect(gameTree(0.6, 0, 4).pReturner).toBe(1);
    // Won 5-3 from advantage: the loser stood on 40, so deuce happened.
    expect(gameTree(0.6, 5, 3).pDeuce).toBe(1);
  });

  it("is symmetric: swapping p and the score mirrors the outcome", () => {
    for (const p of P_GRID) {
      const a = gameTree(p, 2, 1);
      const b = gameTree(1 - p, 1, 2);
      expect(a.pServer).toBeCloseTo(b.pReturner, 12);
      expect(a.pDeuce).toBeCloseTo(b.pDeuce, 12);
    }
  });

  it("break probability is the returner's game probability, not a separate number", () => {
    const t = gameTree(0.62, 2, 2);
    expect(t.pBreak).toBe(t.pReturner);
  });
});

describe("§19 state transitions", () => {
  it("the two branches reconcile to the current state", () => {
    for (const p of P_GRID) {
      for (const [s, r] of STATES.filter(([a, b]) => a < 3 || b < 3)) {
        const here = gameTree(p, s, r);
        const { onServerPoint, onReturnerPoint } = nextStates(p, s, r);
        const recombined =
          onServerPoint.prob * onServerPoint.tree.pServer +
          onReturnerPoint.prob * onReturnerPoint.tree.pServer;
        expect(recombined).toBeCloseTo(here.pServer, 12);
      }
    }
  });
});

describe("score parsing", () => {
  it("reads ordinary and advantage scores", () => {
    expect(pointIndices("30", "40")).toEqual({ srvPts: 2, retPts: 3 });
    expect(pointIndices("A", "40")).toEqual({ srvPts: 4, retPts: 3 });
    expect(pointIndices("0", "0")).toEqual({ srvPts: 0, retPts: 0 });
  });

  it("refuses tiebreak scores rather than reading them as game points", () => {
    // "7" is not a game point; routing it here would price a tiebreak with the
    // game model and silently produce a number that looks plausible.
    expect(pointIndices("7", "5")).toBeNull();
    expect(pointIndices("1", "2")).toBeNull();
  });

  it("refuses states tennis cannot produce", () => {
    expect(pointIndices("A", "A")).toBeNull();
    expect(pointIndices("A", "30")).toBeNull();
    expect(pointIndices(undefined, "30")).toBeNull();
  });
});

describe("§17–§18 market edge", () => {
  it("de-vigs a two-sided book before measuring edge", () => {
    // 1.82 / 2.10 -> raw 0.54945 + 0.47619 = 1.02564 (2.56% overround)
    const r = marketEdge(0.64, 1 / 1.82, 1 / 2.1)!;
    expect(r.overround).toBeCloseTo(0.02564, 4);
    expect(r.fairA + r.fairB).toBeCloseTo(1, 12);
    expect(r.fairA).toBeCloseTo(0.53571, 4);
    // Edge against the RAW price would read 0.64 - 0.54945 = +9.06%; against
    // the de-vigged price it is +10.43%. Skipping the de-vig understates the
    // edge here, and overstates it on the other side of the same book.
    expect(r.edge).toBeCloseTo(0.10429, 4);
  });

  it("refuses a pair that is not a coherent two-way book", () => {
    expect(marketEdge(0.6, 0.55, 0.05)).toBeNull();
    expect(marketEdge(0.6, 0.9, 0.9)).toBeNull();
    expect(marketEdge(0.6, 0, 0.5)).toBeNull();
  });
});

describe("agreement with the engines already in the codebase", () => {
  it("reproduces breakHoldEngine's recursive gameWinProb at every state", () => {
    // Local copy of breakHoldEngine.ts:932 — if this ever diverges, the board
    // and the backtester have started disagreeing about the same game.
    const legacy = (p: number, a: number, b: number): number => {
      if (a >= 4 && a - b >= 2) return 1;
      if (b >= 4 && b - a >= 2) return 0;
      if (a >= 3 && b >= 3) {
        const d = a - b;
        if (d === 0) return (p * p) / (p * p + (1 - p) * (1 - p));
        if (d === 1) return p + (1 - p) * (p * p) / (p * p + (1 - p) * (1 - p));
        return 0;
      }
      return p * legacy(p, a + 1, b) + (1 - p) * legacy(p, a, b + 1);
    };
    for (const p of P_GRID) {
      for (const [s, r] of STATES) {
        // The legacy version returns 0 at advantage-returner rather than the
        // server's real chance of climbing back, so it is compared only where
        // it is defined.
        if (s >= 3 && r >= 3 && s - r < 0) continue;
        expect(gameTree(p, s, r).pServer).toBeCloseTo(legacy(p, s, r), 12);
      }
    }
  });
});
