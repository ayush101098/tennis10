import { describe, it, expect, beforeEach } from "vitest";
import { observePressure, pressureSignal, resetPressure, type PressureObservation } from "../gamePressure";

/**
 * The engine reads polled snapshots, so every test here is a SEQUENCE of
 * snapshots rather than a single call — the sequence is the thing being tested.
 */

const M = "match-1";
let games: [number, number] = [0, 0];
let set = 1;

function snap(server: 1 | 2, srvPts: number, retPts: number, over?: Partial<PressureObservation>) {
  return observePressure({
    matchId: M, server, srvPts, retPts,
    gamesP1: games[0], gamesP2: games[1], setIndex: set, ...over,
  });
}
/** Advance the game score, as the feed would after a game is decided. */
function gameTo(g1: number, g2: number) { games = [g1, g2]; }

beforeEach(() => { resetPressure(); games = [0, 0]; set = 1; });

describe("break-point counting", () => {
  it("counts break points once, not once per poll", () => {
    snap(1, 2, 3);                       // 30-40, a break point
    snap(1, 2, 3);                       // same score polled again
    snap(1, 2, 3);
    const s = snap(1, 2, 3);
    expect(s.bpThisGame).toBe(1);        // NOT 4
    expect(s.atBreakPoint).toBe(true);
  });

  it("counts repeated break points in one game", () => {
    snap(1, 2, 3);                       // 30-40  -> BP 1
    snap(1, 3, 3);                       // deuce  -> saved
    snap(1, 3, 4);                       // AD-OUT -> BP 2
    snap(1, 3, 3);                       // deuce  -> saved
    const s = snap(1, 3, 4);             // AD-OUT -> BP 3
    expect(s.bpThisGame).toBe(3);
    expect(s.bpSavedThisGame).toBe(2);
    expect(s.deuceThisGame).toBe(2);
  });

  it("treats advantage-returner as a break point", () => {
    snap(1, 3, 3);
    const s = snap(1, 3, 4);
    expect(s.atBreakPoint).toBe(true);
    expect(s.bpThisGame).toBe(1);
  });

  it("does not count break points in a tiebreak", () => {
    const s = snap(1, 2, 5, { isTiebreak: true });
    expect(s.bpThisGame).toBe(0);
    expect(s.atBreakPoint).toBe(false);
  });
});

describe("break detection across games", () => {
  it("records a break when the returner takes the game", () => {
    snap(1, 0, 0);                       // P1 serving at 0-0
    snap(1, 0, 3);                       // 0-40
    gameTo(0, 1);                        // P2 won it -> P1 was broken
    const s = snap(2, 0, 0);             // P2 now serving
    expect(s.breaksThisSet).toBe(1);
    expect(s.prevGameWasBreak).toBe(true);
  });

  it("does not record a break when the server holds", () => {
    snap(1, 0, 0);
    snap(1, 3, 0);
    gameTo(1, 0);                        // P1 held
    const s = snap(2, 0, 0);
    expect(s.breaksThisSet).toBe(0);
    expect(s.prevGameWasBreak).toBe(false);
  });

  it("a server winning their own game is a hold, not a break", () => {
    // The game count that moves is the WINNER's, not the server's. Reading it
    // the other way turns every hold into a break and doubles the break count.
    snap(2, 0, 0);                       // P2 serving
    snap(2, 3, 0);                       // 40-0
    gameTo(0, 1);                        // P2's count moves -> P2 held
    const s = snap(1, 0, 0);
    expect(s.breaksThisSet).toBe(0);
    expect(s.prevGameWasBreak).toBe(false);
  });

  it("refuses to guess when whole games were missed between polls", () => {
    snap(1, 0, 0);
    gameTo(3, 1);                        // four games appeared at once
    const s = snap(2, 0, 0);
    // Nothing can be said about who won what, so nothing is recorded.
    expect(s.breaksThisSet).toBe(0);
    expect(s.prevGameWasBreak).toBe(false);
  });

  it("clears per-set counters at a new set", () => {
    snap(1, 0, 0); snap(1, 0, 3);
    gameTo(0, 1); snap(2, 0, 0);
    expect(observePressure({ matchId: M, server: 1, srvPts: 0, retPts: 0,
      gamesP1: 0, gamesP2: 0, setIndex: 2 }).breaksThisSet).toBe(0);
  });
});

describe("GPI accumulation and decay", () => {
  it("rises with pressure and is zero on a quiet game", () => {
    const quiet = snap(1, 1, 0);         // 15-0
    expect(quiet.gpi).toBe(0);
    snap(1, 3, 3);                       // deuce
    const s = snap(1, 3, 4);             // break point
    expect(s.gpi).toBeGreaterThan(0);
  });

  it("decays across a game boundary rather than accumulating forever", () => {
    snap(1, 3, 3); snap(1, 3, 4); snap(1, 3, 3); snap(1, 3, 4);
    const before = (snap(1, 3, 3)).gpi;
    gameTo(1, 0);                        // server survived; new game
    const after = snap(2, 0, 0).gpi;
    expect(after).toBeLessThan(before);
    expect(after).toBeGreaterThan(0);    // but the history is not erased
  });

  it("escalates the regime with breaks", () => {
    snap(1, 0, 0); snap(1, 0, 3);
    gameTo(0, 1); expect(snap(2, 0, 0).regime).toBe("ELEVATED");   // P2 took P1's serve
    snap(2, 0, 3);
    gameTo(1, 1);                        // P1 takes P2's serve — second break of the set
    expect(snap(1, 0, 0).regime).toBe("DOMINANCE");
  });
});

describe("the entry hierarchy", () => {
  const P = 0.62;

  it("plain deuce is a watch, never an entry", () => {
    snap(1, 3, 3);
    const sig = pressureSignal(observePressure({ matchId: M, server: 1, srvPts: 3, retPts: 3,
      gamesP1: 0, gamesP2: 0, setIndex: 1 }), P, 3, 3);
    expect(sig.tier).toBe("S1_WATCH");
    expect(sig.headline).toMatch(/no edge on its own/i);
  });

  it("repeated break points reach S3", () => {
    snap(1, 2, 3); snap(1, 3, 3);
    const s = snap(1, 3, 4);
    expect(pressureSignal(s, P, 3, 4).tier).toBe("S3_STRONG");
  });

  it("a break point right after a break is the strongest state", () => {
    snap(1, 0, 0); snap(1, 0, 3);
    gameTo(0, 1);
    snap(2, 0, 0);
    const s = snap(2, 2, 3);             // new server immediately facing a BP
    const sig = pressureSignal(s, P, 2, 3);
    expect(sig.tier).toBe("S5_EXTREME");
    expect(sig.evidence).toContain("previous game was a break");
  });

  it("two breaks in a set switches to the regime signal, which asks for confirmation", () => {
    snap(1, 0, 0); snap(1, 0, 3); gameTo(0, 1);   // P1 broken
    snap(2, 0, 0); snap(2, 0, 3); gameTo(1, 1);   // P2 broken
    const s = snap(1, 1, 1);
    const sig = pressureSignal(s, P, 1, 1);
    expect(sig.tier).toBe("S6_REGIME");
    expect(sig.headline).toMatch(/confirmation/i);
  });

  it("quiet play produces no signal at all", () => {
    const s = snap(1, 1, 0);
    expect(pressureSignal(s, P, 1, 0).tier).toBe("NONE");
  });
});

describe("probability honesty", () => {
  it("flags a probability outside the calibrated band as untrustworthy", () => {
    const s = snap(1, 3, 0);            // 40-0: model says break is very unlikely
    const sig = pressureSignal(s, 0.7, 3, 0);
    expect(sig.breakProb).toBeLessThan(0.28);
    expect(sig.trustworthy).toBe(false);
  });

  it("accepts a probability inside the band", () => {
    const s = snap(1, 3, 3);
    const sig = pressureSignal(s, 0.55, 3, 3);
    expect(sig.trustworthy).toBe(true);
  });
});
