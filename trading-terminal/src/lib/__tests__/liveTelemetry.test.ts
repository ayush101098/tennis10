import { describe, it, expect, beforeEach } from "vitest";
import {
  observeTelemetry, momentum, workload, resetTelemetry,
  MIN_POINTS_FOR_MOMENTUM,
} from "../liveTelemetry";

const M = "m1";
let g: [number, number] = [0, 0];
let set = 1;

const snap = (p1: number, p2: number, tb = false) =>
  observeTelemetry({ matchId: M, p1Pts: p1, p2Pts: p2, gamesP1: g[0], gamesP2: g[1], setIndex: set, isTiebreak: tb });

beforeEach(() => { resetTelemetry(); g = [0, 0]; set = 1; });

describe("deriving points from the score", () => {
  it("counts a point when the score advances", () => {
    snap(0, 0);
    const t = snap(1, 0);
    expect(t.observedPoints).toBe(1);
    expect(t.p1Points).toBe(1);
  });

  it("does not count a repeated poll of the same score", () => {
    snap(0, 0); snap(1, 0); snap(1, 0); snap(1, 0);
    expect(snap(1, 0).observedPoints).toBe(1);
  });

  it("attributes a whole game correctly", () => {
    snap(0, 0); snap(1, 0); snap(2, 0); snap(2, 1); snap(3, 1);
    const t = snap(3, 2);
    expect(t.observedPoints).toBe(5);
    expect(t.p1Points).toBe(3);
    expect(t.p2Points).toBe(2);
  });

  it("reads an advantage being lost as a point for the other player", () => {
    snap(3, 3); snap(4, 3);              // P1 takes advantage
    const t = snap(3, 3);                // back to deuce -> P2 won that point
    expect(t.p2Points).toBe(1);
    expect(t.observedPoints).toBe(2);
  });

  it("credits the final point of a game to whoever won the game", () => {
    // The score resets to 0-0 on a game, so the winning point is invisible to
    // a naive differ. It is recovered from the game count instead.
    snap(3, 0);
    g = [1, 0];
    const t = snap(0, 0);
    expect(t.gamesPlayed).toBe(1);
    expect(t.p1Points).toBe(1);          // the game-winning point
    expect(t.observedPoints).toBe(1);
  });

  it("records a gap rather than inventing points when the score jumps", () => {
    snap(0, 0);
    const t = snap(3, 2);                // five points at once — unattributable
    expect(t.gaps).toBe(1);
    expect(t.observedPoints).toBe(0);    // nothing fabricated
  });

  it("does not difference across a game boundary", () => {
    snap(3, 2);
    g = [1, 0];
    const t = snap(0, 0);                // 40-30 -> 0-0 is a new game, not -3 points
    expect(t.gaps).toBe(0);
    expect(t.p1Points).toBe(1);          // only the game-winning point
  });

  it("handles tiebreak scoring", () => {
    g = [6, 6];
    snap(0, 0, true); snap(1, 0, true); snap(1, 1, true);
    const t = snap(2, 1, true);
    expect(t.observedPoints).toBe(3);
    expect(t.p1Points).toBe(2);
  });
});

describe("deuce is counted, not estimated", () => {
  it("counts a real deuce once per game", () => {
    snap(3, 3); snap(4, 3);
    const t = snap(3, 3);                // back to deuce in the SAME game
    expect(t.deuceGames).toBe(1);        // not 2
  });

  it("counts deuce in separate games separately", () => {
    snap(3, 3);
    g = [1, 0]; snap(0, 0);
    snap(3, 3);
    expect(snap(3, 3).deuceGames).toBe(2);
  });

  it("reports zero deuce when none occurred — never a fraction of games", () => {
    snap(0, 0); snap(1, 0); snap(2, 0); snap(3, 0);
    g = [1, 0];
    const t = snap(0, 0);
    expect(t.deuceGames).toBe(0);        // the old code would have said 0.3 x games
  });
});

describe("momentum", () => {
  it("is withheld until enough points have been seen", () => {
    snap(0, 0); snap(1, 0);
    expect(momentum(observeTelemetry({ matchId: M, p1Pts: 2, p2Pts: 0, gamesP1: 0, gamesP2: 0, setIndex: 1 }))).toBeNull();
  });

  it("is positive for the player winning the recent run", () => {
    // Feed a long alternating baseline, then a streak for P1.
    let t = snap(0, 0);
    for (let i = 0; i < 6; i++) {
      g = [i, 0]; snap(0, 0);            // game to P1 each time = 1 point each
    }
    for (let i = 6; i < 14; i++) { g = [i, 0]; t = snap(0, 0); }
    const m = momentum(t);
    expect(m).not.toBeNull();
    expect(m!.observedPoints).toBeGreaterThanOrEqual(MIN_POINTS_FOR_MOMENTUM);
    expect(m!.p1).toBeGreaterThan(0);
    expect(m!.p2).toBe(-m!.p1);          // zero-sum, by construction
  });

  it("never reports the same value for both players", () => {
    let t = snap(0, 0);
    for (let i = 0; i < 14; i++) { g = [i, 0]; t = snap(0, 0); }
    const m = momentum(t)!;
    // The old panel showed +20.0% for BOTH — the signature of a fallback.
    expect(m.p1).not.toBe(m.p2);
  });
});

describe("workload", () => {
  it("reports observed points, not games x 4.5", () => {
    snap(0, 0); snap(1, 0); snap(2, 0); snap(2, 1); snap(3, 1);
    const w = workload(observeTelemetry({ matchId: M, p1Pts: 3, p2Pts: 2, gamesP1: 0, gamesP2: 0, setIndex: 1 }))!;
    expect(w.points).toBe(5);
  });

  it("flags itself incomplete when points were missed", () => {
    snap(0, 0); snap(3, 2);              // a gap
    snap(3, 3); snap(4, 3); snap(3, 3); snap(4, 3);
    const w = workload(observeTelemetry({ matchId: M, p1Pts: 3, p2Pts: 3, gamesP1: 0, gamesP2: 0, setIndex: 1 }));
    expect(w?.incomplete).toBe(true);
  });

  it("returns null before there is anything to report", () => {
    expect(workload(snap(0, 0))).toBeNull();
  });
});
