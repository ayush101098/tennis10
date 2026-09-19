import { describe, it, expect } from "vitest";
import { computeTrueProbabilities, computeLiveMatchProbNoServer, holdsAndBreaks, predictTotalGames } from "../breakHoldEngine";

/**
 * computeLiveMatchProbNoServer exists because attachBreakHoldSignals used to
 * withhold the ENTIRE live probability whenever the feed did not report who
 * was serving — which, for the Flashscore fallback, is every live match. These
 * tests are the evidence for the claim the code comments make: that the
 * match/set-level number never actually depended on the server, so computing
 * it without one is not a downgrade in kind, only in the one field
 * (gameHoldProb) that genuinely cannot be known without it.
 */

const NO_SETS: { p1: number; p2: number }[] = [];
const EVEN_GAMES = { p1: 2, p2: 2 };

describe("computeLiveMatchProbNoServer agrees with the server-aware version", () => {
  it("produces the same p1MatchProb regardless of which server it is told", () => {
    const a = computeTrueProbabilities(1, EVEN_GAMES, NO_SETS, null, 3, 0.65, "ATP");
    const b = computeTrueProbabilities(2, EVEN_GAMES, NO_SETS, null, 3, 0.65, "ATP");
    const c = computeLiveMatchProbNoServer(EVEN_GAMES, NO_SETS, null, 3, 0.65, "ATP");
    // Confirms the claim in the code comment: matchWinProbFromScore's
    // p1Serving parameter is not read, so telling it server=1 vs server=2
    // changes nothing about the match/set number.
    expect(a.p1MatchProb).toBeCloseTo(b.p1MatchProb, 10);
    expect(c.p1MatchProb).toBeCloseTo(a.p1MatchProb, 10);
    expect(c.p1SetProb).toBeCloseTo(a.p1SetProb, 10);
  });

  it("labels itself so the UI can tell the two apart", () => {
    const known = computeTrueProbabilities(1, EVEN_GAMES, NO_SETS, null, 3, 0.6, "ATP");
    const unknown = computeLiveMatchProbNoServer(EVEN_GAMES, NO_SETS, null, 3, 0.6, "ATP");
    expect(known.method).toBe("markov-tour-aware");
    expect(unknown.method).toBe("markov-no-server");
  });

  it("does not attribute gameHoldProb to a guessed player", () => {
    const r = computeLiveMatchProbNoServer(EVEN_GAMES, NO_SETS, null, 3, 0.65, "ATP");
    // The server-aware version picks p1Hold or p2Hold; the no-server version
    // reports their average, which sits strictly between the two whenever the
    // players are not identical.
    const asP1 = computeTrueProbabilities(1, EVEN_GAMES, NO_SETS, null, 3, 0.65, "ATP");
    const asP2 = computeTrueProbabilities(2, EVEN_GAMES, NO_SETS, null, 3, 0.65, "ATP");
    expect(r.gameHoldProb).toBeGreaterThan(Math.min(asP1.gameHoldProb, asP2.gameHoldProb) - 1e-9);
    expect(r.gameHoldProb).toBeLessThan(Math.max(asP1.gameHoldProb, asP2.gameHoldProb) + 1e-9);
  });
});

describe("the score still moves the number without a server", () => {
  it("favours whoever is ahead in sets", () => {
    const evenSets = computeLiveMatchProbNoServer({ p1: 3, p2: 3 }, NO_SETS, null, 3, 0.5, "ATP");
    const p1Up = computeLiveMatchProbNoServer({ p1: 0, p2: 0 }, [{ p1: 6, p2: 3 }], null, 3, 0.5, "ATP");
    expect(p1Up.p1MatchProb).toBeGreaterThan(evenSets.p1MatchProb);
  });

  it("moves toward the player ahead in the current set's games", () => {
    const tied = computeLiveMatchProbNoServer({ p1: 2, p2: 2 }, NO_SETS, null, 3, 0.5, "ATP");
    const p1Ahead = computeLiveMatchProbNoServer({ p1: 5, p2: 2 }, NO_SETS, null, 3, 0.5, "ATP");
    expect(p1Ahead.p1MatchProb).toBeGreaterThan(tied.p1MatchProb);
  });

  it("stays at 50/50 for identical players with a level score", () => {
    const r = computeLiveMatchProbNoServer({ p1: 0, p2: 0 }, NO_SETS, null, 3, 0.5, "ATP");
    expect(r.p1MatchProb).toBeCloseTo(0.5, 6);
  });
});

describe("momentum nudge", () => {
  it("a positive P1 momentum raises P1's live probability over the unnudged figure", () => {
    const base = computeLiveMatchProbNoServer(EVEN_GAMES, NO_SETS, null, 3, 0.5, "ATP");
    const withMomentum = computeLiveMatchProbNoServer(EVEN_GAMES, NO_SETS, null, 3, 0.5, "ATP", 0.15);
    expect(withMomentum.p1MatchProb).toBeGreaterThan(base.p1MatchProb);
  });

  it("is symmetric: the same-magnitude opposite momentum lowers it by about as much", () => {
    const base = computeLiveMatchProbNoServer(EVEN_GAMES, NO_SETS, null, 3, 0.5, "ATP");
    const up = computeLiveMatchProbNoServer(EVEN_GAMES, NO_SETS, null, 3, 0.5, "ATP", 0.1);
    const down = computeLiveMatchProbNoServer(EVEN_GAMES, NO_SETS, null, 3, 0.5, "ATP", -0.1);
    expect(up.p1MatchProb - base.p1MatchProb).toBeCloseTo(base.p1MatchProb - down.p1MatchProb, 6);
  });

  it("is capped — an extreme momentum reading cannot dominate the score", () => {
    const huge = computeLiveMatchProbNoServer(EVEN_GAMES, NO_SETS, null, 3, 0.5, "ATP", 5);
    const capped = computeLiveMatchProbNoServer(EVEN_GAMES, NO_SETS, null, 3, 0.5, "ATP", 0.2);
    // Anything beyond the ±0.2 the nudge clamps to should have no further effect.
    expect(huge.p1MatchProb).toBeCloseTo(capped.p1MatchProb, 10);
  });

  it("does nothing when momentum is undefined", () => {
    const a = computeLiveMatchProbNoServer(EVEN_GAMES, NO_SETS, null, 3, 0.6, "ATP");
    const b = computeLiveMatchProbNoServer(EVEN_GAMES, NO_SETS, null, 3, 0.6, "ATP", undefined);
    expect(a.p1MatchProb).toBe(b.p1MatchProb);
  });
});

describe("probability bounds", () => {
  it("never leaves [0, 1] across a range of scorelines and priors", () => {
    for (const p1WinProb of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      for (const games of [{ p1: 0, p2: 0 }, { p1: 5, p2: 6 }, { p1: 6, p2: 0 }]) {
        for (const sets of [NO_SETS, [{ p1: 6, p2: 4 }], [{ p1: 4, p2: 6 }, { p1: 3, p2: 6 }]]) {
          const r = computeLiveMatchProbNoServer(games, sets, null, 3, p1WinProb, "ATP");
          expect(r.p1MatchProb).toBeGreaterThanOrEqual(0);
          expect(r.p1MatchProb).toBeLessThanOrEqual(1);
          expect(r.p2MatchProb).toBeCloseTo(1 - r.p1MatchProb, 10);
        }
      }
    }
  });
});

describe("p1GameProb", () => {
  it("exact when server is known: P1 serving uses P1's own hold rate", () => {
    const r = computeTrueProbabilities(1, { p1: 2, p2: 2 }, [], null, 3, 0.65, "ATP");
    expect(r.p1GameProb).toBeGreaterThan(0.5);   // 0.65 favourite, serving
  });

  it("exact when server is known: P2 serving lowers P1's game chance", () => {
    const r1 = computeTrueProbabilities(1, { p1: 2, p2: 2 }, [], null, 3, 0.65, "ATP");
    const r2 = computeTrueProbabilities(2, { p1: 2, p2: 2 }, [], null, 3, 0.65, "ATP");
    // Same players, same score — P1 receiving must be worse for P1 than serving.
    expect(r2.p1GameProb).toBeLessThan(r1.p1GameProb);
  });

  it("no-server p1GameProb sits between the two server-known values", () => {
    const asP1 = computeTrueProbabilities(1, { p1: 2, p2: 2 }, [], null, 3, 0.65, "ATP");
    const asP2 = computeTrueProbabilities(2, { p1: 2, p2: 2 }, [], null, 3, 0.65, "ATP");
    const unknown = computeLiveMatchProbNoServer({ p1: 2, p2: 2 }, [], null, 3, 0.65, "ATP");
    expect(unknown.p1GameProb).toBeGreaterThan(Math.min(asP1.p1GameProb, asP2.p1GameProb) - 1e-9);
    expect(unknown.p1GameProb).toBeLessThan(Math.max(asP1.p1GameProb, asP2.p1GameProb) + 1e-9);
  });
});

describe("holdsAndBreaks", () => {
  it("counts a clean hold-hold-hold-hold match correctly", () => {
    // P1 serves game 1 (so nextServer for the "game after game 4" = P1, since
    // 4 games alternated back to P1's turn), everyone holds: 1,2,1,2.
    const log: (1 | 2)[] = [1, 2, 1, 2];
    const r = holdsAndBreaks(log, 1); // game 5 in progress, served by P1
    expect(r).toEqual({ p1Holds: 2, p1Breaks: 0, p2Holds: 2, p2Breaks: 0, unattributed: 0 });
  });

  it("counts breaks when the same player wins two games in a row", () => {
    // P1 wins games 1 and 2. Alternation says game 2's server was different
    // from game 1's, so one of these two wins was a break.
    const log: (1 | 2)[] = [1, 1];
    const r = holdsAndBreaks(log, 1); // game 3 served by P1 -> game 2 served by P2 -> game 1 served by P1
    expect(r.p1Holds).toBe(1);   // game 1: P1 served, P1 won
    expect(r.p1Breaks).toBe(1);  // game 2: P2 served, P1 won
    expect(r.p2Holds + r.p2Breaks).toBe(0);
  });

  it("flips entirely when the anchor server is assumed the other way", () => {
    const log: (1 | 2)[] = [1, 2, 1, 2];
    const asP1 = holdsAndBreaks(log, 1);
    const asP2 = holdsAndBreaks(log, 2);
    // Every hold under one anchor is a break under the other — this is exactly
    // why an anchor is required and cannot be guessed: the two readings are
    // not close, they are opposite.
    expect(asP1.p1Holds).toBe(asP2.p1Breaks);
    expect(asP1.p2Holds).toBe(asP2.p2Breaks);
  });

  it("handles an empty log", () => {
    expect(holdsAndBreaks([], 1)).toEqual({ p1Holds: 0, p1Breaks: 0, p2Holds: 0, p2Breaks: 0, unattributed: 0 });
  });
});

describe("predictTotalGames", () => {
  it("predicts close to the textbook ~20-24 games for two even players from 0-0", () => {
    const r = predictTotalGames({ p1: 0, p2: 0 }, [], null, 3, 0.65);
    expect(r.gamesSoFar).toBe(0);
    expect(r.expectedTotal).toBeGreaterThan(15);
    expect(r.expectedTotal).toBeLessThan(30);
  });

  it("counts games already played as a real count, not an estimate", () => {
    const r = predictTotalGames({ p1: 3, p2: 2 }, [{ p1: 6, p2: 4 }], null, 3, 0.5);
    expect(r.gamesSoFar).toBe(10 + 5); // 6+4 completed, 3+2 current set
  });

  it("predicts fewer remaining games for a lopsided server", () => {
    const even = predictTotalGames({ p1: 0, p2: 0 }, [], null, 3, 0.5);
    const lopsided = predictTotalGames({ p1: 0, p2: 0 }, [], null, 3, 0.95);
    // A dominant server closes sets faster (fewer deuces/tiebreaks), so the
    // total games predicted should not be higher than the even case.
    expect(lopsided.expectedTotal).toBeLessThanOrEqual(even.expectedTotal);
  });

  it("predicts zero remaining once the match is over", () => {
    const r = predictTotalGames({ p1: 0, p2: 0 }, [{ p1: 6, p2: 2 }, { p1: 6, p2: 3 }], null, 3, 0.5);
    expect(r.expectedRemaining).toBe(0);
    expect(r.gamesSoFar).toBe(17);
    expect(r.expectedTotal).toBe(17);
  });

  it("always predicts at least as many games as already played", () => {
    for (const games of [{ p1: 0, p2: 0 }, { p1: 5, p2: 4 }, { p1: 6, p2: 6 }]) {
      const r = predictTotalGames(games, [{ p1: 6, p2: 3 }], null, 3, 0.5);
      expect(r.expectedTotal).toBeGreaterThanOrEqual(r.gamesSoFar);
    }
  });
});
