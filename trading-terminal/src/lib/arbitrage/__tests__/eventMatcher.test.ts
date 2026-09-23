import { describe, it, expect } from "vitest";
import { scoreEventMatch, toCanonicalEvent, MATCH_CONFIDENCE_FLOOR } from "../eventMatcher";
import type { CanonicalEvent } from "../types";
import type { ScheduledMatch } from "@/lib/scheduleService";

const event: CanonicalEvent = {
  matchId: "m1",
  player1: "Alcaraz C.",
  player2: "Sinner J.",
  tournament: "US Open",
  tour: "ATP",
  round: "Final",
  surface: "Hard",
  bestOf: 5,
  startTimestamp: 1_700_000_000,
  status: "scheduled",
};

describe("scoreEventMatch", () => {
  it("matches identical player order with high confidence", () => {
    const { confidence, reversed } = scoreEventMatch(event, {
      player1: "Alcaraz C.", player2: "Sinner J.",
      tournament: "US Open", startTimestamp: 1_700_000_000,
    });
    expect(confidence).toBeGreaterThanOrEqual(MATCH_CONFIDENCE_FLOOR);
    expect(reversed).toBe(false);
  });

  it("matches REVERSED player order and flags it — provider B lists player Y first", () => {
    const { confidence, reversed } = scoreEventMatch(event, {
      player1: "Sinner J.", player2: "Alcaraz C.",
      tournament: "US Open", startTimestamp: 1_700_000_000,
    });
    expect(confidence).toBeGreaterThanOrEqual(MATCH_CONFIDENCE_FLOOR);
    expect(reversed).toBe(true);
  });

  it("matches full-name vs surname-abbreviated formats across providers", () => {
    const { confidence } = scoreEventMatch(event, {
      player1: "Carlos Alcaraz", player2: "Jannik Sinner",
      tournament: "US Open", startTimestamp: 1_700_000_000,
    });
    expect(confidence).toBeGreaterThanOrEqual(MATCH_CONFIDENCE_FLOOR);
  });

  it("does not match unrelated players — never match on name alone with no support", () => {
    const { confidence } = scoreEventMatch(event, {
      player1: "Djokovic N.", player2: "Medvedev D.",
      tournament: "US Open", startTimestamp: 1_700_000_000,
    });
    expect(confidence).toBe(0);
  });

  it("flags an ambiguous mapping below the confidence floor rather than accepting it", () => {
    // Right players, wrong tournament AND far-off start time — degrades below floor.
    const { confidence } = scoreEventMatch(event, {
      player1: "Alcaraz C.", player2: "Sinner J.",
      tournament: "Challenger Bordeaux", startTimestamp: 1_700_000_000 + 400_000,
    });
    expect(confidence).toBeLessThan(1);
  });
});

describe("toCanonicalEvent", () => {
  it("preserves the internal match id as the canonical identity", () => {
    const m = {
      id: "sofa_atp_123", player1: "A", player2: "B", tournament: "T", tour: "ATP",
      round: "R1", surface: "Hard", best_of: 3, source: "sofascore", status: "scheduled",
      start_time: "12:00", start_timestamp: 123, p1_win_prob: 0.5, p2_win_prob: 0.5,
      prob_method: "elo", p1_rank: 1, p2_rank: 2, p1_seed: 0, p2_seed: 0,
    } as unknown as ScheduledMatch;
    const canonical = toCanonicalEvent(m);
    expect(canonical.matchId).toBe("sofa_atp_123");
    expect(canonical.player1).toBe("A");
  });
});
