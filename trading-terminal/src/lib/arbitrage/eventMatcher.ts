import type { ScheduledMatch } from "@/lib/scheduleService";
import type { CanonicalEvent } from "./types";

/**
 * Confidence-scored event identity.
 *
 * With one real market provider (Polymarket) matched by exact surname key
 * (see polymarket.ts's fixtureKey), there is no cross-venue ambiguity to
 * resolve yet — matching is exact by construction. This module exists so
 * that adding a second real provider is "write an adapter + call
 * scoreEventMatch for its candidates", not "redesign how matches are
 * identified". Per the spec: never match on player name alone, and flag
 * anything below the confidence floor instead of silently merging it.
 */

export const MATCH_CONFIDENCE_FLOOR = 0.75;

export interface EventMatchCandidate {
  player1: string;
  player2: string;
  tournament?: string;
  startTimestamp?: number;
}

function nameScore(a: string, b: string): number {
  const na = a.trim().toLowerCase();
  const nb = b.trim().toLowerCase();
  if (!na || !nb) return 0;
  if (na === nb) return 1;
  // Surname-anchored partial match (handles "Sabalenka A." vs "Aryna Sabalenka",
  // where the LAST token on one side is a real surname but on the other side
  // is a trailing initial — check both directions rather than assuming which
  // side wrote the full name).
  const lastA = na.split(/\s+/).pop() || na;
  const lastB = nb.split(/\s+/).pop() || nb;
  if ((lastA.length > 2 && nb.includes(lastA)) || (lastB.length > 2 && na.includes(lastB))) return 0.85;
  return 0;
}

/**
 * Score a candidate against a canonical event, allowing for reversed player
 * order (provider A lists X first, provider B lists Y first) — the spec's
 * explicit test case. Weighs player identity (dominant), tournament, and
 * start-time proximity. Returns 0 confidence and reversed:false on no match.
 */
export function scoreEventMatch(
  event: CanonicalEvent,
  candidate: EventMatchCandidate,
): { confidence: number; reversed: boolean } {
  const straight = (nameScore(event.player1, candidate.player1) + nameScore(event.player2, candidate.player2)) / 2;
  const crossed = (nameScore(event.player1, candidate.player2) + nameScore(event.player2, candidate.player1)) / 2;
  const reversed = crossed > straight;
  let playerScore = Math.max(straight, crossed);
  if (playerScore === 0) return { confidence: 0, reversed: false };

  let confidence = playerScore;
  const weight = { player: 0.7, tournament: 0.15, time: 0.15 };
  let total = weight.player;
  let weighted = playerScore * weight.player;

  if (candidate.tournament) {
    const tMatch = event.tournament.trim().toLowerCase() === candidate.tournament.trim().toLowerCase() ? 1 : 0.3;
    weighted += tMatch * weight.tournament;
    total += weight.tournament;
  }
  if (candidate.startTimestamp) {
    const deltaHrs = Math.abs(event.startTimestamp - candidate.startTimestamp) / 3600;
    const tScore = deltaHrs <= 1 ? 1 : deltaHrs <= 6 ? 0.6 : deltaHrs <= 24 ? 0.2 : 0;
    weighted += tScore * weight.time;
    total += weight.time;
  }

  confidence = weighted / total;
  return { confidence, reversed };
}

export function toCanonicalEvent(m: ScheduledMatch): CanonicalEvent {
  return {
    matchId: m.id,
    player1: m.player1,
    player2: m.player2,
    tournament: m.tournament,
    tour: m.tour,
    round: m.round,
    surface: m.surface,
    bestOf: m.best_of,
    startTimestamp: m.start_timestamp,
    status: m.status,
  };
}
