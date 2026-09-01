import type { ScheduledMatch } from "@/lib/scheduleService";
import { kellyFraction } from "@/lib/scheduleService";
import { outcomePrice, type PmFixture } from "@/lib/polymarket";

/**
 * Value priced off POLYMARKET instead of a bookmaker.
 *
 * WHY THIS EXISTS
 *   `attachIntelligence` prices every match against SofaScore's bookmaker
 *   odds. Those endpoints now answer 403 for the per-event market, and the
 *   daily bulk feed returns a stale, disjoint event-id space, so the join
 *   matches nothing: as of 2026-08-31 not one of the 112 US Open matches on
 *   the card carried `prematchOdds`, and a value board with no price is an
 *   empty page. Polymarket is both reachable and the venue this project
 *   actually executes on, so it is the price of record here.
 *
 *   The model side is unchanged — the same True P the terminal trades — and
 *   the same guards apply: no real prior means no opinion, and an edge too
 *   large to be real is quarantined rather than recommended.
 */

/** Edge beyond this is a data fault, not an opportunity. Mirrors attachIntelligence. */
const SUSPECT_EDGE = 0.20;

const round3 = (n: number) => Math.round(n * 1000) / 1000;

/**
 * Both sides of a Polymarket match market, de-vigged.
 *
 * The two outcome prices do not sum to 1 — the spread and stale quotes push
 * the pair either side of it — so they are normalised exactly as the
 * bookmaker path normalises its overround. Skipping this would read the
 * spread itself as edge and manufacture bets out of an efficient market.
 */
function devigged(fixture: PmFixture, p1: string, p2: string): { m1: number; m2: number } | null {
  const market = fixture.match;
  if (!market) return null;
  const raw1 = outcomePrice(market, p1);
  const raw2 = outcomePrice(market, p2);
  if (raw1 == null || raw2 == null) return null;
  const total = raw1 + raw2;
  // A pair that does not roughly sum to 1 is not a two-way match market —
  // most likely an outcome-name match landed on the wrong leg.
  if (total < 0.9 || total > 1.1) return null;
  return { m1: raw1 / total, m2: raw2 / total };
}

/**
 * Best-side value for a match, priced from Polymarket.
 * Returns null when the match cannot be priced or the model has no opinion.
 */
export function polymarketValue(
  m: ScheduledMatch,
  fixture: PmFixture | undefined,
): ScheduledMatch["value"] | null {
  if (!fixture) return null;
  if (m.status === "finished" || m.status === "cancelled") return null;

  // Same bar as the bookmaker path: an unranked field degrades the prior to a
  // coin flip, and a coin flip against a market that knows the players is
  // ignorance, not edge.
  const hasRealPrior =
    m.prob_method !== "unknown" &&
    ((m.p1_rank > 0 && m.p2_rank > 0) || (m.p1_seed > 0 && m.p2_seed > 0));

  const live = m.status === "live";
  const liveTrue = m.liveScore?.trueProbabilities?.p1MatchProb;
  if (live && liveTrue == null && !hasRealPrior) return null;
  if (!live && !hasRealPrior) return null;

  const market = devigged(fixture, m.player1, m.player2);
  if (!market) return null;

  // Live matches are re-priced by the Markov engine on the actual score;
  // pre-match uses the NN+Elo prior.
  const p1True = live && liveTrue != null ? liveTrue : m.p1_win_prob;

  const edge1 = p1True - market.m1;
  const edge2 = (1 - p1True) - market.m2;
  const side: 1 | 2 = edge1 >= edge2 ? 1 : 2;
  const trueP = side === 1 ? p1True : 1 - p1True;
  const marketP = side === 1 ? market.m1 : market.m2;
  const edge = side === 1 ? edge1 : edge2;
  const odds = 1 / marketP;

  return {
    side,
    player: side === 1 ? m.player1 : m.player2,
    trueP: round3(trueP),
    odds: round3(odds),
    marketP: round3(marketP),
    edge: round3(edge),
    kelly: round3(kellyFraction(trueP, odds)),
    live: live && liveTrue != null,
    suspect: edge > SUSPECT_EDGE,
  };
}
