import { kellyFraction } from "@/lib/scheduleService";

/**
 * PARLAY MATH.
 *
 * A parlay multiplies both the payout and the model's error. Two legs at a
 * true 60% pay like 2.78 but land 36% of the time; if the model is 10 points
 * hot on each leg, the combined probability is wrong by far more than 10
 * points. That compounding is the whole reason this module exists as its own
 * unit with its own tests rather than as arithmetic buried in a component.
 *
 * Everything here assumes the legs are INDEPENDENT, which is why
 * `correlationWarnings` exists — the assumption is false often enough that the
 * UI has to say so out loud.
 */

export interface ParlayLeg {
  /** Match id — also the dedupe key: one leg per match. */
  matchId: string;
  /** Who you are backing. */
  player: string;
  opponent: string;
  tournament: string;
  /** Model probability for this leg. */
  trueP: number;
  /** De-vigged market probability for this leg. */
  marketP: number;
  /** Decimal odds for this leg (1 / marketP for a de-vigged price). */
  odds: number;
  live: boolean;
}

export interface ParlayResult {
  legs: number;
  /** Model probability that EVERY leg lands. */
  trueP: number;
  /** Market probability that every leg lands. */
  marketP: number;
  /** Combined decimal odds — the product of the legs. */
  odds: number;
  /** trueP − marketP. Positive means the model likes the ticket. */
  edge: number;
  /** Full-Kelly fraction on the combined price. */
  kelly: number;
  /** Payout multiple on a winning ticket. */
  payout: number;
  /** True when any leg is in play — the price moves under you. */
  hasLive: boolean;
}

const round4 = (n: number) => Math.round(n * 10000) / 10000;

/** Multiply a list, with an explicit identity so an empty parlay is 1, not NaN. */
const product = (xs: number[]) => xs.reduce((a, b) => a * b, 1);

export function priceParlay(legs: ParlayLeg[]): ParlayResult | null {
  if (legs.length === 0) return null;
  const trueP = product(legs.map(l => l.trueP));
  const marketP = product(legs.map(l => l.marketP));
  const odds = product(legs.map(l => l.odds));
  return {
    legs: legs.length,
    trueP: round4(trueP),
    marketP: round4(marketP),
    odds: round4(odds),
    edge: round4(trueP - marketP),
    kelly: round4(kellyFraction(trueP, odds)),
    payout: round4(odds),
    hasLive: legs.some(l => l.live),
  };
}

/**
 * Re-price the ticket with every leg's model probability knocked down by a
 * fixed number of PERCENTAGE POINTS.
 *
 * WHY POINTS AND NOT A FRACTION OF THE GAP
 *   The obvious stress test — shade each leg from its model probability toward
 *   its market probability — is degenerate. At full concession every leg's
 *   True P *is* the market price, so the combined edge is exactly zero by
 *   construction and the break-even point is 100% for every ticket ever built.
 *   It looks like a measurement and carries no information.
 *
 *   Absolute points do carry information, because they map onto something
 *   measured: the model currently sits ~13 points from the market per leg on
 *   the US Open card. Asking "how many points of per-leg overconfidence does
 *   this ticket survive?" has a different answer for every ticket, and a
 *   sharply worse one as legs are added — which is the fact a parlay builder
 *   most needs to convey.
 */
export function stressParlay(legs: ParlayLeg[], pointsPerLeg: number): ParlayResult | null {
  const d = Math.max(0, pointsPerLeg);
  return priceParlay(legs.map(l => ({
    ...l,
    // Clamped: a leg cannot be driven to an impossible probability, and the
    // market price stays the comparison throughout.
    trueP: Math.min(0.99, Math.max(0.01, l.trueP - d)),
  })));
}

/** Per-leg overconfidence, in points, at which the ticket stops being +EV. */
export const MAX_STRESS_POINTS = 0.30;

/**
 * How much per-leg model error the ticket absorbs before it turns -EV.
 *
 * Returns 0 when there is no edge to begin with, and MAX_STRESS_POINTS when
 * the ticket survives even a 30-point-per-leg haircut. Bisection, because the
 * combined probability is a product of clamped terms with no closed form.
 */
export function stressTolerance(legs: ParlayLeg[]): number | null {
  const base = priceParlay(legs);
  if (!base) return null;
  if (base.edge <= 0) return 0;
  if ((stressParlay(legs, MAX_STRESS_POINTS)?.edge ?? -1) > 0) return MAX_STRESS_POINTS;

  let lo = 0, hi = MAX_STRESS_POINTS;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if ((stressParlay(legs, mid)?.edge ?? -1) > 0) lo = mid;
    else hi = mid;
  }
  return round4(lo);
}

/**
 * The model-vs-market gap measured on the current US Open card (mean 12.9pp
 * per leg, see TENNIS_DATA_SOURCES_AUDIT.md). Used as the default stress so
 * the headline edge is never shown without the measured correction beside it.
 */
export const MEASURED_GAP_PP = 0.13;

/**
 * Reasons these legs may not be independent.
 *
 * Independence is what makes multiplying probabilities legal, and tennis
 * breaks it in specific, knowable ways. Returning prose rather than a boolean
 * because the user has to judge each case — the app should not silently drop
 * a leg it merely suspects.
 */
export function correlationWarnings(legs: ParlayLeg[]): string[] {
  const out: string[] = [];

  // Same match twice: not a parlay, a contradiction or a duplicate.
  const byMatch = new Map<string, number>();
  for (const l of legs) byMatch.set(l.matchId, (byMatch.get(l.matchId) ?? 0) + 1);
  if ([...byMatch.values()].some(n => n > 1)) {
    out.push("Two legs from the same match are not independent — they cannot both be priced by multiplication.");
  }

  // Same tournament: players can meet later, and a draw is not independent of
  // itself. Backing both sides of a future meeting is a guaranteed loser.
  const byEvent = new Map<string, number>();
  for (const l of legs) byEvent.set(l.tournament, (byEvent.get(l.tournament) ?? 0) + 1);
  for (const [tournament, n] of byEvent) {
    if (n > 1) {
      out.push(`${n} legs from ${tournament}: players in one draw can meet in a later round, so these legs are correlated. Multiplying overstates the true price.`);
    }
  }

  // Live legs move while you are legging in.
  if (legs.some(l => l.live)) {
    out.push("A live leg re-prices on every point — by the time the last leg is filled, the first has moved.");
  }

  return out;
}

/** Long parlays are where compounding error does the most damage. */
export const LEG_WARNING_THRESHOLD = 4;
