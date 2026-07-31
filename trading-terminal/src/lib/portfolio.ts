/**
 * Portfolio & staking math — the discipline from the Trading Execution Manual,
 * expressed as pure functions so both the calculator page and the value board
 * stake identically.
 *
 * Everything is denominated as a fraction of the portfolio. The manual works in
 * "units" where the whole bankroll = 100 units, so 1 unit = 1% of portfolio.
 */

export const KELLY_FRACTION = 0.25; // quarter Kelly — never full
export const MIN_EDGE = 0.02; // 2% edge floor: below this, no bet
export const STRONG_EDGE = 0.05; // strong-signal threshold
export const SUSPECT_EDGE = 0.2; // >20% edge = bad data, not free money
export const MAX_ENTRY_FRACTION = 0.06; // 6u max initial entry per event
export const MAX_EXPOSURE_FRACTION = 0.15; // 15u max concurrent exposure

/** Full-Kelly fraction for a decimal-odds bet at a given true probability. */
export function kellyFraction(trueProb: number, odds: number): number {
  const b = odds - 1;
  if (b <= 0) return 0;
  const f = (trueProb * b - (1 - trueProb)) / b;
  return Math.max(0, f);
}

/** The manual's three-tier bankroll split for a given portfolio ($). */
export function portfolioTiers(portfolio: number) {
  const unit = portfolio / 100; // 1 unit = 1% of the bankroll
  return {
    unit,
    tierA: portfolio * 0.6, // Core — main trading bankroll
    tierB: portfolio * 0.3, // Hedge reserve
    tierC: portfolio * 0.1, // Emergency
    maxEntry: portfolio * MAX_ENTRY_FRACTION, // 6u max initial entry
    maxExposure: portfolio * MAX_EXPOSURE_FRACTION, // 15u total across events
  };
}

export type StakeClass = "NONE" | "MICRO" | "SMALL" | "MEDIUM" | "LARGE" | "MAX";

export interface StakePlan {
  fullKelly: number; // full-Kelly fraction
  fraction: number; // fraction of portfolio actually staked (¼ Kelly, capped)
  stake: number; // dollar stake for this portfolio
  units: number; // stake in units (1u = 1% of portfolio)
  capped: boolean; // true when the 6u entry cap bound the size down
  classification: StakeClass;
}

function classify(units: number): StakeClass {
  if (units <= 0) return "NONE";
  if (units >= MAX_ENTRY_FRACTION * 100) return "MAX";
  if (units >= 4) return "LARGE";
  if (units >= 2) return "MEDIUM";
  if (units >= 1) return "SMALL";
  return "MICRO";
}

/**
 * Recommended stake for one bet, following the manual: ¼ Kelly, hard 6u
 * per-event cap, and a 2% edge floor below which we simply don't bet.
 */
export function stakePlan(
  portfolio: number,
  trueProb: number,
  odds: number,
  edge: number,
): StakePlan {
  const fullKelly = kellyFraction(trueProb, odds);
  if (edge < MIN_EDGE || fullKelly <= 0) {
    return { fullKelly, fraction: 0, stake: 0, units: 0, capped: false, classification: "NONE" };
  }
  const quarter = fullKelly * KELLY_FRACTION;
  const fraction = Math.min(quarter, MAX_ENTRY_FRACTION);
  const capped = quarter > MAX_ENTRY_FRACTION;
  const stake = portfolio * fraction;
  return {
    fullKelly,
    fraction,
    stake,
    units: fraction * 100,
    capped,
    classification: classify(fraction * 100),
  };
}

/** De-vig two decimal prices into a fair (overround-free) implied probability pair. */
export function devig(oddsA: number, oddsB: number): { a: number; b: number } | null {
  if (oddsA <= 1 || oddsB <= 1) return null;
  const ia = 1 / oddsA;
  const ib = 1 / oddsB;
  const sum = ia + ib;
  return { a: ia / sum, b: ib / sum };
}
