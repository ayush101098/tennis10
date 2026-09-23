/**
 * Arbitrage mathematics — implied probability, stake allocation, commission-
 * adjusted payoff. Pure functions, no I/O, so the what-if calculator in the
 * UI and the detection engine both call these directly instead of each
 * keeping its own copy of the formulas.
 */

/** S = sum of (1 / odds) across n mutually exclusive, collectively exhaustive
 *  outcomes. S < 1 is a theoretical arbitrage; S >= 1 is not. */
export function impliedProbSum(oddsDecimal: number[]): number {
  return oddsDecimal.reduce((sum, o) => sum + 1 / o, 0);
}

/** Arb% = (1/S - 1) * 100. Negative or zero means no arbitrage. */
export function arbPercent(s: number): number {
  return (1 / s - 1) * 100;
}

export interface StakeAllocation {
  stakes: number[];      // one per outcome, in the same order as the input odds
  grossPayout: number;   // equal under every outcome for a true arbitrage
  grossProfit: number;
  roiPct: number;
}

/** Equal-payout stake allocation across n outcomes for total stake B.
 *  Stake_i = B * (1/O_i) / S. Verified against the spec's worked example:
 *  O = [2.10, 2.05], B = 1000 -> stakes ~[492.80, 507.20], profit ~37.20. */
export function allocateStakes(oddsDecimal: number[], totalStake: number): StakeAllocation {
  const s = impliedProbSum(oddsDecimal);
  const stakes = oddsDecimal.map(o => totalStake * (1 / o) / s);
  const grossPayout = totalStake / s;
  const grossProfit = grossPayout - totalStake;
  return { stakes, grossPayout, grossProfit, roiPct: arbPercent(s) };
}

/**
 * Effective decimal return for an exchange-style bet where commission c is
 * charged on NET winnings only (not on the returned stake).
 * O_eff = 1 + (O - 1) * (1 - c)
 */
export function effectiveOddsWithCommission(oddsDecimal: number, commissionRate: number): number {
  return 1 + (oddsDecimal - 1) * (1 - commissionRate);
}

/**
 * Net payoff per outcome for an arbitrary set of positions (payoff-vector
 * form — required for markets that aren't simple two-outcome books, e.g. a
 * 3-way correct-score market, or combining a match-winner leg with a
 * set-handicap leg that only partially covers the same outcome space).
 *
 * xs[j]      stake on position j
 * payoff[j][k] net decimal return per unit stake on position j if outcome k occurs
 * costs[j]   fixed execution cost attached to position j (0 if none/unknown-as-zero
 *            is never assumed by callers — see classify.ts, which refuses to
 *            call this with an unknown cost silently coerced to 0)
 *
 * Returns one net payoff number per outcome k. A candidate is a true
 * arbitrage only if every entry is strictly positive.
 */
export function netPayoffVector(
  stakes: number[],
  payoff: number[][],
  costs: number[] = [],
): number[] {
  const nOutcomes = payoff[0]?.length ?? 0;
  const totalCost = costs.reduce((a, b) => a + b, 0);
  const result: number[] = [];
  for (let k = 0; k < nOutcomes; k++) {
    let net = -totalCost;
    for (let j = 0; j < stakes.length; j++) net += stakes[j] * payoff[j][k];
    result.push(net);
  }
  return result;
}

/** Round a stake to the nearest venue-specific increment (e.g. $1 on
 *  Polymarket's USDC market). Applied only at the presentation edge — every
 *  internal calculation above stays full-precision. */
export function roundToIncrement(value: number, increment: number): number {
  if (increment <= 0) return value;
  return Math.round(value / increment) * increment;
}
