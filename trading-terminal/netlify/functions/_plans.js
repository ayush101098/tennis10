/**
 * Server-side tier table — the authority for how long a payment buys.
 *
 * Mirrors src/lib/plans.ts. The client copy only decides what is displayed;
 * this one decides what a payer actually gets, so any disagreement resolves in
 * favour of this file. Keep the two in step.
 *
 * The payment routes are "send an amount", not "buy SKU X" — an on-chain
 * transfer carries no plan id — so the amount is what selects the tier.
 */

const DAY_MS = 86400000;

const PLANS = [
  { id: "day", label: "Day pass", usd: 19, days: 1 },
  { id: "month", label: "Monthly", usd: 99, days: 30 },
  { id: "year", label: "Yearly", usd: 999, days: 365 },
];

const MIN_PAYMENT_USD = Math.min(...PLANS.map((p) => p.usd));

/** Days of access an amount buys — the best tier it covers, else 0. */
function daysForAmount(usd) {
  let days = 0;
  for (const p of PLANS) if (usd >= p.usd) days = Math.max(days, p.days);
  return days;
}

/** Label for what the amount bought (receipts, admin roster). */
function planNameForAmount(usd) {
  const match = [...PLANS].reverse().find((p) => usd >= p.usd);
  return match ? match.label : "below the minimum";
}

module.exports = { PLANS, MIN_PAYMENT_USD, DAY_MS, daysForAmount, planNameForAmount };
