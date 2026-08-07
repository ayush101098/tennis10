/**
 * Access tiers.
 *
 * The payment routes (PayPal.me, crypto) are both "send an amount" rather than
 * "buy SKU X" — nothing carries a plan id through to settlement. So the AMOUNT
 * decides the window, everywhere, via daysForAmount(). That rule is duplicated
 * server-side in netlify/functions/_plans.js; the two must stay in step, and
 * the server is the authority (the client copy only drives what is displayed).
 *
 * Amounts are floors, not exact matches: crypto payers cannot hit a cent, and
 * an ETH transfer worth $99.40 must not be read as a $19 day pass.
 */

export interface Plan {
  id: "day" | "month" | "year";
  label: string;
  usd: number;
  days: number;
  blurb: string;
  highlight?: boolean;
}

export const PLANS: Plan[] = [
  {
    id: "day",
    label: "Day pass",
    usd: 19,
    days: 1,
    blurb: "24 hours of the full terminal. For a single tournament day.",
  },
  {
    id: "month",
    label: "Monthly",
    usd: 99,
    days: 30,
    blurb: "30 days of everything. Renews when you pay again — never auto-charged.",
    highlight: true,
  },
  {
    id: "year",
    label: "Yearly",
    usd: 999,
    days: 365,
    blurb: "A full season. Works out at $83 a month.",
  },
];

export const planById = (id: Plan["id"]): Plan =>
  PLANS.find(p => p.id === id) || PLANS[1];

/** The cheapest tier — anything below this buys nothing. */
export const MIN_PAYMENT_USD = Math.min(...PLANS.map(p => p.usd));

/**
 * Days of access an amount buys: the best tier it covers.
 *
 * Deliberately generous at the boundary (a $98.90 transfer for the $99 plan
 * still lands, see the tolerance in the verifier) but never upgrades a payer
 * to a tier they did not cover.
 */
export function daysForAmount(usd: number): number {
  let days = 0;
  for (const p of PLANS) if (usd >= p.usd) days = Math.max(days, p.days);
  return days;
}

/** Human label for what an amount bought, for receipts and the admin roster. */
export function planNameForAmount(usd: number): string {
  const match = [...PLANS].reverse().find(p => usd >= p.usd);
  return match ? match.label : "below the minimum";
}
