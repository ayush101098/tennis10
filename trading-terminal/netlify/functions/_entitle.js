/**
 * Shared entitlement grant — one place where "this payment buys 30 days" lives.
 *
 * Stripe, PayPal, Razorpay and the on-chain verifier all end at the same point:
 * a confirmed payment for an email. Triplicating the account-DB write invites
 * exactly the drift that already bit us once (payments stamped with the
 * confirmation time instead of the payment time, which would have handed a late
 * verifier ~59 days instead of 30).
 *
 * Rules enforced here for every provider:
 *   - `ts` is the PAYMENT time, never "now"; expiry is ts + 30 days.
 *   - grants are idempotent on `ref`, so webhook retries can never extend access.
 *   - paidUntil is recomputed from payments + grants, so it cannot drift.
 */

const { store: sharedStore } = require("./_blobs");

const SUBSCRIPTION_DAYS = 30;
const DAY = 86400000;

function blankAccount(email, now) {
  return { email, firstSeen: now, lastLogin: now, loginCount: 0, paidUntil: 0, payments: [], grants: [] };
}

function recompute(a) {
  const fromPayments = a.payments.reduce((m, p) => Math.max(m, p.ts + SUBSCRIPTION_DAYS * DAY), 0);
  const fromGrants = a.grants.reduce((m, g) => Math.max(m, g.until), 0);
  return Math.max(fromPayments, fromGrants);
}

/**
 * Record a confirmed payment and return { paidUntil, warning }.
 *
 * A storage failure is reported as `warning`, never thrown: the money has
 * already changed hands by the time we get here, so bookkeeping must not be
 * able to lock out a paying customer.
 */
async function grantPaid({ email, ref, amountUsd, from, ts }) {
  const paidAt = Number(ts) || Date.now();
  const paidUntil = paidAt + SUBSCRIPTION_DAYS * DAY;
  try {
    const s = sharedStore("accounts");
    if (!s) return { paidUntil, warning: "blob store unavailable" };
    const db = (await s.get("byEmail", { type: "json" })) || {};
    const now = Date.now();
    if (!db[email]) db[email] = blankAccount(email, now);
    if (!db[email].payments.some((p) => p.txHash === ref)) {
      db[email].payments.push({
        txHash: ref,
        amountUsd: Math.round(Number(amountUsd) || 0),
        from: from || "card",
        ts: paidAt,
      });
    }
    db[email].paidUntil = recompute(db[email]);
    await s.setJSON("byEmail", db);
    return { paidUntil: db[email].paidUntil, warning: null };
  } catch (e) {
    return { paidUntil, warning: String((e && e.message) || e).slice(0, 160) };
  }
}

module.exports = { grantPaid, SUBSCRIPTION_DAYS, DAY };
