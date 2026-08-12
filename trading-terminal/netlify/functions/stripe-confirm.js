/**
 * Stripe Checkout confirmation — authoritative, storage-independent.
 *
 *   GET /api/stripe/confirm?session_id=cs_...  ->  { ok, email, paidUntil }
 *
 * The browser returns from Stripe with the session id and calls this, which
 * asks STRIPE whether the session is actually paid. That mirrors how the crypto
 * flow trusts the chain rather than the client: the source of truth is external
 * and cannot be forged from the browser.
 *
 * Deliberately does NOT depend on the webhook or on Netlify Blobs. Webhooks can
 * be delayed or misconfigured, and Blobs is currently unavailable on this site,
 * which would otherwise leave a customer who just paid staring at the paywall.
 * Persisting to the account DB is attempted, but only as bookkeeping — a
 * failure there is logged and never blocks access.
 *
 * Env: STRIPE_SECRET_KEY.
 */

const { store: sharedStore } = require("./_store");

const SUBSCRIPTION_DAYS = 30;
const DAY = 86400000;

const reply = (statusCode, obj) => ({
  statusCode,
  headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  body: JSON.stringify(obj),
});

/** Best-effort mirror into the unified account DB (see functions/account.js). */
async function record(email, session, paidUntil) {
  try {
    const s = sharedStore("accounts");
    if (!s) return "blob store unavailable";
    const db = (await s.get("byEmail", { type: "json" })) || {};
    const now = Date.now();
    if (!db[email]) {
      db[email] = { email, firstSeen: now, lastLogin: now, loginCount: 0, paidUntil: 0, payments: [], grants: [] };
    }
    const ref = `stripe:${session.id}`;
    if (!db[email].payments.some((p) => p.txHash === ref)) {
      db[email].payments.push({
        txHash: ref,
        amountUsd: Math.round((session.amount_total || 0) / 100),
        from: "stripe",
        // Payment time, not confirmation time — expiry is derived as ts + 30d.
        ts: (session.created ? session.created * 1000 : now),
      });
    }
    const fromPayments = db[email].payments.reduce((m, p) => Math.max(m, p.ts + 30 * DAY), 0);
    const fromGrants = db[email].grants.reduce((m, g) => Math.max(m, g.until), 0);
    db[email].paidUntil = Math.max(fromPayments, fromGrants);
    await s.setJSON("byEmail", db);
    return null;
  } catch (e) {
    return String((e && e.message) || e).slice(0, 160);
  }
}

exports.handler = async (event) => {
  if (event.httpMethod !== "GET") return reply(405, { error: "method not allowed" });

  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) return reply(503, { ok: false, reason: "Card payments aren't configured." });

  const sessionId = String((event.queryStringParameters || {}).session_id || "").trim();
  if (!/^cs_[A-Za-z0-9_]+$/.test(sessionId)) {
    return reply(400, { ok: false, reason: "A valid session_id is required." });
  }

  try {
    const Stripe = require("stripe");
    const stripe = new Stripe(key);
    const session = await stripe.checkout.sessions.retrieve(sessionId);

    // `paid` is the only status that may unlock access.
    if (session.payment_status !== "paid") {
      return reply(200, {
        ok: false,
        reason: session.payment_status === "unpaid"
          ? "Payment hasn't completed yet."
          : `Payment status: ${session.payment_status}.`,
      });
    }

    const email = String(
      (session.metadata && session.metadata.email)
      || session.customer_email
      || session.client_reference_id
      || "",
    ).trim().toLowerCase();
    if (!email) return reply(200, { ok: false, reason: "No email attached to that payment." });

    const paidFrom = session.created ? session.created * 1000 : Date.now();
    const paidUntil = paidFrom + SUBSCRIPTION_DAYS * DAY;
    const storeWarning = await record(email, session, paidUntil);
    if (storeWarning) console.error("[stripe-confirm] record failed:", storeWarning);

    return reply(200, {
      ok: true,
      email,
      paidUntil,
      amountUsd: Math.round((session.amount_total || 0) / 100),
      reason: `Card payment verified — $${Math.round((session.amount_total || 0) / 100)}.`,
      ...(storeWarning ? { storeWarning } : {}),
    });
  } catch (e) {
    console.error("[stripe-confirm]", e && e.message);
    return reply(502, {
      ok: false,
      reason: "Could not confirm with Stripe. " + String((e && e.message) || e).slice(0, 140),
    });
  }
};
