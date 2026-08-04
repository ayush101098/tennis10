/**
 * Stripe webhook — keeps entitlements current without the user being present.
 *
 *   POST /api/stripe/webhook   (called by Stripe, not the browser)
 *
 * Handles:
 *   checkout.session.completed  first payment  -> 30 days from the payment
 *   invoice.paid                every renewal  -> extend another 30 days
 *   invoice.payment_failed      logged only; access lapses on its own because
 *                               it is time-boxed, so nothing needs revoking
 *
 * SECURITY: the signature is verified against the RAW request body. Netlify may
 * deliver the body base64-encoded (isBase64Encoded), and decoding it to a UTF-8
 * string before verifying would corrupt the bytes and reject every real event —
 * so decode to a Buffer and hand Stripe the exact bytes it signed. Without a
 * verified signature anyone could POST a fake "paid" event and mint access.
 *
 * Env: STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET (both required).
 */

const { store: sharedStore } = require("./_blobs");

const SUBSCRIPTION_DAYS = 30;
const DAY = 86400000;

const reply = (statusCode, obj) => ({
  statusCode,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(obj),
});

/** Exact bytes Stripe signed. */
function rawBody(event) {
  return event.isBase64Encoded
    ? Buffer.from(event.body || "", "base64")
    : Buffer.from(event.body || "", "utf8");
}

async function grant(email, ref, amountUsd, paidAtMs) {
  const s = sharedStore("accounts");
  if (!s) throw new Error("blob store unavailable");
  const db = (await s.get("byEmail", { type: "json" })) || {};
  const now = Date.now();
  if (!db[email]) {
    db[email] = { email, firstSeen: now, lastLogin: now, loginCount: 0, paidUntil: 0, payments: [], grants: [] };
  }
  // Idempotent: Stripe retries webhooks, and a duplicate row would extend access.
  if (!db[email].payments.some((p) => p.txHash === ref)) {
    db[email].payments.push({ txHash: ref, amountUsd, from: "stripe", ts: paidAtMs });
  }
  const fromPayments = db[email].payments.reduce((m, p) => Math.max(m, p.ts + 30 * DAY), 0);
  const fromGrants = db[email].grants.reduce((m, g) => Math.max(m, g.until), 0);
  db[email].paidUntil = Math.max(fromPayments, fromGrants);
  await s.setJSON("byEmail", db);
  return db[email].paidUntil;
}

const emailFrom = (obj) =>
  String(
    (obj.metadata && obj.metadata.email)
    || obj.customer_email
    || obj.client_reference_id
    || (obj.customer_details && obj.customer_details.email)
    || "",
  ).trim().toLowerCase();

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") return reply(405, { error: "method not allowed" });

  const key = process.env.STRIPE_SECRET_KEY;
  const whSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!key || !whSecret) {
    console.error("[stripe-webhook] missing STRIPE_SECRET_KEY or STRIPE_WEBHOOK_SECRET");
    return reply(503, { error: "stripe not configured" });
  }

  const sig = event.headers["stripe-signature"] || event.headers["Stripe-Signature"];
  if (!sig) return reply(400, { error: "missing stripe-signature" });

  const Stripe = require("stripe");
  const stripe = new Stripe(key);

  let evt;
  try {
    evt = stripe.webhooks.constructEvent(rawBody(event), sig, whSecret);
  } catch (e) {
    // Never trust an unverified payload — this is the whole security boundary.
    console.error("[stripe-webhook] signature verification failed:", e && e.message);
    return reply(400, { error: "invalid signature" });
  }

  try {
    const o = evt.data.object;

    if (evt.type === "checkout.session.completed") {
      if (o.payment_status !== "paid") return reply(200, { received: true, skipped: "unpaid" });
      const email = emailFrom(o);
      if (!email) return reply(200, { received: true, skipped: "no email" });
      const until = await grant(email, `stripe:${o.id}`,
        Math.round((o.amount_total || 0) / 100),
        o.created ? o.created * 1000 : Date.now());
      console.log(`[stripe-webhook] checkout paid ${email} -> ${new Date(until).toISOString()}`);
      return reply(200, { received: true, email, paidUntil: until });
    }

    if (evt.type === "invoice.paid") {
      const email = emailFrom(o)
        || String((o.lines && o.lines.data && o.lines.data[0]
          && o.lines.data[0].metadata && o.lines.data[0].metadata.email) || "").toLowerCase();
      if (!email) return reply(200, { received: true, skipped: "no email" });
      const until = await grant(email, `stripe:${o.id}`,
        Math.round((o.amount_paid || 0) / 100),
        o.created ? o.created * 1000 : Date.now());
      console.log(`[stripe-webhook] renewal ${email} -> ${new Date(until).toISOString()}`);
      return reply(200, { received: true, email, paidUntil: until });
    }

    if (evt.type === "invoice.payment_failed") {
      // No action: access is time-boxed and simply lapses if no payment lands.
      console.warn("[stripe-webhook] payment failed for", emailFrom(o) || "unknown");
      return reply(200, { received: true });
    }

    return reply(200, { received: true, ignored: evt.type });
  } catch (e) {
    // 500 makes Stripe retry, which is what we want for a transient store failure.
    console.error("[stripe-webhook] handling failed:", e && e.message);
    return reply(500, { error: String((e && e.message) || e).slice(0, 200) });
  }
};
