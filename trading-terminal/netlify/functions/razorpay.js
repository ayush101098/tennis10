/**
 * Razorpay — Indian payments (UPI, netbanking, cards, wallets).
 *
 *   POST /api/razorpay { action:"create", email }  -> { orderId, keyId, amount, currency }
 *   POST /api/razorpay { action:"verify", email, razorpay_order_id,
 *                        razorpay_payment_id, razorpay_signature } -> { ok, paidUntil }
 *
 * One-time payment buying 30 days, matching every other provider here.
 * Deliberately NOT a Razorpay Subscription: RBI e-mandate rules make recurring
 * card charges genuinely painful in India, and the entitlement model is already
 * "each payment grants 30 days", so a plain order avoids all of that.
 *
 * VERIFICATION — two independent checks, both must pass:
 *   1. HMAC-SHA256(`${order_id}|${payment_id}`, key_secret) === razorpay_signature
 *      proves the callback really came from Razorpay.
 *   2. A server-side GET of the payment proves it is genuinely captured, for the
 *      right order and the right amount.
 * (2) is the one that actually secures this: an attacker cannot fabricate a
 * captured payment id belonging to our order. It also means a change in
 * Razorpay's signature scheme degrades to a clear, logged failure rather than a
 * silent hole.
 *
 * Env: RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, RAZORPAY_AMOUNT_INR (default 8900).
 */

const crypto = require("crypto");
const { grantPaid } = require("./_entitle");

const API = "https://api.razorpay.com/v1";
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const amountInr = () => Math.max(1, parseInt(process.env.RAZORPAY_AMOUNT_INR || "8900", 10));
// USD figure recorded in the roster so revenue stays comparable across providers
const usdFor = (inr) => Math.round(inr / Number(process.env.USD_INR || "89"));

const reply = (statusCode, obj) => ({
  statusCode,
  headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  body: JSON.stringify(obj),
});

const authHeader = () =>
  "Basic " + Buffer.from(
    `${process.env.RAZORPAY_KEY_ID}:${process.env.RAZORPAY_KEY_SECRET}`,
  ).toString("base64");

/** Timing-safe compare so signature checking can't be probed byte by byte. */
function safeEqual(a, b) {
  const x = Buffer.from(String(a || ""), "utf8");
  const y = Buffer.from(String(b || ""), "utf8");
  if (x.length !== y.length) return false;
  return crypto.timingSafeEqual(x, y);
}

function expectedSignature(orderId, paymentId, secret) {
  return crypto.createHmac("sha256", secret).update(`${orderId}|${paymentId}`).digest("hex");
}

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") return reply(405, { error: "method not allowed" });

  const keyId = process.env.RAZORPAY_KEY_ID;
  const keySecret = process.env.RAZORPAY_KEY_SECRET;
  if (!keyId || !keySecret) {
    return reply(503, {
      ok: false,
      reason: "Razorpay isn't configured yet. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET.",
    });
  }

  let body = {};
  try { body = JSON.parse(event.body || "{}"); } catch { /* ignore */ }
  const email = String(body.email || "").trim().toLowerCase();
  if (!EMAIL_RE.test(email)) return reply(400, { ok: false, reason: "A valid email is required." });

  try {
    // ── create an order ──
    if (body.action === "create") {
      const inr = amountInr();
      const res = await fetch(`${API}/orders`, {
        method: "POST",
        headers: { Authorization: authHeader(), "Content-Type": "application/json" },
        body: JSON.stringify({
          amount: inr * 100,               // paise — smallest currency sub-unit
          currency: "INR",
          receipt: `tt_${Date.now()}`,
          notes: { email, product: "Tennis Alpha Pro (30 days)" },
        }),
      });
      const d = await res.json();
      if (!res.ok || !d.id) {
        console.error("[razorpay] create failed", res.status, JSON.stringify(d).slice(0, 300));
        return reply(502, { ok: false, reason: "Could not start the Razorpay payment." });
      }
      return reply(200, {
        ok: true, orderId: d.id, keyId, amount: d.amount, currency: d.currency, amountInr: inr,
      });
    }

    // ── verify a completed payment ──
    if (body.action === "verify") {
      const orderId = String(body.razorpay_order_id || "").trim();
      const paymentId = String(body.razorpay_payment_id || "").trim();
      const signature = String(body.razorpay_signature || "").trim();
      if (!orderId || !paymentId || !signature) {
        return reply(400, { ok: false, reason: "Missing payment confirmation fields." });
      }

      // 1. signature
      if (!safeEqual(expectedSignature(orderId, paymentId, keySecret), signature)) {
        console.error("[razorpay] signature mismatch for", orderId, paymentId);
        return reply(400, { ok: false, reason: "Payment signature could not be verified." });
      }

      // 2. authoritative: ask Razorpay what actually happened
      const res = await fetch(`${API}/payments/${encodeURIComponent(paymentId)}`, {
        headers: { Authorization: authHeader() },
      });
      const p = await res.json();
      if (!res.ok || !p.id) {
        console.error("[razorpay] fetch payment failed", res.status, JSON.stringify(p).slice(0, 200));
        return reply(502, { ok: false, reason: "Could not confirm the payment with Razorpay." });
      }
      if (p.order_id !== orderId) {
        return reply(400, { ok: false, reason: "That payment belongs to a different order." });
      }
      if (p.status !== "captured") {
        return reply(200, { ok: false, reason: `Payment status: ${p.status}.` });
      }
      const paidInr = Math.round((p.amount || 0) / 100);
      if (paidInr + 1 < amountInr()) {
        return reply(200, { ok: false, reason: `Paid ₹${paidInr}, expected ₹${amountInr()}.` });
      }

      const ts = p.created_at ? p.created_at * 1000 : Date.now();
      const { paidUntil, warning } = await grantPaid({
        email, ref: `razorpay:${p.id}`, amountUsd: usdFor(paidInr), from: "razorpay", ts,
      });
      if (warning) console.error("[razorpay] entitlement store:", warning);

      return reply(200, {
        ok: true, email, paidUntil, amountInr: paidInr,
        reason: `Payment verified — ₹${paidInr}.`,
        ...(warning ? { storeWarning: warning } : {}),
      });
    }

    return reply(400, { ok: false, reason: "action must be create or verify" });
  } catch (e) {
    console.error("[razorpay]", e && e.message);
    return reply(502, { ok: false, reason: "Razorpay error. " + String((e && e.message) || e).slice(0, 140) });
  }
};

// exported for unit tests
exports._expectedSignature = expectedSignature;
