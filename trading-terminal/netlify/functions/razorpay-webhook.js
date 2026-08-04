/**
 * Razorpay webhook — catches payments the browser never confirmed.
 *
 *   POST /api/razorpay/webhook   (called by Razorpay, not the browser)
 *
 * The normal path is the client calling /api/razorpay { action: "verify" } right
 * after the checkout modal closes. This exists for the case that matters most:
 * the customer's money was taken but their browser died, lost network, or they
 * closed the tab before that call landed. Without it they would be charged and
 * still see a paywall.
 *
 * SECURITY: HMAC-SHA256 of the RAW body with RAZORPAY_WEBHOOK_SECRET, compared
 * against x-razorpay-signature. Netlify may deliver the body base64-encoded, so
 * decode to a Buffer — converting to a UTF-8 string first corrupts the bytes and
 * would reject every genuine event.
 *
 * Env: RAZORPAY_WEBHOOK_SECRET.
 */

const crypto = require("crypto");
const { grantPaid } = require("./_entitle");

const reply = (statusCode, obj) => ({
  statusCode,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(obj),
});

const rawBody = (event) =>
  event.isBase64Encoded
    ? Buffer.from(event.body || "", "base64")
    : Buffer.from(event.body || "", "utf8");

function safeEqual(a, b) {
  const x = Buffer.from(String(a || ""), "utf8");
  const y = Buffer.from(String(b || ""), "utf8");
  if (x.length !== y.length) return false;
  return crypto.timingSafeEqual(x, y);
}

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") return reply(405, { error: "method not allowed" });

  const secret = process.env.RAZORPAY_WEBHOOK_SECRET;
  if (!secret) {
    console.error("[razorpay-webhook] RAZORPAY_WEBHOOK_SECRET not set");
    return reply(503, { error: "webhook not configured" });
  }

  const sig = event.headers["x-razorpay-signature"] || event.headers["X-Razorpay-Signature"];
  if (!sig) return reply(400, { error: "missing signature" });

  const raw = rawBody(event);
  const expected = crypto.createHmac("sha256", secret).update(raw).digest("hex");
  if (!safeEqual(expected, sig)) {
    console.error("[razorpay-webhook] signature verification failed");
    return reply(400, { error: "invalid signature" });
  }

  let evt = {};
  try { evt = JSON.parse(raw.toString("utf8")); } catch {
    return reply(400, { error: "invalid JSON" });
  }

  try {
    if (evt.event === "payment.captured") {
      const p = ((evt.payload || {}).payment || {}).entity || {};
      const email = String(
        (p.notes && (p.notes.email || p.notes.Email)) || p.email || "",
      ).trim().toLowerCase();
      if (!email) return reply(200, { received: true, skipped: "no email on payment" });

      const inr = Math.round((p.amount || 0) / 100);
      const { paidUntil, warning } = await grantPaid({
        email,
        ref: `razorpay:${p.id}`,          // same ref as the client path -> idempotent
        amountUsd: Math.round(inr / Number(process.env.USD_INR || "89")),
        from: "razorpay",
        ts: p.created_at ? p.created_at * 1000 : Date.now(),
      });
      if (warning) {
        // 500 so Razorpay retries a transient storage failure
        console.error("[razorpay-webhook] store failed:", warning);
        return reply(500, { error: warning });
      }
      console.log(`[razorpay-webhook] captured ${email} -> ${new Date(paidUntil).toISOString()}`);
      return reply(200, { received: true, email, paidUntil });
    }

    return reply(200, { received: true, ignored: evt.event });
  } catch (e) {
    console.error("[razorpay-webhook]", e && e.message);
    return reply(500, { error: String((e && e.message) || e).slice(0, 200) });
  }
};
