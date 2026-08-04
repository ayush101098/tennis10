/**
 * PayPal — international / US card payments.
 *
 * Chosen alongside Razorpay because an Indian merchant can open a PayPal
 * Business account immediately (no approval wait), US customers already trust
 * it, and — importantly — they can pay by CARD as a guest without owning a
 * PayPal account, which is the friction this was meant to remove.
 *
 *   POST /api/paypal  { action:"create", email }        -> { id }
 *   POST /api/paypal  { action:"capture", orderId, email } -> { ok, paidUntil }
 *
 * One-time $100 payment buying 30 days, matching the crypto and Stripe models.
 * Deliberately NOT a PayPal subscription: subscriptions need a Product+Plan
 * created up front, and the entitlement model here is already "each payment
 * grants 30 days", so a plain order is both simpler and consistent.
 *
 * The capture step is authoritative — the server asks PayPal to capture and
 * only grants access if PayPal reports COMPLETED and the amount matches. The
 * browser cannot fake that.
 *
 * Env: PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET, PAYPAL_ENV (sandbox|live).
 */

const { grantPaid } = require("./_entitle");

const PRICE_USD = 100;
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const api = () =>
  (process.env.PAYPAL_ENV || "live").toLowerCase() === "sandbox"
    ? "https://api-m.sandbox.paypal.com"
    : "https://api-m.paypal.com";

const reply = (statusCode, obj) => ({
  statusCode,
  headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  body: JSON.stringify(obj),
});

function siteUrlFrom(event) {
  if (process.env.SITE_URL) return process.env.SITE_URL.replace(/\/$/, "");
  if (process.env.URL) return process.env.URL.replace(/\/$/, "");
  const proto = event.headers["x-forwarded-proto"] || "https";
  const host = event.headers.host || event.headers.Host;
  return host ? `${proto}://${host}` : "";
}

async function accessToken() {
  const id = process.env.PAYPAL_CLIENT_ID;
  const secret = process.env.PAYPAL_CLIENT_SECRET;
  const res = await fetch(`${api()}/v1/oauth2/token`, {
    method: "POST",
    headers: {
      Authorization: "Basic " + Buffer.from(`${id}:${secret}`).toString("base64"),
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: "grant_type=client_credentials",
  });
  const d = await res.json();
  if (!res.ok || !d.access_token) {
    throw new Error(`paypal auth ${res.status}: ${(d.error_description || d.error || "").slice(0, 120)}`);
  }
  return d.access_token;
}

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") return reply(405, { error: "method not allowed" });

  if (!process.env.PAYPAL_CLIENT_ID || !process.env.PAYPAL_CLIENT_SECRET) {
    return reply(503, {
      ok: false,
      reason: "PayPal isn't configured yet. Set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET.",
    });
  }

  let body = {};
  try { body = JSON.parse(event.body || "{}"); } catch { /* ignore */ }
  const email = String(body.email || "").trim().toLowerCase();
  if (!EMAIL_RE.test(email)) return reply(400, { ok: false, reason: "A valid email is required." });

  try {
    const token = await accessToken();

    // ── create the order ──
    if (body.action === "create") {
      const res = await fetch(`${api()}/v2/checkout/orders`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          intent: "CAPTURE",
          purchase_units: [{
            amount: { currency_code: "USD", value: PRICE_USD.toFixed(2) },
            description: "Tennis Intelligence Terminal — Pro, 30 days",
            custom_id: email,
          }],
          application_context: {
            brand_name: "Tennis Intelligence Terminal",
            user_action: "PAY_NOW",
            shipping_preference: "NO_SHIPPING",
            // Redirect flow rather than the JS SDK: this is a static export, so
            // avoiding a client-side SDK (and shipping the client id into the
            // bundle) keeps it simpler and matches the Stripe return pattern.
            return_url: `${siteUrlFrom(event)}/terminal?paypal=success`,
            cancel_url: `${siteUrlFrom(event)}/?paypal=cancelled`,
          },
        }),
      });
      const d = await res.json();
      if (!res.ok || !d.id) {
        console.error("[paypal] create failed", res.status, JSON.stringify(d).slice(0, 300));
        return reply(502, { ok: false, reason: "Could not start PayPal checkout." });
      }
      const approve = (d.links || []).find(
        (l) => l.rel === "approve" || l.rel === "payer-action",
      );
      if (!approve) {
        console.error("[paypal] no approve link", JSON.stringify(d).slice(0, 300));
        return reply(502, { ok: false, reason: "PayPal did not return an approval link." });
      }
      return reply(200, { ok: true, id: d.id, url: approve.href });
    }

    // ── capture: the authoritative step ──
    if (body.action === "capture") {
      const orderId = String(body.orderId || "").trim();
      if (!orderId) return reply(400, { ok: false, reason: "orderId required" });

      const res = await fetch(`${api()}/v2/checkout/orders/${encodeURIComponent(orderId)}/capture`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      });
      const d = await res.json();

      // PayPal returns 422 ORDER_ALREADY_CAPTURED on a double-submit; that is a
      // success from the user's point of view, so re-read the order instead of
      // failing someone who has already paid.
      let order = d;
      if (!res.ok) {
        const issue = ((d.details || [])[0] || {}).issue || "";
        if (issue !== "ORDER_ALREADY_CAPTURED") {
          console.error("[paypal] capture failed", res.status, JSON.stringify(d).slice(0, 300));
          return reply(502, { ok: false, reason: "PayPal could not complete that payment." });
        }
        const re = await fetch(`${api()}/v2/checkout/orders/${encodeURIComponent(orderId)}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        order = await re.json();
      }

      if (order.status !== "COMPLETED") {
        return reply(200, { ok: false, reason: `Payment status: ${order.status || "unknown"}.` });
      }

      const cap = (((order.purchase_units || [])[0] || {}).payments || {}).captures || [];
      const paid = cap[0] || {};
      const amount = parseFloat((paid.amount || {}).value || "0");
      if (!(amount + 0.5 >= PRICE_USD)) {
        return reply(200, { ok: false, reason: `Payment was $${amount.toFixed(2)}, expected $${PRICE_USD}.` });
      }

      // trust PayPal's own timestamp for when the money moved
      const ts = paid.create_time ? Date.parse(paid.create_time) : Date.now();
      const { paidUntil, warning } = await grantPaid({
        email, ref: `paypal:${paid.id || orderId}`, amountUsd: amount, from: "paypal", ts,
      });
      if (warning) console.error("[paypal] entitlement store:", warning);

      return reply(200, {
        ok: true, email, paidUntil, amountUsd: amount,
        reason: `PayPal payment verified — $${amount.toFixed(2)}.`,
        ...(warning ? { storeWarning: warning } : {}),
      });
    }

    return reply(400, { ok: false, reason: "action must be create or capture" });
  } catch (e) {
    console.error("[paypal]", e && e.message);
    return reply(502, { ok: false, reason: "PayPal error. " + String((e && e.message) || e).slice(0, 140) });
  }
};
