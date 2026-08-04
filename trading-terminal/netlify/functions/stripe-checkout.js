/**
 * Stripe Checkout — start a card subscription for the terminal.
 *
 *   POST /api/stripe/checkout  { email }  ->  { url }
 *
 * Creates a $PRO_PRICE/month subscription Checkout Session and returns the
 * hosted URL for the browser to redirect to. The price is defined inline with
 * price_data, so there is nothing to configure in the Stripe dashboard first.
 *
 * The email is carried in three places on purpose — customer_email (prefills
 * and receipts), metadata.email and subscription_data.metadata.email — because
 * the webhook and the confirm endpoint each read it from a different object,
 * and an entitlement with no email attached is unusable.
 *
 * Env: STRIPE_SECRET_KEY (required), SITE_URL (optional, for return URLs).
 */

const PRO_PRICE_USD = 100;
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const reply = (statusCode, obj) => ({
  statusCode,
  headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  body: JSON.stringify(obj),
});

function siteUrlFrom(event) {
  if (process.env.SITE_URL) return process.env.SITE_URL.replace(/\/$/, "");
  if (process.env.URL) return process.env.URL.replace(/\/$/, "");   // Netlify sets this
  const proto = event.headers["x-forwarded-proto"] || "https";
  const host = event.headers.host || event.headers.Host;
  return host ? `${proto}://${host}` : "";
}

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") return reply(405, { error: "method not allowed" });

  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) {
    return reply(503, {
      ok: false,
      reason: "Card payments aren't configured yet. Set STRIPE_SECRET_KEY in the site env.",
    });
  }

  let body = {};
  try { body = JSON.parse(event.body || "{}"); } catch { /* ignore */ }
  const email = String(body.email || "").trim().toLowerCase();
  if (!EMAIL_RE.test(email)) {
    return reply(400, { ok: false, reason: "A valid email is required." });
  }

  try {
    const Stripe = require("stripe");
    const stripe = new Stripe(key);
    const site = siteUrlFrom(event);

    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      customer_email: email,
      client_reference_id: email,
      metadata: { email },
      subscription_data: { metadata: { email } },
      line_items: [{
        quantity: 1,
        price_data: {
          currency: "usd",
          unit_amount: PRO_PRICE_USD * 100,
          recurring: { interval: "month" },
          product_data: {
            name: "Tennis Intelligence Terminal — Pro",
            description:
              "Live True P for tennis and table tennis, edge vs bookmaker, "
              + "Value Board, hedge timing and the bet tracker.",
          },
        },
      }],
      // session_id lets the return page confirm entitlement straight from
      // Stripe, so access does not depend on the webhook having landed yet.
      success_url: `${site}/terminal?checkout=success&session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${site}/?checkout=cancelled`,
      allow_promotion_codes: true,
    });

    return reply(200, { ok: true, url: session.url, id: session.id });
  } catch (e) {
    console.error("[stripe-checkout]", e && e.message);
    return reply(500, {
      ok: false,
      reason: "Could not start checkout. " + String((e && e.message) || e).slice(0, 160),
    });
  }
};
