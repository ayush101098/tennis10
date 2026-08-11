import { NextRequest, NextResponse } from "next/server";

/**
 * Local-dev mirror of netlify/functions/stripe-checkout.js. The static export
 * strips src/app/api, so production uses the Netlify function via the
 * netlify.toml redirect; this exists so the flow is testable with `next dev`.
 */

export const dynamic = "force-dynamic";

const PRO_PRICE_USD = 100;
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export async function POST(req: NextRequest) {
  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) {
    return NextResponse.json(
      { ok: false, reason: "Card payments aren't configured — set STRIPE_SECRET_KEY in .env.local." },
      { status: 503 },
    );
  }

  let body: Record<string, unknown> = {};
  try { body = await req.json(); } catch { /* ignore */ }
  const email = String(body.email || "").trim().toLowerCase();
  if (!EMAIL_RE.test(email)) {
    return NextResponse.json({ ok: false, reason: "A valid email is required." }, { status: 400 });
  }

  try {
    const Stripe = (await import("stripe")).default;
    const stripe = new Stripe(key);
    const site = process.env.SITE_URL || req.nextUrl.origin;

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
      success_url: `${site}/terminal?checkout=success&session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${site}/?checkout=cancelled`,
      allow_promotion_codes: true,
    });

    return NextResponse.json({ ok: true, url: session.url, id: session.id });
  } catch (e) {
    return NextResponse.json(
      { ok: false, reason: "Could not start checkout. " + String(e).slice(0, 160) },
      { status: 500 },
    );
  }
}
