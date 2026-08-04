import { NextRequest, NextResponse } from "next/server";

/**
 * Local-dev mirror of netlify/functions/stripe-confirm.js — asks Stripe whether
 * a Checkout session is genuinely paid. Authoritative and storage-independent,
 * the same way the crypto flow trusts the chain rather than the browser.
 */

export const dynamic = "force-dynamic";

const SUBSCRIPTION_DAYS = 30;

export async function GET(req: NextRequest) {
  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) {
    return NextResponse.json({ ok: false, reason: "Card payments aren't configured." }, { status: 503 });
  }
  const sessionId = String(req.nextUrl.searchParams.get("session_id") || "").trim();
  if (!/^cs_[A-Za-z0-9_]+$/.test(sessionId)) {
    return NextResponse.json({ ok: false, reason: "A valid session_id is required." }, { status: 400 });
  }

  try {
    const Stripe = (await import("stripe")).default;
    const stripe = new Stripe(key);
    const session = await stripe.checkout.sessions.retrieve(sessionId);

    if (session.payment_status !== "paid") {
      return NextResponse.json({ ok: false, reason: `Payment status: ${session.payment_status}.` });
    }
    const email = String(
      session.metadata?.email || session.customer_email || session.client_reference_id || "",
    ).trim().toLowerCase();
    if (!email) {
      return NextResponse.json({ ok: false, reason: "No email attached to that payment." });
    }
    const paidFrom = session.created ? session.created * 1000 : Date.now();
    const paidUntil = paidFrom + SUBSCRIPTION_DAYS * 86400000;
    const amountUsd = Math.round((session.amount_total || 0) / 100);

    // Bookkeeping only — never block access on it.
    try {
      await fetch(new URL("/api/account", req.nextUrl.origin), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email, txHash: `stripe:${session.id}`, amountUsd, from: "stripe", ts: paidFrom,
        }),
      });
    } catch { /* access is already granted below */ }

    return NextResponse.json({
      ok: true, email, paidUntil, amountUsd,
      reason: `Card payment verified — $${amountUsd}.`,
    });
  } catch (e) {
    return NextResponse.json(
      { ok: false, reason: "Could not confirm with Stripe. " + String(e).slice(0, 140) },
      { status: 502 },
    );
  }
}
