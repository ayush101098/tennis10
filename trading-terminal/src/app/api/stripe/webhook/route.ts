import { NextRequest, NextResponse } from "next/server";

/**
 * Local-dev mirror of netlify/functions/stripe-webhook.js.
 *
 * Test it with the Stripe CLI, which supplies real signed events:
 *   stripe listen --forward-to localhost:3111/api/stripe/webhook
 *   stripe trigger checkout.session.completed
 *
 * The signature is verified against the RAW body (req.text(), never req.json()),
 * because re-serialising the payload changes the bytes and every event would be
 * rejected.
 */

export const dynamic = "force-dynamic";

const SUBSCRIPTION_DAYS = 30;

export async function POST(req: NextRequest) {
  const key = process.env.STRIPE_SECRET_KEY;
  const whSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!key || !whSecret) {
    return NextResponse.json({ error: "stripe not configured" }, { status: 503 });
  }
  const sig = req.headers.get("stripe-signature");
  if (!sig) return NextResponse.json({ error: "missing stripe-signature" }, { status: 400 });

  const raw = await req.text();   // MUST be the raw body, not parsed JSON

  const Stripe = (await import("stripe")).default;
  const stripe = new Stripe(key);

  let evt: import("stripe").Stripe.Event;
  try {
    evt = stripe.webhooks.constructEvent(raw, sig, whSecret);
  } catch (e) {
    return NextResponse.json(
      { error: "invalid signature: " + String(e).slice(0, 120) }, { status: 400 },
    );
  }

  const grant = async (email: string, ref: string, amountUsd: number, ts: number) => {
    await fetch(new URL("/api/account", req.nextUrl.origin), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, txHash: ref, amountUsd, from: "stripe", ts }),
    });
    return ts + SUBSCRIPTION_DAYS * 86400000;
  };

  try {
    if (evt.type === "checkout.session.completed") {
      const o = evt.data.object;
      if (o.payment_status !== "paid") return NextResponse.json({ received: true, skipped: "unpaid" });
      const email = String(o.metadata?.email || o.customer_email || o.client_reference_id || "")
        .trim().toLowerCase();
      if (!email) return NextResponse.json({ received: true, skipped: "no email" });
      const until = await grant(email, `stripe:${o.id}`,
        Math.round((o.amount_total || 0) / 100), (o.created || 0) * 1000 || Date.now());
      return NextResponse.json({ received: true, email, paidUntil: until });
    }

    if (evt.type === "invoice.paid") {
      const o = evt.data.object as unknown as {
        id: string; amount_paid?: number; created?: number;
        metadata?: Record<string, string>; customer_email?: string;
      };
      const email = String(o.metadata?.email || o.customer_email || "").trim().toLowerCase();
      if (!email) return NextResponse.json({ received: true, skipped: "no email" });
      const until = await grant(email, `stripe:${o.id}`,
        Math.round((o.amount_paid || 0) / 100), (o.created || 0) * 1000 || Date.now());
      return NextResponse.json({ received: true, email, paidUntil: until });
    }

    return NextResponse.json({ received: true, ignored: evt.type });
  } catch (e) {
    // 500 so Stripe retries a transient failure.
    return NextResponse.json({ error: String(e).slice(0, 200) }, { status: 500 });
  }
}
