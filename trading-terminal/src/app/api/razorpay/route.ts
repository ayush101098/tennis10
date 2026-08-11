import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";

/** Local-dev mirror of netlify/functions/razorpay.js (see it for the full
 *  contract and the two-step verification rationale). */
export const dynamic = "force-dynamic";

const API = "https://api.razorpay.com/v1";
const amountInr = () => Math.max(1, parseInt(process.env.RAZORPAY_AMOUNT_INR || "8900", 10));

const auth = () => "Basic " + Buffer.from(
  `${process.env.RAZORPAY_KEY_ID}:${process.env.RAZORPAY_KEY_SECRET}`).toString("base64");

function safeEqual(a: string, b: string) {
  const x = Buffer.from(a, "utf8"), y = Buffer.from(b, "utf8");
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}

export async function POST(req: NextRequest) {
  const keyId = process.env.RAZORPAY_KEY_ID, keySecret = process.env.RAZORPAY_KEY_SECRET;
  if (!keyId || !keySecret) {
    return NextResponse.json(
      { ok: false, reason: "Razorpay isn't configured — set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET." },
      { status: 503 });
  }
  let body: Record<string, unknown> = {};
  try { body = await req.json(); } catch { /* ignore */ }
  const email = String(body.email || "").trim().toLowerCase();
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return NextResponse.json({ ok: false, reason: "A valid email is required." }, { status: 400 });
  }
  try {
    if (body.action === "create") {
      const inr = amountInr();
      const res = await fetch(`${API}/orders`, {
        method: "POST", headers: { Authorization: auth(), "Content-Type": "application/json" },
        body: JSON.stringify({
          amount: inr * 100, currency: "INR", receipt: `tt_${Date.now()}`,
          notes: { email, product: "Tennis Intelligence Terminal Pro (30 days)" },
        }),
      });
      const d = await res.json();
      if (!res.ok || !d.id) {
        return NextResponse.json({ ok: false, reason: "Could not start the Razorpay payment." }, { status: 502 });
      }
      return NextResponse.json({ ok: true, orderId: d.id, keyId, amount: d.amount, currency: d.currency, amountInr: inr });
    }

    if (body.action === "verify") {
      const orderId = String(body.razorpay_order_id || "");
      const paymentId = String(body.razorpay_payment_id || "");
      const signature = String(body.razorpay_signature || "");
      if (!orderId || !paymentId || !signature) {
        return NextResponse.json({ ok: false, reason: "Missing payment confirmation fields." }, { status: 400 });
      }
      const expected = crypto.createHmac("sha256", keySecret).update(`${orderId}|${paymentId}`).digest("hex");
      if (!safeEqual(expected, signature)) {
        return NextResponse.json({ ok: false, reason: "Payment signature could not be verified." }, { status: 400 });
      }
      const res = await fetch(`${API}/payments/${encodeURIComponent(paymentId)}`, { headers: { Authorization: auth() } });
      const p = await res.json();
      if (!res.ok || !p.id) {
        return NextResponse.json({ ok: false, reason: "Could not confirm the payment with Razorpay." }, { status: 502 });
      }
      if (p.order_id !== orderId) {
        return NextResponse.json({ ok: false, reason: "That payment belongs to a different order." }, { status: 400 });
      }
      if (p.status !== "captured") {
        return NextResponse.json({ ok: false, reason: `Payment status: ${p.status}.` });
      }
      const paidInr = Math.round((p.amount || 0) / 100);
      const ts = p.created_at ? p.created_at * 1000 : Date.now();
      const paidUntil = ts + 30 * 86400000;
      try {
        await fetch(new URL("/api/account", req.nextUrl.origin), {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email, txHash: `razorpay:${p.id}`, from: "razorpay", ts,
            amountUsd: Math.round(paidInr / Number(process.env.USD_INR || "89")),
          }),
        });
      } catch { /* bookkeeping only */ }
      return NextResponse.json({ ok: true, email, paidUntil, amountInr: paidInr, reason: `Payment verified — ₹${paidInr}.` });
    }
    return NextResponse.json({ ok: false, reason: "action must be create or verify" }, { status: 400 });
  } catch (e) {
    return NextResponse.json({ ok: false, reason: "Razorpay error. " + String(e).slice(0, 140) }, { status: 502 });
  }
}
