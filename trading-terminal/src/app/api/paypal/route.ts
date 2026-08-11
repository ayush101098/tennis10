import { NextRequest, NextResponse } from "next/server";

/** Local-dev mirror of netlify/functions/paypal.js (see it for the full contract).
 *  Production uses the Netlify function via the netlify.toml redirect. */
export const dynamic = "force-dynamic";

const PRICE_USD = 100;
const api = () =>
  (process.env.PAYPAL_ENV || "live").toLowerCase() === "sandbox"
    ? "https://api-m.sandbox.paypal.com" : "https://api-m.paypal.com";

async function token() {
  const res = await fetch(`${api()}/v1/oauth2/token`, {
    method: "POST",
    headers: {
      Authorization: "Basic " + Buffer.from(
        `${process.env.PAYPAL_CLIENT_ID}:${process.env.PAYPAL_CLIENT_SECRET}`).toString("base64"),
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: "grant_type=client_credentials",
  });
  const d = await res.json();
  if (!res.ok || !d.access_token) throw new Error(`paypal auth ${res.status}`);
  return d.access_token as string;
}

export async function POST(req: NextRequest) {
  if (!process.env.PAYPAL_CLIENT_ID || !process.env.PAYPAL_CLIENT_SECRET) {
    return NextResponse.json(
      { ok: false, reason: "PayPal isn't configured — set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET." },
      { status: 503 });
  }
  let body: Record<string, unknown> = {};
  try { body = await req.json(); } catch { /* ignore */ }
  const email = String(body.email || "").trim().toLowerCase();
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return NextResponse.json({ ok: false, reason: "A valid email is required." }, { status: 400 });
  }
  try {
    const t = await token();
    const site = process.env.SITE_URL || req.nextUrl.origin;
    if (body.action === "create") {
      const res = await fetch(`${api()}/v2/checkout/orders`, {
        method: "POST",
        headers: { Authorization: `Bearer ${t}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          intent: "CAPTURE",
          purchase_units: [{ amount: { currency_code: "USD", value: PRICE_USD.toFixed(2) }, custom_id: email }],
          application_context: {
            brand_name: "Tennis Intelligence Terminal", user_action: "PAY_NOW",
            shipping_preference: "NO_SHIPPING",
            return_url: `${site}/terminal?paypal=success`, cancel_url: `${site}/?paypal=cancelled`,
          },
        }),
      });
      const d = await res.json();
      const approve = (d.links || []).find((l: { rel: string }) => l.rel === "approve" || l.rel === "payer-action");
      if (!res.ok || !d.id || !approve) {
        return NextResponse.json({ ok: false, reason: "Could not start PayPal checkout." }, { status: 502 });
      }
      return NextResponse.json({ ok: true, id: d.id, url: approve.href });
    }
    if (body.action === "capture") {
      const orderId = String(body.orderId || "");
      const res = await fetch(`${api()}/v2/checkout/orders/${encodeURIComponent(orderId)}/capture`, {
        method: "POST", headers: { Authorization: `Bearer ${t}`, "Content-Type": "application/json" },
      });
      const d = await res.json();
      if (d.status !== "COMPLETED") {
        return NextResponse.json({ ok: false, reason: `Payment status: ${d.status || "unknown"}.` });
      }
      const paid = (((d.purchase_units || [])[0] || {}).payments || {}).captures?.[0] || {};
      const amount = parseFloat(paid.amount?.value || "0");
      const ts = paid.create_time ? Date.parse(paid.create_time) : Date.now();
      const paidUntil = ts + 30 * 86400000;
      try {
        await fetch(new URL("/api/account", req.nextUrl.origin), {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, txHash: `paypal:${paid.id || orderId}`, amountUsd: amount, from: "paypal", ts }),
        });
      } catch { /* bookkeeping only */ }
      return NextResponse.json({ ok: true, email, paidUntil, amountUsd: amount, reason: `PayPal payment verified — $${amount.toFixed(2)}.` });
    }
    return NextResponse.json({ ok: false, reason: "action must be create or capture" }, { status: 400 });
  } catch (e) {
    return NextResponse.json({ ok: false, reason: "PayPal error. " + String(e).slice(0, 140) }, { status: 502 });
  }
}
