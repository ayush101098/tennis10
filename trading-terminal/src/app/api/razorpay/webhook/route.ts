import { adapt } from "@/lib/netlifyAdapter";
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { handler } = require("../../../../../netlify/functions/razorpay-webhook");

/**
 * razorpay/webhook — served by the shared handler in
 * netlify/functions/razorpay-webhook.js.
 *
 * WHY IT WAS MISSING
 *   On Netlify this path is its own redirect, so it never needed a Next route.
 *   On Vercel the routes ARE the backend, and without this file
 *   POST /api/razorpay/webhook returns 404 — Razorpay's payment confirmations
 *   would be accepted by Razorpay and silently dropped by us, which is the
 *   worst possible failure mode for a payment webhook: the customer is charged
 *   and never entitled.
 */
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export const POST = adapt(handler);
