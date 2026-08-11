import { adapt } from "@/lib/netlifyAdapter";
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { handler } = require("../../../../netlify/functions/razorpay");

/**
 * razorpay — served by the shared handler in netlify/functions/razorpay.js.
 *
 * Deliberately not a reimplementation: one handler, one behaviour, whichever
 * host runs it. See src/lib/netlifyAdapter.ts.
 */
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export const POST = adapt(handler);
