import { adapt } from "@/lib/netlifyAdapter";
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { handler } = require("../../../../../netlify/functions/stripe-confirm");

/**
 * stripe/confirm — served by the shared handler in netlify/functions/stripe-confirm.js.
 *
 * Deliberately not a reimplementation: one handler, one behaviour, whichever
 * host runs it. See src/lib/netlifyAdapter.ts.
 */
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export const GET = adapt(handler);
export const POST = adapt(handler);
