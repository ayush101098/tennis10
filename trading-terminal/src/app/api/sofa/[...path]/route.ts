import { adapt } from "@/lib/netlifyAdapter";
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { handler } = require("../../../../../netlify/functions/sofa-proxy");

/**
 * sofa/[...path] — served by the shared handler in netlify/functions/sofa-proxy.js.
 *
 * Deliberately not a reimplementation: one handler, one behaviour, whichever
 * host runs it. See src/lib/netlifyAdapter.ts.
 */
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export const GET = adapt(handler);
export const POST = adapt(handler);
