import { adapt } from "@/lib/netlifyAdapter";
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { handler } = require("../../../../netlify/functions/google-auth");

/**
 * google-auth — served by the shared handler in netlify/functions/google-auth.js.
 *
 * This route was missing: the endpoint existed only as a Netlify function and a
 * netlify.toml redirect, both of which stopped applying when hosting moved to
 * Vercel. /api/google-auth returned the HTML 404 page, so any Google sign-in
 * would have failed with a parse error rather than a message.
 */
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export const POST = adapt(handler);
