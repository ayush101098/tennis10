import { adapt } from "@/lib/netlifyAdapter";
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { handler } = require("../../../../../netlify/functions/pm-proxy");

/**
 * pm/[...path] — served by the shared handler in netlify/functions/pm-proxy.js.
 *
 * WHY THIS STOPPED BEING A SEPARATE IMPLEMENTATION
 *   It was the last standalone route, and it sent `no-cache, no-store`. On
 *   Netlify that did not matter, because the redirect sends /api/pm/* to the
 *   function and the route is stripped at build time. On VERCEL this route IS
 *   the backend — so the standalone copy would have made every Polymarket
 *   request a serverless invocation with no CDN caching, reintroducing exactly
 *   the per-visitor upstream cost the proxy was written to remove, on the host
 *   we are moving to.
 *
 *   Sharing the handler means the host-aware cache lifetimes (gamma 30s,
 *   clob 3s, data 60s, lb 300s) apply on both, and there is one place where
 *   that policy lives.
 */
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export const GET = adapt(handler);
