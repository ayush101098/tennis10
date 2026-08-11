import { NextRequest, NextResponse } from "next/server";

/**
 * Run a Netlify Function handler as a Next.js route.
 *
 * WHY THIS EXISTS
 *   There were two implementations of every endpoint: netlify/functions/*.js
 *   (production) and the route.ts files under src/app/api (local dev). They
 *   drifted badly: device binding, the free trial, payment claims and the
 *   leads resync existed only in the Netlify copy, and the Next copies
 *   persisted to local JSON files — which on a serverless host means every
 *   signup and payment silently disappears.
 *
 *   Rewriting fifteen handlers would have produced a third thing to keep in
 *   sync. Instead the Next route calls the SAME handler, so there is exactly one
 *   implementation and the drift cannot come back.
 *
 * The Netlify handler contract is a plain function:
 *   (event) => { statusCode, headers?, body? }
 * so the adapter's whole job is translating the request and response shapes.
 */

export interface NetlifyEvent {
  httpMethod: string;
  headers: Record<string, string>;
  body: string | null;
  queryStringParameters: Record<string, string>;
  path: string;
  rawQuery: string;
  rawUrl: string;
  isBase64Encoded: boolean;
}

export interface NetlifyResult {
  statusCode?: number;
  headers?: Record<string, string>;
  body?: string;
}

type Handler = (event: NetlifyEvent, context?: unknown) => Promise<NetlifyResult> | NetlifyResult;

/** Wrap a handler as a Next route method (export const GET = adapt(h)). */
export function adapt(handler: Handler) {
  return async function route(req: NextRequest): Promise<NextResponse> {
    const url = new URL(req.url);

    // GET/HEAD have no body to read, and calling .text() on them throws.
    const body = req.method === "GET" || req.method === "HEAD" ? null : await req.text();

    const headers: Record<string, string> = {};
    req.headers.forEach((v, k) => { headers[k] = v; });

    const queryStringParameters: Record<string, string> = {};
    url.searchParams.forEach((v, k) => { queryStringParameters[k] = v; });

    const event: NetlifyEvent = {
      httpMethod: req.method,
      headers,
      body,
      queryStringParameters,
      // The proxies parse their upstream path out of this, so it must be the
      // full request path exactly as Netlify would report it.
      path: url.pathname,
      rawQuery: url.search.replace(/^\?/, ""),
      rawUrl: req.url,
      isBase64Encoded: false,
    };

    try {
      const res = await handler(event, {});
      return new NextResponse(res.body ?? null, {
        status: res.statusCode ?? 200,
        headers: res.headers ?? { "Content-Type": "application/json" },
      });
    } catch (e) {
      // A thrown handler must not become an opaque 500 with no explanation —
      // these endpoints take money.
      console.error("[netlifyAdapter] handler threw:", e);
      return NextResponse.json(
        { ok: false, reason: "Internal error" },
        { status: 500, headers: { "Cache-Control": "no-store" } },
      );
    }
  };
}
