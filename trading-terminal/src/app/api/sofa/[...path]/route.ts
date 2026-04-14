import { NextRequest, NextResponse } from "next/server";

/**
 * Server-side proxy for SofaScore API.
 *
 * SofaScore's Varnish CDN blocks ALL programmatic HTTP requests via TLS
 * fingerprinting (returns 403 for curl, Node.js fetch, etc.).
 *
 * This API route forwards requests to a local Python micro-server
 * (sofa_proxy.py) running on port 3001 that uses `tls_client` to
 * impersonate Chrome's TLS handshake.
 *
 * On Netlify (static export) the equivalent Netlify Function at
 * /netlify/functions/sofa-proxy handles the same path.
 */

const SOFA_PROXY = process.env.SOFA_PROXY_URL || "http://127.0.0.1:3001";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path } = await params;
  const sofaPath = path.join("/");
  const url = `${SOFA_PROXY}/${sofaPath}`;

  try {
    const res = await fetch(url, { cache: "no-store" });

    if (!res.ok) {
      return NextResponse.json(
        { error: `SofaScore proxy returned ${res.status}` },
        { status: res.status },
      );
    }

    const data = await res.json();
    return NextResponse.json(data, {
      headers: {
        "Cache-Control": "public, s-maxage=30, stale-while-revalidate=60",
      },
    });
  } catch (err: unknown) {
    console.error("[sofa-proxy route]", err);
    return NextResponse.json(
      { error: "SofaScore proxy unavailable — is sofa_proxy.py running on port 3001?" },
      { status: 502 },
    );
  }
}
