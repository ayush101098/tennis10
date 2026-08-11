import { NextRequest, NextResponse } from "next/server";

/**
 * Server-side proxy for Polymarket's public data APIs, so the terminal's
 * smart-money / edge dashboard reads wallet history through localhost instead
 * of the browser hitting Polymarket directly (avoids CORS + rate-limit noise
 * and keeps one caching seam).
 *
 * Path prefix selects the upstream host:
 *   /api/pm/data/<path>  -> https://data-api.polymarket.com/<path>   (trades, positions, activity)
 *   /api/pm/lb/<path>    -> https://lb-api.polymarket.com/<path>     (leaderboard)
 *   /api/pm/clob/<path>  -> https://clob.polymarket.com/<path>       (books, markets)
 *   /api/pm/gamma/<path> -> https://gamma-api.polymarket.com/<path>  (events, markets)
 *
 * The incoming query string is forwarded verbatim.
 */

const HOSTS: Record<string, string> = {
  data: "https://data-api.polymarket.com",
  lb: "https://lb-api.polymarket.com",
  clob: "https://clob.polymarket.com",
  gamma: "https://gamma-api.polymarket.com",
};

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path } = await params;
  const [key, ...rest] = path;
  const base = HOSTS[key];
  if (!base) {
    return NextResponse.json(
      { error: `unknown polymarket host '${key}'; use one of ${Object.keys(HOSTS).join(", ")}` },
      { status: 400 },
    );
  }

  const qs = req.nextUrl.search;
  const url = `${base}/${rest.join("/")}${qs}`;

  try {
    const res = await fetch(url, {
      cache: "no-store",
      headers: { "User-Agent": "tennis10-terminal/2.0", Accept: "application/json" },
    });
    if (!res.ok) {
      return NextResponse.json(
        { error: `polymarket ${key} returned ${res.status}`, url },
        { status: res.status },
      );
    }
    const data = await res.json();
    return NextResponse.json(data, {
      headers: { "Cache-Control": "no-cache, no-store, must-revalidate" },
    });
  } catch (err: unknown) {
    console.error("[pm-proxy route]", err);
    return NextResponse.json({ error: "polymarket proxy unavailable", url }, { status: 502 });
  }
}
