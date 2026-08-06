import { NextRequest, NextResponse } from "next/server";

/**
 * Server-side ESPN scoreboard proxy (dev twin of netlify/functions/espn-proxy.js).
 *
 * ESPN answers browser-shaped requests with 403, and a 403 carries no CORS
 * headers — so fetching it straight from the page fails and the match board
 * renders empty. Fetched from the server instead, where none of that applies.
 *
 * The static export strips this route; Netlify serves the same path via the
 * function + redirect in netlify.toml.
 */

const ESPN = "https://site.api.espn.com/apis/site/v2/sports/tennis";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest, ctx: { params: { path: string[] } }) {
  const path = (ctx.params.path || []).join("/");
  // not an open relay — scoreboard endpoints only
  if (!/^(atp|wta)\/scoreboard$/.test(path)) {
    return NextResponse.json({ error: "unsupported path", path }, { status: 400 });
  }
  const qs = req.nextUrl.search || "";
  try {
    const res = await fetch(`${ESPN}/${path}${qs}`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!res.ok) return NextResponse.json({ error: `espn ${res.status}` }, { status: 502 });
    return NextResponse.json(await res.json(), { headers: { "Cache-Control": "no-store" } });
  } catch (e) {
    return NextResponse.json({ error: String(e).slice(0, 160) }, { status: 502 });
  }
}
