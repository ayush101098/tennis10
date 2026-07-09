import { NextRequest, NextResponse } from "next/server";

/**
 * Lightweight live-presence counter.
 *
 * Each open terminal tab POSTs a heartbeat with a stable per-tab id every ~15s.
 * We keep the last-seen timestamp per id in module memory and report how many
 * ids were seen inside the freshness window. No database, no auth — good enough
 * to show "who's on the page right now" for a locally-run / single-instance
 * deployment. (On a static export with no server this route is absent and the
 * client simply hides the indicator.)
 */

export const dynamic = "force-dynamic";

const WINDOW_MS = 30_000; // an id is "online" if seen within the last 30s
const seen = new Map<string, number>();

function prune(now: number) {
  for (const [id, ts] of seen) {
    if (now - ts > WINDOW_MS) seen.delete(id);
  }
}

export async function POST(req: NextRequest) {
  const now = Date.now();
  let id = "";
  try {
    const body = await req.json();
    id = typeof body?.id === "string" ? body.id : "";
  } catch {
    /* ignore malformed body */
  }
  if (id) seen.set(id, now);
  prune(now);
  return NextResponse.json({ count: seen.size }, {
    headers: { "Cache-Control": "no-store" },
  });
}

export async function GET() {
  const now = Date.now();
  prune(now);
  return NextResponse.json({ count: seen.size }, {
    headers: { "Cache-Control": "no-store" },
  });
}
