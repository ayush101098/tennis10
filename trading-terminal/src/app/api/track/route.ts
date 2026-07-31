import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

/**
 * Local-dev pageview tracking (mirrors netlify/functions/track.js). Appends to a
 * gitignored JSON ring so traffic is visible during `next dev`. Stripped from the
 * static export; production uses the Netlify function via the netlify.toml redirect.
 */

export const dynamic = "force-dynamic";

const FILE = path.join(process.cwd(), "analytics.local.json");
const CAP = 5000;

export async function POST(req: NextRequest) {
  let b: Record<string, unknown> = {};
  try { b = await req.json(); } catch { /* ignore */ }
  const rec = {
    ts: Date.now(),
    path: String(b.path || "/").slice(0, 120),
    ref: String(b.ref || "").slice(0, 200),
    vid: String(b.vid || "").slice(0, 40),
  };
  let events: unknown[] = [];
  try { events = JSON.parse(await fs.readFile(FILE, "utf8")); } catch { /* new */ }
  events.push(rec);
  if (events.length > CAP) events = events.slice(-CAP);
  await fs.writeFile(FILE, JSON.stringify(events));
  return NextResponse.json({ ok: true }, { headers: { "Cache-Control": "no-store" } });
}
