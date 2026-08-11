import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

/**
 * Table-tennis intelligence feed for the /tt terminal.
 *
 * Serves the tabletennis/ project's artifacts straight from disk (the repo is
 * a monorepo sibling): predictions.json (pre-match model), live_predictions.json
 * (8s in-play poller — analytic recursion + character residual) and metrics.json
 * (held-out model transparency). Local-dev only, like /api/sofa — the pipeline
 * (`python -m tabletennis.pipeline --serve` + `python -m tabletennis.live`)
 * must be running for fresh data; stale files are still served with their
 * generated_ts so the client can flag staleness.
 */

export const dynamic = "force-dynamic";

const SITE = path.resolve(process.cwd(), "..", "tabletennis", "site");

async function readJson(name: string): Promise<unknown | null> {
  try {
    return JSON.parse(await fs.readFile(path.join(SITE, name), "utf8"));
  } catch {
    return null;
  }
}

export async function GET() {
  const [predictions, live, metrics] = await Promise.all([
    readJson("predictions.json"),
    readJson("live_predictions.json"),
    readJson("metrics.json"),
  ]);
  return NextResponse.json(
    { predictions, live, metrics },
    { headers: { "Cache-Control": "no-store" } },
  );
}
