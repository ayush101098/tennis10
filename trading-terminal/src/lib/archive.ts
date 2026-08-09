import { promises as fs } from "fs";
import path from "path";

/**
 * Build-time reader for the committed match archive (data/matches/*.json).
 *
 * Server-only: it touches the filesystem, so it must never be imported from a
 * client component. Its whole purpose is to put real match content into the
 * exported HTML — the board was fetched entirely after hydration, so the page
 * Google received contained zero matches.
 */

export interface ArchivedMatch {
  id: string;
  date: string;
  tour: string;
  tournament: string;
  surface: string;
  round: string;
  bestOf: number;
  player1: string;
  player2: string;
  p1Rank: number;
  p2Rank: number;
  p1Prob: number;
  p2Prob: number;
  probMethod: string;
  odds1?: number;
  odds2?: number;
  edge1?: number;
  edge2?: number;
  status: string;
  startTs: number;
  winner?: 1 | 2;
}

const DIR = path.resolve(process.cwd(), "data", "matches");

const localDate = (d = new Date()): string =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

/** Every archived day, newest first. Empty when nothing has been archived yet. */
export async function archivedDays(): Promise<string[]> {
  try {
    const files = await fs.readdir(DIR);
    return files.filter(f => f.endsWith(".json")).map(f => f.replace(".json", "")).sort().reverse();
  } catch {
    return [];
  }
}

export async function matchesForDay(day: string): Promise<ArchivedMatch[]> {
  try {
    return JSON.parse(await fs.readFile(path.join(DIR, `${day}.json`), "utf8"));
  } catch {
    return [];
  }
}

/** Tour tier, duplicated from scheduleService — this module must stay
 *  server-only and importing the client module here would drag it along. */
function tier(tour: string): number {
  const t = (tour || "").toUpperCase();
  if (t.startsWith("ATP") || t.startsWith("WTA")) return 0;
  if (t.includes("CHALLENGER") || t.includes("W125")) return 1;
  return 2;
}

/**
 * The board's opening state, rendered into the HTML at build time.
 *
 * Ordered the way the live board orders it — tour tier before start time — so
 * hydration replaces like with like AND the indexed markup leads with the
 * matches people actually search for, rather than whichever ITF W15 happens to
 * start earliest. Finished matches are dropped: a static build can be hours
 * old, and listing completed matches as "today" would be wrong on the page and
 * wrong in the index.
 */
export async function todaysBoard(limit = 80): Promise<ArchivedMatch[]> {
  const day = localDate();
  const all = await matchesForDay(day);
  return all
    .filter(m => m.status !== "finished" && m.status !== "cancelled")
    .sort((a, b) => tier(a.tour) - tier(b.tour) || a.startTs - b.startTs)
    .slice(0, limit);
}
