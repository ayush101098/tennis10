/**
 * Snapshot today's and tomorrow's matches into the committed archive.
 *
 * WHY AN ARCHIVE AT ALL
 *   The site is a static export: `next build` generates exactly the pages
 *   generateStaticParams returns, from whatever data exists at build time. So
 *   anything not written down disappears at the next deploy — yesterday's match
 *   pages would evaporate rather than becoming the permanent long-tail assets
 *   they are supposed to be. The archive is the memory.
 *
 * WHY IT RUNS HERE AND NOT ON NETLIFY
 *   SofaScore only answers the operator's machine (sofa_proxy.py impersonates
 *   Chrome's TLS handshake). A Netlify build cannot reach it. This runs beside
 *   push_sofa.py, writes JSON into the repo, and the commit triggers the build.
 *
 *   npx tsx scripts/archive-matches.ts [--base http://127.0.0.1:3001-backed site]
 */

import { promises as fs } from "fs";
import path from "path";
import {
  setApiBase, fetchScheduleClient, type ScheduledMatch,
} from "../src/lib/scheduleService";

const ARCHIVE = path.resolve(process.cwd(), "data", "matches");

/** Trimmed record — the page needs facts and model output, not the raw feed. */
export interface ArchivedMatch {
  id: string;
  date: string;              // YYYY-MM-DD (local tour date)
  tour: string;              // ATP | WTA | CH | W125 | ITF
  tournament: string;
  surface: string;
  round: string;
  bestOf: number;
  player1: string;
  player2: string;
  p1Rank: number;
  p2Rank: number;
  p1Prob: number;            // model True P
  p2Prob: number;
  probMethod: string;
  odds1?: number;            // de-vigged bookmaker price, when known
  odds2?: number;
  edge1?: number;
  edge2?: number;
  status: string;
  startTs: number;
  winner?: 1 | 2;            // filled once the match finishes
}

const slugify = (s: string): string =>
  s.normalize("NFD").replace(/[̀-ͯ]/g, "")
    .toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");

function toArchived(m: ScheduledMatch, date: string): ArchivedMatch {
  const o = m.prematchOdds;
  const rec: ArchivedMatch = {
    id: m.id,
    date,
    tour: m.tour,
    tournament: m.tournament,
    surface: m.surface,
    round: m.round || "",
    bestOf: m.best_of,
    player1: m.player1,
    player2: m.player2,
    p1Rank: m.p1_rank,
    p2Rank: m.p2_rank,
    p1Prob: m.p1_win_prob,
    p2Prob: m.p2_win_prob,
    probMethod: m.prob_method,
    status: m.status,
    startTs: m.start_timestamp,
  };
  if (o) {
    rec.odds1 = o.p1;
    rec.odds2 = o.p2;
    // Edge against the de-vigged price, the same quantity the terminal shows.
    const imp1 = 1 / o.p1, imp2 = 1 / o.p2;
    const vig = imp1 + imp2;
    rec.edge1 = Math.round((m.p1_win_prob - imp1 / vig) * 1e4) / 1e4;
    rec.edge2 = Math.round((m.p2_win_prob - imp2 / vig) * 1e4) / 1e4;
  }
  return rec;
}

async function main() {
  const baseArg = process.argv.indexOf("--base");
  const base = baseArg > -1 ? process.argv[baseArg + 1] : "https://tennisalpha.in";
  setApiBase(base);
  console.log(`archiving from ${base}`);

  const data = await fetchScheduleClient();
  await fs.mkdir(ARCHIVE, { recursive: true });

  for (const [day, matches] of [
    [data.today_date, data.today],
    [data.tomorrow_date, data.tomorrow],
  ] as [string, ScheduledMatch[]][]) {
    if (!matches.length) { console.log(`  ${day}: nothing to archive`); continue; }

    const file = path.join(ARCHIVE, `${day}.json`);
    // Merge, never truncate: a later run must not drop matches an earlier one
    // saw, and a finished match must keep the result it recorded.
    let existing: ArchivedMatch[] = [];
    try { existing = JSON.parse(await fs.readFile(file, "utf8")); } catch { /* first run */ }
    const byId = new Map(existing.map(r => [r.id, r]));
    for (const m of matches) {
      const rec = toArchived(m, day);
      const prev = byId.get(rec.id);
      // Keep a recorded winner — the live feed drops finished matches quickly.
      if (prev?.winner) rec.winner = prev.winner;
      if (m.status === "finished" && m.liveScore) {
        // Sets won = completed sets each player took. The feed exposes the set
        // list, not a tally, and a finished match with equal sets is a
        // retirement — recording a winner there would be a guess.
        const sets = m.liveScore.completedSets || [];
        const w1 = sets.filter(x => x.p1 > x.p2).length;
        const w2 = sets.filter(x => x.p2 > x.p1).length;
        if (w1 !== w2) rec.winner = w1 > w2 ? 1 : 2;
      }
      byId.set(rec.id, rec);
    }
    const out = [...byId.values()].sort((a, b) => a.startTs - b.startTs);
    await fs.writeFile(file, JSON.stringify(out, null, 0) + "\n");
    console.log(`  ${day}: ${out.length} matches (${matches.length} seen this run)`);
  }
}

export { slugify };

main().catch(e => { console.error(String(e?.message || e)); process.exit(1); });
