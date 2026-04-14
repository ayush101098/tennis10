/**
 * Client-side tennis schedule service.
 * Fetches ATP / WTA matches from ESPN API (CORS-enabled, no backend).
 * Filters by actual match date, computes Elo-based probabilities.
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export interface ScheduledMatch {
  id: string;
  player1: string;
  player2: string;
  p1_rank: number;
  p2_rank: number;
  p1_seed: number;
  p2_seed: number;
  tournament: string;
  round: string;
  tour: string;
  surface: string;
  best_of: number;
  source: string;
  status: "scheduled" | "live" | "finished" | "cancelled";
  start_time: string;
  start_timestamp: number;
  p1_win_prob: number;
  p2_win_prob: number;
  prob_method: string;
  venue?: string;
  score?: { p1_sets: number[]; p2_sets: number[]; winner?: 1 | 2 };
}

export interface ScheduleData {
  today: ScheduledMatch[];
  tomorrow: ScheduledMatch[];
  today_date: string;
  tomorrow_date: string;
  fetched_at: number;
}

// ─── ESPN ────────────────────────────────────────────────────────────────────

const ESPN = "https://site.api.espn.com/apis/site/v2/sports/tennis";

function detectSurface(event: Record<string, unknown>): string {
  const v = (event.venue as Record<string, unknown>)?.surface as string || "";
  const n = ((event.name as string) || "").toLowerCase();
  const vs = v.toLowerCase();
  if (vs.includes("clay") || vs.includes("terre")) return "Clay";
  if (vs.includes("grass") || vs.includes("lawn")) return "Grass";
  if (vs.includes("hard") || vs.includes("deco") || vs.includes("plexi") || vs.includes("laykold")) return "Hard";
  if (/roland garros|barcelona|monte.carlo|rome|madrid|geneva|lyon|hamburg|buenos aires|rio|umag|bastad|kitzb|gstaad|parma|prague|bucharest|palermo|rabat|bogota|oeiras|marrakech|cordoba|houston|winston.salem clay/i.test(n)) return "Clay";
  if (/wimbledon|halle|queen|s-hertogenbosch|nottingham|mallorca|bad homburg|eastbourne|berlin grass|stuttgart grass/i.test(n)) return "Grass";
  return "Hard";
}

function detectBestOf(name: string, tour: string): number {
  const n = name.toLowerCase();
  if (tour === "ATP" && /australian open|roland garros|french open|wimbledon|us open/i.test(n)) return 5;
  return 3;
}

/** Get local YYYY-MM-DD for a Date object (not UTC!) */
function localDateStr(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${dd}`;
}

async function fetchESPN(
  tour: "atp" | "wta",
  targetDate: string, // YYYY-MM-DD — only return matches on this local date
): Promise<ScheduledMatch[]> {
  const espnDate = targetDate.replace(/-/g, "");
  const out: ScheduledMatch[] = [];
  try {
    const res = await fetch(`${ESPN}/${tour}/scoreboard?dates=${espnDate}`);
    if (!res.ok) return [];
    const data = await res.json();

    for (const event of data.events || []) {
      const tName = event.name || "";
      const surface = detectSurface(event);
      const bestOf = detectBestOf(tName, tour.toUpperCase());

      // Gather all competitions
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const allComps: any[] = [];
      for (const g of event.groupings || []) {
        for (const c of g.competitions || []) allComps.push(c);
      }
      if (!allComps.length) {
        for (const c of event.competitions || []) allComps.push(c);
      }

      for (const comp of allComps) {
        // ── Filter by actual match date ──
        const rawDate = comp.startDate || comp.date || "";
        if (!rawDate) continue;
        let matchDt: Date;
        try {
          matchDt = new Date(rawDate);
        } catch {
          continue;
        }
        const matchLocalDate = localDateStr(matchDt);
        if (matchLocalDate !== targetDate) continue;

        // ── Singles only: type=athlete ──
        const competitors = comp.competitors || [];
        if (competitors.length < 2) continue;
        // Skip doubles (type=team with roster)
        if (competitors[0]?.type === "team") continue;

        let p1d = competitors[0];
        let p2d = competitors[1];
        for (const c of competitors) {
          if (c.order === 1) p1d = c;
          else if (c.order === 2) p2d = c;
        }

        const p1a = typeof p1d.athlete === "object" ? p1d.athlete : null;
        const p2a = typeof p2d.athlete === "object" ? p2d.athlete : null;
        const p1Name = p1a?.displayName || p1a?.fullName || "";
        const p2Name = p2a?.displayName || p2a?.fullName || "";
        if (!p1Name || !p2Name) continue;

        const p1Seed = parseInt(p1d.seed || "0") || 0;
        const p2Seed = parseInt(p2d.seed || "0") || 0;
        let p1Rank = 0, p2Rank = 0;
        if (p1a?.rankings?.length) p1Rank = p1a.rankings[0].current || 0;
        if (p2a?.rankings?.length) p2Rank = p2a.rankings[0].current || 0;
        if (!p1Rank && p1Seed) p1Rank = p1Seed;
        if (!p2Rank && p2Seed) p2Rank = p2Seed;

        const state = comp.status?.type?.state || "pre";
        const status: ScheduledMatch["status"] =
          state === "in" ? "live" : state === "post" ? "finished" : "scheduled";

        const startTime = matchDt.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
        const startTs = matchDt.getTime() / 1000;

        let roundName = "";
        if (comp.round?.displayName) roundName = comp.round.displayName;
        else if (comp.notes?.length) roundName = comp.notes[0].headline || "";

        let score: ScheduledMatch["score"];
        if (status !== "scheduled") {
          const p1s = (p1d.linescores || []).map((l: { value: number }) => l.value);
          const p2s = (p2d.linescores || []).map((l: { value: number }) => l.value);
          if (p1s.length || p2s.length) {
            score = { p1_sets: p1s, p2_sets: p2s, winner: p1d.winner ? 1 : p2d.winner ? 2 : undefined };
          }
        }

        const venue = comp.venue?.fullName || "";
        const { p1_prob, p2_prob, method } = computeProb(p1Rank, p2Rank, p1Seed, p2Seed, surface, bestOf);

        out.push({
          id: `espn_${tour}_${event.id}_${comp.id}`,
          player1: p1Name, player2: p2Name,
          p1_rank: p1Rank, p2_rank: p2Rank,
          p1_seed: p1Seed, p2_seed: p2Seed,
          tournament: tName, round: roundName,
          tour: tour.toUpperCase(), surface, best_of: bestOf,
          source: "espn", status, start_time: startTime,
          start_timestamp: startTs,
          p1_win_prob: p1_prob, p2_win_prob: p2_prob,
          prob_method: method, venue, score,
        });
      }
    }
  } catch (e) {
    console.warn(`ESPN ${tour} fetch failed`, e);
  }
  return out;
}

// ─── Elo probability ─────────────────────────────────────────────────────────

function elo(rank: number): number {
  if (rank <= 0) return 1700;
  if (rank === 1) return 2400;
  return Math.max(1300, 2400 - 180 * Math.log(rank));
}

function eloP(e1: number, e2: number): number {
  return 1 / (1 + Math.pow(10, (e2 - e1) / 400));
}

function computeProb(
  r1: number, r2: number, s1: number, s2: number,
  surface: string, bestOf: number,
): { p1_prob: number; p2_prob: number; method: string } {
  let method = "ranking";
  let p: number;

  if (r1 > 0 && r2 > 0) {
    p = eloP(elo(r1), elo(r2));
  } else if (s1 > 0 && s2 > 0) {
    method = "seed"; p = eloP(elo(s1), elo(s2));
  } else if (r1 > 0 || r2 > 0) {
    p = eloP(elo(r1 || 150), elo(r2 || 150));
  } else if (s1 > 0 || s2 > 0) {
    method = "seed"; p = eloP(elo(s1 || 16), elo(s2 || 16));
  } else {
    return { p1_prob: 0.5, p2_prob: 0.5, method: "unknown" };
  }

  // Surface adjustment
  if (surface === "Clay") p = p * 0.92 + 0.04;
  else if (surface === "Grass") p = p * 1.03 - 0.015;

  // Bo5 adjustment
  if (bestOf === 5) {
    const q = 1 - p;
    p = Math.pow(p, 1.22) / (Math.pow(p, 1.22) + Math.pow(q, 1.22));
  }

  p = Math.max(0.03, Math.min(0.97, p));
  return { p1_prob: Math.round(p * 1e4) / 1e4, p2_prob: Math.round((1 - p) * 1e4) / 1e4, method };
}

// ─── Public API ──────────────────────────────────────────────────────────────

export function probToOdds(p: number): number {
  return p > 0 ? Math.round((1 / p) * 100) / 100 : 99;
}

export function kellyFraction(trueProb: number, odds: number): number {
  const b = odds - 1;
  if (b <= 0) return 0;
  const f = (trueProb * b - (1 - trueProb)) / b;
  return Math.max(0, f);
}

export async function fetchScheduleClient(): Promise<ScheduleData> {
  const now = new Date();
  const tom = new Date(now);
  tom.setDate(tom.getDate() + 1);

  const todayStr = localDateStr(now);
  const tomorrowStr = localDateStr(tom);

  const [at, wt, ato, wto] = await Promise.all([
    fetchESPN("atp", todayStr),
    fetchESPN("wta", todayStr),
    fetchESPN("atp", tomorrowStr),
    fetchESPN("wta", tomorrowStr),
  ]);

  const order = { live: 0, scheduled: 1, finished: 2, cancelled: 3 };
  const sort = (a: ScheduledMatch, b: ScheduledMatch) => {
    const d = (order[a.status] ?? 2) - (order[b.status] ?? 2);
    return d !== 0 ? d : (a.start_timestamp || 9e9) - (b.start_timestamp || 9e9);
  };

  const today = [...at, ...wt].sort(sort);
  const tomorrow = [...ato, ...wto].sort(sort);

  return { today, tomorrow, today_date: todayStr, tomorrow_date: tomorrowStr, fetched_at: Date.now() };
}
