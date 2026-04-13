/**
 * Client-side tennis schedule service.
 * Fetches today/tomorrow matches directly from ESPN API (CORS-enabled).
 * Computes ranking-based win probabilities using Elo conversion.
 * No backend required — works on static Netlify deploy.
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
  tour: string; // ATP, WTA
  surface: string; // Hard, Clay, Grass
  best_of: number;
  source: string;
  status: "scheduled" | "live" | "finished" | "cancelled";
  start_time: string;
  start_timestamp: number;
  p1_win_prob: number;
  p2_win_prob: number;
  prob_method: string;
  // Score (for live/finished)
  score?: {
    p1_sets: number[];
    p2_sets: number[];
    winner?: 1 | 2;
  };
}

export interface ScheduleData {
  today: ScheduledMatch[];
  tomorrow: ScheduledMatch[];
  today_date: string;
  tomorrow_date: string;
  fetched_at: number;
}

// ─── ESPN API ────────────────────────────────────────────────────────────────

const ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis";
const TOURS = ["atp", "wta"] as const;

// Surface detection from event/venue data
function detectSurface(event: Record<string, unknown>): string {
  const venue = event.venue as Record<string, unknown> | undefined;
  const name = ((event.name as string) || "").toLowerCase();
  const venueSurface = (venue?.surface as string || "").toLowerCase();

  // Check venue surface field
  if (venueSurface.includes("clay") || venueSurface.includes("terre")) return "Clay";
  if (venueSurface.includes("grass") || venueSurface.includes("lawn")) return "Grass";
  if (venueSurface.includes("hard") || venueSurface.includes("decoturf") || venueSurface.includes("plexicushion") || venueSurface.includes("laykold")) return "Hard";

  // Detect from tournament name
  if (name.includes("roland garros") || name.includes("barcelona") || 
      name.includes("monte carlo") || name.includes("rome") || name.includes("madrid") ||
      name.includes("geneva") || name.includes("lyon") || name.includes("hamburg") ||
      name.includes("buenos aires") || name.includes("rio") || name.includes("umag") ||
      name.includes("bastad") || name.includes("kitzbühel") || name.includes("gstaad") ||
      name.includes("parma") || name.includes("prague") || name.includes("bucharest") ||
      name.includes("palermo") || name.includes("rabat") || name.includes("bogota") ||
      name.includes("oeiras")) return "Clay";
  if (name.includes("wimbledon") || name.includes("halle") || name.includes("queen") ||
      name.includes("stuttgart") || name.includes("eastbourne") || name.includes("berlin") ||
      name.includes("s-hertogenbosch") || name.includes("nottingham") || name.includes("mallorca") ||
      name.includes("bad homburg")) return "Grass";
  return "Hard";
}

// Detect best-of from event context
function detectBestOf(eventName: string, tour: string): number {
  const n = eventName.toLowerCase();
  if (tour === "ATP" && (
    n.includes("australian open") || n.includes("roland garros") || 
    n.includes("french open") || n.includes("wimbledon") || n.includes("us open")
  )) return 5;
  return 3;
}

interface ESPNCompetitor {
  id?: string;
  order?: number;
  winner?: boolean;
  seed?: string;
  athlete?: {
    displayName?: string;
    shortName?: string;
    fullName?: string;
    rankings?: Array<{ current?: number }>;
  } | string;
  linescores?: Array<{ value: number; winner: boolean }>;
}

interface ESPNCompetition {
  id?: string;
  date?: string;
  startDate?: string;
  status?: {
    type?: {
      state?: string;
      description?: string;
      completed?: boolean;
    };
  };
  competitors?: ESPNCompetitor[];
  notes?: Array<{ headline?: string }>;
  description?: string;
  round?: { displayName?: string };
}

async function fetchESPNTour(
  tour: (typeof TOURS)[number],
  dateStr: string, // YYYYMMDD
  eventDate: string, // YYYY-MM-DD for display
): Promise<ScheduledMatch[]> {
  const matches: ScheduledMatch[] = [];
  try {
    const res = await fetch(
      `${ESPN_BASE}/${tour}/scoreboard?dates=${dateStr}`,
    );
    if (!res.ok) return [];
    const data = await res.json();

    for (const event of data.events || []) {
      const tournamentName = event.name || "";
      const surface = detectSurface(event);
      const bestOf = detectBestOf(tournamentName, tour.toUpperCase());

      // Collect all competitions from groupings or direct
      const allComps: ESPNCompetition[] = [];
      for (const g of event.groupings || []) {
        for (const c of g.competitions || []) {
          allComps.push(c);
        }
      }
      if (allComps.length === 0) {
        for (const c of event.competitions || []) {
          allComps.push(c);
        }
      }

      for (const comp of allComps) {
        const competitors = comp.competitors || [];
        if (competitors.length < 2) continue;

        // Sort by order
        let p1Data = competitors[0];
        let p2Data = competitors[1];
        for (const c of competitors) {
          if (c.order === 1) p1Data = c;
          else if (c.order === 2) p2Data = c;
        }

        // Player names
        const p1Ath = typeof p1Data.athlete === "object" ? p1Data.athlete : null;
        const p2Ath = typeof p2Data.athlete === "object" ? p2Data.athlete : null;
        const p1Name = p1Ath?.displayName || p1Ath?.fullName || "";
        const p2Name = p2Ath?.displayName || p2Ath?.fullName || "";
        if (!p1Name || !p2Name) continue;

        // Seeds
        const p1Seed = parseInt(p1Data.seed || "0") || 0;
        const p2Seed = parseInt(p2Data.seed || "0") || 0;

        // Rankings (ESPN sometimes has them)
        let p1Rank = 0;
        let p2Rank = 0;
        if (p1Ath?.rankings?.length) p1Rank = p1Ath.rankings[0].current || 0;
        if (p2Ath?.rankings?.length) p2Rank = p2Ath.rankings[0].current || 0;
        // If rank unknown but seed known, use seed as rough rank proxy
        if (!p1Rank && p1Seed) p1Rank = p1Seed;
        if (!p2Rank && p2Seed) p2Rank = p2Seed;

        // Status
        const stateStr = comp.status?.type?.state || "pre";
        const status: ScheduledMatch["status"] =
          stateStr === "in" ? "live" : stateStr === "post" ? "finished" : "scheduled";

        // Start time
        const rawDate = comp.startDate || comp.date || event.date || "";
        let startTime = "";
        let startTs = 0;
        if (rawDate) {
          try {
            const dt = new Date(rawDate);
            startTime = dt.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
            startTs = dt.getTime() / 1000;
          } catch { /* ignore */ }
        }

        // Round
        let roundName = "";
        if (comp.round?.displayName) roundName = comp.round.displayName;
        else if (comp.notes?.length) roundName = comp.notes[0].headline || "";
        else if (comp.description) roundName = comp.description;

        // Score for live/finished
        let score: ScheduledMatch["score"] = undefined;
        if (status !== "scheduled") {
          const p1Sets = (p1Data.linescores || []).map((l) => l.value);
          const p2Sets = (p2Data.linescores || []).map((l) => l.value);
          const winner = p1Data.winner ? 1 : p2Data.winner ? 2 : undefined;
          if (p1Sets.length > 0 || p2Sets.length > 0) {
            score = { p1_sets: p1Sets, p2_sets: p2Sets, winner };
          }
        }

        // Compute probability
        const { p1_prob, p2_prob, method } = computeProbability(
          p1Rank, p2Rank, p1Seed, p2Seed, surface, bestOf,
        );

        const matchId = `espn_${tour}_${event.id || ""}_${comp.id || ""}`;

        matches.push({
          id: matchId,
          player1: p1Name,
          player2: p2Name,
          p1_rank: p1Rank,
          p2_rank: p2Rank,
          p1_seed: p1Seed,
          p2_seed: p2Seed,
          tournament: tournamentName,
          round: roundName,
          tour: tour.toUpperCase(),
          surface,
          best_of: bestOf,
          source: "espn",
          status,
          start_time: startTime,
          start_timestamp: startTs,
          p1_win_prob: p1_prob,
          p2_win_prob: p2_prob,
          prob_method: method,
          score,
        });
      }
    }
  } catch (err) {
    console.warn(`ESPN ${tour} fetch failed:`, err);
  }
  return matches;
}

// ─── Probability Engine (Elo-based) ──────────────────────────────────────────

function rankingToElo(rank: number): number {
  if (rank <= 0) return 1700;
  if (rank === 1) return 2400;
  return Math.max(1300, 2400 - 180 * Math.log(rank));
}

function eloWinProb(elo1: number, elo2: number): number {
  return 1.0 / (1.0 + Math.pow(10, (elo2 - elo1) / 400));
}

function surfaceAdj(p: number, surface: string): number {
  if (surface === "Clay") return p * 0.92 + 0.04; // compress toward 50%
  if (surface === "Grass") return p * 1.03 - 0.015; // amplify
  return p;
}

function bestOfAdj(p: number, bestOf: number): number {
  if (bestOf !== 5) return p;
  const q = 1 - p;
  return Math.pow(p, 1.22) / (Math.pow(p, 1.22) + Math.pow(q, 1.22));
}

function computeProbability(
  p1Rank: number,
  p2Rank: number,
  p1Seed: number,
  p2Seed: number,
  surface: string,
  bestOf: number,
): { p1_prob: number; p2_prob: number; method: string } {
  let method = "ranking";
  let p1: number;

  if (p1Rank > 0 && p2Rank > 0) {
    p1 = eloWinProb(rankingToElo(p1Rank), rankingToElo(p2Rank));
  } else if (p1Seed > 0 && p2Seed > 0) {
    method = "seed";
    p1 = eloWinProb(rankingToElo(p1Seed), rankingToElo(p2Seed));
  } else if (p1Rank > 0 || p2Rank > 0) {
    const r1 = p1Rank || 150;
    const r2 = p2Rank || 150;
    p1 = eloWinProb(rankingToElo(r1), rankingToElo(r2));
  } else if (p1Seed > 0 || p2Seed > 0) {
    method = "seed";
    const s1 = p1Seed || 16;
    const s2 = p2Seed || 16;
    p1 = eloWinProb(rankingToElo(s1), rankingToElo(s2));
  } else {
    return { p1_prob: 0.5, p2_prob: 0.5, method: "unknown" };
  }

  p1 = surfaceAdj(p1, surface);
  p1 = bestOfAdj(p1, bestOf);
  p1 = Math.max(0.03, Math.min(0.97, p1));

  return {
    p1_prob: Math.round(p1 * 10000) / 10000,
    p2_prob: Math.round((1 - p1) * 10000) / 10000,
    method,
  };
}

// ─── Main Fetch ──────────────────────────────────────────────────────────────

function formatDate(d: Date): string {
  return d.toISOString().slice(0, 10); // YYYY-MM-DD
}

function formatESPNDate(d: Date): string {
  return d.toISOString().slice(0, 10).replace(/-/g, ""); // YYYYMMDD
}

export async function fetchScheduleClient(): Promise<ScheduleData> {
  const now = new Date();
  const tomorrow = new Date(now);
  tomorrow.setDate(tomorrow.getDate() + 1);

  const todayStr = formatESPNDate(now);
  const tomorrowStr = formatESPNDate(tomorrow);

  // Fetch all tours for both days in parallel
  const [atpToday, wtaToday, atpTomorrow, wtaTomorrow] = await Promise.all([
    fetchESPNTour("atp", todayStr, formatDate(now)),
    fetchESPNTour("wta", todayStr, formatDate(now)),
    fetchESPNTour("atp", tomorrowStr, formatDate(tomorrow)),
    fetchESPNTour("wta", tomorrowStr, formatDate(tomorrow)),
  ]);

  const todayMatches = [...atpToday, ...wtaToday];
  const tomorrowMatches = [...atpTomorrow, ...wtaTomorrow];

  // Sort: live first, then scheduled, then finished. Within each group by start time.
  const statusOrder = { live: 0, scheduled: 1, finished: 2, cancelled: 3 };
  const sorter = (a: ScheduledMatch, b: ScheduledMatch) => {
    const sa = statusOrder[a.status] ?? 2;
    const sb = statusOrder[b.status] ?? 2;
    if (sa !== sb) return sa - sb;
    return (a.start_timestamp || 9e9) - (b.start_timestamp || 9e9);
  };
  todayMatches.sort(sorter);
  tomorrowMatches.sort(sorter);

  return {
    today: todayMatches,
    tomorrow: tomorrowMatches,
    today_date: formatDate(now),
    tomorrow_date: formatDate(tomorrow),
    fetched_at: Date.now(),
  };
}
