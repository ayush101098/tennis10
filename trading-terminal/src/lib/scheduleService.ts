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
  /** Live-match extras (only present when status === "live") */
  liveScore?: {
    server: 1 | 2;                  // who is serving (from ESPN possession)
    completedSets: { p1: number; p2: number }[];  // finished sets (games)
    currentSetGames: { p1: number; p2: number };   // games in current set
    statusDetail: string;            // "1st Set", "2nd Set", etc.
    scoreText: string;               // e.g. "Kopriva leads Engel 4-2"
    /** Point-level data from SofaScore (when matched) */
    pointScore?: { p1: string; p2: string }; // "0"/"15"/"30"/"40"/"A"
    tiebreakScore?: { p1: number; p2: number };
    /** Live match statistics from SofaScore */
    stats?: LiveMatchStats;
    sofaId?: number;                 // SofaScore event ID for stats lookup
    /** Real bookmaker odds from SofaScore */
    bookmakerOdds?: { p1: number; p2: number; source?: string };
  };
}

/** Real-time match statistics from SofaScore */
export interface LiveMatchStats {
  p1_aces: number; p2_aces: number;
  p1_doubleFaults: number; p2_doubleFaults: number;
  p1_firstServePercent: number; p2_firstServePercent: number;
  p1_firstServeWon: number; p2_firstServeWon: number;
  p1_secondServeWon: number; p2_secondServeWon: number;
  p1_breakPointsConverted: string; p2_breakPointsConverted: string; // e.g. "3/5"
  p1_totalPointsWon: number; p2_totalPointsWon: number;
}

export interface ScheduleData {
  today: ScheduledMatch[];
  tomorrow: ScheduledMatch[];
  today_date: string;
  tomorrow_date: string;
  fetched_at: number;
}

// ─── Rankings lookup (loaded once from /rankings.json) ───────────────────────

interface RankEntry { rank: number; points: number }
interface RankingsFile {
  atp: Record<string, RankEntry>;
  wta: Record<string, RankEntry>;
  atp_date: string;
  wta_date: string;
}

// Normalise name for fuzzy matching: lowercase, strip hyphens/accents, collapse spaces
function normName(n: string): string {
  return n
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "")   // strip accents
    .replace(/-/g, " ")                                   // hyphens → space
    .replace(/\s+/g, " ")                                 // collapse spaces
    .trim();
}

let _rankingsPromise: Promise<Map<string, number>> | null = null;

function loadRankings(): Promise<Map<string, number>> {
  if (_rankingsPromise) return _rankingsPromise;
  _rankingsPromise = (async () => {
    const map = new Map<string, number>();
    try {
      const res = await fetch("/rankings.json");
      if (!res.ok) return map;
      const data: RankingsFile = await res.json();
      for (const [name, entry] of Object.entries(data.atp)) {
        map.set(normName(name), entry.rank);
      }
      for (const [name, entry] of Object.entries(data.wta)) {
        map.set(normName(name), entry.rank);
      }
      console.log(`[rankings] loaded ${map.size} players (ATP ${data.atp_date}, WTA ${data.wta_date})`);
    } catch (e) {
      console.warn("[rankings] failed to load /rankings.json", e);
    }
    return map;
  })();
  return _rankingsPromise;
}

function lookupRank(nameMap: Map<string, number>, displayName: string): number {
  const key = normName(displayName);
  const r = nameMap.get(key);
  if (r) return r;
  // Try last-name first-name swap (Sackmann uses "First Last", ESPN uses "First Last" too usually)
  // Also try just last name for common cases
  const parts = key.split(" ");
  if (parts.length >= 2) {
    // Try "last first" order
    const swapped = parts.slice(1).join(" ") + " " + parts[0];
    const r2 = nameMap.get(swapped);
    if (r2) return r2;
  }
  return 0;
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
  rankMap: Map<string, number>,
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

        // Rank: try ESPN data first, then Sackmann rankings lookup, then seed fallback
        let p1Rank = 0, p2Rank = 0;
        if (p1a?.rankings?.length) p1Rank = p1a.rankings[0].current || 0;
        if (p2a?.rankings?.length) p2Rank = p2a.rankings[0].current || 0;
        if (!p1Rank && p1Name) p1Rank = lookupRank(rankMap, p1Name);
        if (!p2Rank && p2Name) p2Rank = lookupRank(rankMap, p2Name);
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

        // ── Live score extras (server, set/game breakdown) ──
        let liveScore: ScheduledMatch["liveScore"];
        if (status === "live") {
          // Server from 'possession' field (true = currently serving)
          const p1Serving = !!p1d.possession;
          const server: 1 | 2 = p1Serving ? 1 : 2;

          // Build completed sets vs current set from linescores
          const p1ls: { value: number; winner?: boolean }[] = p1d.linescores || [];
          const p2ls: { value: number; winner?: boolean }[] = p2d.linescores || [];
          const completedSets: { p1: number; p2: number }[] = [];
          let currentSetGames = { p1: 0, p2: 0 };

          const maxSets = Math.max(p1ls.length, p2ls.length);
          for (let si = 0; si < maxSets; si++) {
            const g1 = p1ls[si]?.value ?? 0;
            const g2 = p2ls[si]?.value ?? 0;
            const isCompleted = p1ls[si]?.winner !== undefined || p2ls[si]?.winner !== undefined;
            if (isCompleted) {
              completedSets.push({ p1: g1, p2: g2 });
            } else {
              // Current (incomplete) set — last one without a winner
              currentSetGames = { p1: g1, p2: g2 };
            }
          }

          const statusDetail = comp.status?.type?.detail || "";
          const scoreText = comp.notes?.[0]?.text || "";

          liveScore = { server, completedSets, currentSetGames, statusDetail, scoreText };
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
          prob_method: method, venue, score, liveScore,
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

// ─── SofaScore Scheduled Events (ITF + Challenger + WTA 125) ─────────────────

const SOFA_SCHEDULED = "/api/sofa/sport/tennis/scheduled-events";

/** SofaScore category slugs we pull that ESPN doesn't cover */
const SOFA_EXTRA_CATS = new Set(["itf-men", "itf-women", "challenger", "wta-125"]);

function sofaCatToTour(slug: string): string {
  if (slug === "itf-men") return "ITF M";
  if (slug === "itf-women") return "ITF W";
  if (slug === "challenger") return "CHAL";
  if (slug === "wta-125") return "W125";
  return slug.toUpperCase();
}

function sofaStatusToMatch(code: number): ScheduledMatch["status"] {
  if (code === 0) return "scheduled";
  if (code >= 6 && code <= 14) return "live";     // 6–10 = set 1-5, 12=halted
  if (code >= 100) return "finished";
  if (code === 70 || code === 60) return "cancelled"; // canceled / postponed
  return "scheduled";
}

function sofaGroundToSurface(gt?: string): string {
  if (!gt) return "Hard";
  const g = gt.toLowerCase();
  if (g.includes("clay")) return "Clay";
  if (g.includes("grass")) return "Grass";
  return "Hard";
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function sofaEventToMatch(evt: any, rankMap: Map<string, number>): ScheduledMatch | null {
  // Singles only — skip doubles (type===2 or "/" in name)
  const p1Name: string = evt.homeTeam?.name || "";
  const p2Name: string = evt.awayTeam?.name || "";
  if (!p1Name || !p2Name) return null;
  if (p1Name.includes("/") || p2Name.includes("/")) return null;
  if (evt.homeTeam?.type === 2 || evt.awayTeam?.type === 2) return null;

  const catSlug: string = evt.tournament?.category?.slug || "";
  if (!SOFA_EXTRA_CATS.has(catSlug)) return null;

  // Filter by eventFilters (if available) to ensure singles
  const ef = evt.eventFilters;
  if (ef?.category && Array.isArray(ef.category) && !ef.category.includes("singles")) return null;

  const tour = sofaCatToTour(catSlug);
  const tName: string = evt.tournament?.uniqueTournament?.name || evt.tournament?.name || "";
  const surface = sofaGroundToSurface(evt.groundType);
  const bestOf = 3; // ITF/Challenger/W125 are always best of 3

  const status = sofaStatusToMatch(evt.status?.code || 0);
  const startTs = evt.startTimestamp || 0;
  const startTime = startTs ? new Date(startTs * 1000).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }) : "";

  const p1Rank = lookupRank(rankMap, p1Name);
  const p2Rank = lookupRank(rankMap, p2Name);
  const { p1_prob, p2_prob, method } = computeProb(p1Rank, p2Rank, 0, 0, surface, bestOf);

  const roundName = evt.roundInfo?.name || "";
  const sofaId = evt.id as number;

  // Build live score from SofaScore data
  let liveScore: ScheduledMatch["liveScore"];
  let score: ScheduledMatch["score"];

  const hs = evt.homeScore || {};
  const as = evt.awayScore || {};

  if (status === "live" || status === "finished") {
    // Extract set scores
    const p1Sets: number[] = [];
    const p2Sets: number[] = [];
    for (let si = 1; si <= 5; si++) {
      const pk = `period${si}` as string;
      if (hs[pk] !== undefined) {
        p1Sets.push(hs[pk] ?? 0);
        p2Sets.push(as[pk] ?? 0);
      }
    }

    if (p1Sets.length) {
      score = {
        p1_sets: p1Sets,
        p2_sets: p2Sets,
        winner: status === "finished" ? ((hs.current ?? 0) > (as.current ?? 0) ? 1 : 2) : undefined,
      };
    }

    if (status === "live") {
      // Completed vs current set
      const completedSets: { p1: number; p2: number }[] = [];
      let currentSetGames = { p1: 0, p2: 0 };

      for (let si = 0; si < p1Sets.length; si++) {
        const g1 = p1Sets[si];
        const g2 = p2Sets[si];
        // Last set with no clear winner is the current set
        if (si === p1Sets.length - 1 && !(g1 >= 6 && g1 - g2 >= 2) && !(g2 >= 6 && g2 - g1 >= 2) && !(g1 === 7 || g2 === 7)) {
          currentSetGames = { p1: g1, p2: g2 };
        } else {
          completedSets.push({ p1: g1, p2: g2 });
        }
      }

      // Server: firstToServe indicates who is CURRENTLY serving (1=home, 2=away)
      let server: 1 | 2 = 1;
      if (evt.firstToServe) {
        // In SofaScore, firstToServe = current server. P1 is always home for sofa-sourced matches.
        server = evt.firstToServe === 1 ? 1 : 2;
      }

      // Point score
      let pointScore: { p1: string; p2: string } | undefined;
      if (hs.point !== undefined && as.point !== undefined) {
        pointScore = { p1: String(hs.point), p2: String(as.point) };
      }

      // Tiebreak score
      let tiebreakScore: { p1: number; p2: number } | undefined;
      const numSets = completedSets.length + 1;
      const tbKey = `period${numSets}TieBreak`;
      if (hs[tbKey] !== undefined) {
        tiebreakScore = { p1: hs[tbKey] ?? 0, p2: as[tbKey] ?? 0 };
      }

      liveScore = {
        server,
        completedSets,
        currentSetGames,
        statusDetail: evt.status?.description || "",
        scoreText: `${p1Name} vs ${p2Name}`,
        pointScore,
        tiebreakScore,
        sofaId,
      };
    }
  }

  return {
    id: `sofa_${catSlug}_${sofaId}`,
    player1: p1Name,
    player2: p2Name,
    p1_rank: p1Rank,
    p2_rank: p2Rank,
    p1_seed: 0,
    p2_seed: 0,
    tournament: tName,
    round: roundName,
    tour,
    surface,
    best_of: bestOf,
    source: "sofascore",
    status,
    start_time: startTime,
    start_timestamp: startTs,
    p1_win_prob: p1_prob,
    p2_win_prob: p2_prob,
    prob_method: method,
    score,
    liveScore,
  };
}

/**
 * Category-specific endpoints for ITF (the generic scheduled-events endpoint
 * does NOT include ITF events — only ATP, WTA, Challenger, W125).
 */
const SOFA_CAT_URLS: Record<string, string> = {
  "itf-men": "/api/sofa/category/785/scheduled-events",
  "itf-women": "/api/sofa/category/213/scheduled-events",
};

/** Cache SofaScore scheduled data per date — 12s TTL for schedule, 2s for live polling */
const _sofaSchedCache: Record<string, { data: ScheduledMatch[]; ts: number }> = {};
const SOFA_SCHED_TTL = 12_000;
const SOFA_SCHED_LIVE_TTL = 2_000;

/** Fetch a single SofaScore scheduled endpoint, return raw events array */
async function fetchSofaEndpoint(url: string): Promise<unknown[]> {
  try {
    const res = await fetch(url);
    if (!res.ok) return [];
    const json = await res.json();
    return json.events || [];
  } catch {
    return [];
  }
}

async function fetchSofaScheduled(
  targetDate: string,
  rankMap: Map<string, number>,
  forceFresh = false,
): Promise<ScheduledMatch[]> {
  const cached = _sofaSchedCache[targetDate];
  const ttl = forceFresh ? SOFA_SCHED_LIVE_TTL : SOFA_SCHED_TTL;
  if (cached && Date.now() - cached.ts < ttl) return cached.data;

  try {
    // Fetch generic endpoint + category-specific ITF endpoints in parallel
    const [genericEvents, itfMenEvents, itfWomenEvents] = await Promise.all([
      fetchSofaEndpoint(`${SOFA_SCHEDULED}/${targetDate}`),
      fetchSofaEndpoint(`${SOFA_CAT_URLS["itf-men"]}/${targetDate}`),
      fetchSofaEndpoint(`${SOFA_CAT_URLS["itf-women"]}/${targetDate}`),
    ]);

    // Merge all events, deduplicate by SofaScore event ID
    const seen = new Set<number>();
    const allEvents: unknown[] = [];
    for (const evt of [...genericEvents, ...itfMenEvents, ...itfWomenEvents]) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const id = (evt as any).id as number;
      if (id && !seen.has(id)) {
        seen.add(id);
        allEvents.push(evt);
      }
    }

    const matches: ScheduledMatch[] = [];
    for (const evt of allEvents) {
      const m = sofaEventToMatch(evt, rankMap);
      if (m) matches.push(m);
    }
    _sofaSchedCache[targetDate] = { data: matches, ts: Date.now() };
    console.log(`[sofascore] ${targetDate}: ${matches.length} ITF/Chal/W125 singles (generic=${genericEvents.length}, itfM=${itfMenEvents.length}, itfW=${itfWomenEvents.length})`);
    return matches;
  } catch (e) {
    console.warn("[sofascore] scheduled fetch failed", e);
    return cached?.data || [];
  }
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

  const rankMap = await loadRankings();

  const [at, wt, ato, wto, sofaToday, sofaTomorrow] = await Promise.all([
    fetchESPN("atp", todayStr, rankMap),
    fetchESPN("wta", todayStr, rankMap),
    fetchESPN("atp", tomorrowStr, rankMap),
    fetchESPN("wta", tomorrowStr, rankMap),
    fetchSofaScheduled(todayStr, rankMap),
    fetchSofaScheduled(tomorrowStr, rankMap),
  ]);

  const order = { live: 0, scheduled: 1, finished: 2, cancelled: 3 };
  const sort = (a: ScheduledMatch, b: ScheduledMatch) => {
    const d = (order[a.status] ?? 2) - (order[b.status] ?? 2);
    return d !== 0 ? d : (a.start_timestamp || 9e9) - (b.start_timestamp || 9e9);
  };

  const today = [...at, ...wt, ...sofaToday].sort(sort);
  const tomorrow = [...ato, ...wto, ...sofaTomorrow].sort(sort);

  return { today, tomorrow, today_date: todayStr, tomorrow_date: tomorrowStr, fetched_at: Date.now() };
}

/** Fetch fresh live score for a single match by re-querying today's scoreboard */
export async function fetchLiveScore(matchId: string): Promise<ScheduledMatch | null> {
  const now = new Date();
  const todayStr = localDateStr(now);
  const rankMap = await loadRankings();

  // SofaScore-sourced matches: refresh from SofaScore scheduled API
  if (matchId.startsWith("sofa_")) {
    const sofaMatches = await fetchSofaScheduled(todayStr, rankMap, true);
    const match = sofaMatches.find(m => m.id === matchId) ?? null;
    if (match && match.status === "live" && match.liveScore?.sofaId) {
      // Fetch stats + live feed + odds IN PARALLEL for speed
      const [stats, sofaEvents, odds] = await Promise.all([
        fetchSofaStats(match.liveScore.sofaId).catch(() => null),
        fetchSofaLive(),
        fetchSofaOdds(match.liveScore.sofaId).catch(() => null),
      ]);
      if (stats) match.liveScore.stats = stats;
      if (odds) match.liveScore.bookmakerOdds = odds;
      const sofaMatch = sofaEvents.find(se => se.id === match.liveScore!.sofaId);
      if (sofaMatch) {
        const p1IsHome = matchNames(match.player1, sofaMatch.homeTeam.name);
        const hPt = sofaMatch.homeScore.point;
        const aPt = sofaMatch.awayScore.point;
        if (hPt !== undefined && aPt !== undefined) {
          match.liveScore.pointScore = {
            p1: String(p1IsHome ? hPt : aPt),
            p2: String(p1IsHome ? aPt : hPt),
          };
        }
      }
    }
    return match;
  }

  // ESPN-sourced matches
  const all = await Promise.all([
    fetchESPN("atp", todayStr, rankMap),
    fetchESPN("wta", todayStr, rankMap),
  ]);
  const flat = all.flat();
  const match = flat.find(m => m.id === matchId) ?? null;
  if (match && match.status === "live" && match.liveScore) {
    // Enrich with SofaScore point-level data
    await enrichWithSofaScore(match);
  }
  return match;
}

/* ═══════════════════════════════════════════════════════════════════════════════
   SofaScore Integration — point-level scores + match statistics
   ═══════════════════════════════════════════════════════════════════════════ */

const SOFA_LIVE = "/api/sofa/sport/tennis/events/live";

interface SofaLiveEvent {
  id: number;
  homeTeam: { name: string; id: number };
  awayTeam: { name: string; id: number };
  homeScore: {
    current: number;
    period1?: number; period2?: number; period3?: number; period4?: number; period5?: number;
    point?: string;
    period1TieBreak?: number; period2TieBreak?: number; period3TieBreak?: number;
    period4TieBreak?: number; period5TieBreak?: number;
  };
  awayScore: {
    current: number;
    period1?: number; period2?: number; period3?: number; period4?: number; period5?: number;
    point?: string;
    period1TieBreak?: number; period2TieBreak?: number; period3TieBreak?: number;
    period4TieBreak?: number; period5TieBreak?: number;
  };
  firstToServe?: number; // who is currently serving: 1 = home, 2 = away
  status: { code: number; description: string; type: string };
}

/** Fuzzy name matching: normalize and compare last names */
function matchNames(espnName: string, sofaName: string): boolean {
  const a = normName(espnName);
  const b = normName(sofaName);
  if (a === b) return true;
  // Compare last names
  const aLast = a.split(" ").pop() || "";
  const bLast = b.split(" ").pop() || "";
  if (aLast.length > 2 && aLast === bLast) return true;
  // Try first 3 chars of last name for accented variants
  if (aLast.length > 3 && bLast.length > 3 && aLast.slice(0, 4) === bLast.slice(0, 4)) return true;
  return false;
}

/** Cache SofaScore data — ultra-fast for local use */
let _sofaCache: { data: SofaLiveEvent[]; ts: number } | null = null;
const SOFA_CACHE_TTL = 2_000; // 2 seconds — fastest local

async function fetchSofaLive(): Promise<SofaLiveEvent[]> {
  if (_sofaCache && Date.now() - _sofaCache.ts < SOFA_CACHE_TTL) {
    return _sofaCache.data;
  }
  try {
    const res = await fetch(SOFA_LIVE);
    if (!res.ok) return _sofaCache?.data || [];
    const json = await res.json();
    const events: SofaLiveEvent[] = json.events || [];
    _sofaCache = { data: events, ts: Date.now() };
    return events;
  } catch (e) {
    console.warn("[sofascore] fetch failed", e);
    return _sofaCache?.data || [];
  }
}

/** Fetch match statistics from SofaScore */
export async function fetchSofaStats(sofaId: number): Promise<LiveMatchStats | null> {
  try {
    const res = await fetch(`/api/sofa/event/${sofaId}/statistics`);
    if (!res.ok) return null;
    const json = await res.json();
    // Statistics are grouped by period — use "ALL" or the last group
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const groups: any[] = json.statistics || [];
    if (!groups.length) return null;

    // Find the "All" period group, or use the first one
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const allGroup = groups.find((g: any) => g.period === "ALL") || groups[0];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const statsGroups: any[] = allGroup.groups || [];

    const stats: Partial<LiveMatchStats> = {};

    for (const group of statsGroups) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      for (const item of group.statisticsItems || []) {
        const key = item.name as string;
        const home = item.home as string;
        const away = item.away as string;
        switch (key) {
          case "Aces": stats.p1_aces = parseInt(home) || 0; stats.p2_aces = parseInt(away) || 0; break;
          case "Double faults": stats.p1_doubleFaults = parseInt(home) || 0; stats.p2_doubleFaults = parseInt(away) || 0; break;
          case "First serve percentage": stats.p1_firstServePercent = parseInt(home) || 0; stats.p2_firstServePercent = parseInt(away) || 0; break;
          case "First serve points won": stats.p1_firstServeWon = parseInt(home) || 0; stats.p2_firstServeWon = parseInt(away) || 0; break;
          case "Second serve points won": stats.p1_secondServeWon = parseInt(home) || 0; stats.p2_secondServeWon = parseInt(away) || 0; break;
          case "Break points won": stats.p1_breakPointsConverted = home || "0/0"; stats.p2_breakPointsConverted = away || "0/0"; break;
          case "Total points won": stats.p1_totalPointsWon = parseInt(home) || 0; stats.p2_totalPointsWon = parseInt(away) || 0; break;
        }
      }
    }

    return {
      p1_aces: stats.p1_aces || 0,
      p2_aces: stats.p2_aces || 0,
      p1_doubleFaults: stats.p1_doubleFaults || 0,
      p2_doubleFaults: stats.p2_doubleFaults || 0,
      p1_firstServePercent: stats.p1_firstServePercent || 0,
      p2_firstServePercent: stats.p2_firstServePercent || 0,
      p1_firstServeWon: stats.p1_firstServeWon || 0,
      p2_firstServeWon: stats.p2_firstServeWon || 0,
      p1_secondServeWon: stats.p1_secondServeWon || 0,
      p2_secondServeWon: stats.p2_secondServeWon || 0,
      p1_breakPointsConverted: stats.p1_breakPointsConverted || "0/0",
      p2_breakPointsConverted: stats.p2_breakPointsConverted || "0/0",
      p1_totalPointsWon: stats.p1_totalPointsWon || 0,
      p2_totalPointsWon: stats.p2_totalPointsWon || 0,
    };
  } catch (e) {
    console.warn("[sofascore] stats fetch failed", e);
    return null;
  }
}

/** Convert fractional odds string (e.g. "9/4") to decimal odds (e.g. 3.25) */
function fractionalToDecimal(frac: string): number {
  if (!frac) return 0;
  const parts = frac.split("/");
  if (parts.length !== 2) return parseFloat(frac) || 0;
  const num = parseFloat(parts[0]);
  const den = parseFloat(parts[1]);
  if (!den) return 0;
  return num / den + 1; // fractional→decimal = numerator/denominator + 1
}

/** Fetch real bookmaker odds from SofaScore for a given event */
async function fetchSofaOdds(sofaId: number): Promise<{ p1: number; p2: number; source?: string } | null> {
  try {
    const res = await fetch(`/api/sofa/event/${sofaId}/odds/1/all`);
    if (!res.ok) return null;
    const json = await res.json();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const markets: any[] = json.markets || [];
    if (!markets.length) return null;

    // Use first market (typically most prominent bookmaker)
    const market = markets[0];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const choices: any[] = market.choices || [];
    if (choices.length < 2) return null;

    // choices: name "1" = home, name "2" = away
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const homeChoice = choices.find((c: any) => c.name === "1");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const awayChoice = choices.find((c: any) => c.name === "2");
    if (!homeChoice || !awayChoice) return null;

    const homeOdds = fractionalToDecimal(homeChoice.fractionalValue);
    const awayOdds = fractionalToDecimal(awayChoice.fractionalValue);
    if (homeOdds <= 1 || awayOdds <= 1) return null;

    const source = market.marketName || market.sourceName || "SofaScore";
    // For sofa-sourced matches, P1 = home. Caller maps if needed.
    return { p1: homeOdds, p2: awayOdds, source };
  } catch (e) {
    console.warn("[sofascore] odds fetch failed", e);
    return null;
  }
}

/** Enrich an ESPN match with SofaScore point-level data */
async function enrichWithSofaScore(match: ScheduledMatch): Promise<void> {
  if (!match.liveScore || match.status !== "live") return;

  const sofaEvents = await fetchSofaLive();
  // Find matching SofaScore event by player names
  const sofaMatch = sofaEvents.find(se => {
    const homeMatch = matchNames(match.player1, se.homeTeam.name) || matchNames(match.player2, se.homeTeam.name);
    const awayMatch = matchNames(match.player1, se.awayTeam.name) || matchNames(match.player2, se.awayTeam.name);
    return homeMatch && awayMatch;
  });

  if (!sofaMatch) return;

  // Determine which SofaScore player maps to ESPN P1/P2
  const p1IsHome = matchNames(match.player1, sofaMatch.homeTeam.name);
  const homeScore = sofaMatch.homeScore;
  const awayScore = sofaMatch.awayScore;
  const p1Score = p1IsHome ? homeScore : awayScore;
  const p2Score = p1IsHome ? awayScore : homeScore;

  // Point scores
  if (p1Score.point !== undefined && p2Score.point !== undefined) {
    match.liveScore.pointScore = {
      p1: p1Score.point,
      p2: p2Score.point,
    };
  }

  // Server from SofaScore — firstToServe indicates who is CURRENTLY serving (1=home, 2=away)
  if (sofaMatch.firstToServe) {
    const sofaServerIsHome = sofaMatch.firstToServe === 1;
    // Map SofaScore home/away to our P1/P2
    match.liveScore.server = (sofaServerIsHome && p1IsHome) || (!sofaServerIsHome && !p1IsHome) ? 1 : 2;
  }

  // Check for tiebreak scores
  const numSets = (match.liveScore.completedSets?.length || 0) + 1;
  const tbKey = `period${numSets}TieBreak` as keyof typeof homeScore;
  if (homeScore[tbKey] !== undefined) {
    const hTB = homeScore[tbKey] as number;
    const aTB = (awayScore[tbKey] as number) || 0;
    match.liveScore.tiebreakScore = {
      p1: p1IsHome ? hTB : aTB,
      p2: p1IsHome ? aTB : hTB,
    };
  }

  // Store sofaId for stats fetch
  match.liveScore.sofaId = sofaMatch.id;

  // Fetch live stats + bookmaker odds
  try {
    const [stats, odds] = await Promise.all([
      fetchSofaStats(sofaMatch.id),
      fetchSofaOdds(sofaMatch.id),
    ]);
    if (odds) {
      // Map home/away odds to P1/P2
      match.liveScore.bookmakerOdds = p1IsHome
        ? odds
        : { p1: odds.p2, p2: odds.p1, source: odds.source };
    }
    if (stats) {
      // If P1 is away in SofaScore, swap the stats
      if (!p1IsHome) {
        match.liveScore.stats = {
          p1_aces: stats.p2_aces, p2_aces: stats.p1_aces,
          p1_doubleFaults: stats.p2_doubleFaults, p2_doubleFaults: stats.p1_doubleFaults,
          p1_firstServePercent: stats.p2_firstServePercent, p2_firstServePercent: stats.p1_firstServePercent,
          p1_firstServeWon: stats.p2_firstServeWon, p2_firstServeWon: stats.p1_firstServeWon,
          p1_secondServeWon: stats.p2_secondServeWon, p2_secondServeWon: stats.p1_secondServeWon,
          p1_breakPointsConverted: stats.p2_breakPointsConverted, p2_breakPointsConverted: stats.p1_breakPointsConverted,
          p1_totalPointsWon: stats.p2_totalPointsWon, p2_totalPointsWon: stats.p1_totalPointsWon,
        };
      } else {
        match.liveScore.stats = stats;
      }
    }
  } catch { /* stats are optional */ }
}
