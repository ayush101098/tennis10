import type { MatchSetupPayload, PointStatsInput } from "./types";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8888";
const WS_BASE = API.replace("http", "ws");

export async function setupMatch(payload: MatchSetupPayload) {
  const res = await fetch(`${API}/match/setup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function sendPoint(
  matchId: string,
  winner: "SERVER" | "RECEIVER",
  oddsServer: number,
  oddsReceiver: number,
  stats?: PointStatsInput,
) {
  const body: Record<string, unknown> = {
    winner,
    market_odds_server: oddsServer,
    market_odds_receiver: oddsReceiver,
  };
  if (stats) body.stats = stats;

  const res = await fetch(`${API}/match/${matchId}/point`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

export async function sendOdds(matchId: string, breakOdds: number, holdOdds: number) {
  const res = await fetch(`${API}/match/${matchId}/odds`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ break_odds: breakOdds, hold_odds: holdOdds }),
  });
  return res.json();
}

export async function fetchMatches() {
  const res = await fetch(`${API}/matches`);
  return res.json();
}

export async function fetchState(matchId: string) {
  const res = await fetch(`${API}/match/${matchId}/state`);
  return res.json();
}

export function createWs(matchId: string): WebSocket {
  return new WebSocket(`${WS_BASE}/ws/${matchId}`);
}

// ─── Live Match Feed ─────────────────────────────────────────────────────────

export interface LiveMatch {
  id: string;
  player1: string;
  player2: string;
  tournament: string;
  round: string;
  tour: string;  // ATP, WTA, ITF-M, ITF-W, Challenger
  source: string;
  score: {
    sets_p1: number;
    sets_p2: number;
    games_p1: number;
    games_p2: number;
    point_text: string;
  } | null;
  server: number;
  status: string;
}

export async function fetchLiveMatches(): Promise<{
  matches: LiveMatch[];
  sources: string[];
}> {
  const res = await fetch(`${API}/live/matches`);
  return res.json();
}

export async function attachLiveFeed(
  matchId: string,
  player1?: string,
  player2?: string,
): Promise<{ status: string; source?: string; error?: string }> {
  const params = new URLSearchParams();
  if (player1) params.set("player1", player1);
  if (player2) params.set("player2", player2);
  const res = await fetch(`${API}/live/attach/${matchId}?${params}`, {
    method: "POST",
  });
  return res.json();
}

export async function detachLiveFeed(matchId: string) {
  const res = await fetch(`${API}/live/detach/${matchId}`, { method: "POST" });
  return res.json();
}

export async function fetchFeedStatus(matchId: string) {
  const res = await fetch(`${API}/live/feed/${matchId}/status`);
  return res.json();
}

// ─── Schedule (Today + Tomorrow) ─────────────────────────────────────────────

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
  status: string;
  start_time: string;
  start_timestamp: number;
  p1_win_prob: number;
  p2_win_prob: number;
  prob_method: string;
}

export interface ScheduleResponse {
  today: ScheduledMatch[];
  tomorrow: ScheduledMatch[];
  today_date: string;
  tomorrow_date: string;
}

export async function fetchSchedule(): Promise<ScheduleResponse> {
  const res = await fetch(`${API}/schedule`);
  return res.json();
}
