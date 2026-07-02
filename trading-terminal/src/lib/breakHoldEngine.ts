/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *  BREAK / HOLD SIGNAL ENGINE — Industry-Grade In-Play Tennis Analytics
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *  Computes break and hold probabilities from:
 *   1. Markov chain game-level model (exact recursive solution)
 *   2. Live SofaScore serve/return statistics
 *   3. Score-state context (pressure, set importance, momentum)
 *   4. Serve efficiency decomposition (1st%, 1st won%, 2nd won%, DF rate)
 *   5. Return pressure index (break point conversion, return points won)
 *   6. Game state machine with phase classification
 *
 *  Outputs a BreakHoldSignals object attached to every live match.
 */

import type { LiveMatchStats } from "./scheduleService";

// ─── Output Types ────────────────────────────────────────────────────────────

export interface BreakHoldSignals {
  /** Who is serving: 1 = P1, 2 = P2 */
  server: 1 | 2;

  // ── Core Probabilities ──
  /** Probability the server holds this game (0–1) */
  holdProb: number;
  /** Probability the returner breaks this game (0–1) = 1 - holdProb */
  breakProb: number;

  // ── Serve Efficiency Rating (SER) ──
  /** Server's composite serve efficiency 0–100 */
  serverSER: number;
  /** SER tier: ELITE / STRONG / AVERAGE / WEAK / CRISIS */
  serverSERTier: SERTier;
  /** Breakdown: first serve %, first serve points won %, second serve points won % */
  serveBreakdown: {
    firstServeIn: number;   // % of first serves in
    firstServeWon: number;  // % of first serve points won
    secondServeWon: number; // % of second serve points won
    doubleFaultRate: number; // DFs per service game
    aceRate: number;         // aces per service game
  };

  // ── Return Pressure Index (RPI) ──
  /** Returner's composite return pressure 0–100 */
  returnerRPI: number;
  /** RPI tier: DOMINANT / AGGRESSIVE / NEUTRAL / PASSIVE / ABSENT */
  returnerRPITier: RPITier;
  /** Breakdown: return points won %, break point conversion, 2nd serve return % */
  returnBreakdown: {
    returnPointsWon: number;    // % of return points won
    breakPointConversion: number; // % of break points converted
    secondServeReturnWon: number; // % of 2nd serve return points won
  };

  // ── Game State ──
  /** Current game phase classification */
  gamePhase: GamePhase;
  /** Whether we're at a break point right now */
  isBreakPoint: boolean;
  /** Number of break points in this game (cumulative) */
  breakPointCount: number;
  /** Points until potential break point (0 = already at BP) */
  pointsToBreakPoint: number;

  // ── Pressure Indices ──
  /** Server pressure: 0 = comfortable, 100 = extreme danger */
  serverPressure: number;
  /** Context multiplier from set/match situation */
  contextPressure: number;
  /** Composite danger level combining all factors */
  dangerLevel: DangerLevel;

  // ── Momentum ──
  /** Server's recent serve hold trend: +ve = improving, -ve = declining */
  serveTrend: number;
  /** Returner's recent return form trend */
  returnTrend: number;

  // ── Actionable Signals ──
  /** Primary signal for traders */
  primarySignal: TradeSignal;
  /** All active signals sorted by strength */
  signals: TradeSignal[];
}

export type SERTier = "ELITE" | "STRONG" | "AVERAGE" | "WEAK" | "CRISIS";
export type RPITier = "DOMINANT" | "AGGRESSIVE" | "NEUTRAL" | "PASSIVE" | "ABSENT";
export type GamePhase =
  | "NEW_GAME"           // 0-0
  | "SERVER_BUILDING"    // Server ahead (30-0, 40-0, 40-15)
  | "RETURNER_BUILDING"  // Returner ahead (0-30, 0-40, 15-30)
  | "CONTESTED"          // Close (15-15, 30-30, deuce)
  | "BREAK_POINT"        // Returner has BP (30-40, 40-AD, 0-40)
  | "GAME_POINT"         // Server has GP (40-0, 40-15, 40-30, AD-40)
  | "TIEBREAK"           // In tiebreak
  | "BETWEEN_GAMES";     // Score 0-0 at start

export type DangerLevel = "SAFE" | "ALERT" | "WARNING" | "DANGER" | "CRITICAL";

export interface TradeSignal {
  type: "BREAK_IMMINENT" | "HOLD_STRONG" | "SERVE_CRISIS" | "RETURN_DOMINANCE"
      | "PRESSURE_BUILDING" | "BREAK_POINT_LIVE" | "GAME_POINT_HOLD"
      | "DF_CRISIS" | "ACE_POWER" | "SECOND_SERVE_VULN"
      | "BREAK_BACK_RISK" | "SET_ON_SERVE" | "CLUTCH_SERVER";
  strength: number;  // 0–100
  label: string;     // Human-readable signal text
  emoji: string;     // Visual indicator
  actionable: boolean;
}

// ─── Constants ───────────────────────────────────────────────────────────────

/**
 * Tour-aware baselines.
 * ATP is the empirically-sourced anchor (2023-2024 season data). WTA / ITF
 * are derived two ways and reconciled:
 *   1. Proportionally scaled from the same Barnett-Clarke point/game-win
 *      formulas used in hierarchical_model.py (TOUR_SERVE_DEFAULTS), so the
 *      client (TS) and server (Python) engines agree on relative tour gaps.
 *   2. aceRate / dfRate / bpConversion are sense-checked against published
 *      pro-tennis analytics ranges (these aren't modeled in hierarchical_model.py).
 *
 * This replaces the old single ATP-only TOUR_AVGS constant, which meant every
 * WTA/ITF match was silently scored against ATP serve-hold expectations.
 */
export type TourAvgs = {
  firstServeIn: number;
  firstServeWon: number;
  secondServeWon: number;
  holdRate: number;
  breakRate: number;
  aceRate: number;    // per service game
  dfRate: number;     // per service game
  returnPointsWon: number;
  bpConversion: number;
};

const ATP_AVGS: TourAvgs = {
  firstServeIn: 62, firstServeWon: 73, secondServeWon: 52,
  holdRate: 82, breakRate: 18,
  aceRate: 0.80, dfRate: 0.35,
  returnPointsWon: 37, bpConversion: 42,
};

const TOUR_AVGS_BY_TOUR: Record<string, TourAvgs> = {
  ATP: ATP_AVGS,
  WTA: {
    firstServeIn: 58.5, firstServeWon: 65.3, secondServeWon: 47.9,
    holdRate: 67, breakRate: 33,
    aceRate: 0.30, dfRate: 0.50,
    returnPointsWon: 44, bpConversion: 45,
  },
  "ITF M": {
    firstServeIn: 56.5, firstServeWon: 62.3, secondServeWon: 44.9,
    holdRate: 59, breakRate: 41,
    aceRate: 0.45, dfRate: 0.55,
    returnPointsWon: 47, bpConversion: 46,
  },
  "ITF W": {
    firstServeIn: 55, firstServeWon: 58, secondServeWon: 41,
    holdRate: 54, breakRate: 46,
    aceRate: 0.15, dfRate: 0.65,
    returnPointsWon: 50, bpConversion: 48,
  },
  CHALLENGER: {
    firstServeIn: 60, firstServeWon: 68, secondServeWon: 49,
    holdRate: 73, breakRate: 27,
    aceRate: 0.55, dfRate: 0.45,
    returnPointsWon: 41, bpConversion: 44,
  },
  W125: {
    firstServeIn: 57, firstServeWon: 62, secondServeWon: 45,
    holdRate: 61, breakRate: 39,
    aceRate: 0.22, dfRate: 0.57,
    returnPointsWon: 46.5, bpConversion: 46,
  },
};

/** Resolve a tour string (e.g. "ATP", "WTA", "ITF M", "ITF-W", "W125") to its baselines. */
export function resolveTourAvgs(tour?: string): TourAvgs {
  if (!tour) return ATP_AVGS;
  const t = tour.toUpperCase().replace(/-/g, " ").trim();
  if (TOUR_AVGS_BY_TOUR[t]) return TOUR_AVGS_BY_TOUR[t];
  if (t.includes("ITF") && t.includes("W")) return TOUR_AVGS_BY_TOUR["ITF W"];
  if (t.includes("ITF")) return TOUR_AVGS_BY_TOUR["ITF M"];
  if (t.includes("CHALL")) return TOUR_AVGS_BY_TOUR.CHALLENGER;
  if (t.includes("125") || t.includes("W100") || t.includes("W75") || t.includes("W50") || t.includes("W35")) return TOUR_AVGS_BY_TOUR.W125;
  if (t.includes("WTA")) return TOUR_AVGS_BY_TOUR.WTA;
  if (t.includes("ATP")) return ATP_AVGS;
  return ATP_AVGS;
}

/** Point score mapping */
const PT = { "0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4 } as Record<string, number>;
const PT_LABELS = ["0", "15", "30", "40", "AD"];

// ─── Main Entry Point ────────────────────────────────────────────────────────

export function computeBreakHoldSignals(
  server: 1 | 2,
  pointScore: { p1: string; p2: string } | undefined,
  currentSetGames: { p1: number; p2: number },
  completedSets: { p1: number; p2: number }[],
  stats: LiveMatchStats | null,
  tiebreakScore: { p1: number; p2: number } | undefined,
  bestOf: number,
  p1WinProb: number,  // pre-match Elo probability
  tour: string = "ATP",  // ATP / WTA / ITF M / ITF W / Challenger / W125 — fixes ATP-only bias
): BreakHoldSignals {

  const avgs = resolveTourAvgs(tour);
  const isTiebreak = !!(tiebreakScore && currentSetGames.p1 === 6 && currentSetGames.p2 === 6);
  const returner: 1 | 2 = server === 1 ? 2 : 1;

  // ── Parse point score ──
  let srvPts = 0, retPts = 0;
  if (isTiebreak && tiebreakScore) {
    srvPts = server === 1 ? tiebreakScore.p1 : tiebreakScore.p2;
    retPts = server === 1 ? tiebreakScore.p2 : tiebreakScore.p1;
  } else if (pointScore) {
    const p1Pts = PT[pointScore.p1] ?? 0;
    const p2Pts = PT[pointScore.p2] ?? 0;
    srvPts = server === 1 ? p1Pts : p2Pts;
    retPts = server === 1 ? p2Pts : p1Pts;
  }

  // ── Compute Serve Point Win Rate ──
  const srvPtWin = computeServePointWinRate(server, stats, p1WinProb);
  const retPtWin = 1 - srvPtWin;

  // ── Markov Hold/Break from current game score ──
  const holdFromHere = isTiebreak
    ? tbWinProb(srvPtWin, srvPts, retPts)
    : gameWinProb(srvPtWin, srvPts, retPts);
  let breakProb = 1 - holdFromHere;
  let holdProb = holdFromHere;

  // ── Serve Efficiency Rating ──
  const serveBreakdown = computeServeBreakdown(server, stats, avgs);
  const serverSER = computeSER(serveBreakdown);
  const serverSERTier = classifySER(serverSER);

  // ── Return Pressure Index ──
  const returnBreakdown = computeReturnBreakdown(returner, stats, avgs);
  const returnerRPI = computeRPI(returnBreakdown);
  const returnerRPITier = classifyRPI(returnerRPI);

  // ── Game Phase Classification ──
  const gamePhase = classifyGamePhase(srvPts, retPts, isTiebreak);

  // ── Break Point Detection ──
  const isBreakPoint = !isTiebreak && retPts >= 3 && retPts > srvPts;
  const breakPointCount = isBreakPoint ? 1 : 0;
  const pointsToBreakPoint = computePointsToBreakPoint(srvPts, retPts, isTiebreak);

  // ── Context Pressure ──
  const contextPressure = computeContextPressure(
    server, currentSetGames, completedSets, bestOf, gamePhase
  );

  // ── Apply Context & Stats Adjustments to Break Prob ──
  const adjustments = computeAdjustments(
    server, serverSER, returnerRPI, contextPressure,
    serveBreakdown, returnBreakdown, stats, gamePhase,
    currentSetGames, completedSets, bestOf
  );
  breakProb = Math.max(0.01, Math.min(0.99, breakProb + adjustments.totalAdjustment));
  holdProb = 1 - breakProb;

  // ── Server Pressure Index ──
  const serverPressure = computeServerPressure(breakProb, gamePhase, contextPressure, serverSER);

  // ── Danger Level ──
  const dangerLevel = classifyDanger(serverPressure, breakProb, gamePhase);

  // ── Trends ──
  const { serveTrend, returnTrend } = computeTrends(server, stats, avgs);

  // ── Generate Signals ──
  const signals = generateSignals(
    server, breakProb, holdProb, serverSER, serverSERTier,
    returnerRPI, returnerRPITier, gamePhase, isBreakPoint,
    serverPressure, contextPressure, dangerLevel,
    serveBreakdown, returnBreakdown, stats,
    currentSetGames, completedSets, bestOf,
    srvPts, retPts, isTiebreak, avgs
  );

  const primarySignal = signals[0] || {
    type: "HOLD_STRONG" as const, strength: 50, label: "Neutral — no strong signal",
    emoji: "⚖️", actionable: false,
  };

  return {
    server, holdProb, breakProb,
    serverSER, serverSERTier, serveBreakdown,
    returnerRPI, returnerRPITier, returnBreakdown,
    gamePhase, isBreakPoint, breakPointCount, pointsToBreakPoint,
    serverPressure, contextPressure, dangerLevel,
    serveTrend, returnTrend,
    primarySignal, signals,
  };
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SERVE POINT WIN RATE — Composite from live stats or Elo fallback
// ═══════════════════════════════════════════════════════════════════════════════

function computeServePointWinRate(
  server: 1 | 2, stats: LiveMatchStats | null, p1WinProb: number
): number {
  if (stats) {
    const firstPct = (server === 1 ? stats.p1_firstServePercent : stats.p2_firstServePercent) || 60;
    const firstWon = (server === 1 ? stats.p1_firstServeWon : stats.p2_firstServeWon) || 65;
    const secondWon = (server === 1 ? stats.p1_secondServeWon : stats.p2_secondServeWon) || 45;
    const p = (firstPct / 100) * (firstWon / 100) + (1 - firstPct / 100) * (secondWon / 100);
    return clamp(p, 0.35, 0.85);
  }
  // Elo-derived fallback
  const prob = server === 1 ? p1WinProb : (1 - p1WinProb);
  return clamp(0.55 + (prob - 0.5) * 0.3, 0.45, 0.75);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SERVE EFFICIENCY RATING (SER) — 0–100 composite
// ═══════════════════════════════════════════════════════════════════════════════

interface ServeBreakdown {
  firstServeIn: number;
  firstServeWon: number;
  secondServeWon: number;
  doubleFaultRate: number;
  aceRate: number;
}

function computeServeBreakdown(server: 1 | 2, stats: LiveMatchStats | null, avgs: TourAvgs): ServeBreakdown {
  if (!stats) {
    return {
      firstServeIn: avgs.firstServeIn,
      firstServeWon: avgs.firstServeWon,
      secondServeWon: avgs.secondServeWon,
      doubleFaultRate: avgs.dfRate,
      aceRate: avgs.aceRate,
    };
  }

  const totalPts = Math.max(1, stats.p1_totalPointsWon + stats.p2_totalPointsWon);
  const srvPts = Math.max(1, totalPts / 2); // approximate service points

  return {
    firstServeIn: server === 1 ? stats.p1_firstServePercent : stats.p2_firstServePercent,
    firstServeWon: server === 1 ? stats.p1_firstServeWon : stats.p2_firstServeWon,
    secondServeWon: server === 1 ? stats.p1_secondServeWon : stats.p2_secondServeWon,
    doubleFaultRate: (server === 1 ? stats.p1_doubleFaults : stats.p2_doubleFaults) / srvPts * 4, // per service game (~4 pts)
    aceRate: (server === 1 ? stats.p1_aces : stats.p2_aces) / srvPts * 4,
  };
}

function computeSER(b: ServeBreakdown): number {
  // Weighted composite: 1st% in (15%), 1st won% (30%), 2nd won% (25%), DF penalty (15%), Ace bonus (15%)
  const firstInScore = clamp((b.firstServeIn - 45) / 30, 0, 1) * 100;   // 45-75% → 0-100
  const firstWonScore = clamp((b.firstServeWon - 55) / 25, 0, 1) * 100; // 55-80% → 0-100
  const secondWonScore = clamp((b.secondServeWon - 30) / 35, 0, 1) * 100; // 30-65% → 0-100
  const dfPenalty = clamp(1 - b.doubleFaultRate / 1.5, 0, 1) * 100;     // 0 DFs/game = 100, 1.5 = 0
  const aceBonus = clamp(b.aceRate / 2, 0, 1) * 100;                     // 2+ aces/game = 100

  return Math.round(
    firstInScore * 0.15 +
    firstWonScore * 0.30 +
    secondWonScore * 0.25 +
    dfPenalty * 0.15 +
    aceBonus * 0.15
  );
}

function classifySER(ser: number): SERTier {
  if (ser >= 80) return "ELITE";
  if (ser >= 65) return "STRONG";
  if (ser >= 45) return "AVERAGE";
  if (ser >= 25) return "WEAK";
  return "CRISIS";
}


// ═══════════════════════════════════════════════════════════════════════════════
//  RETURN PRESSURE INDEX (RPI) — 0–100 composite
// ═══════════════════════════════════════════════════════════════════════════════

interface ReturnBreakdown {
  returnPointsWon: number;
  breakPointConversion: number;
  secondServeReturnWon: number;
}

function computeReturnBreakdown(returner: 1 | 2, stats: LiveMatchStats | null, avgs: TourAvgs): ReturnBreakdown {
  if (!stats) {
    return {
      returnPointsWon: avgs.returnPointsWon,
      breakPointConversion: avgs.bpConversion,
      secondServeReturnWon: 100 - avgs.secondServeWon,
    };
  }

  // Return points won = opponent's serve points lost
  const oppFirstWon = returner === 1 ? stats.p2_firstServeWon : stats.p1_firstServeWon;
  const oppSecondWon = returner === 1 ? stats.p2_secondServeWon : stats.p1_secondServeWon;
  const oppFirstPct = returner === 1 ? stats.p2_firstServePercent : stats.p1_firstServePercent;
  const retPtsWon = (oppFirstPct / 100) * (100 - oppFirstWon) + (1 - oppFirstPct / 100) * (100 - oppSecondWon);

  // Break point conversion
  const bpc = returner === 1 ? stats.p1_breakPointsConverted : stats.p2_breakPointsConverted;
  const [bpWon, bpTotal] = (bpc || "0/0").split("/").map(Number);
  const bpConv = bpTotal > 0 ? (bpWon / bpTotal) * 100 : avgs.bpConversion;

  // 2nd serve return won = 100 - opponent's 2nd serve won %
  const secondRetWon = 100 - (oppSecondWon || 50);

  return {
    returnPointsWon: Math.round(retPtsWon),
    breakPointConversion: Math.round(bpConv),
    secondServeReturnWon: Math.round(secondRetWon),
  };
}

function computeRPI(b: ReturnBreakdown): number {
  // Weighted: return pts won (40%), BP conversion (35%), 2nd serve return (25%)
  const retPtsScore = clamp((b.returnPointsWon - 25) / 25, 0, 1) * 100;   // 25-50% → 0-100
  const bpConvScore = clamp((b.breakPointConversion - 20) / 50, 0, 1) * 100; // 20-70% → 0-100
  const secondRetScore = clamp((b.secondServeReturnWon - 30) / 35, 0, 1) * 100; // 30-65% → 0-100

  return Math.round(
    retPtsScore * 0.40 +
    bpConvScore * 0.35 +
    secondRetScore * 0.25
  );
}

function classifyRPI(rpi: number): RPITier {
  if (rpi >= 75) return "DOMINANT";
  if (rpi >= 55) return "AGGRESSIVE";
  if (rpi >= 35) return "NEUTRAL";
  if (rpi >= 18) return "PASSIVE";
  return "ABSENT";
}


// ═══════════════════════════════════════════════════════════════════════════════
//  GAME PHASE CLASSIFIER
// ═══════════════════════════════════════════════════════════════════════════════

function classifyGamePhase(srvPts: number, retPts: number, isTiebreak: boolean): GamePhase {
  if (isTiebreak) return "TIEBREAK";
  if (srvPts === 0 && retPts === 0) return "NEW_GAME";

  // Break point: returner at 40 (3+) and ahead
  if (retPts >= 3 && retPts > srvPts) return "BREAK_POINT";

  // Game point: server at 40 (3+) and ahead
  if (srvPts >= 3 && srvPts > retPts) return "GAME_POINT";

  // Contested: deuce or close
  if (srvPts >= 3 && retPts >= 3) return "CONTESTED";

  // Server building: server leads meaningfully
  if (srvPts > retPts && srvPts >= 2) return "SERVER_BUILDING";

  // Returner building
  if (retPts > srvPts && retPts >= 2) return "RETURNER_BUILDING";

  // Default
  return srvPts >= retPts ? "SERVER_BUILDING" : "RETURNER_BUILDING";
}


// ═══════════════════════════════════════════════════════════════════════════════
//  CONTEXT PRESSURE — Set & match situation multiplier
// ═══════════════════════════════════════════════════════════════════════════════

function computeContextPressure(
  server: 1 | 2,
  currentSetGames: { p1: number; p2: number },
  completedSets: { p1: number; p2: number }[],
  bestOf: number,
  gamePhase: GamePhase,
): number {
  let pressure = 0;
  const setsToWin = bestOf === 5 ? 3 : 2;
  const srvGames = server === 1 ? currentSetGames.p1 : currentSetGames.p2;
  const retGames = server === 1 ? currentSetGames.p2 : currentSetGames.p1;
  const srvSets = completedSets.filter(s => (server === 1 ? s.p1 > s.p2 : s.p2 > s.p1)).length;
  const retSets = completedSets.filter(s => (server === 1 ? s.p2 > s.p1 : s.p1 > s.p2)).length;

  // Serving to stay in set
  if (retGames >= 5 && retGames > srvGames && srvGames < 6) {
    pressure += 25;
  }
  // Serving for the set
  if (srvGames >= 5 && srvGames > retGames && retGames < 6) {
    pressure -= 10; // server has positive context
  }

  // Set score pressure
  if (retSets === setsToWin - 1 && srvSets < retSets) {
    pressure += 20; // server behind in sets, this set critical
  }
  if (retSets === setsToWin - 1 && retGames >= 5 && retGames > srvGames) {
    pressure += 30; // match point scenario
  }

  // Decisive set
  if (srvSets === setsToWin - 1 && retSets === setsToWin - 1) {
    pressure += 15; // final set, everything matters
  }

  // Tight game score in set
  if (Math.abs(srvGames - retGames) <= 1 && srvGames + retGames >= 8) {
    pressure += 10; // tight late in set
  }

  // Break point amplification
  if (gamePhase === "BREAK_POINT") pressure += 15;

  return clamp(pressure, -20, 100);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  ADJUSTMENTS — Stats & context corrections to Markov break probability
// ═══════════════════════════════════════════════════════════════════════════════

function computeAdjustments(
  server: 1 | 2,
  serverSER: number,
  returnerRPI: number,
  contextPressure: number,
  serveBD: ServeBreakdown,
  returnBD: ReturnBreakdown,
  stats: LiveMatchStats | null,
  gamePhase: GamePhase,
  currentSetGames: { p1: number; p2: number },
  completedSets: { p1: number; p2: number }[],
  bestOf: number,
): { totalAdjustment: number; components: Record<string, number> } {
  const components: Record<string, number> = {};
  const setsToWin = bestOf === 5 ? 3 : 2;

  // 1. SER vs RPI mismatch: if returner outclasses server → break more likely
  const serRpiDelta = (returnerRPI - (100 - serverSER)) / 100;
  components.serRpiMismatch = serRpiDelta * 0.08;

  // 2. Double fault crisis
  if (serveBD.doubleFaultRate > 1.0) {
    components.dfCrisis = 0.06;
  } else if (serveBD.doubleFaultRate > 0.6) {
    components.dfPressure = 0.03;
  } else {
    components.dfNone = 0;
  }

  // 3. Second serve vulnerability (returner attacks weak second serve)
  if (serveBD.secondServeWon < 40) {
    components.secondServeVuln = 0.04;
  } else if (serveBD.secondServeWon < 48) {
    components.secondServeSoft = 0.02;
  } else {
    components.secondServe = 0;
  }

  // 4. Context pressure
  components.context = contextPressure * 0.001;

  // 5. Break point conversion momentum
  if (stats) {
    const retBPC = server === 1 ? stats.p2_breakPointsConverted : stats.p1_breakPointsConverted;
    const [bpW, bpT] = (retBPC || "0/0").split("/").map(Number);
    if (bpT >= 3 && bpW / bpT > 0.55) {
      components.bpMomentum = 0.05; // returner is a clutch converter today
    } else if (bpT >= 3 && bpW / bpT < 0.20) {
      components.bpMomentum = -0.04; // server saves break points well
    } else {
      components.bpMomentum = 0;
    }
  }

  // 6. First serve in-rate crash (below 50% = trouble)
  if (serveBD.firstServeIn < 50) {
    components.firstServeCollapse = 0.04;
  } else if (serveBD.firstServeIn < 55) {
    components.firstServeLow = 0.02;
  } else {
    components.firstServe = 0;
  }

  // 7. Ace power offset (server with lots of aces is harder to break)
  if (serveBD.aceRate > 1.5) {
    components.acePower = -0.04;
  } else if (serveBD.aceRate > 1.0) {
    components.aceBonus = -0.02;
  } else {
    components.ace = 0;
  }

  // 8. Return dominance: returner winning > 45% of return points
  if (returnBD.returnPointsWon > 45) {
    components.returnDominance = 0.04;
  } else if (returnBD.returnPointsWon > 40) {
    components.returnStrong = 0.02;
  } else {
    components.return = 0;
  }

  // 9. Serve-to-stay-in-set choking factor
  const srvGames = server === 1 ? currentSetGames.p1 : currentSetGames.p2;
  const retGames = server === 1 ? currentSetGames.p2 : currentSetGames.p1;
  if (retGames >= 5 && retGames > srvGames && gamePhase !== "GAME_POINT") {
    components.stayInSet = 0.04;
  }

  // 10. Match-point pressure amplification
  const retSets = completedSets.filter(s => (server === 1 ? s.p2 > s.p1 : s.p1 > s.p2)).length;
  if (retSets === setsToWin - 1 && retGames >= 5 && retGames > srvGames) {
    components.matchPoint = 0.06;
  }

  const totalAdjustment = Object.values(components).reduce((s, v) => s + v, 0);
  return { totalAdjustment: clamp(totalAdjustment, -0.15, 0.25), components };
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SERVER PRESSURE & DANGER CLASSIFICATION
// ═══════════════════════════════════════════════════════════════════════════════

function computeServerPressure(
  breakProb: number, gamePhase: GamePhase, contextPressure: number, ser: number
): number {
  // Base from break probability
  let pressure = breakProb * 80;

  // Phase multipliers
  if (gamePhase === "BREAK_POINT") pressure += 20;
  else if (gamePhase === "RETURNER_BUILDING") pressure += 8;
  else if (gamePhase === "GAME_POINT") pressure -= 15;
  else if (gamePhase === "SERVER_BUILDING") pressure -= 10;

  // Context
  pressure += contextPressure * 0.3;

  // SER penalty: weak server = more pressure
  if (ser < 35) pressure += 10;
  else if (ser > 70) pressure -= 8;

  return clamp(pressure, 0, 100);
}

function classifyDanger(pressure: number, breakProb: number, phase: GamePhase): DangerLevel {
  if (phase === "BREAK_POINT" && breakProb > 0.55) return "CRITICAL";
  if (pressure >= 80 || (phase === "BREAK_POINT" && breakProb > 0.40)) return "CRITICAL";
  if (pressure >= 60 || breakProb > 0.50) return "DANGER";
  if (pressure >= 40 || breakProb > 0.35) return "WARNING";
  if (pressure >= 20) return "ALERT";
  return "SAFE";
}


// ═══════════════════════════════════════════════════════════════════════════════
//  BREAK POINT PROXIMITY
// ═══════════════════════════════════════════════════════════════════════════════

function computePointsToBreakPoint(srvPts: number, retPts: number, isTB: boolean): number {
  if (isTB) return 99; // N/A in tiebreak
  if (retPts >= 3 && retPts > srvPts) return 0; // already at break point

  // Minimum points returner needs to reach 40 (3) while being ahead
  const retNeeds = Math.max(0, 3 - retPts);
  if (retNeeds === 0 && retPts === srvPts) return 1; // deuce → need 1 more (AD)
  return retNeeds;
}


// ═══════════════════════════════════════════════════════════════════════════════
//  TRENDS — Serve/return form direction
// ═══════════════════════════════════════════════════════════════════════════════

function computeTrends(
  server: 1 | 2, stats: LiveMatchStats | null, avgs: TourAvgs
): { serveTrend: number; returnTrend: number } {
  if (!stats) return { serveTrend: 0, returnTrend: 0 };

  const srvFirst = server === 1 ? stats.p1_firstServePercent : stats.p2_firstServePercent;
  const srvFirstW = server === 1 ? stats.p1_firstServeWon : stats.p2_firstServeWon;
  const srvSecondW = server === 1 ? stats.p1_secondServeWon : stats.p2_secondServeWon;

  // Serve trend: compare to tour average
  const srvComposite = (srvFirst * 0.3 + srvFirstW * 0.4 + srvSecondW * 0.3);
  const avgComposite = (avgs.firstServeIn * 0.3 + avgs.firstServeWon * 0.4 + avgs.secondServeWon * 0.3);
  const serveTrend = (srvComposite - avgComposite) / avgComposite;

  // Return trend: returner's effectiveness
  const retFirst = server === 1 ? stats.p2_firstServeWon : stats.p1_firstServeWon;
  const retSecondW = server === 1 ? stats.p2_secondServeWon : stats.p1_secondServeWon;
  const retComposite = (100 - retFirst) * 0.5 + (100 - retSecondW) * 0.5;
  const avgRet = (100 - avgs.firstServeWon) * 0.5 + (100 - avgs.secondServeWon) * 0.5;
  const returnTrend = (retComposite - avgRet) / Math.max(1, avgRet);

  return {
    serveTrend: clamp(serveTrend, -0.5, 0.5),
    returnTrend: clamp(returnTrend, -0.5, 0.5),
  };
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SIGNAL GENERATOR — Produces actionable trade signals
// ═══════════════════════════════════════════════════════════════════════════════

function generateSignals(
  server: 1 | 2,
  breakProb: number, holdProb: number,
  serverSER: number, serTier: SERTier,
  returnerRPI: number, rpiTier: RPITier,
  phase: GamePhase, isBreakPoint: boolean,
  serverPressure: number, contextPressure: number,
  danger: DangerLevel,
  serveBD: ServeBreakdown, returnBD: ReturnBreakdown,
  stats: LiveMatchStats | null,
  csGames: { p1: number; p2: number },
  completedSets: { p1: number; p2: number }[],
  bestOf: number,
  srvPts: number, retPts: number, isTB: boolean,
  avgs: TourAvgs,
): TradeSignal[] {
  const signals: TradeSignal[] = [];
  const srvName = server === 1 ? "P1" : "P2";
  const retName = server === 1 ? "P2" : "P1";
  const srvGames = server === 1 ? csGames.p1 : csGames.p2;
  const retGames = server === 1 ? csGames.p2 : csGames.p1;
  const setLabel = `${csGames.p1}-${csGames.p2}`;

  // ── Live Break Point ──
  if (isBreakPoint) {
    const convPct = Math.round(breakProb * 100);
    signals.push({
      type: "BREAK_POINT_LIVE",
      strength: Math.min(100, Math.round(breakProb * 120)),
      label: `🔴 BREAK POINT! ${retName} converts at ${convPct}%${stats ? ` · Srv 1st%: ${serveBD.firstServeIn}%` : ""}`,
      emoji: "🔴",
      actionable: true,
    });
  }

  // ── Break Imminent (high break prob, not yet at BP) ──
  if (!isBreakPoint && breakProb > 0.40 && phase !== "GAME_POINT" && !isTB) {
    signals.push({
      type: "BREAK_IMMINENT",
      strength: Math.round(breakProb * 100),
      label: `⚠️ Break risk ${Math.round(breakProb * 100)}% — ${srvName} serve under pressure at ${PT_LABELS[Math.min(srvPts,4)]}-${PT_LABELS[Math.min(retPts,4)]}`,
      emoji: "⚠️",
      actionable: true,
    });
  }

  // ── Serve Crisis ──
  if (serTier === "CRISIS") {
    signals.push({
      type: "SERVE_CRISIS",
      strength: Math.round(90 - serverSER),
      label: `🆘 SERVE CRISIS — SER ${serverSER}/100 · 1st%: ${serveBD.firstServeIn}% · DFs: ${serveBD.doubleFaultRate.toFixed(1)}/game`,
      emoji: "🆘",
      actionable: true,
    });
  } else if (serTier === "WEAK") {
    signals.push({
      type: "SERVE_CRISIS",
      strength: Math.round(70 - serverSER),
      label: `⚡ Serve weak — SER ${serverSER}/100 · 2nd won: ${serveBD.secondServeWon}%`,
      emoji: "⚡",
      actionable: breakProb > 0.30,
    });
  }

  // ── Return Dominance ──
  if (rpiTier === "DOMINANT") {
    signals.push({
      type: "RETURN_DOMINANCE",
      strength: returnerRPI,
      label: `🎯 ${retName} RETURN DOMINANT — RPI ${returnerRPI}/100 · Ret pts won: ${returnBD.returnPointsWon}%`,
      emoji: "🎯",
      actionable: true,
    });
  } else if (rpiTier === "AGGRESSIVE" && breakProb > 0.30) {
    signals.push({
      type: "RETURN_DOMINANCE",
      strength: returnerRPI,
      label: `📈 ${retName} aggressive return — RPI ${returnerRPI}/100 · BP conv: ${returnBD.breakPointConversion}%`,
      emoji: "📈",
      actionable: true,
    });
  }

  // ── DF Crisis Signal ──
  if (serveBD.doubleFaultRate > 1.0) {
    const totalDFs = stats ? (server === 1 ? stats.p1_doubleFaults : stats.p2_doubleFaults) : 0;
    signals.push({
      type: "DF_CRISIS",
      strength: Math.min(85, Math.round(serveBD.doubleFaultRate * 50)),
      label: `🔥 DF CRISIS — ${totalDFs} DFs (${serveBD.doubleFaultRate.toFixed(1)}/game) · Break prob ${Math.round(breakProb*100)}%`,
      emoji: "🔥",
      actionable: true,
    });
  }

  // ── Ace Power (hold strength) ──
  if (serveBD.aceRate > 1.5 && holdProb > 0.75) {
    signals.push({
      type: "ACE_POWER",
      strength: Math.round(holdProb * 100),
      label: `💪 ACE POWER — ${serveBD.aceRate.toFixed(1)} aces/game · Hold ${Math.round(holdProb*100)}%`,
      emoji: "💪",
      actionable: false,
    });
  }

  // ── Second Serve Vulnerability ──
  if (serveBD.secondServeWon < 38 && !isBreakPoint) {
    signals.push({
      type: "SECOND_SERVE_VULN",
      strength: Math.round((50 - serveBD.secondServeWon) * 2),
      label: `🎯 2nd serve vulnerability — ${serveBD.secondServeWon}% won · Target 2nd serves`,
      emoji: "🎯",
      actionable: true,
    });
  }

  // ── Game Point Hold ──
  if (phase === "GAME_POINT" && holdProb > 0.80) {
    signals.push({
      type: "GAME_POINT_HOLD",
      strength: Math.round(holdProb * 100),
      label: `✅ Game point — Hold ${Math.round(holdProb*100)}%${serveBD.aceRate > 1 ? " · Ace threat" : ""}`,
      emoji: "✅",
      actionable: false,
    });
  }

  // ── Pressure Building ──
  if (phase === "RETURNER_BUILDING" && retPts >= 2 && srvPts <= 1) {
    signals.push({
      type: "PRESSURE_BUILDING",
      strength: Math.round(breakProb * 80),
      label: `📊 Pressure building ${PT_LABELS[Math.min(srvPts,4)]}-${PT_LABELS[Math.min(retPts,4)]} · Break ${Math.round(breakProb*100)}% · ${retName} momentum`,
      emoji: "📊",
      actionable: breakProb > 0.30,
    });
  }

  // ── Hold Strong ──
  if (holdProb > 0.85 && phase !== "BREAK_POINT" && !isTB) {
    signals.push({
      type: "HOLD_STRONG",
      strength: Math.round(holdProb * 100),
      label: `🛡 ${srvName} hold strong at ${Math.round(holdProb*100)}%${serTier === "ELITE" ? " · ELITE serve" : serTier === "STRONG" ? " · Strong serve" : ""}`,
      emoji: "🛡",
      actionable: false,
    });
  }

  // ── Clutch Server (saves BPs well) ──
  if (stats) {
    const srvBPC = server === 1 ? stats.p1_breakPointsConverted : stats.p2_breakPointsConverted;
    const [, bpFaced] = (srvBPC || "0/0").split("/").map(Number);
    const srvBPS = server === 1 ? stats.p2_breakPointsConverted : stats.p1_breakPointsConverted;
    const [bpConv, bpTot] = (srvBPS || "0/0").split("/").map(Number);
    if (bpTot >= 3 && bpConv / bpTot < 0.25) {
      signals.push({
        type: "CLUTCH_SERVER",
        strength: Math.round((1 - bpConv / bpTot) * 60),
        label: `🧊 ${srvName} clutch — saved ${bpTot - bpConv}/${bpTot} BPs (${Math.round((1-bpConv/bpTot)*100)}%)`,
        emoji: "🧊",
        actionable: false,
      });
    }
  }

  // ── Break Back Risk ──
  if (stats) {
    // If the current server just broke, they might lose concentration
    const srvBreaksToday = server === 1 ? stats.p1_breakPointsConverted : stats.p2_breakPointsConverted;
    const [bW] = (srvBreaksToday || "0/0").split("/").map(Number);
    if (bW > 0) {
      // Server broke earlier — check if hold rate is average or below
      const srvHoldPct = serveBD.firstServeIn * serveBD.firstServeWon / 100;
      if (srvHoldPct < avgs.holdRate * 0.85 && breakProb > 0.25) {
        signals.push({
          type: "BREAK_BACK_RISK",
          strength: Math.round(breakProb * 70),
          label: `🔄 Break-back risk — ${srvName} broke but hold rate below avg · Break ${Math.round(breakProb*100)}%`,
          emoji: "🔄",
          actionable: true,
        });
      }
    }
  }

  // ── Set On Serve ──
  const totalSetGames = csGames.p1 + csGames.p2;
  if (totalSetGames >= 6 && Math.abs(csGames.p1 - csGames.p2) <= 1 && !isTB) {
    signals.push({
      type: "SET_ON_SERVE",
      strength: 30,
      label: `⚖️ Set on serve at ${setLabel} — next break decides set`,
      emoji: "⚖️",
      actionable: true,
    });
  }

  // Sort by strength descending
  signals.sort((a, b) => b.strength - a.strength);

  return signals;
}


// ═══════════════════════════════════════════════════════════════════════════════
//  MARKOV CHAIN — Exact recursive game/tiebreak probability
// ═══════════════════════════════════════════════════════════════════════════════

function gameWinProb(p: number, pts1: number, pts2: number): number {
  if (pts1 >= 4 && pts1 - pts2 >= 2) return 1;
  if (pts2 >= 4 && pts2 - pts1 >= 2) return 0;
  if (pts1 >= 3 && pts2 >= 3) {
    const d = pts1 - pts2;
    if (d === 0) return (p * p) / (p * p + (1 - p) * (1 - p));
    if (d === 1) return p + (1 - p) * (p * p) / (p * p + (1 - p) * (1 - p));
    return 0;
  }
  return p * gameWinProb(p, pts1 + 1, pts2) + (1 - p) * gameWinProb(p, pts1, pts2 + 1);
}

function tbWinProb(p: number, a: number, b: number): number {
  if (a >= 7 && a - b >= 2) return 1;
  if (b >= 7 && b - a >= 2) return 0;
  if (a >= 6 && b >= 6) return (p * p) / (p * p + (1 - p) * (1 - p));
  return p * tbWinProb(p, a + 1, b) + (1 - p) * tbWinProb(p, a, b + 1);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  TRUE PROBABILITY ENGINE — Set & Match level Markov chain
//  Faithful TS port of hierarchical_model.py (Barnett & Clarke 2005), so the
//  client-side engine (this file, used on the static Netlify deployment) and
//  the Python trading_server engine produce the *same* numbers when both are
//  available. Tour-aware via the `avgs`/`tour` baselines above.
// ═══════════════════════════════════════════════════════════════════════════════

function comb(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  k = Math.min(k, n - k);
  let result = 1;
  for (let i = 0; i < k; i++) result = (result * (n - i)) / (i + 1);
  return result;
}

/** Closed-form P(hold) from a single point-win probability — mirrors hierarchical_model.game_win_prob */
function gameHoldFromPointProb(p: number): number {
  const q = 1 - p;
  const hold = p ** 4 + 4 * p ** 4 * q + 10 * p ** 4 * q ** 2 + 20 * p ** 3 * q ** 3 * (p ** 2 / (p ** 2 + q ** 2));
  return clamp(hold, 0, 1);
}

/** P(player wins set) given their hold-as-server prob and opponent's hold-as-server prob. Mirrors set_win_prob. */
function setWinProbability(pHoldServer: number, pHoldReturner: number): number {
  const pAvg = (pHoldServer + (1 - pHoldReturner)) / 2;
  const qAvg = 1 - pAvg;

  let pSet = 0;
  for (let oppGames = 0; oppGames < 5; oppGames++) {
    const total = 6 + oppGames - 1;
    pSet += comb(total, oppGames) * pAvg ** 6 * qAvg ** oppGames;
  }
  const p55 = comb(10, 5) * pAvg ** 5 * qAvg ** 5;
  const p75 = p55 * pAvg ** 2;
  const p66 = comb(12, 6) * pAvg ** 6 * qAvg ** 6;
  const pTb = clamp(0.5 + 0.8 * (pAvg - 0.5), 0, 1);

  return clamp(pSet + p75 + p66 * pTb, 0, 1);
}

/** P(player wins match) from a single set-win probability. Mirrors match_win_prob_from_set_prob. */
function matchWinProbFromSetProb(pSet: number, bestOf: number): number {
  if (bestOf >= 5) return pSet ** 3 * (6 * pSet ** 2 - 15 * pSet + 10);
  return pSet ** 2 * (3 - 2 * pSet);
}

/**
 * P(player1 wins the match) needing need1 more sets, opponent needing need2,
 * with per-set win probability pSet. Negative binomial race:
 *   P = Σ_{j=0}^{need2-1} C(need1+j-1, j) · pSet^need1 · (1-pSet)^j
 * (The previous implementation of this formula was wrong — it multiplied every
 * term by (1-pSet)^need2, inverting/deflating probabilities badly enough that a
 * player leading 6-0, 1-0 was priced at 16% to win.)
 */
function matchProbRemaining(pSet: number, need1: number, need2: number): number {
  if (need1 <= 0) return 1;
  if (need2 <= 0) return 0;
  let p = 0;
  for (let j = 0; j < need2; j++) {
    p += comb(need1 + j - 1, j) * pSet ** need1 * (1 - pSet) ** j;
  }
  return clamp(p, 0, 1);
}

/**
 * P(player1 wins the CURRENT set) from the game score — exact recursion over
 * the games race (first to 6 win-by-2, tiebreak at 6-6), with pGame = P1's
 * average game-win probability. Depth is bounded (≤ 13 games), no memo needed.
 */
function setWinFromGames(g1: number, g2: number, pGame: number, pTb: number): number {
  if (g1 >= 6 && g1 - g2 >= 2) return 1;
  if (g2 >= 6 && g2 - g1 >= 2) return 0;
  if (g1 >= 7) return 1;
  if (g2 >= 7) return 0;
  if (g1 === 6 && g2 === 6) return pTb;
  return pGame * setWinFromGames(g1 + 1, g2, pGame, pTb)
       + (1 - pGame) * setWinFromGames(g1, g2 + 1, pGame, pTb);
}

/**
 * Score-conditioned match win probability — the key live-trading number.
 * Conditions on the sets already banked AND the game score of the set in
 * progress: P(match) = P(win this set)·P(race | set won) + P(lose it)·P(race | set lost).
 */
function matchWinProbFromScore(
  setsP1: number, setsP2: number,
  gamesP1: number, gamesP2: number,
  p1Serving: boolean,
  p1PointWin: number, p2PointWin: number,
  bestOf: number,
): number {
  const setsNeeded = Math.ceil(bestOf / 2);
  if (setsP1 >= setsNeeded) return 1;
  if (setsP2 >= setsNeeded) return 0;

  const p1Need = setsNeeded - setsP1;
  const p2Need = setsNeeded - setsP2;

  const p1Hold = gameHoldFromPointProb(p1PointWin);
  const p2Hold = gameHoldFromPointProb(p2PointWin);

  const p1SetAsServer = setWinProbability(p1Hold, p2Hold);
  const p1SetAsReturner = setWinProbability(1 - p2Hold, 1 - p1Hold);
  const p1Set = clamp((p1SetAsServer + p1SetAsReturner) / 2, 0.05, 0.95);

  // Current set: exact games-race recursion instead of a ±2pp/game nudge
  const pGame = clamp((p1Hold + (1 - p2Hold)) / 2, 0.05, 0.95);
  const pTb = clamp(0.5 + 0.8 * (pGame - 0.5), 0.05, 0.95);
  const pThisSet = setWinFromGames(gamesP1, gamesP2, pGame, pTb);

  const pMatch =
    pThisSet * matchProbRemaining(p1Set, p1Need - 1, p2Need) +
    (1 - pThisSet) * matchProbRemaining(p1Set, p1Need, p2Need - 1);

  return clamp(pMatch, 0, 1);
}

export interface TrueProbabilities {
  /** P(player1 wins match) — combines Markov set/game math with current live score state */
  p1MatchProb: number;
  p2MatchProb: number;
  /** P(player1 wins the set currently in progress) */
  p1SetProb: number;
  /** P(current server holds this game) — same number as BreakHoldSignals.holdProb */
  gameHoldProb: number;
  /** Which tour baseline was used */
  tour: string;
  method: "markov-tour-aware";
}

/**
 * Per-player intrinsic point-win-on-serve rate, derived from live stats when
 * available, else from the pre-match Elo prior — same formula the engine
 * already used for the *current server*, generalised to either player so we
 * can run the Markov set/match recursion (which needs both players' rates).
 */
function pointWinRateFor(
  player: 1 | 2, stats: LiveMatchStats | null, p1WinProb: number,
): number {
  if (stats) {
    const firstPct = (player === 1 ? stats.p1_firstServePercent : stats.p2_firstServePercent) || 60;
    const firstWon = (player === 1 ? stats.p1_firstServeWon : stats.p2_firstServeWon) || 65;
    const secondWon = (player === 1 ? stats.p1_secondServeWon : stats.p2_secondServeWon) || 45;
    const p = (firstPct / 100) * (firstWon / 100) + (1 - firstPct / 100) * (secondWon / 100);
    return clamp(p, 0.35, 0.85);
  }
  const prob = player === 1 ? p1WinProb : 1 - p1WinProb;
  return clamp(0.55 + (prob - 0.5) * 0.3, 0.45, 0.75);
}

/**
 * Main entry point: True P at game / set / match level, tour-aware.
 * This is what should drive the "edge vs bookmaker" calculation for every
 * tour, not just ATP.
 */
export function computeTrueProbabilities(
  server: 1 | 2,
  currentSetGames: { p1: number; p2: number },
  completedSets: { p1: number; p2: number }[],
  stats: LiveMatchStats | null,
  bestOf: number,
  p1WinProb: number,
  tour: string = "ATP",
): TrueProbabilities {
  const p1PointWin = pointWinRateFor(1, stats, p1WinProb);
  const p2PointWin = pointWinRateFor(2, stats, p1WinProb);

  const setsP1 = completedSets.filter((s) => s.p1 > s.p2).length;
  const setsP2 = completedSets.filter((s) => s.p2 > s.p1).length;

  const p1MatchProb = matchWinProbFromScore(
    setsP1, setsP2, currentSetGames.p1, currentSetGames.p2,
    server === 1, p1PointWin, p2PointWin, bestOf,
  );

  const p1Hold = gameHoldFromPointProb(p1PointWin);
  const p2Hold = gameHoldFromPointProb(p2PointWin);
  const p1SetAsServer = setWinProbability(p1Hold, p2Hold);
  const p1SetAsReturner = setWinProbability(1 - p2Hold, 1 - p1Hold);
  const p1SetProb = clamp((p1SetAsServer + p1SetAsReturner) / 2 + (currentSetGames.p1 - currentSetGames.p2) * 0.02, 0.05, 0.95);

  const gameHoldProb = server === 1 ? p1Hold : p2Hold;

  return {
    p1MatchProb,
    p2MatchProb: 1 - p1MatchProb,
    p1SetProb,
    gameHoldProb,
    tour,
    method: "markov-tour-aware",
  };
}


// ═══════════════════════════════════════════════════════════════════════════════
//  HEDGE ENGINE — when to hedge against a live position
//  TS port of trading_server/hedge_engine.py's three triggers:
//    1. Trend break  (score swings against the favoured side)
//    2. Adverse move (odds move >25% against the reference entry state)
//    3. Deuce loss   (DEUCE → AD-OUT)
//  The Python backend hedges against live exchange game-odds inside an open
//  position; here (no backend / no placed bet) we evaluate the same triggers
//  against the bookmaker match-odds captured at the moment each match first
//  reached a "tradeable" state (30-15 / 15-30 / 30-30 / DEUCE — the same
//  states trading_engine.py treats as entry points), so the signal tells you
//  "hedge now" if you'd entered around there.
// ═══════════════════════════════════════════════════════════════════════════════

export type HedgeTrigger = "TREND_BREAK" | "ADVERSE_MOVE" | "DEUCE_LOSS";

export interface HedgeAlert {
  shouldHedge: boolean;
  trigger?: HedgeTrigger;
  reason: string;
  urgency: "LOW" | "HIGH" | "IMMEDIATE";
  /** hedge_size = (entry_size × entry_odds) / current_odds — multiplier on a unit stake */
  hedgeSizeMultiplier?: number;
}

const TREND_BREAKS = new Set([
  "30-15>30-30", "15-30>30-30", "40-30>DEUCE", "30-40>DEUCE", "AD-IN>DEUCE",
]);

const ENTRY_ELIGIBLE_STATES = new Set(["30-15", "15-30", "30-30", "DEUCE"]);

/** Mirrors ScoreState._recompute()'s game_state_key — from the SERVER's perspective. */
function gameStateKey(srvPts: number, retPts: number, isTiebreak: boolean): string {
  if (isTiebreak) return "TIEBREAK";
  if (srvPts >= 3 && retPts >= 3) {
    if (srvPts === retPts) return "DEUCE";
    return srvPts > retPts ? "AD-IN" : "AD-OUT";
  }
  const m: Record<number, string> = { 0: "0", 1: "15", 2: "30", 3: "40" };
  return `${m[srvPts] ?? "40"}-${m[retPts] ?? "40"}`;
}

interface HedgeMemory { prevStateKey?: string; referenceOdds?: number }
const _hedgeMemory = new Map<string, HedgeMemory>();

/**
 * Evaluate the hedge triggers for a given match, polled once per tick.
 * `currentOddsAgainstFavourite` = live market odds on the side you'd be
 * holding a backed position against (i.e. the side that benefits from a
 * trend break) — pass the bookmaker odds for the returner/non-favoured side.
 */
export function evaluateHedgeSignal(
  matchId: string,
  srvPts: number, retPts: number, isTiebreak: boolean,
  currentOddsAgainstFavourite: number | undefined,
): HedgeAlert {
  const stateKey = gameStateKey(srvPts, retPts, isTiebreak);
  const mem = _hedgeMemory.get(matchId) || {};
  const prevStateKey = mem.prevStateKey;

  // Capture a reference odds line the first time we hit an entry-eligible state
  if (ENTRY_ELIGIBLE_STATES.has(stateKey) && mem.referenceOdds === undefined && currentOddsAgainstFavourite) {
    mem.referenceOdds = currentOddsAgainstFavourite;
  }
  // Reset reference odds once we leave the game (new game started)
  if (stateKey === "0-0") {
    mem.referenceOdds = undefined;
  }

  let alert: HedgeAlert = { shouldHedge: false, reason: "", urgency: "LOW" };

  // 1. Trend break
  if (prevStateKey && TREND_BREAKS.has(`${prevStateKey}>${stateKey}`)) {
    alert = {
      shouldHedge: true,
      trigger: "TREND_BREAK",
      reason: `Trend break: ${prevStateKey} → ${stateKey}`,
      urgency: "HIGH",
      hedgeSizeMultiplier: mem.referenceOdds && currentOddsAgainstFavourite
        ? roundTo(mem.referenceOdds / currentOddsAgainstFavourite, 2) : undefined,
    };
  }
  // 2. Adverse odds move > 25%
  else if (mem.referenceOdds && currentOddsAgainstFavourite && mem.referenceOdds > 0) {
    const move = Math.abs(currentOddsAgainstFavourite - mem.referenceOdds) / mem.referenceOdds;
    if (move > 0.25 && currentOddsAgainstFavourite > mem.referenceOdds) {
      alert = {
        shouldHedge: true,
        trigger: "ADVERSE_MOVE",
        reason: `Odds moved ${Math.round(move * 100)}% against entry`,
        urgency: "HIGH",
        hedgeSizeMultiplier: roundTo(mem.referenceOdds / currentOddsAgainstFavourite, 2),
      };
    }
  }
  // 3. Deuce loss
  if (stateKey === "AD-OUT" && prevStateKey === "DEUCE") {
    alert = {
      shouldHedge: true,
      trigger: "DEUCE_LOSS",
      reason: "Deuce → AD-OUT — hedge immediately",
      urgency: "IMMEDIATE",
      hedgeSizeMultiplier: mem.referenceOdds && currentOddsAgainstFavourite
        ? roundTo(mem.referenceOdds / currentOddsAgainstFavourite, 2) : undefined,
    };
  }

  mem.prevStateKey = stateKey;
  _hedgeMemory.set(matchId, mem);
  return alert;
}

function roundTo(v: number, dp: number): number {
  const f = 10 ** dp;
  return Math.round(v * f) / f;
}


// ═══════════════════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}
