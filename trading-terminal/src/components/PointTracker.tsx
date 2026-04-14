"use client";

import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import type { ScheduledMatch } from "@/lib/scheduleService";
import { probToOdds, kellyFraction, fetchLiveScore } from "@/lib/scheduleService";

/* ═══════════════════════════════════════════════════════════════════════════════
   Tennis scoring engine + Markov match-win probability
   ═══════════════════════════════════════════════════════════════════════════ */

const PT_LABELS = ["0", "15", "30", "40", "AD"];

interface GameScore { p1: number; p2: number }
interface SetScore  { p1: number; p2: number }
interface MatchState {
  sets: { p1: number; p2: number }[];
  currentSet: SetScore;
  currentGame: GameScore;
  server: 1 | 2;
  tiebreak: boolean;
  matchOver: boolean;
  winner?: 1 | 2;
  setsToWin: number;
}

interface PointEntry {
  id: number;
  winner: 1 | 2;
  server: 1 | 2;
  scoreBefore: string;
  scoreAfter: string;
  p1WinProb: number;
  timestamp: number;
  isBreak: boolean;
  isSetPoint: boolean;
  isMatchPoint: boolean;
  ewma1: number;
  ewma2: number;
  fatigue1: number;
  fatigue2: number;
  breakOppProb: number;
  rallyIntensity: number;
}

/* ═══════════════════════════════════════════════════════════════════════════════
   EWMA Momentum Engine
   α_fast = 0.15, α_slow = 0.05
   Signal = fast - slow → positive = player gaining momentum
   ═══════════════════════════════════════════════════════════════════════════ */

const EWMA_ALPHA_FAST = 0.15;
const EWMA_ALPHA_SLOW = 0.05;

interface EwmaState {
  fast1: number; slow1: number;
  fast2: number; slow2: number;
}

function initEwma(): EwmaState {
  return { fast1: 0.5, slow1: 0.5, fast2: 0.5, slow2: 0.5 };
}

function updateEwma(prev: EwmaState, winner: 1 | 2): EwmaState {
  const v1 = winner === 1 ? 1 : 0;
  const v2 = winner === 2 ? 1 : 0;
  return {
    fast1: EWMA_ALPHA_FAST * v1 + (1 - EWMA_ALPHA_FAST) * prev.fast1,
    slow1: EWMA_ALPHA_SLOW * v1 + (1 - EWMA_ALPHA_SLOW) * prev.slow1,
    fast2: EWMA_ALPHA_FAST * v2 + (1 - EWMA_ALPHA_FAST) * prev.fast2,
    slow2: EWMA_ALPHA_SLOW * v2 + (1 - EWMA_ALPHA_SLOW) * prev.slow2,
  };
}

function ewmaMomentum(e: EwmaState, player: 1 | 2): number {
  return player === 1 ? e.fast1 - e.slow1 : e.fast2 - e.slow2;
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Fatigue Model
   Accumulates per point, extra load in deuce/TB, recovery on changeovers
   ═══════════════════════════════════════════════════════════════════════════ */

interface FatigueState {
  totalPoints: number;
  deuceGames: number;
  tiebreaks: number;
  setsPlayed: number;
  p1Load: number;
  p2Load: number;
}

function initFatigue(): FatigueState {
  return { totalPoints: 0, deuceGames: 0, tiebreaks: 0, setsPlayed: 0, p1Load: 0, p2Load: 0 };
}

function updateFatigue(prev: FatigueState, state: MatchState, prevState: MatchState): FatigueState {
  const f = { ...prev };
  f.totalPoints++;
  const baseFatigue = 0.003;
  const inDeuce = !state.tiebreak && state.currentGame.p1 >= 3 && state.currentGame.p2 >= 3;
  const deucePenalty = inDeuce ? 0.005 : 0;
  const tbPenalty = state.tiebreak ? 0.004 : 0;
  const gameDepth = state.currentGame.p1 + state.currentGame.p2;
  const rallyBonus = gameDepth > 6 ? 0.003 : 0;

  f.p1Load += baseFatigue + deucePenalty + tbPenalty + rallyBonus;
  f.p2Load += baseFatigue + deucePenalty + tbPenalty + rallyBonus;

  const newGame = state.currentGame.p1 === 0 && state.currentGame.p2 === 0
    && prevState.currentGame.p1 + prevState.currentGame.p2 > 0;
  if (newGame) { f.p1Load *= 0.97; f.p2Load *= 0.97; }

  if (state.sets.length > prevState.sets.length) {
    f.setsPlayed = state.sets.length;
    f.p1Load *= 0.90;
    f.p2Load *= 0.90;
  }

  if (inDeuce && !(prevState.currentGame.p1 >= 3 && prevState.currentGame.p2 >= 3)) f.deuceGames++;
  if (state.tiebreak && !prevState.tiebreak) f.tiebreaks++;

  return f;
}

function fatigueIndex(f: FatigueState, player: 1 | 2): number {
  return Math.min(1, player === 1 ? f.p1Load : f.p2Load);
}

function fatigueLabel(idx: number): { label: string; color: string } {
  if (idx < 0.15) return { label: "FRESH", color: "text-terminal-green" };
  if (idx < 0.35) return { label: "NORMAL", color: "text-terminal-green" };
  if (idx < 0.55) return { label: "TIRING", color: "text-terminal-yellow" };
  if (idx < 0.75) return { label: "FATIGUED", color: "text-terminal-yellow" };
  return { label: "EXHAUSTED", color: "text-terminal-red" };
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Break Opportunity Predictor
   EWMA of returner + server fatigue + score pressure + game depth
   ═══════════════════════════════════════════════════════════════════════════ */

function breakOppProb(
  ewma: EwmaState, fatigue: FatigueState, state: MatchState,
  pServe1: number, pServe2: number,
): number {
  if (state.tiebreak || state.matchOver) return 0;
  const server = state.server;
  const returner = server === 1 ? 2 : 1;
  const serverPct = server === 1 ? pServe1 : pServe2;
  const baseBreakProb = 1 - holdProb(serverPct);

  const returnerMom = ewmaMomentum(ewma, returner);
  const momentumAdj = returnerMom * 0.3;
  const serverFat = fatigueIndex(fatigue, server);
  const fatigueAdj = serverFat * 0.15;
  const gameDiff = server === 1
    ? state.currentSet.p2 - state.currentSet.p1
    : state.currentSet.p1 - state.currentSet.p2;
  const pressureAdj = gameDiff > 0 ? gameDiff * 0.03 : 0;
  const gs = state.currentGame;
  const depth = gs.p1 + gs.p2;
  const depthAdj = depth >= 6 ? 0.05 : depth >= 4 ? 0.02 : 0;

  return Math.max(0, Math.min(0.85, baseBreakProb + momentumAdj + fatigueAdj + pressureAdj + depthAdj));
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Position Signals — GAME / SET / MATCH entry & exit
   ═══════════════════════════════════════════════════════════════════════════ */

type SignalType = "ENTRY" | "EXIT" | "HOLD" | "HEDGE" | "NO_POSITION";
type SignalStrength = "STRONG" | "MODERATE" | "WEAK";

interface PositionSignal {
  level: "GAME" | "SET" | "MATCH";
  type: SignalType;
  strength: SignalStrength;
  side: 1 | 2 | null;
  reason: string;
  edgePct: number;
  stopLoss?: number;
  target?: number;
}

function computePositionSignals(
  currentP1Prob: number, preMatchP1Prob: number,
  ewma: EwmaState, fatigue: FatigueState,
  state: MatchState, history: PointEntry[], breakOpp: number,
): PositionSignal[] {
  const signals: PositionSignal[] = [];
  if (state.matchOver) return signals;

  const probShift = currentP1Prob - preMatchP1Prob;
  const p1Mom = ewmaMomentum(ewma, 1);
  const p2Mom = ewmaMomentum(ewma, 2);
  const p1Fat = fatigueIndex(fatigue, 1);
  const p2Fat = fatigueIndex(fatigue, 2);
  const recentProbs = history.slice(-10).map(h => h.p1WinProb);
  const probVol = recentProbs.length > 2 ? stdDev(recentProbs) : 0;

  // ── MATCH ──
  const matchEdge = Math.abs(probShift);
  if (matchEdge > 0.08 && history.length >= 10) {
    const fav = currentP1Prob > preMatchP1Prob ? 1 : 2;
    const mom = fav === 1 ? p1Mom : p2Mom;
    const oppFat = fav === 1 ? p2Fat : p1Fat;
    if (mom > 0.05) {
      signals.push({
        level: "MATCH", type: "ENTRY",
        strength: matchEdge > 0.15 ? "STRONG" : "MODERATE",
        side: fav,
        reason: `P shifted ${(probShift * 100).toFixed(0)}% from pre-match, mom ${mom > 0 ? "+" : ""}${(mom * 100).toFixed(0)}%${oppFat > 0.4 ? ", opp fatigued" : ""}`,
        edgePct: matchEdge * 100,
        stopLoss: fav === 1 ? Math.max(0.25, currentP1Prob - 0.15) : Math.min(0.75, currentP1Prob + 0.15),
        target: fav === 1 ? Math.min(0.95, currentP1Prob + 0.10) : Math.max(0.05, currentP1Prob - 0.10),
      });
    }
  }
  if (matchEdge > 0.05 && history.length >= 8) {
    const wasFav = probShift > 0 ? 1 : 2;
    const reversed = wasFav === 1 ? p1Mom < -0.08 : p2Mom < -0.08;
    if (reversed) {
      signals.push({
        level: "MATCH", type: "EXIT", strength: "STRONG", side: wasFav,
        reason: `Momentum reversed: ${wasFav === 1 ? "P1" : "P2"} losing grip, EWMA -${(Math.abs(wasFav === 1 ? p1Mom : p2Mom) * 100).toFixed(0)}%`,
        edgePct: -matchEdge * 50,
      });
    }
  }

  // ── SET ──
  if (history.length > 0) {
    const lastPt = history[history.length - 1];
    if (lastPt.isBreak) {
      signals.push({
        level: "SET", type: "ENTRY", strength: "STRONG", side: lastPt.winner,
        reason: `BREAK! ${lastPt.winner === 1 ? "P1" : "P2"} broke serve — back for set`,
        edgePct: 12,
        stopLoss: lastPt.winner === 1 ? currentP1Prob - 0.12 : currentP1Prob + 0.12,
      });
    }
  }
  if (history.length >= 2) {
    const last4 = history.slice(-4);
    const breaks = last4.filter(p => p.isBreak);
    if (breaks.length >= 2 && breaks[breaks.length - 1].winner !== breaks[breaks.length - 2].winner) {
      signals.push({
        level: "SET", type: "EXIT", strength: "MODERATE", side: null,
        reason: "Break back! Service breaks trading — reduce set exposure",
        edgePct: -3,
      });
    }
  }

  // ── GAME ──
  if (breakOpp > 0.35 && !state.tiebreak) {
    const returner = state.server === 1 ? 2 : 1;
    signals.push({
      level: "GAME", type: "ENTRY",
      strength: breakOpp > 0.50 ? "STRONG" : "MODERATE",
      side: returner,
      reason: `Break opp ${(breakOpp * 100).toFixed(0)}% — returner ${returner === 1 ? "P1" : "P2"}, server ${state.server === 1 ? (p1Fat > 0.4 ? "fatigued" : "stable") : (p2Fat > 0.4 ? "fatigued" : "stable")}`,
      edgePct: (breakOpp - 0.25) * 40,
    });
  }
  if (breakOpp < 0.15 && state.currentGame.p1 + state.currentGame.p2 >= 2) {
    signals.push({
      level: "GAME", type: "EXIT", strength: "WEAK", side: null,
      reason: `Server stabilized — hold prob ${((1 - breakOpp) * 100).toFixed(0)}%, close game pos`,
      edgePct: -1,
    });
  }

  // ── HEDGE ──
  if (probVol > 0.06 && history.length >= 15) {
    signals.push({
      level: "MATCH", type: "HEDGE", strength: "MODERATE", side: null,
      reason: `High volatility (σ=${(probVol * 100).toFixed(1)}%) — consider delta neutral hedge`,
      edgePct: 0,
    });
  }

  return signals;
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Delta Neutral Hedge Calculator
   ═══════════════════════════════════════════════════════════════════════════ */

interface HedgeCalc {
  action: string;
  player: string;
  hedgeStake: number;
  profitIfWin: number;
  profitIfLose: number;
  netPosition: number;
  hedgeOdds: number;
}

function calcDeltaNeutralHedge(
  originalSide: 1 | 2, originalStake: number, originalOdds: number,
  currentP1Prob: number, p1Name: string, p2Name: string,
): HedgeCalc {
  const currentOdds1 = probToOdds(currentP1Prob);
  const currentOdds2 = probToOdds(1 - currentP1Prob);
  const hedgeOn = originalSide === 1 ? 2 : 1;
  const hOdds = hedgeOn === 1 ? currentOdds1 : currentOdds2;
  const origProfit = originalStake * (originalOdds - 1);
  const hStake = Math.max(0, (originalStake * originalOdds) / hOdds);

  let pWin: number, pLose: number;
  if (originalSide === 1) {
    pWin = origProfit - hStake;
    pLose = -originalStake + hStake * (hOdds - 1);
  } else {
    pWin = -originalStake + hStake * (hOdds - 1);
    pLose = origProfit - hStake;
  }

  return {
    action: "BACK",
    player: hedgeOn === 1 ? p1Name : p2Name,
    hedgeStake: Math.round(hStake * 100) / 100,
    profitIfWin: Math.round(pWin * 100) / 100,
    profitIfLose: Math.round(pLose * 100) / 100,
    netPosition: Math.round((pWin + pLose) / 2 * 100) / 100,
    hedgeOdds: hOdds,
  };
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Scoring Engine
   ═══════════════════════════════════════════════════════════════════════════ */

function initState(bestOf: number): MatchState {
  return {
    sets: [], currentSet: { p1: 0, p2: 0 }, currentGame: { p1: 0, p2: 0 },
    server: 1, tiebreak: false, matchOver: false, setsToWin: bestOf === 5 ? 3 : 2,
  };
}

function formatScore(s: MatchState): string {
  const sets = s.sets.map(st => `${st.p1}-${st.p2}`).join(" ");
  const cs = `${s.currentSet.p1}-${s.currentSet.p2}`;
  if (s.tiebreak) return `${sets} ${cs} TB(${s.currentGame.p1}-${s.currentGame.p2})`.trim();
  const p1 = s.currentGame.p1, p2 = s.currentGame.p2;
  let pts: string;
  if (p1 <= 3 && p2 <= 3) pts = `${PT_LABELS[p1]}-${PT_LABELS[p2]}`;
  else if (p1 > p2) pts = "AD-40";
  else if (p2 > p1) pts = "40-AD";
  else pts = "40-40";
  return `${sets} ${cs} ${pts}`.trim();
}

function advancePoint(state: MatchState, pointWinner: 1 | 2): MatchState {
  const s = JSON.parse(JSON.stringify(state)) as MatchState;
  if (s.matchOver) return s;
  if (s.tiebreak) {
    if (pointWinner === 1) s.currentGame.p1++; else s.currentGame.p2++;
    const tb1 = s.currentGame.p1, tb2 = s.currentGame.p2, tot = tb1 + tb2;
    if (tot === 1 || (tot > 1 && (tot - 1) % 2 === 0)) s.server = s.server === 1 ? 2 : 1;
    if (tb1 >= 7 && tb1 - tb2 >= 2) { s.currentSet.p1++; finishSet(s); }
    else if (tb2 >= 7 && tb2 - tb1 >= 2) { s.currentSet.p2++; finishSet(s); }
  } else {
    if (pointWinner === 1) s.currentGame.p1++; else s.currentGame.p2++;
    const g1 = s.currentGame.p1, g2 = s.currentGame.p2;
    let gameWon = false, gameWinner: 1 | 2 = 1;
    if (g1 >= 4 && g1 - g2 >= 2) { gameWon = true; gameWinner = 1; }
    else if (g2 >= 4 && g2 - g1 >= 2) { gameWon = true; gameWinner = 2; }
    if (gameWon) {
      if (gameWinner === 1) s.currentSet.p1++; else s.currentSet.p2++;
      s.currentGame = { p1: 0, p2: 0 };
      s.server = s.server === 1 ? 2 : 1;
      const cs = s.currentSet;
      if (cs.p1 >= 6 && cs.p1 - cs.p2 >= 2) finishSet(s);
      else if (cs.p2 >= 6 && cs.p2 - cs.p1 >= 2) finishSet(s);
      else if (cs.p1 === 6 && cs.p2 === 6) s.tiebreak = true;
    }
  }
  return s;
}

function finishSet(s: MatchState) {
  s.sets.push({ ...s.currentSet });
  s.currentSet = { p1: 0, p2: 0 };
  s.currentGame = { p1: 0, p2: 0 };
  s.tiebreak = false;
  const p1S = s.sets.filter(st => st.p1 > st.p2).length;
  const p2S = s.sets.filter(st => st.p2 > st.p1).length;
  if (p1S >= s.setsToWin) { s.matchOver = true; s.winner = 1; }
  else if (p2S >= s.setsToWin) { s.matchOver = true; s.winner = 2; }
}

/* ── Markov ── */

function matchWinProb(pS1: number, pS2: number, state: MatchState): number {
  if (state.matchOver) return state.winner === 1 ? 1 : 0;
  const p1Sets = state.sets.filter(s => s.p1 > s.p2).length;
  const p2Sets = state.sets.filter(s => s.p2 > s.p1).length;
  const pOnServe = state.server === 1 ? pS1 : (1 - pS2);
  const pGame = gameWinProb(pOnServe, state.currentGame.p1, state.currentGame.p2, state.tiebreak);
  const pSetW = setWinProb(pS1, pS2, state.currentSet.p1 + 1, state.currentSet.p2, state.server === 1 ? 2 : 1, state.setsToWin);
  const pSetL = setWinProb(pS1, pS2, state.currentSet.p1, state.currentSet.p2 + 1, state.server === 1 ? 2 : 1, state.setsToWin);
  const pSet = pGame * pSetW + (1 - pGame) * pSetL;
  const pMatchW = matchFromSets(pS1, pS2, p1Sets + 1, p2Sets, state.setsToWin);
  const pMatchL = matchFromSets(pS1, pS2, p1Sets, p2Sets + 1, state.setsToWin);
  return pSet * pMatchW + (1 - pSet) * pMatchL;
}

function gameWinProb(p: number, pts1: number, pts2: number, isTB: boolean): number {
  if (isTB) return tbWinProb(p, pts1, pts2);
  if (pts1 >= 4 && pts1 - pts2 >= 2) return 1;
  if (pts2 >= 4 && pts2 - pts1 >= 2) return 0;
  if (pts1 >= 3 && pts2 >= 3) {
    const d = pts1 - pts2;
    if (d === 0) return (p * p) / (p * p + (1 - p) * (1 - p));
    if (d === 1) return p + (1 - p) * (p * p) / (p * p + (1 - p) * (1 - p));
    return 0;
  }
  return p * gameWinProb(p, pts1 + 1, pts2, false) + (1 - p) * gameWinProb(p, pts1, pts2 + 1, false);
}

function tbWinProb(p: number, a: number, b: number): number {
  if (a >= 7 && a - b >= 2) return 1;
  if (b >= 7 && b - a >= 2) return 0;
  if (a >= 6 && b >= 6) return (p * p) / (p * p + (1 - p) * (1 - p));
  return p * tbWinProb(p, a + 1, b) + (1 - p) * tbWinProb(p, a, b + 1);
}

function setWinProb(pS1: number, pS2: number, g1: number, g2: number, ns: 1 | 2, stw: number): number {
  if (g1 >= 6 && g1 - g2 >= 2) return 1;
  if (g2 >= 6 && g2 - g1 >= 2) return 0;
  if (g1 === 6 && g2 === 6) { const p = ns === 1 ? pS1 : (1 - pS2); return (p * p) / (p * p + (1 - p) * (1 - p)); }
  if (g1 > 6 || g2 > 6) return g1 > g2 ? 1 : 0;
  const pG = ns === 1 ? holdProb(pS1) : (1 - holdProb(pS2));
  const n2: 1 | 2 = ns === 1 ? 2 : 1;
  return pG * setWinProb(pS1, pS2, g1 + 1, g2, n2, stw) + (1 - pG) * setWinProb(pS1, pS2, g1, g2 + 1, n2, stw);
}

function holdProb(p: number): number { return gameWinProb(p, 0, 0, false); }

function matchFromSets(pS1: number, pS2: number, s1: number, s2: number, stw: number): number {
  if (s1 >= stw) return 1;
  if (s2 >= stw) return 0;
  const pSet = setWinProb(pS1, pS2, 0, 0, 1, stw);
  const n1 = stw - s1, n2 = stw - s2, tot = n1 + n2 - 1;
  let pr = 0;
  for (let w = n1; w <= tot; w++) pr += binom(tot, w) * Math.pow(pSet, w) * Math.pow(1 - pSet, tot - w);
  return pr;
}

function binom(n: number, k: number): number {
  if (k > n) return 0; let r = 1;
  for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1);
  return r;
}

function stdDev(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Game Log — Markov probability at each game transition
   ═══════════════════════════════════════════════════════════════════════════ */

interface GameLogEntry {
  id: number;
  setNum: number;
  p1Games: number;
  p2Games: number;
  server: 1 | 2;
  p1WinProb: number;
  label: string;        // e.g. "Set 1: 3-2"
  isBreak: boolean;     // game was a break of serve
  isTiebreak: boolean;
}

/** Build a MatchState from ESPN live score data (game-level only, no point data) */
function buildStateFromLiveScore(
  liveScore: NonNullable<ScheduledMatch["liveScore"]>,
  bestOf: number,
): MatchState {
  const setsToWin = bestOf === 5 ? 3 : 2;
  const sets: { p1: number; p2: number }[] = liveScore.completedSets.map(s => ({ ...s }));
  const currentSet = { ...liveScore.currentSetGames };
  const tiebreak = currentSet.p1 === 6 && currentSet.p2 === 6;

  // Determine server — ESPN gives us this directly
  const server = liveScore.server;

  // Check if match is over
  const p1SetsWon = sets.filter(s => s.p1 > s.p2).length;
  const p2SetsWon = sets.filter(s => s.p2 > s.p1).length;
  const matchOver = p1SetsWon >= setsToWin || p2SetsWon >= setsToWin;
  const winner = matchOver ? (p1SetsWon >= setsToWin ? 1 : 2) : undefined;

  return {
    sets,
    currentSet,
    currentGame: { p1: 0, p2: 0 }, // ESPN doesn't provide point-level score
    server,
    tiebreak,
    matchOver,
    winner,
    setsToWin,
  };
}

/** Build game log with Markov probabilities from a MatchState's history */
function buildGameLog(
  liveScore: NonNullable<ScheduledMatch["liveScore"]>,
  pS1: number, pS2: number, bestOf: number,
): GameLogEntry[] {
  const entries: GameLogEntry[] = [];
  const setsToWin = bestOf === 5 ? 3 : 2;
  let id = 0;

  // Helper: compute Markov probability at a given score state
  const probAt = (
    completedSets: { p1: number; p2: number }[],
    curSet: { p1: number; p2: number },
    server: 1 | 2,
  ): number => {
    const tmpState: MatchState = {
      sets: completedSets,
      currentSet: curSet,
      currentGame: { p1: 0, p2: 0 },
      server,
      tiebreak: curSet.p1 === 6 && curSet.p2 === 6,
      matchOver: false,
      setsToWin,
    };
    return matchWinProb(pS1, pS2, tmpState);
  };

  // Walk through completed sets game by game
  let prevServer: 1 | 2 = liveScore.server; // approximate start server
  // We can't know the exact starting server, so we infer from total games played
  // Total games in completed sets
  let totalGamesPlayed = 0;
  for (const s of liveScore.completedSets) totalGamesPlayed += s.p1 + s.p2;
  totalGamesPlayed += liveScore.currentSetGames.p1 + liveScore.currentSetGames.p2;
  // Current server has served (totalGamesPlayed % 2 === 0) times since start
  // If totalGamesPlayed is even, server started. If odd, other player started.
  const startServer: 1 | 2 = (totalGamesPlayed % 2 === 0) ? liveScore.server : (liveScore.server === 1 ? 2 : 1);

  // Walk each completed set
  const allSets = [...liveScore.completedSets, liveScore.currentSetGames];
  const completedBefore: { p1: number; p2: number }[] = [];
  let runningServer = startServer;

  for (let si = 0; si < allSets.length; si++) {
    const setData = allSets[si];
    const isCurrentSet = si === allSets.length - 1 && si >= liveScore.completedSets.length;
    const maxGames = setData.p1 + setData.p2;

    // Simulate game-by-game progression through this set
    let g1 = 0, g2 = 0;
    // We don't know which player won each game, but we know the final score
    // For the game log, reconstruct with assumption: alternate holds/breaks
    // But actually we just show the probability at each game score point

    // Add entry at start of set
    entries.push({
      id: id++,
      setNum: si + 1,
      p1Games: 0,
      p2Games: 0,
      server: runningServer,
      p1WinProb: probAt(completedBefore, { p1: 0, p2: 0 }, runningServer),
      label: `Set ${si + 1}: 0-0`,
      isBreak: false,
      isTiebreak: false,
    });

    // We can generate intermediate game score entries for each possible game count
    for (let g = 1; g <= maxGames; g++) {
      // We don't know the exact order, so generate entries for each game boundary
      // Use a simple heuristic: distribute games proportionally
      const frac = g / maxGames;
      g1 = Math.round(frac * setData.p1);
      g2 = Math.round(frac * setData.p2);
      // Clamp
      g1 = Math.min(g1, setData.p1);
      g2 = Math.min(g2, setData.p2);
      if (g1 + g2 !== g) {
        // Adjust: prefer the leader
        if (g1 + g2 < g) {
          if (setData.p1 >= setData.p2 && g1 < setData.p1) g1++;
          else if (g2 < setData.p2) g2++;
          else g1++;
        }
      }

      const curServer: 1 | 2 = (entries.length % 2 === 0) ? startServer : (startServer === 1 ? 2 : 1);
      const isTB = g1 === 6 && g2 === 6;

      entries.push({
        id: id++,
        setNum: si + 1,
        p1Games: g1,
        p2Games: g2,
        server: curServer,
        p1WinProb: probAt(completedBefore, { p1: g1, p2: g2 }, curServer),
        label: isTB ? `Set ${si + 1}: TB` : `Set ${si + 1}: ${g1}-${g2}`,
        isBreak: false, // Can't determine without game-by-game data
        isTiebreak: isTB,
      });
    }

    if (!isCurrentSet) {
      completedBefore.push({ ...setData });
      // After a set, server alternates based on total games in the set
      if (maxGames % 2 === 1) runningServer = runningServer === 1 ? 2 : 1;
    }
  }

  return entries;
}

/* ═══════════════════════════════════════════════════════════════════════════════
   COMPONENT
   ═══════════════════════════════════════════════════════════════════════════ */

export default function PointTracker({ match }: { match: ScheduledMatch }) {
  const [state, setState] = useState<MatchState>(() => initState(match.best_of));
  const [history, setHistory] = useState<PointEntry[]>([]);
  const [ewma, setEwma] = useState<EwmaState>(initEwma);
  const [fatigue, setFatigue] = useState<FatigueState>(initFatigue);
  const [hedgeSide, setHedgeSide] = useState<1 | 2>(1);
  const [hedgeStake, setHedgeStake] = useState(100);
  const [hedgeOdds, setHedgeOdds] = useState(2.0);
  const [subTab, setSubTab] = useState<"live" | "signals" | "hedge">("live");
  const [mode, setMode] = useState<"auto" | "manual">(() => match.status === "live" && match.liveScore ? "auto" : "manual");
  const [liveMatch, setLiveMatch] = useState<ScheduledMatch>(match);
  const [lastRefresh, setLastRefresh] = useState<number>(Date.now());
  const [syncing, setSyncing] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auto-build state from live score when in auto mode
  const liveState = useMemo(() => {
    if (mode === "auto" && liveMatch.liveScore) {
      return buildStateFromLiveScore(liveMatch.liveScore, liveMatch.best_of);
    }
    return null;
  }, [mode, liveMatch.liveScore, liveMatch.best_of]);

  // Game log for live matches
  const gameLog = useMemo(() => {
    if (liveMatch.liveScore) {
      const pS1 = Math.max(0.50, Math.min(0.75, 0.55 + (liveMatch.p1_win_prob - 0.5) * 0.3));
      const pS2 = Math.max(0.50, Math.min(0.75, 0.55 + (liveMatch.p2_win_prob - 0.5) * 0.3));
      return buildGameLog(liveMatch.liveScore, pS1, pS2, liveMatch.best_of);
    }
    return [];
  }, [liveMatch]);

  // Use live state when in auto mode, manual state otherwise
  const activeState = liveState || state;

  // Poll for live score updates every 15 seconds
  useEffect(() => {
    if (mode !== "auto" || liveMatch.status !== "live") return;
    const poll = async () => {
      setSyncing(true);
      try {
        const fresh = await fetchLiveScore(liveMatch.id);
        if (fresh) {
          setLiveMatch(fresh);
          setLastRefresh(Date.now());
        }
      } catch { /* */ }
      finally { setSyncing(false); }
    };
    pollRef.current = setInterval(poll, 15_000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [mode, liveMatch.id, liveMatch.status]);

  // When match prop changes, update liveMatch
  useEffect(() => {
    setLiveMatch(match);
    if (match.status === "live" && match.liveScore && mode === "auto") {
      setLastRefresh(Date.now());
    }
  }, [match, mode]);

  const pServe1 = useMemo(() => Math.max(0.50, Math.min(0.75, 0.55 + (match.p1_win_prob - 0.5) * 0.3)), [match.p1_win_prob]);
  const pServe2 = useMemo(() => Math.max(0.50, Math.min(0.75, 0.55 + (match.p2_win_prob - 0.5) * 0.3)), [match.p2_win_prob]);
  const currentP1WinProb = useMemo(() => matchWinProb(pServe1, pServe2, activeState), [pServe1, pServe2, activeState]);
  const currentBreakOpp = useMemo(() => breakOppProb(ewma, fatigue, activeState, pServe1, pServe2), [ewma, fatigue, activeState, pServe1, pServe2]);
  const signals = useMemo(() => computePositionSignals(currentP1WinProb, match.p1_win_prob, ewma, fatigue, activeState, history, currentBreakOpp), [currentP1WinProb, match.p1_win_prob, ewma, fatigue, activeState, history, currentBreakOpp]);
  const hedge = useMemo(() => calcDeltaNeutralHedge(hedgeSide, hedgeStake, hedgeOdds, currentP1WinProb, match.player1, match.player2), [hedgeSide, hedgeStake, hedgeOdds, currentP1WinProb, match.player1, match.player2]);

  const logPoint = useCallback((winner: 1 | 2) => {
    if (state.matchOver) return;
    const scoreBefore = formatScore(state);
    const isBreakOpp = state.server !== winner;
    const newState = advancePoint(state, winner);
    const scoreAfter = formatScore(newState);
    const gameEnded = newState.currentGame.p1 === 0 && newState.currentGame.p2 === 0 && scoreBefore !== scoreAfter;
    const isBreak = isBreakOpp && gameEnded && !state.tiebreak;
    const newP1Prob = matchWinProb(pServe1, pServe2, newState);
    const newEwma = updateEwma(ewma, winner);
    const newFatigue = updateFatigue(fatigue, newState, state);
    const brkOpp = breakOppProb(newEwma, newFatigue, newState, pServe1, pServe2);
    const gd = newState.currentGame.p1 + newState.currentGame.p2;

    setHistory(prev => [...prev, {
      id: prev.length + 1, winner, server: state.server, scoreBefore, scoreAfter,
      p1WinProb: newP1Prob, timestamp: Date.now(), isBreak, isSetPoint: false,
      isMatchPoint: newState.matchOver, ewma1: newEwma.fast1, ewma2: newEwma.fast2,
      fatigue1: fatigueIndex(newFatigue, 1), fatigue2: fatigueIndex(newFatigue, 2),
      breakOppProb: brkOpp, rallyIntensity: Math.min(1, gd / 10),
    }]);
    setState(newState);
    setEwma(newEwma);
    setFatigue(newFatigue);
  }, [state, ewma, fatigue, pServe1, pServe2]);

  const undo = useCallback(() => {
    if (history.length === 0) return;
    const prev = history.slice(0, -1);
    let s = initState(match.best_of), e = initEwma(), f = initFatigue();
    for (const entry of prev) { const ps = s; s = advancePoint(s, entry.winner); e = updateEwma(e, entry.winner); f = updateFatigue(f, s, ps); }
    setHistory(prev); setState(s); setEwma(e); setFatigue(f);
  }, [history, match.best_of]);

  const reset = useCallback(() => {
    setState(initState(match.best_of)); setHistory([]); setEwma(initEwma()); setFatigue(initFatigue());
  }, [match.best_of]);

  const p1PtsWon = history.filter(p => p.winner === 1).length;
  const p2PtsWon = history.filter(p => p.winner === 2).length;
  const p1SrvW = history.filter(p => p.server === 1 && p.winner === 1).length;
  const p1SrvT = history.filter(p => p.server === 1).length;
  const p2SrvW = history.filter(p => p.server === 2 && p.winner === 2).length;
  const p2SrvT = history.filter(p => p.server === 2).length;
  const brk1 = history.filter(p => p.isBreak && p.winner === 1).length;
  const brk2 = history.filter(p => p.isBreak && p.winner === 2).length;

  const probHist = useMemo(() => {
    const pts: { id: number; p: number; e1: number; e2: number }[] = [{ id: 0, p: match.p1_win_prob, e1: 0.5, e2: 0.5 }];
    history.forEach(h => pts.push({ id: h.id, p: h.p1WinProb, e1: h.ewma1, e2: h.ewma2 }));
    return pts.slice(-40);
  }, [history, match.p1_win_prob]);

  const mom1 = ewmaMomentum(ewma, 1), mom2 = ewmaMomentum(ewma, 2);
  const fat1 = fatigueIndex(fatigue, 1), fat2 = fatigueIndex(fatigue, 2);
  const fl1 = fatigueLabel(fat1), fl2 = fatigueLabel(fat2);
  const odds1 = probToOdds(currentP1WinProb), odds2 = probToOdds(1 - currentP1WinProb);
  const entrySigs = signals.filter(s => s.type === "ENTRY");
  const exitSigs = signals.filter(s => s.type === "EXIT");
  const hedgeSigs = signals.filter(s => s.type === "HEDGE");

  const p1Short = match.player1.split(" ").pop()!;
  const p2Short = match.player2.split(" ").pop()!;

  return (
    <div className="p-3 space-y-2 text-[11px]">
      {/* Header */}
      <div className="text-center">
        <div className="text-xs font-bold text-terminal-cyan mb-1">🎾 POINT TRACKER</div>
        <div className="text-slate-200 font-medium">{match.player1} vs {match.player2}</div>
        <div className="text-[10px] text-terminal-muted">{match.tournament} · {match.surface} · Bo{match.best_of}</div>
      </div>

      {/* Mode toggle + sync status */}
      <div className="flex items-center justify-between px-2 py-1.5 border border-terminal-border rounded bg-terminal-panel/40">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setMode(mode === "auto" ? "manual" : "auto")}
            className={`flex items-center gap-1 px-2 py-0.5 rounded text-[9px] font-bold transition ${
              mode === "auto"
                ? "bg-terminal-red/20 text-terminal-red border border-terminal-red/40"
                : "bg-terminal-blue/20 text-terminal-blue border border-terminal-blue/40"
            }`}
          >
            {mode === "auto" ? "🔴 LIVE" : "✏️ MANUAL"}
          </button>
          {mode === "auto" && (
            <span className="text-[8px] text-terminal-muted">
              {syncing ? "⟳ syncing…" : `Last: ${Math.round((Date.now() - lastRefresh) / 1000)}s ago`}
            </span>
          )}
        </div>
        {mode === "auto" && liveMatch.liveScore?.statusDetail && (
          <span className="text-[9px] text-terminal-yellow font-mono">{liveMatch.liveScore.statusDetail}</span>
        )}
      </div>

      {/* Scoreboard */}
      <div className="border border-terminal-border rounded overflow-hidden">
        <div className="bg-terminal-panel/80 px-3 py-2">
          <div className="grid grid-cols-[1fr_repeat(5,28px)_40px] gap-1 items-center text-[11px] font-mono">
            <div />
            {[0,1,2,3,4].map(i => (
              <div key={i} className="text-center text-[8px] text-terminal-muted">
                {i < activeState.sets.length ? `S${i+1}` : i === activeState.sets.length ? "S*" : ""}
              </div>
            ))}
            <div className="text-center text-[8px] text-terminal-muted">PTS</div>
          </div>
          {([1,2] as const).map(pl => {
            const name = pl === 1 ? p1Short : p2Short;
            return (
              <div key={pl} className="grid grid-cols-[1fr_repeat(5,28px)_40px] gap-1 items-center text-[11px] font-mono mt-0.5">
                <div className="flex items-center gap-1 truncate">
                  {activeState.server === pl && <span className="text-terminal-yellow text-[8px]">●</span>}
                  <span className={`truncate ${activeState.winner === pl ? "text-terminal-green font-bold" : "text-slate-200"}`}>{name}</span>
                </div>
                {[0,1,2,3,4].map(i => (
                  <div key={i} className="text-center font-bold">
                    {i < activeState.sets.length
                      ? <span className={(pl === 1 ? activeState.sets[i].p1 > activeState.sets[i].p2 : activeState.sets[i].p2 > activeState.sets[i].p1) ? "text-terminal-green" : "text-slate-400"}>
                          {pl === 1 ? activeState.sets[i].p1 : activeState.sets[i].p2}
                        </span>
                      : i === activeState.sets.length ? <span className="text-slate-200">{pl === 1 ? activeState.currentSet.p1 : activeState.currentSet.p2}</span> : ""}
                  </div>
                ))}
                <div className="text-center text-terminal-yellow font-bold">
                  {activeState.tiebreak ? (pl === 1 ? activeState.currentGame.p1 : activeState.currentGame.p2) :
                    ((pl === 1 ? activeState.currentGame.p1 : activeState.currentGame.p2) <= 3
                      ? PT_LABELS[pl === 1 ? activeState.currentGame.p1 : activeState.currentGame.p2]
                      : "AD")}
                </div>
              </div>
            );
          })}
        </div>
        <div className="px-3 py-1 border-t border-terminal-border bg-terminal-bg text-[9px] text-terminal-muted text-center">
          {activeState.matchOver
            ? <span className="text-terminal-green font-bold">MATCH OVER — {activeState.winner === 1 ? match.player1 : match.player2} wins!</span>
            : <>{activeState.tiebreak ? "TIEBREAK · " : ""}{activeState.server === 1 ? match.player1 : match.player2} serving</>}
        </div>
      </div>

      {/* Point buttons — manual mode only */}
      {mode === "manual" && !activeState.matchOver && (
        <div className="grid grid-cols-2 gap-2">
          <button onClick={() => logPoint(1)} className="py-2 rounded border border-terminal-green/40 bg-terminal-green/10 hover:bg-terminal-green/20 text-terminal-green font-bold text-[11px] transition active:scale-95">
            {activeState.server === 1 ? "🟢" : "🔴"} {p1Short} wins pt
          </button>
          <button onClick={() => logPoint(2)} className="py-2 rounded border border-terminal-cyan/40 bg-terminal-cyan/10 hover:bg-terminal-cyan/20 text-terminal-cyan font-bold text-[11px] transition active:scale-95">
            {activeState.server === 2 ? "🟢" : "🔴"} {p2Short} wins pt
          </button>
        </div>
      )}

      {/* Undo/Reset — manual mode only */}
      {mode === "manual" && (
        <div className="flex gap-2">
          <button onClick={undo} disabled={history.length === 0} className="flex-1 text-[9px] py-1 rounded border border-terminal-border text-terminal-muted hover:text-slate-300 disabled:opacity-30">↩ Undo</button>
          <button onClick={reset} disabled={history.length === 0} className="flex-1 text-[9px] py-1 rounded border border-terminal-border text-terminal-muted hover:text-slate-300 disabled:opacity-30">⟲ Reset</button>
        </div>
      )}

      {/* Sub-tabs */}
      <div className="flex border border-terminal-border rounded overflow-hidden">
        {(["live","signals","hedge"] as const).map(t => (
          <button key={t} onClick={() => setSubTab(t)}
            className={`flex-1 text-[9px] font-bold uppercase py-1.5 transition ${
              subTab === t
                ? t === "live" ? "bg-terminal-cyan/10 text-terminal-cyan border-b-2 border-terminal-cyan"
                  : t === "signals" ? "bg-terminal-yellow/10 text-terminal-yellow border-b-2 border-terminal-yellow"
                  : "bg-terminal-green/10 text-terminal-green border-b-2 border-terminal-green"
                : "text-terminal-muted hover:text-slate-300"
            }`}>
            {t === "live" ? "📊 Live" : t === "signals" ? `⚡ Signals${entrySigs.length ? ` (${entrySigs.length})` : ""}` : "🛡 Hedge"}
          </button>
        ))}
      </div>

      {/* ═══ LIVE TAB ═══ */}
      {subTab === "live" && (
        <div className="space-y-2">
          {/* Win probability bar */}
          <Sec title="LIVE WIN PROBABILITY">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between"><span className={currentP1WinProb >= 0.5 ? "text-terminal-green font-bold" : "text-slate-300"}>{match.player1}</span><span className={currentP1WinProb >= 0.5 ? "text-terminal-green font-bold" : "text-slate-400"}>{pct(currentP1WinProb)}</span></div>
              <div className="h-3 bg-terminal-border rounded-full overflow-hidden flex">
                <div className="h-full bg-terminal-green transition-all duration-300" style={{ width: pct(currentP1WinProb) }} />
                <div className="h-full bg-terminal-cyan transition-all duration-300" style={{ width: pct(1 - currentP1WinProb) }} />
              </div>
              <div className="flex items-center justify-between"><span className={currentP1WinProb < 0.5 ? "text-terminal-cyan font-bold" : "text-slate-300"}>{match.player2}</span><span className={currentP1WinProb < 0.5 ? "text-terminal-cyan font-bold" : "text-slate-400"}>{pct(1-currentP1WinProb)}</span></div>
            </div>
            <div className="text-[9px] text-terminal-muted text-center mt-1">Fair: {odds1.toFixed(2)} / {odds2.toFixed(2)} · Pre: {pct(match.p1_win_prob)} / {pct(match.p2_win_prob)}</div>
          </Sec>

          {/* EWMA Momentum */}
          <Sec title="EWMA MOMENTUM (α=0.15/0.05)">
            <div className="grid grid-cols-3 gap-1 text-center text-[10px]">
              <div>
                <div className={`text-[14px] font-bold ${mom1 > 0.03 ? "text-terminal-green" : mom1 < -0.03 ? "text-terminal-red" : "text-slate-400"}`}>{mom1 > 0 ? "+" : ""}{(mom1*100).toFixed(1)}%</div>
                <div className="text-[8px] text-terminal-muted">{p1Short}</div>
                <div className="text-[8px] text-terminal-muted">Rate: {(ewma.fast1*100).toFixed(0)}%</div>
              </div>
              <div>
                <div className="flex justify-center gap-0.5 mt-1">{history.slice(-8).map((p,i) => <div key={i} className={`w-2 h-2 rounded-full ${p.winner === 1 ? "bg-terminal-green" : "bg-terminal-cyan"}`} />)}</div>
                <div className="text-[7px] text-terminal-muted mt-1">Last 8</div>
              </div>
              <div>
                <div className={`text-[14px] font-bold ${mom2 > 0.03 ? "text-terminal-cyan" : mom2 < -0.03 ? "text-terminal-red" : "text-slate-400"}`}>{mom2 > 0 ? "+" : ""}{(mom2*100).toFixed(1)}%</div>
                <div className="text-[8px] text-terminal-muted">{p2Short}</div>
                <div className="text-[8px] text-terminal-muted">Rate: {(ewma.fast2*100).toFixed(0)}%</div>
              </div>
            </div>
            <div className="mt-2 h-2 bg-terminal-border rounded-full overflow-hidden flex">
              <div className="h-full bg-terminal-green transition-all duration-500" style={{ width: `${ewma.fast1*100}%` }} />
              <div className="h-full bg-terminal-cyan transition-all duration-500" style={{ width: `${ewma.fast2*100}%` }} />
            </div>
          </Sec>

          {/* Fatigue */}
          <Sec title="FATIGUE INDEX">
            <div className="grid grid-cols-2 gap-3">
              {[{n:p1Short,f:fat1,l:fl1,k:1},{n:p2Short,f:fat2,l:fl2,k:2}].map(({n,f,l,k}) => (
                <div key={k} className="text-center">
                  <div className="text-[8px] text-terminal-muted mb-1">{n}</div>
                  <div className="relative h-2 bg-terminal-border rounded-full overflow-hidden">
                    <div className={`h-full transition-all duration-500 ${f < 0.35 ? "bg-terminal-green" : f < 0.6 ? "bg-terminal-yellow" : "bg-terminal-red"}`} style={{ width: `${f*100}%` }} />
                  </div>
                  <div className={`text-[9px] font-bold mt-0.5 ${l.color}`}>{l.label}</div>
                  <div className="text-[8px] text-terminal-muted">{(f*100).toFixed(0)}%</div>
                </div>
              ))}
            </div>
            <div className="text-[8px] text-terminal-muted text-center mt-1">{fatigue.totalPoints} pts · {fatigue.deuceGames} deuce · {fatigue.tiebreaks} TB</div>
          </Sec>

          {/* Break opportunity */}
          {!activeState.tiebreak && !activeState.matchOver && (
            <Sec title="BREAK OPPORTUNITY">
              <div className="text-center">
                <div className={`text-[18px] font-bold ${currentBreakOpp > 0.50 ? "text-terminal-red animate-pulse" : currentBreakOpp > 0.35 ? "text-terminal-yellow" : "text-terminal-muted"}`}>{(currentBreakOpp*100).toFixed(0)}%</div>
                <div className="text-[8px] text-terminal-muted">{activeState.server === 1 ? p2Short : p1Short} break chance</div>
                <div className="h-2 mt-1 bg-terminal-border rounded-full overflow-hidden">
                  <div className={`h-full transition-all duration-300 ${currentBreakOpp > 0.50 ? "bg-terminal-red" : currentBreakOpp > 0.35 ? "bg-terminal-yellow" : "bg-terminal-blue"}`} style={{ width: `${currentBreakOpp*100}%` }} />
                </div>
                <div className="flex justify-between text-[7px] text-terminal-muted mt-0.5"><span>Low</span><span>EWMA + Fatigue + Pressure</span><span>High</span></div>
              </div>
            </Sec>
          )}

          {/* Dual chart */}
          {probHist.length > 1 && (
            <Sec title="PROBABILITY + EWMA TIMELINE">
              <DualChart data={probHist} p1Name={match.player1} />
            </Sec>
          )}

          {/* Game Log — live Markov probabilities per game */}
          {gameLog.length > 1 && mode === "auto" && (
            <Sec title={`GAME LOG — MARKOV PROBABILITY (${gameLog.length})`}>
              <div className="max-h-[160px] overflow-y-auto space-y-0.5">
                {gameLog.slice().reverse().map(g => {
                  const probColor = g.p1WinProb >= 0.6 ? "text-terminal-green" : g.p1WinProb <= 0.4 ? "text-terminal-cyan" : "text-slate-400";
                  return (
                    <div key={g.id} className={`flex items-center gap-1.5 text-[9px] py-0.5 px-1 rounded ${g.isTiebreak ? "bg-terminal-yellow/10" : ""}`}>
                      <span className="text-terminal-muted w-[40px] shrink-0 font-mono">{g.label.replace(/^Set \d+: /, "")}</span>
                      <span className="text-[7px] text-terminal-muted shrink-0">S{g.setNum}</span>
                      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${g.server === 1 ? "bg-terminal-green" : "bg-terminal-cyan"}`} />
                      <span className="text-[8px] text-terminal-muted">{g.server === 1 ? p1Short : p2Short} srv</span>
                      {g.isTiebreak && <span className="text-terminal-yellow text-[7px] font-bold">TB</span>}
                      <span className={`font-mono font-bold ml-auto shrink-0 ${probColor}`}>{pct(g.p1WinProb)}</span>
                    </div>
                  );
                })}
              </div>
              {/* Mini probability chart from game log */}
              {gameLog.length > 2 && (
                <div className="mt-2">
                  <GameLogChart data={gameLog} p1Name={p1Short} />
                </div>
              )}
            </Sec>
          )}

          {/* Stats */}
          {history.length > 0 && (
            <Sec title="MATCH STATISTICS">
              <div className="grid grid-cols-3 gap-y-1 text-[10px] text-center">
                <div className="text-terminal-green font-bold">{p1PtsWon}</div><div className="text-terminal-muted">Points Won</div><div className="text-terminal-cyan font-bold">{p2PtsWon}</div>
                <div className="text-terminal-green font-bold">{p1SrvT > 0 ? `${Math.round(p1SrvW/p1SrvT*100)}%` : "—"}</div><div className="text-terminal-muted">Serve Pts Won</div><div className="text-terminal-cyan font-bold">{p2SrvT > 0 ? `${Math.round(p2SrvW/p2SrvT*100)}%` : "—"}</div>
                <div className="text-terminal-green font-bold">{brk1}</div><div className="text-terminal-muted">Breaks</div><div className="text-terminal-cyan font-bold">{brk2}</div>
              </div>
            </Sec>
          )}

          {/* Point log */}
          {history.length > 0 && (
            <Sec title={`POINT LOG (${history.length})`}>
              <div className="max-h-[120px] overflow-y-auto space-y-0.5">
                {history.slice(-12).reverse().map(p => (
                  <div key={p.id} className={`flex items-center gap-1.5 text-[9px] py-0.5 px-1 rounded ${p.isBreak ? "bg-terminal-red/10" : p.isMatchPoint ? "bg-terminal-yellow/10" : ""}`}>
                    <span className="text-terminal-muted w-[16px] shrink-0">#{p.id}</span>
                    <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${p.winner === 1 ? "bg-terminal-green" : "bg-terminal-cyan"}`} />
                    <span className={`font-medium ${p.winner === 1 ? "text-terminal-green" : "text-terminal-cyan"}`}>{p.winner === 1 ? p1Short : p2Short}</span>
                    {p.server !== p.winner && !p.isBreak && <span className="text-terminal-red text-[7px]">RET</span>}
                    {p.isBreak && <span className="text-terminal-red text-[7px] font-bold">BREAK!</span>}
                    {p.isMatchPoint && <span className="text-terminal-yellow text-[7px] font-bold">MATCH!</span>}
                    <span className="text-terminal-muted ml-auto shrink-0 font-mono">{pct(p.p1WinProb)}</span>
                  </div>
                ))}
              </div>
            </Sec>
          )}
        </div>
      )}

      {/* ═══ SIGNALS TAB ═══ */}
      {subTab === "signals" && (
        <div className="space-y-2">
          <Sec title="POSITION SIGNALS">
            {signals.length === 0
              ? <div className="text-center text-terminal-muted text-[10px] py-3">{history.length < 5 ? "Log more points to generate signals…" : "No actionable signals"}</div>
              : <div className="space-y-1.5">{signals.map((s,i) => <SigCard key={i} sig={s} p1={match.player1} p2={match.player2} />)}</div>
            }
          </Sec>

          <Sec title="SIGNAL RULES">
            <div className="space-y-1 text-[9px]">
              <RuleRow icon="📈" lv="MATCH" rule="ENTRY when P shifts >8% + EWMA confirms" />
              <RuleRow icon="📉" lv="MATCH" rule="EXIT when momentum reverses (EWMA < -8%)" />
              <RuleRow icon="🎯" lv="SET" rule="ENTRY on break — back the breaker" />
              <RuleRow icon="↔️" lv="SET" rule="EXIT on break-back — reduce exposure" />
              <RuleRow icon="🔴" lv="GAME" rule="ENTRY when break opp >35% (EWMA+fatigue)" />
              <RuleRow icon="🟢" lv="GAME" rule="EXIT when server stabilizes, hold >85%" />
              <RuleRow icon="🛡" lv="MATCH" rule="HEDGE when σ >6% — delta neutral" />
            </div>
          </Sec>

          <Sec title="POSITION SUMMARY">
            <div className="grid grid-cols-3 gap-2 text-center text-[10px]">
              <div className="border border-terminal-green/30 rounded p-1.5"><div className="text-[14px] font-bold text-terminal-green">{entrySigs.length}</div><div className="text-[8px] text-terminal-muted">ENTRY</div></div>
              <div className="border border-terminal-red/30 rounded p-1.5"><div className="text-[14px] font-bold text-terminal-red">{exitSigs.length}</div><div className="text-[8px] text-terminal-muted">EXIT</div></div>
              <div className="border border-terminal-yellow/30 rounded p-1.5"><div className="text-[14px] font-bold text-terminal-yellow">{hedgeSigs.length}</div><div className="text-[8px] text-terminal-muted">HEDGE</div></div>
            </div>
          </Sec>

          {entrySigs.length > 0 && (
            <Sec title="STOP-LOSS LEVELS">
              <div className="space-y-1">
                {entrySigs.filter(s => s.stopLoss).map((s,i) => (
                  <div key={i} className="flex items-center gap-2 text-[9px] py-0.5">
                    <span className={`px-1 rounded text-[7px] font-bold ${s.level === "MATCH" ? "bg-terminal-blue/20 text-terminal-blue" : s.level === "SET" ? "bg-terminal-cyan/20 text-terminal-cyan" : "bg-terminal-yellow/20 text-terminal-yellow"}`}>{s.level}</span>
                    <span className="text-slate-300">{s.side === 1 ? p1Short : p2Short}</span>
                    <span className="text-terminal-muted">→</span>
                    <span className="text-terminal-red font-mono">Stop: {pct(s.stopLoss!)}</span>
                    {s.target && <span className="text-terminal-green font-mono ml-auto">TP: {pct(s.target)}</span>}
                  </div>
                ))}
              </div>
            </Sec>
          )}
        </div>
      )}

      {/* ═══ HEDGE TAB ═══ */}
      {subTab === "hedge" && (
        <div className="space-y-2">
          <Sec title="DELTA NEUTRAL HEDGE (Account B)">
            <div className="text-[9px] text-terminal-muted mb-2 text-center">Opposing position on 2nd account to neutralize risk</div>

            <div className="space-y-2 mb-3">
              <div className="text-[9px] font-bold text-terminal-yellow uppercase">Original Position (Account A)</div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[8px] text-terminal-muted block mb-0.5">Backed</label>
                  <select value={hedgeSide} onChange={e => setHedgeSide(parseInt(e.target.value) as 1|2)}
                    className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[10px] text-slate-200 outline-none">
                    <option value={1}>{match.player1}</option>
                    <option value={2}>{match.player2}</option>
                  </select>
                </div>
                <div>
                  <label className="text-[8px] text-terminal-muted block mb-0.5">Stake ($)</label>
                  <input type="number" value={hedgeStake} onChange={e => setHedgeStake(parseFloat(e.target.value) || 0)}
                    className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[10px] text-slate-200 outline-none" />
                </div>
              </div>
              <div>
                <label className="text-[8px] text-terminal-muted block mb-0.5">Entry Odds (decimal)</label>
                <input type="number" step="0.01" value={hedgeOdds} onChange={e => setHedgeOdds(parseFloat(e.target.value) || 1.01)}
                  className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[10px] text-slate-200 outline-none" />
              </div>
            </div>

            <div className="text-[9px] font-bold text-terminal-green uppercase mb-1">Hedge (Account B)</div>
            <div className={`p-2 rounded border ${hedge.netPosition >= 0 ? "border-terminal-green/40 bg-terminal-green/5" : "border-terminal-red/40 bg-terminal-red/5"}`}>
              <div className="grid grid-cols-2 gap-y-1.5 text-[10px]">
                <div className="text-terminal-muted">Action</div><div className="text-slate-200 font-bold">BACK {hedge.player}</div>
                <div className="text-terminal-muted">Hedge Stake</div><div className="text-terminal-yellow font-bold font-mono">${hedge.hedgeStake.toFixed(2)}</div>
                <div className="text-terminal-muted">Fair Odds</div><div className="text-slate-200 font-mono">{hedge.hedgeOdds.toFixed(2)}</div>
              </div>
            </div>

            <div className="mt-2 space-y-1">
              <div className="text-[9px] font-bold text-terminal-muted uppercase">P&L Scenarios</div>
              <div className={`flex items-center justify-between text-[10px] p-1.5 rounded ${hedge.profitIfWin >= 0 ? "bg-terminal-green/5" : "bg-terminal-red/5"}`}>
                <span className="text-terminal-muted">{hedgeSide === 1 ? match.player1 : match.player2} wins</span>
                <span className={`font-mono font-bold ${hedge.profitIfWin >= 0 ? "text-terminal-green" : "text-terminal-red"}`}>{hedge.profitIfWin >= 0 ? "+" : ""}{hedge.profitIfWin.toFixed(2)}</span>
              </div>
              <div className={`flex items-center justify-between text-[10px] p-1.5 rounded ${hedge.profitIfLose >= 0 ? "bg-terminal-green/5" : "bg-terminal-red/5"}`}>
                <span className="text-terminal-muted">{hedgeSide === 1 ? match.player2 : match.player1} wins</span>
                <span className={`font-mono font-bold ${hedge.profitIfLose >= 0 ? "text-terminal-green" : "text-terminal-red"}`}>{hedge.profitIfLose >= 0 ? "+" : ""}{hedge.profitIfLose.toFixed(2)}</span>
              </div>
              <div className={`flex items-center justify-between text-[10px] p-2 rounded border ${hedge.netPosition >= 0 ? "border-terminal-green/40 bg-terminal-green/10" : "border-terminal-red/40 bg-terminal-red/10"}`}>
                <span className="font-bold text-slate-200">Net Position</span>
                <span className={`font-mono font-bold text-[12px] ${hedge.netPosition >= 0 ? "text-terminal-green" : "text-terminal-red"}`}>{hedge.netPosition >= 0 ? "+" : ""}{hedge.netPosition.toFixed(2)}</span>
              </div>
            </div>

            <div className="mt-2 text-[9px] text-terminal-muted border-t border-terminal-border pt-2">
              <div className="font-bold text-terminal-yellow mb-1">WHEN TO HEDGE (Account B):</div>
              <ul className="space-y-0.5">
                <li>• EWMA momentum reverses against your position</li>
                <li>• Opponent breaks back → breaks trading</li>
                <li>• Probability shifts &gt;12% against entry</li>
                <li>• Server fatigue &gt;60% on backed player&apos;s serve</li>
                <li>• Volatility σ &gt;6% — choppy market</li>
              </ul>
            </div>
          </Sec>

          {hedgeSigs.length > 0 && (
            <Sec title="⚠️ ACTIVE HEDGE SIGNALS">
              {hedgeSigs.map((s,i) => <SigCard key={i} sig={s} p1={match.player1} p2={match.player2} />)}
            </Sec>
          )}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Sub-components
   ═══════════════════════════════════════════════════════════════════════════ */

function SigCard({ sig, p1, p2 }: { sig: PositionSignal; p1: string; p2: string }) {
  const bg = sig.type === "ENTRY" ? "bg-terminal-green/10 border-terminal-green/30" : sig.type === "EXIT" ? "bg-terminal-red/10 border-terminal-red/30" : sig.type === "HEDGE" ? "bg-terminal-yellow/10 border-terminal-yellow/30" : "bg-terminal-panel/30 border-terminal-border";
  const tc = sig.type === "ENTRY" ? "text-terminal-green" : sig.type === "EXIT" ? "text-terminal-red" : sig.type === "HEDGE" ? "text-terminal-yellow" : "text-terminal-muted";
  const lc = sig.level === "MATCH" ? "bg-terminal-blue/20 text-terminal-blue" : sig.level === "SET" ? "bg-terminal-cyan/20 text-terminal-cyan" : "bg-terminal-yellow/20 text-terminal-yellow";
  const icon = sig.strength === "STRONG" ? "🔥" : sig.strength === "MODERATE" ? "⚡" : "—";
  return (
    <div className={`p-2 rounded border ${bg}`}>
      <div className="flex items-center gap-1.5 mb-0.5">
        <span className={`text-[7px] px-1 rounded font-bold ${lc}`}>{sig.level}</span>
        <span className={`text-[10px] font-bold ${tc}`}>{icon} {sig.type}</span>
        {sig.side && <span className="text-[9px] text-slate-300 ml-auto">{sig.side === 1 ? p1.split(" ").pop() : p2.split(" ").pop()}</span>}
      </div>
      <div className="text-[9px] text-terminal-muted">{sig.reason}</div>
      {sig.edgePct !== 0 && (
        <div className={`text-[8px] mt-0.5 ${sig.edgePct > 0 ? "text-terminal-green" : "text-terminal-red"}`}>
          Edge: {sig.edgePct > 0 ? "+" : ""}{sig.edgePct.toFixed(1)}%{sig.stopLoss ? ` · Stop: ${pct(sig.stopLoss)}` : ""}{sig.target ? ` · TP: ${pct(sig.target)}` : ""}
        </div>
      )}
    </div>
  );
}

function RuleRow({ icon, lv, rule }: { icon: string; lv: string; rule: string }) {
  return (
    <div className="flex items-start gap-1.5">
      <span className="shrink-0">{icon}</span>
      <span className="text-terminal-cyan font-bold shrink-0">{lv}</span>
      <span className="text-terminal-muted">{rule}</span>
    </div>
  );
}

function DualChart({ data, p1Name }: { data: { id: number; p: number; e1: number; e2: number }[]; p1Name: string }) {
  const h = 60, w = 260, px = 2, py = 4, iw = w - px * 2, ih = h - py * 2;
  const mk = (fn: (d: typeof data[0]) => number) =>
    data.map((d, i) => `${px + (i / Math.max(data.length - 1, 1)) * iw},${py + (1 - fn(d)) * ih}`);
  const pp = mk(d => d.p), e1p = mk(d => d.e1), e2p = mk(d => d.e2);
  const lastP = data[data.length - 1]?.p ?? 0.5;
  return (
    <div className="relative">
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[60px]">
        <line x1={px} y1={h/2} x2={w-px} y2={h/2} stroke="rgba(255,255,255,0.08)" strokeDasharray="2,2" />
        <polyline fill="none" stroke="#4ade80" strokeWidth="0.7" strokeDasharray="2,1" points={e1p.join(" ")} opacity={0.5} />
        <polyline fill="none" stroke="#22d3ee" strokeWidth="0.7" strokeDasharray="2,1" points={e2p.join(" ")} opacity={0.5} />
        <polyline fill="none" stroke={lastP >= 0.5 ? "#4ade80" : "#22d3ee"} strokeWidth="1.8" points={pp.join(" ")} />
        <circle cx={parseFloat(pp[pp.length-1]?.split(",")[0]||"0")} cy={parseFloat(pp[pp.length-1]?.split(",")[1]||"30")} r="3" fill={lastP >= 0.5 ? "#4ade80" : "#22d3ee"} />
      </svg>
      <div className="flex justify-between text-[8px] text-terminal-muted mt-0.5">
        <span>—— Prob · - - EWMA</span>
        <span>{pct(lastP)} {p1Name.split(" ").pop()}</span>
      </div>
    </div>
  );
}

function GameLogChart({ data, p1Name }: { data: GameLogEntry[]; p1Name: string }) {
  const h = 50, w = 260, px = 2, py = 4, iw = w - px * 2, ih = h - py * 2;
  const pts = data.map((d, i) => `${px + (i / Math.max(data.length - 1, 1)) * iw},${py + (1 - d.p1WinProb) * ih}`);
  const lastP = data[data.length - 1]?.p1WinProb ?? 0.5;

  // Find set boundaries for vertical dividers
  const setBounds: number[] = [];
  for (let i = 1; i < data.length; i++) {
    if (data[i].setNum !== data[i - 1].setNum) {
      setBounds.push(px + (i / Math.max(data.length - 1, 1)) * iw);
    }
  }

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[50px]">
        <line x1={px} y1={h/2} x2={w-px} y2={h/2} stroke="rgba(255,255,255,0.08)" strokeDasharray="2,2" />
        {setBounds.map((x, i) => (
          <line key={i} x1={x} y1={py} x2={x} y2={h-py} stroke="rgba(255,255,255,0.15)" strokeDasharray="1,2" />
        ))}
        <polyline fill="none" stroke={lastP >= 0.5 ? "#4ade80" : "#22d3ee"} strokeWidth="1.5" points={pts.join(" ")} />
        <circle cx={parseFloat(pts[pts.length-1]?.split(",")[0]||"0")} cy={parseFloat(pts[pts.length-1]?.split(",")[1]||"25")} r="2.5" fill={lastP >= 0.5 ? "#4ade80" : "#22d3ee"} />
      </svg>
      <div className="flex justify-between text-[7px] text-terminal-muted mt-0.5">
        <span>Game-by-game Markov P</span>
        <span>{pct(lastP)} {p1Name}</span>
      </div>
    </div>
  );
}

function Sec({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border border-terminal-border rounded overflow-hidden">
      <div className="px-2 py-1 bg-terminal-panel/50 border-b border-terminal-border">
        <span className="text-[9px] font-bold text-terminal-muted uppercase tracking-wider">{title}</span>
      </div>
      <div className="p-2">{children}</div>
    </div>
  );
}

function pct(v: number): string { return `${Math.round(v * 100)}%`; }
