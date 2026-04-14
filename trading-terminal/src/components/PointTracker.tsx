"use client";

import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import type { ScheduledMatch, LiveMatchStats } from "@/lib/scheduleService";
import { probToOdds, kellyFraction, fetchLiveScore, fetchSofaStats } from "@/lib/scheduleService";

/* ═══════════════════════════════════════════════════════════════════════════════
   Tennis scoring engine + Markov match-win probability
   ═══════════════════════════════════════════════════════════════════════════ */

const PT_LABELS = ["0", "15", "30", "40", "AD"];
const PT_MAP: Record<string, number> = { "0": 0, "15": 1, "30": 2, "40": 3, "A": 4 };

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

/** Build EWMA state from live stats — derive from total points won ratio */
function ewmaFromStats(stats: LiveMatchStats): EwmaState {
  const tot = (stats.p1_totalPointsWon + stats.p2_totalPointsWon) || 1;
  const p1Rate = stats.p1_totalPointsWon / tot;
  const p2Rate = stats.p2_totalPointsWon / tot;
  // Use 1st serve % as momentum proxy: high 1st serve % = strong serving momentum
  const srvMom1 = Math.min(1, (stats.p1_firstServePercent || 50) / 100);
  const srvMom2 = Math.min(1, (stats.p2_firstServePercent || 50) / 100);
  return {
    fast1: p1Rate * 0.6 + srvMom1 * 0.4,
    slow1: p1Rate,
    fast2: p2Rate * 0.6 + srvMom2 * 0.4,
    slow2: p2Rate,
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

/** Build fatigue from live match state — estimate from total points + sets played */
function fatigueFromLive(state: MatchState, totalPtsWon: number): FatigueState {
  const totalPoints = totalPtsWon || ((state.sets.reduce((a, s) => a + s.p1 + s.p2, 0) + state.currentSet.p1 + state.currentSet.p2) * 4.5);
  const setsPlayed = state.sets.length;
  // Count tiebreaks from completed sets
  const tiebreaks = state.sets.filter(s => s.p1 + s.p2 >= 13).length;
  // Estimate deuce games as ~30% of games
  const totalGames = state.sets.reduce((a, s) => a + s.p1 + s.p2, 0) + state.currentSet.p1 + state.currentSet.p2;
  const deuceGames = Math.round(totalGames * 0.3);
  // Load estimation: 0.003 per point + 0.005 per deuce + 0.004 per TB point
  const baseLoad = totalPoints * 0.003;
  const deuceLoad = deuceGames * 0.005 * 3; // ~3 extra deuce points avg
  const tbLoad = tiebreaks * 0.004 * 10; // ~10 TB points avg
  const setRecovery = setsPlayed * 0.1;
  const load = Math.min(1, baseLoad + deuceLoad + tbLoad - setRecovery);
  return {
    totalPoints: Math.round(totalPoints),
    deuceGames,
    tiebreaks,
    setsPlayed,
    p1Load: load,
    p2Load: load,
  };
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
   Break Opportunity Predictor — Markov-based from current game score
   Uses gameWinProb() to compute EXACT break probability from the current
   point score, then adjusts for momentum / fatigue / match context.
   ═══════════════════════════════════════════════════════════════════════════ */

function breakOppProb(
  ewma: EwmaState, fatigue: FatigueState, state: MatchState,
  pServe1: number, pServe2: number,
  stats?: LiveMatchStats | null,
): number {
  if (state.tiebreak || state.matchOver) return 0;
  const server = state.server;
  const returner = server === 1 ? 2 : 1;
  const gs = state.currentGame;
  const cs = state.currentSet;

  // ── Step 1: Determine serve point win rate ──
  let srvPtWin: number;
  if (stats) {
    const srvFirstPct = server === 1 ? stats.p1_firstServePercent : stats.p2_firstServePercent;
    const srvFirstWon = server === 1 ? stats.p1_firstServeWon : stats.p2_firstServeWon;
    const srvSecondWon = server === 1 ? stats.p1_secondServeWon : stats.p2_secondServeWon;
    const firstP = (srvFirstPct || 60) / 100;
    const firstW = (srvFirstWon || 65) / 100;
    const secondW = (srvSecondWon || 45) / 100;
    srvPtWin = firstP * firstW + (1 - firstP) * secondW;
  } else {
    // Elo-derived serve point win probability
    srvPtWin = server === 1 ? pServe1 : pServe2;
  }

  // ── Step 2: Markov break probability from CURRENT game score ──
  // gameWinProb gives P(server wins game from this score)
  // So break prob = 1 - P(server holds from here)
  const srvPts = server === 1 ? gs.p1 : gs.p2;
  const retPts = server === 1 ? gs.p2 : gs.p1;
  const holdFromHere = gameWinProb(srvPtWin, srvPts, retPts, false);
  let breakProb = 1 - holdFromHere;

  // ── Step 3: Contextual adjustments ──
  // Momentum: returner on a roll → increase break chance
  const returnerMom = ewmaMomentum(ewma, returner);
  breakProb += returnerMom * 0.12;

  // Fatigue: tired server → harder to hold
  const serverFat = fatigueIndex(fatigue, server);
  breakProb += serverFat * 0.06;

  // Set pressure: server trailing → psychological pressure
  const srvGames = server === 1 ? cs.p1 : cs.p2;
  const retGames = server === 1 ? cs.p2 : cs.p1;
  if (retGames >= 5 && retGames > srvGames) {
    breakProb += 0.04; // serving to stay in set
  } else if (retGames > srvGames && retGames >= 3) {
    breakProb += 0.02; // trailing in set
  }

  // ── Step 4: Stats-based adjustments (when SofaScore data available) ──
  if (stats) {
    // Double fault pressure: high DF rate → serve is unreliable
    const dfs = server === 1 ? stats.p1_doubleFaults : stats.p2_doubleFaults;
    const totalPts = (stats.p1_totalPointsWon + stats.p2_totalPointsWon) || 1;
    const dfRate = dfs / totalPts;
    if (dfRate > 0.08) breakProb += 0.05;      // DF crisis
    else if (dfRate > 0.05) breakProb += 0.03;  // elevated DF rate

    // Second serve vulnerability: low 2nd serve win % → breaks more likely
    const srvSecondWon = server === 1 ? stats.p1_secondServeWon : stats.p2_secondServeWon;
    if ((srvSecondWon || 50) < 40) breakProb += 0.03;

    // Break point conversion history (secondary signal)
    const bpc = server === 1 ? stats.p2_breakPointsConverted : stats.p1_breakPointsConverted;
    const [bpWon, bpTotal] = (bpc || "0/0").split("/").map(Number);
    if (bpTotal >= 2) {
      const bpConvRate = bpWon / bpTotal;
      if (bpConvRate > 0.50) breakProb += 0.04;       // excellent converter
      else if (bpConvRate > 0.35) breakProb += 0.02;   // good converter
    }
  }

  return Math.max(0, Math.min(0.95, breakProb));
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Position Signals — GAME / SET / MATCH entry & exit
   Works from score state alone (auto mode) + stats bonus + manual history
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
  stats?: LiveMatchStats | null,
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

  const matchEdge = Math.abs(probShift);
  const cs = state.currentSet;
  const gs = state.currentGame;
  const p1SetsWon = state.sets.filter(s => s.p1 > s.p2).length;
  const p2SetsWon = state.sets.filter(s => s.p2 > s.p1).length;
  const totalGames = state.sets.reduce((a, s) => a + s.p1 + s.p2, 0) + cs.p1 + cs.p2;

  // ═══ MATCH LEVEL — always active from Markov probability ═══

  // 1) Probability shift — fires from score state alone (no history/stats gate)
  if (matchEdge > 0.05) {
    const fav = currentP1Prob > preMatchP1Prob ? 1 : 2;
    const scoreCtx = `${state.sets.map(s => `${s.p1}-${s.p2}`).join(" ")} ${cs.p1}-${cs.p2}`.trim();
    signals.push({
      level: "MATCH", type: "ENTRY",
      strength: matchEdge > 0.15 ? "STRONG" : matchEdge > 0.08 ? "MODERATE" : "WEAK",
      side: fav,
      reason: `Markov P shifted ${probShift > 0 ? "+" : ""}${(probShift * 100).toFixed(0)}% from pre-match [${scoreCtx}]${stats ? ` · pts ${stats.p1_totalPointsWon}-${stats.p2_totalPointsWon}` : ""}`,
      edgePct: matchEdge * 100,
      stopLoss: fav === 1 ? Math.max(0.20, currentP1Prob - 0.15) : Math.min(0.80, currentP1Prob + 0.15),
      target: fav === 1 ? Math.min(0.95, currentP1Prob + 0.10) : Math.max(0.05, currentP1Prob - 0.10),
    });
  }

  // 2) Set lead signal — one player leads in sets
  if (p1SetsWon !== p2SetsWon && state.sets.length > 0) {
    const leader = p1SetsWon > p2SetsWon ? 1 : 2;
    const setDiff = Math.abs(p1SetsWon - p2SetsWon);
    signals.push({
      level: "MATCH", type: "ENTRY",
      strength: setDiff >= 2 || (setDiff === 1 && (leader === 1 ? currentP1Prob : 1 - currentP1Prob) > 0.65) ? "STRONG" : "MODERATE",
      side: leader,
      reason: `Set lead ${p1SetsWon}-${p2SetsWon} · ${leader === 1 ? "P1" : "P2"} Markov ${Math.round((leader === 1 ? currentP1Prob : 1 - currentP1Prob) * 100)}%`,
      edgePct: setDiff * 8 + (matchEdge * 50),
    });
  }

  // 3) Match point detection
  if (p1SetsWon === state.setsToWin - 1 && cs.p1 >= 5 && cs.p1 > cs.p2) {
    signals.push({ level: "MATCH", type: "ENTRY", strength: "STRONG", side: 1,
      reason: `P1 serving for match · ${cs.p1}-${cs.p2} in set ${state.sets.length + 1}`, edgePct: 15 });
  }
  if (p2SetsWon === state.setsToWin - 1 && cs.p2 >= 5 && cs.p2 > cs.p1) {
    signals.push({ level: "MATCH", type: "ENTRY", strength: "STRONG", side: 2,
      reason: `P2 serving for match · ${cs.p1}-${cs.p2} in set ${state.sets.length + 1}`, edgePct: 15 });
  }

  // 4) Dominant probability — one player is heavy favourite from score
  if (currentP1Prob > 0.80) {
    signals.push({ level: "MATCH", type: "ENTRY", strength: "STRONG", side: 1,
      reason: `P1 dominant at ${Math.round(currentP1Prob * 100)}% — look for value on P2 if odds are long`, edgePct: (currentP1Prob - 0.5) * 60 });
  } else if (currentP1Prob < 0.20) {
    signals.push({ level: "MATCH", type: "ENTRY", strength: "STRONG", side: 2,
      reason: `P2 dominant at ${Math.round((1 - currentP1Prob) * 100)}% — look for value on P1 if odds are long`, edgePct: (0.5 - currentP1Prob) * 60 });
  }

  // 5) Momentum reversal EXIT — score moving one way but underdog is closing
  if (matchEdge > 0.05) {
    const wasFav = probShift > 0 ? 1 : 2;
    const momCheck = wasFav === 1 ? p1Mom : p2Mom;
    // In auto mode without stats, detect reversal from score: fav losing current set
    const favGames = wasFav === 1 ? cs.p1 : cs.p2;
    const oppGames = wasFav === 1 ? cs.p2 : cs.p1;
    const scorePressure = oppGames > favGames + 1;
    const reversed = stats
      ? (wasFav === 1 ? stats.p1_totalPointsWon < stats.p2_totalPointsWon : stats.p2_totalPointsWon < stats.p1_totalPointsWon)
      : (momCheck < -0.05 || scorePressure);
    if (reversed) {
      signals.push({
        level: "MATCH", type: "EXIT", strength: scorePressure ? "STRONG" : "MODERATE", side: wasFav,
        reason: stats
          ? `${wasFav === 1 ? "P1" : "P2"} trailing in total pts (${stats.p1_totalPointsWon}-${stats.p2_totalPointsWon}) despite Markov advantage`
          : `${wasFav === 1 ? "P1" : "P2"} losing current set ${cs.p1}-${cs.p2} despite match advantage`,
        edgePct: -matchEdge * 50,
      });
    }
  }

  // ═══ SET LEVEL — from score patterns ═══

  // 6) Break advantage detection — use actual net break count from serve alternation
  {
    const totalGamesInSet = cs.p1 + cs.p2;
    if (totalGamesInSet >= 2) {
      // Determine who served first in this set from current server + games played
      const firstServerInSet: 1 | 2 = totalGamesInSet % 2 === 0
        ? state.server
        : (state.server === 1 ? 2 : 1);
      // Expected game diff if all service games were held
      const p1SrvGames = firstServerInSet === 1
        ? Math.ceil(totalGamesInSet / 2)
        : Math.floor(totalGamesInSet / 2);
      const expectedDiff = p1SrvGames - (totalGamesInSet - p1SrvGames);
      const actualDiff = cs.p1 - cs.p2;
      // Net break advantage: positive = P1 has break lead
      const netBreaks = (actualDiff - expectedDiff) / 2;

      if (Math.abs(netBreaks) >= 1) {
        const leader = netBreaks > 0 ? 1 : 2;
        const breakCount = Math.abs(netBreaks);
        signals.push({
          level: "SET", type: "ENTRY",
          strength: breakCount >= 2 ? "STRONG" : "MODERATE",
          side: leader,
          reason: `${leader === 1 ? "P1" : "P2"} has ${breakCount > 1 ? breakCount + "× " : ""}break advantage at ${cs.p1}-${cs.p2} — back for set`,
          edgePct: breakCount * 7,
          stopLoss: leader === 1 ? Math.max(0.25, currentP1Prob - 0.12) : Math.min(0.75, currentP1Prob + 0.12),
        });
      }
    }
  }

  // 7) Close to winning set — 5-3, 5-4, 6-5 (server advantage)
  if ((cs.p1 >= 5 && cs.p1 > cs.p2 && cs.p2 < 6) || (cs.p2 >= 5 && cs.p2 > cs.p1 && cs.p1 < 6)) {
    const setLeader = cs.p1 > cs.p2 ? 1 : 2;
    const isServing = state.server === setLeader;
    signals.push({
      level: "SET", type: isServing ? "ENTRY" : "HOLD",
      strength: isServing ? "STRONG" : "MODERATE",
      side: setLeader,
      reason: `${setLeader === 1 ? "P1" : "P2"} ${isServing ? "serving for set" : "close to set"} at ${cs.p1}-${cs.p2}`,
      edgePct: isServing ? 10 : 5,
    });
  }

  // 8) Tiebreak — high volatility moment
  if (state.tiebreak) {
    const tbLead = gs.p1 - gs.p2;
    if (Math.abs(tbLead) >= 3) {
      const leader = tbLead > 0 ? 1 : 2;
      signals.push({ level: "SET", type: "ENTRY", strength: "STRONG", side: leader,
        reason: `TB ${gs.p1}-${gs.p2} · ${leader === 1 ? "P1" : "P2"} has mini-break advantage`, edgePct: Math.abs(tbLead) * 4 });
    } else {
      signals.push({ level: "SET", type: "HEDGE", strength: "MODERATE", side: null,
        reason: `Tiebreak ${gs.p1}-${gs.p2} — high variance, consider hedge`, edgePct: 0 });
    }
  }

  // 9) Even set — no edge, hold
  if (cs.p1 === cs.p2 && cs.p1 >= 3 && !state.tiebreak) {
    signals.push({ level: "SET", type: "HOLD", strength: "WEAK", side: null,
      reason: `Set level at ${cs.p1}-${cs.p2} — wait for break opportunity`, edgePct: 0 });
  }

  // ═══ SET LEVEL — from point history (manual mode) ═══
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

  // ═══ GAME LEVEL — from break opportunity + score context ═══

  // 10) Break point / pressure from game score (Markov-accurate)
  if (!state.tiebreak) {
    const retPts = state.server === 1 ? gs.p2 : gs.p1;
    const srvPts = state.server === 1 ? gs.p1 : gs.p2;
    const returner = state.server === 1 ? 2 : 1;
    const scoreLabel = `${PT_LABELS[Math.min(gs.p1,4)]}-${PT_LABELS[Math.min(gs.p2,4)]}`;

    if (retPts >= 3 && retPts > srvPts) {
      // Actual break point! (30-40, 40-AD)
      const convPct = Math.round(breakOpp * 100);
      signals.push({
        level: "GAME", type: "ENTRY", strength: "STRONG", side: returner,
        reason: `🔴 BREAK POINT ${scoreLabel}! Conv: ${convPct}%${stats ? ` · srv 1st%: ${state.server === 1 ? stats.p1_firstServePercent : stats.p2_firstServePercent}% · DFs: ${state.server === 1 ? stats.p1_doubleFaults : stats.p2_doubleFaults}` : ""}`,
        edgePct: Math.max(15, convPct * 0.4),
      });
    } else if (retPts >= 3 && retPts === srvPts) {
      // Deuce — next point critical
      signals.push({
        level: "GAME", type: "ENTRY", strength: "MODERATE", side: returner,
        reason: `Deuce — ${Math.round(breakOpp * 100)}% break chance · server under pressure`,
        edgePct: 8,
      });
    } else if (retPts >= 2 && srvPts <= 1 && retPts > srvPts) {
      // Building toward break (0-30, 15-30)
      signals.push({
        level: "GAME", type: "ENTRY", strength: "MODERATE", side: returner,
        reason: `Pressure ${scoreLabel} · break opp ${Math.round(breakOpp * 100)}%${srvPts === 0 && retPts >= 2 ? " · server in trouble" : ""}`,
        edgePct: Math.max(6, Math.round(breakOpp * 20)),
      });
    } else if (srvPts >= 3 && srvPts > retPts) {
      // Server has game point — break unlikely
      signals.push({
        level: "GAME", type: "EXIT", strength: "MODERATE", side: null,
        reason: `Game point ${scoreLabel} · hold likely ${Math.round((1 - breakOpp) * 100)}%`,
        edgePct: -3,
      });
    }

    // 10b) Serving to stay in set — extra pressure signal
    const srvGames = state.server === 1 ? cs.p1 : cs.p2;
    const retGames = state.server === 1 ? cs.p2 : cs.p1;
    if (retGames >= 5 && retGames > srvGames && srvGames < 6) {
      signals.push({
        level: "GAME", type: "ENTRY", strength: breakOpp > 0.30 ? "STRONG" : "MODERATE",
        side: returner,
        reason: `Server must hold to stay in set (${cs.p1}-${cs.p2}) · break ${Math.round(breakOpp * 100)}%`,
        edgePct: Math.max(8, Math.round(breakOpp * 30)),
      });
    }
  }

  // 11) Break opportunity from Markov model (score-state aware)
  if (breakOpp > 0.25 && !state.tiebreak) {
    const returner = state.server === 1 ? 2 : 1;
    // Only show if not already covered by signal #10 (avoid duplicate)
    const retPts = state.server === 1 ? gs.p2 : gs.p1;
    const srvPts = state.server === 1 ? gs.p1 : gs.p2;
    const alreadySignaled = (retPts >= 3 && retPts > srvPts) || (retPts >= 2 && srvPts <= 1 && retPts > srvPts) || (retPts >= 3 && retPts === srvPts);
    if (!alreadySignaled) {
      signals.push({
        level: "GAME", type: "ENTRY",
        strength: breakOpp > 0.45 ? "STRONG" : breakOpp > 0.35 ? "MODERATE" : "WEAK",
        side: returner,
        reason: stats
          ? `Break opp ${(breakOpp * 100).toFixed(0)}% [Markov] · DFs:${state.server === 1 ? stats.p1_doubleFaults : stats.p2_doubleFaults} 1st%:${state.server === 1 ? stats.p1_firstServePercent : stats.p2_firstServePercent}%`
          : `Break opp ${(breakOpp * 100).toFixed(0)}% · hold ${((1 - breakOpp) * 100).toFixed(0)}% [Markov${p1Fat > 0.3 || p2Fat > 0.3 ? " + fatigue" : ""}]`,
        edgePct: (breakOpp - 0.18) * 40,
      });
    }
  }

  // 12) Server stabilized EXIT
  if (breakOpp < 0.12 && gs.p1 + gs.p2 >= 2 && !state.tiebreak) {
    signals.push({
      level: "GAME", type: "EXIT", strength: "WEAK", side: null,
      reason: `Server stable — hold ${((1 - breakOpp) * 100).toFixed(0)}%, close game position`,
      edgePct: -1,
    });
  }

  // ═══ STATS-BASED SIGNALS (bonus when SofaScore is available) ═══
  if (stats) {
    const totalPts = (stats.p1_totalPointsWon + stats.p2_totalPointsWon) || 1;
    const p1PtShare = stats.p1_totalPointsWon / totalPts;

    // Serve dominance
    const srvGap = (stats.p1_firstServePercent || 60) - (stats.p2_firstServePercent || 60);
    if (Math.abs(srvGap) > 15) {
      const dominant = srvGap > 0 ? 1 : 2;
      signals.push({
        level: "MATCH", type: "ENTRY",
        strength: Math.abs(srvGap) > 25 ? "STRONG" : "MODERATE",
        side: dominant,
        reason: `Serve dom: 1st% ${stats.p1_firstServePercent}% vs ${stats.p2_firstServePercent}%`,
        edgePct: Math.abs(srvGap) * 0.3,
      });
    }

    // DF crisis
    if (stats.p1_doubleFaults >= 4 || stats.p2_doubleFaults >= 4) {
      const dfCrisis = stats.p1_doubleFaults >= stats.p2_doubleFaults ? 1 : 2;
      const dfCount = dfCrisis === 1 ? stats.p1_doubleFaults : stats.p2_doubleFaults;
      signals.push({
        level: "MATCH", type: "ENTRY", strength: dfCount >= 6 ? "STRONG" : "MODERATE",
        side: dfCrisis === 1 ? 2 : 1,
        reason: `DF crisis: ${dfCrisis === 1 ? "P1" : "P2"} has ${dfCount} DFs`, edgePct: dfCount * 1.5,
      });
    }

    // Points imbalance
    if (totalPts > 30 && Math.abs(p1PtShare - 0.5) > 0.06) {
      const dominant = p1PtShare > 0.5 ? 1 : 2;
      signals.push({
        level: "MATCH", type: "ENTRY",
        strength: Math.abs(p1PtShare - 0.5) > 0.12 ? "STRONG" : "MODERATE",
        side: dominant,
        reason: `Points: ${stats.p1_totalPointsWon}-${stats.p2_totalPointsWon} (${Math.round(p1PtShare * 100)}%-${Math.round((1 - p1PtShare) * 100)}%)`,
        edgePct: Math.abs(p1PtShare - 0.5) * 100,
      });
    }

    // Break points converted
    const [bp1W] = (stats.p1_breakPointsConverted || "0/0").split("/").map(Number);
    const [bp2W] = (stats.p2_breakPointsConverted || "0/0").split("/").map(Number);
    if (Math.abs(bp1W - bp2W) >= 1) {
      const breaker = bp1W > bp2W ? 1 : 2;
      signals.push({
        level: "SET", type: "ENTRY", strength: (breaker === 1 ? bp1W : bp2W) >= 3 ? "STRONG" : "MODERATE",
        side: breaker,
        reason: `BPs: P1 ${stats.p1_breakPointsConverted} · P2 ${stats.p2_breakPointsConverted}`,
        edgePct: Math.abs(bp1W - bp2W) * 4,
      });
    }

    // Tight match hedge
    if (totalPts > 40 && Math.abs(stats.p1_totalPointsWon - stats.p2_totalPointsWon) <= 3) {
      signals.push({
        level: "MATCH", type: "HEDGE", strength: "MODERATE", side: null,
        reason: `Tight: pts ${stats.p1_totalPointsWon}-${stats.p2_totalPointsWon} — high variance, hedge`,
        edgePct: 0,
      });
    }
  }

  // ═══ HEDGE — from score proximity ═══
  if (probVol > 0.06 && history.length >= 15) {
    signals.push({
      level: "MATCH", type: "HEDGE", strength: "MODERATE", side: null,
      reason: `High volatility σ=${(probVol * 100).toFixed(1)}% — consider hedge`,
      edgePct: 0,
    });
  }
  // Score-based hedge: match is close overall
  if (Math.abs(currentP1Prob - 0.5) < 0.08 && totalGames >= 8) {
    signals.push({
      level: "MATCH", type: "HEDGE", strength: "MODERATE", side: null,
      reason: `Match on knife-edge (${Math.round(currentP1Prob * 100)}%-${Math.round((1 - currentP1Prob) * 100)}%) — hedge recommended`,
      edgePct: 0,
    });
  }

  // ═══ ALWAYS: Current position assessment ═══
  if (signals.filter(s => s.type === "ENTRY").length === 0 && !state.tiebreak) {
    // No entry signals — show hold status
    const fav = currentP1Prob >= 0.5 ? 1 : 2;
    const favProb = Math.max(currentP1Prob, 1 - currentP1Prob);
    signals.push({
      level: "MATCH", type: "HOLD", strength: "WEAK", side: fav,
      reason: `No clear edge — ${fav === 1 ? "P1" : "P2"} at ${Math.round(favProb * 100)}%. Wait for break/score shift.`,
      edgePct: 0,
    });
  }

  return signals;
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Delta Neutral Hedge Calculator — with live P&L tracking
   ═══════════════════════════════════════════════════════════════════════════ */

interface HedgeCalc {
  action: string;
  player: string;
  hedgeStake: number;
  profitIfWin: number;
  profitIfLose: number;
  netPosition: number;
  hedgeOdds: number;
  kellyEdge: number;         // Kelly edge % vs current fair odds
  impliedProb: number;       // implied probability from entry odds
  currentFairOdds: number;   // Markov-derived fair odds for the original side
  roi: number;               // current ROI %
}

function calcDeltaNeutralHedge(
  originalSide: 1 | 2, originalStake: number, originalOdds: number,
  currentP1Prob: number, p1Name: string, p2Name: string,
  currentLiveOdds?: number, // live bookmaker odds for hedge side
): HedgeCalc {
  const currentOdds1 = probToOdds(currentP1Prob);
  const currentOdds2 = probToOdds(1 - currentP1Prob);
  const hedgeOn = originalSide === 1 ? 2 : 1;
  const hOdds = currentLiveOdds || (hedgeOn === 1 ? currentOdds1 : currentOdds2);
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

  const fairOddsForSide = originalSide === 1 ? currentOdds1 : currentOdds2;
  const impliedProb = 1 / originalOdds;
  const trueProb = originalSide === 1 ? currentP1Prob : 1 - currentP1Prob;
  const kellyEdge = kellyFraction(trueProb, originalOdds) * 100;
  // ROI = (expected profit / stake) * 100
  const expectedPL = trueProb * pWin + (1 - trueProb) * pLose;
  const roi = originalStake > 0 ? (expectedPL / originalStake) * 100 : 0;

  return {
    action: "BACK",
    player: hedgeOn === 1 ? p1Name : p2Name,
    hedgeStake: Math.round(hStake * 100) / 100,
    profitIfWin: Math.round(pWin * 100) / 100,
    profitIfLose: Math.round(pLose * 100) / 100,
    netPosition: Math.round((pWin + pLose) / 2 * 100) / 100,
    hedgeOdds: hOdds,
    kellyEdge: Math.round(kellyEdge * 100) / 100,
    impliedProb,
    currentFairOdds: fairOddsForSide,
    roi: Math.round(roi * 100) / 100,
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
  label: string;
  isBreak: boolean;
  isTiebreak: boolean;
}

function buildStateFromLiveScore(
  liveScore: NonNullable<ScheduledMatch["liveScore"]>,
  bestOf: number,
): MatchState {
  const setsToWin = bestOf === 5 ? 3 : 2;
  const sets = liveScore.completedSets.map(s => ({ ...s }));
  const currentSet = { ...liveScore.currentSetGames };
  const tiebreak = currentSet.p1 === 6 && currentSet.p2 === 6;
  const server = liveScore.server;

  // Point score from SofaScore
  let currentGame = { p1: 0, p2: 0 };
  if (liveScore.tiebreakScore && tiebreak) {
    currentGame = { ...liveScore.tiebreakScore };
  } else if (liveScore.pointScore) {
    currentGame = { p1: PT_MAP[liveScore.pointScore.p1] ?? 0, p2: PT_MAP[liveScore.pointScore.p2] ?? 0 };
  }

  const p1SetsWon = sets.filter(s => s.p1 > s.p2).length;
  const p2SetsWon = sets.filter(s => s.p2 > s.p1).length;
  const matchOver = p1SetsWon >= setsToWin || p2SetsWon >= setsToWin;
  const winner = matchOver ? (p1SetsWon >= setsToWin ? 1 : 2) : undefined;

  return { sets, currentSet, currentGame, server, tiebreak, matchOver, winner, setsToWin };
}

function buildGameLog(
  liveScore: NonNullable<ScheduledMatch["liveScore"]>,
  pS1: number, pS2: number, bestOf: number,
): GameLogEntry[] {
  const entries: GameLogEntry[] = [];
  const setsToWin = bestOf === 5 ? 3 : 2;
  let id = 0;

  const probAt = (
    completedSets: { p1: number; p2: number }[],
    curSet: { p1: number; p2: number },
    server: 1 | 2,
  ): number => {
    const tmpState: MatchState = {
      sets: completedSets, currentSet: curSet, currentGame: { p1: 0, p2: 0 },
      server, tiebreak: curSet.p1 === 6 && curSet.p2 === 6, matchOver: false, setsToWin,
    };
    return matchWinProb(pS1, pS2, tmpState);
  };

  let totalGamesPlayed = 0;
  for (const s of liveScore.completedSets) totalGamesPlayed += s.p1 + s.p2;
  totalGamesPlayed += liveScore.currentSetGames.p1 + liveScore.currentSetGames.p2;
  const startServer: 1 | 2 = (totalGamesPlayed % 2 === 0) ? liveScore.server : (liveScore.server === 1 ? 2 : 1);

  const allSets = [...liveScore.completedSets, liveScore.currentSetGames];
  const completedBefore: { p1: number; p2: number }[] = [];
  let runningServer = startServer;

  for (let si = 0; si < allSets.length; si++) {
    const setData = allSets[si];
    const isCurrentSet = si === allSets.length - 1 && si >= liveScore.completedSets.length;
    const maxGames = setData.p1 + setData.p2;

    entries.push({
      id: id++, setNum: si + 1, p1Games: 0, p2Games: 0, server: runningServer,
      p1WinProb: probAt(completedBefore, { p1: 0, p2: 0 }, runningServer),
      label: `Set ${si + 1}: 0-0`, isBreak: false, isTiebreak: false,
    });

    for (let g = 1; g <= maxGames; g++) {
      const frac = g / maxGames;
      let g1 = Math.round(frac * setData.p1);
      let g2 = Math.round(frac * setData.p2);
      g1 = Math.min(g1, setData.p1);
      g2 = Math.min(g2, setData.p2);
      if (g1 + g2 < g) {
        if (setData.p1 >= setData.p2 && g1 < setData.p1) g1++;
        else if (g2 < setData.p2) g2++;
        else g1++;
      }

      const curServer: 1 | 2 = (entries.length % 2 === 0) ? startServer : (startServer === 1 ? 2 : 1);
      const isTB = g1 === 6 && g2 === 6;

      entries.push({
        id: id++, setNum: si + 1, p1Games: g1, p2Games: g2, server: curServer,
        p1WinProb: probAt(completedBefore, { p1: g1, p2: g2 }, curServer),
        label: isTB ? `Set ${si + 1}: TB` : `Set ${si + 1}: ${g1}-${g2}`,
        isBreak: false, isTiebreak: isTB,
      });
    }

    if (!isCurrentSet) {
      completedBefore.push({ ...setData });
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
  const [hedgeLiveOdds, setHedgeLiveOdds] = useState<string>("");
  const [subTab, setSubTab] = useState<"live" | "signals" | "hedge">("live");
  const [mode, setMode] = useState<"auto" | "manual">(() => match.status === "live" && match.liveScore ? "auto" : "manual");
  const [liveMatch, setLiveMatch] = useState<ScheduledMatch>(match);
  const [lastRefresh, setLastRefresh] = useState<number>(Date.now());
  const [syncing, setSyncing] = useState(false);
  const [manualPtScore, setManualPtScore] = useState<{ p1: string; p2: string } | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auto-build state from live score when in auto mode
  const liveState = useMemo(() => {
    if (mode === "auto" && liveMatch.liveScore) {
      const ls = buildStateFromLiveScore(liveMatch.liveScore, liveMatch.best_of);
      // Apply manual point override if user set one
      if (manualPtScore) {
        ls.currentGame = {
          p1: PT_MAP[manualPtScore.p1] ?? 0,
          p2: PT_MAP[manualPtScore.p2] ?? 0,
        };
      }
      return ls;
    }
    return null;
  }, [mode, liveMatch.liveScore, liveMatch.best_of, manualPtScore]);

  // Game log for live matches
  const gameLog = useMemo(() => {
    if (liveMatch.liveScore) {
      const pS1 = Math.max(0.50, Math.min(0.75, 0.55 + (liveMatch.p1_win_prob - 0.5) * 0.3));
      const pS2 = Math.max(0.50, Math.min(0.75, 0.55 + (liveMatch.p2_win_prob - 0.5) * 0.3));
      return buildGameLog(liveMatch.liveScore, pS1, pS2, liveMatch.best_of);
    }
    return [];
  }, [liveMatch]);

  const activeState = liveState || state;

  // Live stats from SofaScore
  const liveStats = liveMatch.liveScore?.stats || null;

  // Build EWMA from real stats in auto mode
  const activeEwma = useMemo(() => {
    if (mode === "auto" && liveStats) return ewmaFromStats(liveStats);
    return ewma;
  }, [mode, liveStats, ewma]);

  // Build fatigue from live state in auto mode
  const activeFatigue = useMemo(() => {
    if (mode === "auto" && liveState && liveStats) {
      return fatigueFromLive(liveState, liveStats.p1_totalPointsWon + liveStats.p2_totalPointsWon);
    }
    if (mode === "auto" && liveState) {
      return fatigueFromLive(liveState, 0);
    }
    return fatigue;
  }, [mode, liveState, liveStats, fatigue]);

  // Poll for live score updates every 10 seconds
  useEffect(() => {
    if (mode !== "auto" || liveMatch.status !== "live") return;
    const poll = async () => {
      setSyncing(true);
      try {
        const fresh = await fetchLiveScore(liveMatch.id);
        if (fresh) {
          setLiveMatch(fresh);
          setLastRefresh(Date.now());
          // Clear manual point override when new data arrives (score changed)
          if (fresh.liveScore?.pointScore) {
            setManualPtScore(null);
          }
        }
      } catch { /* */ }
      finally { setSyncing(false); }
    };
    poll(); // immediate first poll
    pollRef.current = setInterval(poll, 10_000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [mode, liveMatch.id, liveMatch.status]);

  // When match prop changes, update liveMatch
  useEffect(() => {
    setLiveMatch(match);
    if (match.status === "live" && match.liveScore && mode === "auto") {
      setLastRefresh(Date.now());
    }
  }, [match, mode]);

  const pServe1 = useMemo(() => {
    // Use real serve stats if available
    if (liveStats && mode === "auto") {
      const firstP = (liveStats.p1_firstServePercent || 60) / 100;
      const firstW = (liveStats.p1_firstServeWon || 65) / 100;
      const secondW = (liveStats.p1_secondServeWon || 45) / 100;
      return Math.max(0.45, Math.min(0.80, firstP * firstW + (1 - firstP) * secondW));
    }
    return Math.max(0.50, Math.min(0.75, 0.55 + (match.p1_win_prob - 0.5) * 0.3));
  }, [match.p1_win_prob, liveStats, mode]);

  const pServe2 = useMemo(() => {
    if (liveStats && mode === "auto") {
      const firstP = (liveStats.p2_firstServePercent || 60) / 100;
      const firstW = (liveStats.p2_firstServeWon || 65) / 100;
      const secondW = (liveStats.p2_secondServeWon || 45) / 100;
      return Math.max(0.45, Math.min(0.80, firstP * firstW + (1 - firstP) * secondW));
    }
    return Math.max(0.50, Math.min(0.75, 0.55 + (match.p2_win_prob - 0.5) * 0.3));
  }, [match.p2_win_prob, liveStats, mode]);

  const currentP1WinProb = useMemo(() => matchWinProb(pServe1, pServe2, activeState), [pServe1, pServe2, activeState]);
  const currentBreakOpp = useMemo(() => breakOppProb(activeEwma, activeFatigue, activeState, pServe1, pServe2, liveStats), [activeEwma, activeFatigue, activeState, pServe1, pServe2, liveStats]);
  const signals = useMemo(() => computePositionSignals(currentP1WinProb, match.p1_win_prob, activeEwma, activeFatigue, activeState, history, currentBreakOpp, liveStats), [currentP1WinProb, match.p1_win_prob, activeEwma, activeFatigue, activeState, history, currentBreakOpp, liveStats]);
  const hedge = useMemo(() => {
    const liveOdds = hedgeLiveOdds ? parseFloat(hedgeLiveOdds) : undefined;
    return calcDeltaNeutralHedge(hedgeSide, hedgeStake, hedgeOdds, currentP1WinProb, match.player1, match.player2, liveOdds && liveOdds > 1 ? liveOdds : undefined);
  }, [hedgeSide, hedgeStake, hedgeOdds, hedgeLiveOdds, currentP1WinProb, match.player1, match.player2]);

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

  const mom1 = ewmaMomentum(activeEwma, 1), mom2 = ewmaMomentum(activeEwma, 2);
  const fat1 = fatigueIndex(activeFatigue, 1), fat2 = fatigueIndex(activeFatigue, 2);
  const fl1 = fatigueLabel(fat1), fl2 = fatigueLabel(fat2);
  const odds1 = probToOdds(currentP1WinProb), odds2 = probToOdds(1 - currentP1WinProb);
  const entrySigs = signals.filter(s => s.type === "ENTRY");
  const exitSigs = signals.filter(s => s.type === "EXIT");
  const hedgeSigs = signals.filter(s => s.type === "HEDGE");

  const p1Short = match.player1.split(" ").pop()!;
  const p2Short = match.player2.split(" ").pop()!;

  // Has SofaScore data?
  const hasSofa = !!(liveMatch.liveScore?.pointScore || liveMatch.liveScore?.stats);

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
              {syncing ? "⟳ syncing…" : `${Math.round((Date.now() - lastRefresh) / 1000)}s ago`}
              {hasSofa && <span className="text-terminal-green ml-1">● SOFA</span>}
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

      {/* Manual point input — available in auto mode for fine-tuning */}
      {mode === "auto" && !activeState.matchOver && (
        <div className="border border-terminal-border rounded p-2 bg-terminal-panel/30">
          <div className="text-[8px] text-terminal-muted font-bold uppercase mb-1">📝 MANUAL POINT OVERRIDE (for current game)</div>
          <div className="grid grid-cols-2 gap-2">
            {([1,2] as const).map(pl => (
              <div key={pl}>
                <div className="text-[8px] text-terminal-muted mb-0.5">{pl === 1 ? p1Short : p2Short}</div>
                <div className="flex gap-1">
                  {(activeState.tiebreak ? ["0","1","2","3","4","5","6","7","8","9"] : ["0","15","30","40","A"]).map(pt => {
                    const active = manualPtScore
                      ? (pl === 1 ? manualPtScore.p1 === pt : manualPtScore.p2 === pt)
                      : (liveMatch.liveScore?.pointScore ? (pl === 1 ? liveMatch.liveScore.pointScore.p1 === pt : liveMatch.liveScore.pointScore.p2 === pt) : pt === "0");
                    return (
                      <button key={pt} onClick={() => {
                        const current = manualPtScore || liveMatch.liveScore?.pointScore || { p1: "0", p2: "0" };
                        setManualPtScore(pl === 1 ? { p1: pt, p2: current.p2 } : { p1: current.p1, p2: pt });
                      }}
                      className={`px-1.5 py-0.5 rounded text-[8px] font-mono transition ${
                        active
                          ? "bg-terminal-yellow/30 text-terminal-yellow border border-terminal-yellow/50"
                          : "bg-terminal-bg border border-terminal-border text-terminal-muted hover:text-slate-300"
                      }`}>
                        {pt}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
          {manualPtScore && (
            <button onClick={() => setManualPtScore(null)} className="mt-1 text-[7px] text-terminal-red hover:text-terminal-red/80">
              ✕ Clear override
            </button>
          )}
        </div>
      )}

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

          {/* Live Match Stats from SofaScore */}
          {liveStats && mode === "auto" && (
            <Sec title="📡 LIVE MATCH STATISTICS">
              <div className="grid grid-cols-3 gap-y-1 text-[10px] text-center">
                <div className="text-terminal-green font-bold">{liveStats.p1_aces}</div><div className="text-terminal-muted">Aces</div><div className="text-terminal-cyan font-bold">{liveStats.p2_aces}</div>
                <div className="text-terminal-green font-bold">{liveStats.p1_doubleFaults}</div><div className="text-terminal-muted">Double Faults</div><div className="text-terminal-cyan font-bold">{liveStats.p2_doubleFaults}</div>
                <div className="text-terminal-green font-bold">{liveStats.p1_firstServePercent}%</div><div className="text-terminal-muted">1st Serve %</div><div className="text-terminal-cyan font-bold">{liveStats.p2_firstServePercent}%</div>
                <div className="text-terminal-green font-bold">{liveStats.p1_firstServeWon}%</div><div className="text-terminal-muted">1st Srv Won</div><div className="text-terminal-cyan font-bold">{liveStats.p2_firstServeWon}%</div>
                <div className="text-terminal-green font-bold">{liveStats.p1_secondServeWon}%</div><div className="text-terminal-muted">2nd Srv Won</div><div className="text-terminal-cyan font-bold">{liveStats.p2_secondServeWon}%</div>
                <div className="text-terminal-green font-bold">{liveStats.p1_breakPointsConverted}</div><div className="text-terminal-muted">Break Pts</div><div className="text-terminal-cyan font-bold">{liveStats.p2_breakPointsConverted}</div>
                <div className="text-terminal-green font-bold">{liveStats.p1_totalPointsWon}</div><div className="text-terminal-muted">Total Pts Won</div><div className="text-terminal-cyan font-bold">{liveStats.p2_totalPointsWon}</div>
              </div>
              <div className="mt-1.5 h-2 bg-terminal-border rounded-full overflow-hidden flex">
                <div className="h-full bg-terminal-green transition-all duration-300" style={{ width: `${((liveStats.p1_totalPointsWon / Math.max(1, liveStats.p1_totalPointsWon + liveStats.p2_totalPointsWon)) * 100)}%` }} />
                <div className="h-full bg-terminal-cyan transition-all duration-300" style={{ width: `${((liveStats.p2_totalPointsWon / Math.max(1, liveStats.p1_totalPointsWon + liveStats.p2_totalPointsWon)) * 100)}%` }} />
              </div>
              <div className="text-[8px] text-terminal-muted text-center mt-0.5">Points share: {p1Short} {Math.round(liveStats.p1_totalPointsWon / Math.max(1, liveStats.p1_totalPointsWon + liveStats.p2_totalPointsWon) * 100)}% — {p2Short} {Math.round(liveStats.p2_totalPointsWon / Math.max(1, liveStats.p1_totalPointsWon + liveStats.p2_totalPointsWon) * 100)}%</div>
            </Sec>
          )}

          {/* EWMA Momentum */}
          <Sec title={`EWMA MOMENTUM${liveStats ? " (from live stats)" : " (α=0.15/0.05)"}`}>
            <div className="grid grid-cols-3 gap-1 text-center text-[10px]">
              <div>
                <div className={`text-[14px] font-bold ${mom1 > 0.03 ? "text-terminal-green" : mom1 < -0.03 ? "text-terminal-red" : "text-slate-400"}`}>{mom1 > 0 ? "+" : ""}{(mom1*100).toFixed(1)}%</div>
                <div className="text-[8px] text-terminal-muted">{p1Short}</div>
                <div className="text-[8px] text-terminal-muted">Rate: {(activeEwma.fast1*100).toFixed(0)}%</div>
              </div>
              <div>
                {mode === "manual" ? (
                  <>
                    <div className="flex justify-center gap-0.5 mt-1">{history.slice(-8).map((p,i) => <div key={i} className={`w-2 h-2 rounded-full ${p.winner === 1 ? "bg-terminal-green" : "bg-terminal-cyan"}`} />)}</div>
                    <div className="text-[7px] text-terminal-muted mt-1">Last 8</div>
                  </>
                ) : (
                  <div className="text-[8px] text-terminal-muted mt-2">
                    {liveStats ? `${liveStats.p1_totalPointsWon}-${liveStats.p2_totalPointsWon} pts` : "—"}
                  </div>
                )}
              </div>
              <div>
                <div className={`text-[14px] font-bold ${mom2 > 0.03 ? "text-terminal-cyan" : mom2 < -0.03 ? "text-terminal-red" : "text-slate-400"}`}>{mom2 > 0 ? "+" : ""}{(mom2*100).toFixed(1)}%</div>
                <div className="text-[8px] text-terminal-muted">{p2Short}</div>
                <div className="text-[8px] text-terminal-muted">Rate: {(activeEwma.fast2*100).toFixed(0)}%</div>
              </div>
            </div>
            <div className="mt-2 h-2 bg-terminal-border rounded-full overflow-hidden flex">
              <div className="h-full bg-terminal-green transition-all duration-500" style={{ width: `${activeEwma.fast1*100}%` }} />
              <div className="h-full bg-terminal-cyan transition-all duration-500" style={{ width: `${activeEwma.fast2*100}%` }} />
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
            <div className="text-[8px] text-terminal-muted text-center mt-1">{activeFatigue.totalPoints} pts · {activeFatigue.deuceGames} deuce · {activeFatigue.tiebreaks} TB</div>
          </Sec>

          {/* Break opportunity */}
          {!activeState.tiebreak && !activeState.matchOver && (
            <Sec title={`BREAK OPPORTUNITY${liveStats ? " (live)" : " (Markov)"}`}>
              <div className="text-center">
                <div className={`text-[18px] font-bold ${currentBreakOpp > 0.50 ? "text-terminal-red animate-pulse" : currentBreakOpp > 0.35 ? "text-terminal-yellow" : currentBreakOpp > 0.25 ? "text-terminal-blue" : "text-terminal-muted"}`}>{(currentBreakOpp*100).toFixed(0)}%</div>
                <div className="text-[8px] text-terminal-muted">{activeState.server === 1 ? p2Short : p1Short} break chance</div>
                <div className="text-[7px] text-terminal-muted mt-0.5">
                  Score: {PT_LABELS[Math.min(activeState.currentGame.p1, 4)]}-{PT_LABELS[Math.min(activeState.currentGame.p2, 4)]} · Hold: {((1 - currentBreakOpp) * 100).toFixed(0)}%
                </div>
                {liveStats && (
                  <div className="text-[7px] text-terminal-muted mt-0.5">
                    Srv DFs: {activeState.server === 1 ? liveStats.p1_doubleFaults : liveStats.p2_doubleFaults} · 1st%: {activeState.server === 1 ? liveStats.p1_firstServePercent : liveStats.p2_firstServePercent}% · 2nd%: {activeState.server === 1 ? liveStats.p1_secondServeWon : liveStats.p2_secondServeWon}%
                  </div>
                )}
                <div className="h-2 mt-1 bg-terminal-border rounded-full overflow-hidden">
                  <div className={`h-full transition-all duration-300 ${currentBreakOpp > 0.50 ? "bg-terminal-red" : currentBreakOpp > 0.35 ? "bg-terminal-yellow" : "bg-terminal-blue"}`} style={{ width: `${currentBreakOpp*100}%` }} />
                </div>
                <div className="flex justify-between text-[7px] text-terminal-muted mt-0.5"><span>Hold</span><span>Markov + {liveStats ? "Stats" : "Momentum"}</span><span>Break</span></div>
              </div>
            </Sec>
          )}

          {/* Dual chart (manual mode) */}
          {mode === "manual" && probHist.length > 1 && (
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
              {gameLog.length > 2 && (
                <div className="mt-2">
                  <GameLogChart data={gameLog} p1Name={p1Short} />
                </div>
              )}
            </Sec>
          )}

          {/* Stats (manual mode) */}
          {mode === "manual" && history.length > 0 && (
            <Sec title="MATCH STATISTICS">
              <div className="grid grid-cols-3 gap-y-1 text-[10px] text-center">
                <div className="text-terminal-green font-bold">{p1PtsWon}</div><div className="text-terminal-muted">Points Won</div><div className="text-terminal-cyan font-bold">{p2PtsWon}</div>
                <div className="text-terminal-green font-bold">{p1SrvT > 0 ? `${Math.round(p1SrvW/p1SrvT*100)}%` : "—"}</div><div className="text-terminal-muted">Serve Pts Won</div><div className="text-terminal-cyan font-bold">{p2SrvT > 0 ? `${Math.round(p2SrvW/p2SrvT*100)}%` : "—"}</div>
                <div className="text-terminal-green font-bold">{brk1}</div><div className="text-terminal-muted">Breaks</div><div className="text-terminal-cyan font-bold">{brk2}</div>
              </div>
            </Sec>
          )}

          {/* Point log (manual mode) */}
          {mode === "manual" && history.length > 0 && (
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
              ? <div className="text-center text-terminal-muted text-[10px] py-3">
                  {mode === "auto" ? "Syncing live score…" : (history.length < 2 ? "Log points to generate signals…" : "No actionable signals")}
                </div>
              : <div className="space-y-1.5">{signals.map((s,i) => <SigCard key={i} sig={s} p1={match.player1} p2={match.player2} />)}</div>
            }
          </Sec>

          <Sec title="SIGNAL RULES">
            <div className="space-y-1 text-[9px]">
              <RuleRow icon="📈" lv="MATCH" rule="ENTRY when Markov P shifts >5% from pre-match" />
              <RuleRow icon="📉" lv="MATCH" rule="EXIT when trailing in set despite match advantage" />
              <RuleRow icon="🏆" lv="MATCH" rule="ENTRY on set lead or serving for match" />
              <RuleRow icon="🔴" lv="GAME" rule="ENTRY on break point from live game score" />
              <RuleRow icon="🎯" lv="GAME" rule={`ENTRY when break opp >28%${liveStats ? " (from real serve %)" : " (Elo model)"}`} />
              <RuleRow icon="🟢" lv="GAME" rule="EXIT when server stable — hold >85%" />
              <RuleRow icon="↔️" lv="SET" rule="EXIT on break-back — reduce exposure" />
              <RuleRow icon="🔷" lv="SET" rule="ENTRY on game lead + break advantage" />
              <RuleRow icon="💪" lv="MATCH" rule={`ENTRY on serve dominance${liveStats ? " (1st% gap >15%)" : ""}`} />
              <RuleRow icon="🛡" lv="MATCH" rule="HEDGE on tiebreak, knife-edge, or tight pts" />
              <RuleRow icon="⏸" lv="MATCH" rule="HOLD when no clear edge — wait" />
            </div>
          </Sec>

          <Sec title="POSITION SUMMARY">
            <div className="grid grid-cols-4 gap-1.5 text-center text-[10px]">
              <div className="border border-terminal-green/30 rounded p-1.5"><div className="text-[14px] font-bold text-terminal-green">{entrySigs.length}</div><div className="text-[8px] text-terminal-muted">ENTRY</div></div>
              <div className="border border-terminal-red/30 rounded p-1.5"><div className="text-[14px] font-bold text-terminal-red">{exitSigs.length}</div><div className="text-[8px] text-terminal-muted">EXIT</div></div>
              <div className="border border-terminal-yellow/30 rounded p-1.5"><div className="text-[14px] font-bold text-terminal-yellow">{hedgeSigs.length}</div><div className="text-[8px] text-terminal-muted">HEDGE</div></div>
              <div className="border border-terminal-blue/30 rounded p-1.5"><div className="text-[14px] font-bold text-terminal-blue">{signals.filter(s => s.type === "HOLD").length}</div><div className="text-[8px] text-terminal-muted">HOLD</div></div>
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
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[8px] text-terminal-muted block mb-0.5">Entry Odds (decimal)</label>
                  <input type="number" step="0.01" value={hedgeOdds} onChange={e => setHedgeOdds(parseFloat(e.target.value) || 1.01)}
                    className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[10px] text-slate-200 outline-none" />
                </div>
                <div>
                  <label className="text-[8px] text-terminal-muted block mb-0.5">Live Hedge Odds (optional)</label>
                  <input type="number" step="0.01" value={hedgeLiveOdds} placeholder="auto"
                    onChange={e => setHedgeLiveOdds(e.target.value)}
                    className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[10px] text-slate-200 outline-none placeholder:text-terminal-border" />
                </div>
              </div>
            </div>

            {/* Kelly Edge + ROI indicators */}
            <div className="grid grid-cols-3 gap-1 text-center mb-2">
              <div className={`p-1 rounded border ${hedge.kellyEdge > 0 ? "border-terminal-green/30 bg-terminal-green/5" : "border-terminal-red/30 bg-terminal-red/5"}`}>
                <div className={`text-[12px] font-bold font-mono ${hedge.kellyEdge > 0 ? "text-terminal-green" : "text-terminal-red"}`}>{hedge.kellyEdge > 0 ? "+" : ""}{hedge.kellyEdge}%</div>
                <div className="text-[7px] text-terminal-muted">Kelly Edge</div>
              </div>
              <div className="p-1 rounded border border-terminal-border">
                <div className="text-[12px] font-bold font-mono text-terminal-yellow">{hedge.currentFairOdds.toFixed(2)}</div>
                <div className="text-[7px] text-terminal-muted">Fair Odds</div>
              </div>
              <div className={`p-1 rounded border ${hedge.roi >= 0 ? "border-terminal-green/30 bg-terminal-green/5" : "border-terminal-red/30 bg-terminal-red/5"}`}>
                <div className={`text-[12px] font-bold font-mono ${hedge.roi >= 0 ? "text-terminal-green" : "text-terminal-red"}`}>{hedge.roi >= 0 ? "+" : ""}{hedge.roi}%</div>
                <div className="text-[7px] text-terminal-muted">Exp ROI</div>
              </div>
            </div>

            <div className="text-[9px] font-bold text-terminal-green uppercase mb-1">Hedge (Account B)</div>
            <div className={`p-2 rounded border ${hedge.netPosition >= 0 ? "border-terminal-green/40 bg-terminal-green/5" : "border-terminal-red/40 bg-terminal-red/5"}`}>
              <div className="grid grid-cols-2 gap-y-1.5 text-[10px]">
                <div className="text-terminal-muted">Action</div><div className="text-slate-200 font-bold">BACK {hedge.player}</div>
                <div className="text-terminal-muted">Hedge Stake</div><div className="text-terminal-yellow font-bold font-mono">${hedge.hedgeStake.toFixed(2)}</div>
                <div className="text-terminal-muted">Hedge Odds</div><div className="text-slate-200 font-mono">{hedge.hedgeOdds.toFixed(2)} {hedgeLiveOdds ? "(live)" : "(fair)"}</div>
                <div className="text-terminal-muted">Implied vs True</div><div className="text-slate-200 font-mono">{pct(hedge.impliedProb)} → {pct(hedgeSide === 1 ? currentP1WinProb : 1 - currentP1WinProb)}</div>
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
                <li>• Kelly edge turns negative (fair odds &lt; entry odds)</li>
                <li>• Opponent breaks — serve dominance collapses</li>
                <li>• Probability shifts &gt;12% against entry</li>
                <li>• Server DFs &gt;4 on backed player&apos;s serve</li>
                <li>• Total points share &lt;47% — structural disadvantage</li>
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
  const bg = sig.type === "ENTRY" ? "bg-terminal-green/10 border-terminal-green/30" : sig.type === "EXIT" ? "bg-terminal-red/10 border-terminal-red/30" : sig.type === "HEDGE" ? "bg-terminal-yellow/10 border-terminal-yellow/30" : "bg-terminal-blue/10 border-terminal-blue/30";
  const tc = sig.type === "ENTRY" ? "text-terminal-green" : sig.type === "EXIT" ? "text-terminal-red" : sig.type === "HEDGE" ? "text-terminal-yellow" : "text-terminal-blue";
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
