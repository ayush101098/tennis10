"use client";

import { useState, useCallback, useMemo } from "react";
import type { ScheduledMatch } from "@/lib/scheduleService";
import { probToOdds, kellyFraction } from "@/lib/scheduleService";

/* ═══════════════════════════════════════════════════════════════════════════════
   Tennis scoring engine + Markov match-win probability
   ═══════════════════════════════════════════════════════════════════════════ */

// Point labels in tennis order
const PT_LABELS = ["0", "15", "30", "40", "AD"];

interface GameScore { p1: number; p2: number } // points within current game
interface SetScore  { p1: number; p2: number } // games within current set
interface MatchState {
  sets: { p1: number; p2: number }[];   // completed sets
  currentSet: SetScore;
  currentGame: GameScore;
  server: 1 | 2;                        // who is serving
  tiebreak: boolean;
  matchOver: boolean;
  winner?: 1 | 2;
  setsToWin: number;                    // 2 (Bo3) or 3 (Bo5)
}

interface PointEntry {
  id: number;
  winner: 1 | 2;
  server: 1 | 2;
  scoreBefore: string;
  scoreAfter: string;
  p1WinProb: number;
  timestamp: number;
  isBreak: boolean;        // receiver won the game on this point
  isSetPoint: boolean;
  isMatchPoint: boolean;
}

function initState(bestOf: number): MatchState {
  return {
    sets: [],
    currentSet: { p1: 0, p2: 0 },
    currentGame: { p1: 0, p2: 0 },
    server: 1,
    tiebreak: false,
    matchOver: false,
    setsToWin: bestOf === 5 ? 3 : 2,
  };
}

function formatScore(s: MatchState): string {
  const sets = s.sets.map(st => `${st.p1}-${st.p2}`).join(" ");
  const cs = `${s.currentSet.p1}-${s.currentSet.p2}`;
  if (s.tiebreak) {
    return `${sets} ${cs} TB(${s.currentGame.p1}-${s.currentGame.p2})`.trim();
  }
  const p1 = s.currentGame.p1;
  const p2 = s.currentGame.p2;
  let pts: string;
  if (p1 <= 3 && p2 <= 3) {
    pts = `${PT_LABELS[p1]}-${PT_LABELS[p2]}`;
  } else if (p1 > p2) {
    pts = "AD-40";
  } else if (p2 > p1) {
    pts = "40-AD";
  } else {
    pts = "40-40";
  }
  return `${sets} ${cs} ${pts}`.trim();
}

function advancePoint(state: MatchState, pointWinner: 1 | 2): MatchState {
  const s = JSON.parse(JSON.stringify(state)) as MatchState;
  if (s.matchOver) return s;

  if (s.tiebreak) {
    // Tiebreak scoring
    if (pointWinner === 1) s.currentGame.p1++;
    else s.currentGame.p2++;

    const tb1 = s.currentGame.p1;
    const tb2 = s.currentGame.p2;
    const totalPts = tb1 + tb2;

    // Switch server every 2 points (except after first point)
    if (totalPts === 1 || (totalPts > 1 && (totalPts - 1) % 2 === 0)) {
      s.server = s.server === 1 ? 2 : 1;
    }

    if (tb1 >= 7 && tb1 - tb2 >= 2) {
      s.currentSet.p1++;
      finishSet(s, 1);
    } else if (tb2 >= 7 && tb2 - tb1 >= 2) {
      s.currentSet.p2++;
      finishSet(s, 2);
    }
  } else {
    // Regular game scoring
    if (pointWinner === 1) s.currentGame.p1++;
    else s.currentGame.p2++;

    const g1 = s.currentGame.p1;
    const g2 = s.currentGame.p2;

    let gameWon = false;
    let gameWinner: 1 | 2 = 1;

    if (g1 >= 4 && g1 - g2 >= 2) {
      gameWon = true; gameWinner = 1;
    } else if (g2 >= 4 && g2 - g1 >= 2) {
      gameWon = true; gameWinner = 2;
    }

    if (gameWon) {
      if (gameWinner === 1) s.currentSet.p1++;
      else s.currentSet.p2++;
      s.currentGame = { p1: 0, p2: 0 };
      s.server = s.server === 1 ? 2 : 1;

      const cs = s.currentSet;
      // Check for set win
      if (cs.p1 >= 6 && cs.p1 - cs.p2 >= 2) {
        finishSet(s, 1);
      } else if (cs.p2 >= 6 && cs.p2 - cs.p1 >= 2) {
        finishSet(s, 2);
      } else if (cs.p1 === 6 && cs.p2 === 6) {
        // Start tiebreak
        s.tiebreak = true;
      }
    }
  }

  return s;
}

function finishSet(s: MatchState, setWinner: 1 | 2) {
  s.sets.push({ ...s.currentSet });
  s.currentSet = { p1: 0, p2: 0 };
  s.currentGame = { p1: 0, p2: 0 };
  s.tiebreak = false;

  const p1Sets = s.sets.filter(st => st.p1 > st.p2).length;
  const p2Sets = s.sets.filter(st => st.p2 > st.p1).length;
  if (p1Sets >= s.setsToWin) { s.matchOver = true; s.winner = 1; }
  else if (p2Sets >= s.setsToWin) { s.matchOver = true; s.winner = 2; }
}

/* ── Markov win probability ── */

function matchWinProb(
  pServe1: number, // P(server wins point when P1 serves)
  pServe2: number, // P(server wins point when P2 serves)
  state: MatchState,
): number {
  // Use a recursive/simplified approach:
  // We compute P(P1 wins match) from current state forward.
  // For speed, use set-level → game-level decomposition.

  if (state.matchOver) return state.winner === 1 ? 1 : 0;

  const p1Sets = state.sets.filter(s => s.p1 > s.p2).length;
  const p2Sets = state.sets.filter(s => s.p2 > s.p1).length;

  // P(P1 wins this game)
  const pOnServe = state.server === 1 ? pServe1 : (1 - pServe2);
  const pGame = gameWinProb(pOnServe, state.currentGame.p1, state.currentGame.p2, state.tiebreak);

  // P(P1 wins this set | wins this game) and P(P1 wins this set | loses this game)
  const pSetIfWin = setWinProb(pServe1, pServe2, state.currentSet.p1 + 1, state.currentSet.p2, state.server === 1 ? 2 : 1, state.setsToWin);
  const pSetIfLose = setWinProb(pServe1, pServe2, state.currentSet.p1, state.currentSet.p2 + 1, state.server === 1 ? 2 : 1, state.setsToWin);

  const pSet = pGame * pSetIfWin + (1 - pGame) * pSetIfLose;

  // P(match) from sets
  const pMatchIfWin = matchFromSets(pServe1, pServe2, p1Sets + 1, p2Sets, state.setsToWin);
  const pMatchIfLose = matchFromSets(pServe1, pServe2, p1Sets, p2Sets + 1, state.setsToWin);

  return pSet * pMatchIfWin + (1 - pSet) * pMatchIfLose;
}

function gameWinProb(p: number, pts1: number, pts2: number, isTB: boolean): number {
  if (isTB) return tbWinProb(p, pts1, pts2);
  // Standard game: P(win from pts1, pts2)
  if (pts1 >= 4 && pts1 - pts2 >= 2) return 1;
  if (pts2 >= 4 && pts2 - pts1 >= 2) return 0;
  if (pts1 >= 3 && pts2 >= 3) {
    // Deuce territory: P(win deuce game) = p²/(p²+(1-p)²) when level
    const diff = pts1 - pts2;
    if (diff === 0) return (p * p) / (p * p + (1 - p) * (1 - p));
    if (diff === 1) return p + (1 - p) * (p * p) / (p * p + (1 - p) * (1 - p)); // AD in
    return (1 - p) * 0; // shouldn't reach but safety
  }
  // Pre-deuce: enumerate
  let prob = 0;
  // P(win) = p * P(win|pts1+1, pts2) + (1-p) * P(win|pts1, pts2+1)
  return p * gameWinProb(p, pts1 + 1, pts2, false) + (1 - p) * gameWinProb(p, pts1, pts2 + 1, false);
}

function tbWinProb(p: number, pts1: number, pts2: number): number {
  if (pts1 >= 7 && pts1 - pts2 >= 2) return 1;
  if (pts2 >= 7 && pts2 - pts1 >= 2) return 0;
  if (pts1 >= 6 && pts2 >= 6) {
    // Repeating deuce-like pattern
    return (p * p) / (p * p + (1 - p) * (1 - p));
  }
  return p * tbWinProb(p, pts1 + 1, pts2) + (1 - p) * tbWinProb(p, pts1, pts2 + 1);
}

function setWinProb(pS1: number, pS2: number, g1: number, g2: number, nextServer: 1 | 2, _setsToWin: number): number {
  if (g1 >= 6 && g1 - g2 >= 2) return 1;
  if (g2 >= 6 && g2 - g1 >= 2) return 0;
  if (g1 === 6 && g2 === 6) {
    // Tiebreak: P1 serves if nextServer===1
    const pTB = nextServer === 1 ? pS1 : (1 - pS2);
    return (pTB * pTB) / (pTB * pTB + (1 - pTB) * (1 - pTB)); // simplified
  }
  if (g1 > 6 || g2 > 6) return g1 > g2 ? 1 : 0;
  const pGame = nextServer === 1 ? holdProb(pS1) : (1 - holdProb(pS2));
  const ns: (1|2) = nextServer === 1 ? 2 : 1;
  return pGame * setWinProb(pS1, pS2, g1 + 1, g2, ns, _setsToWin)
       + (1 - pGame) * setWinProb(pS1, pS2, g1, g2 + 1, ns, _setsToWin);
}

function holdProb(pServe: number): number {
  // P(server wins a game) from 0-0
  return gameWinProb(pServe, 0, 0, false);
}

function matchFromSets(pS1: number, pS2: number, s1: number, s2: number, setsToWin: number): number {
  if (s1 >= setsToWin) return 1;
  if (s2 >= setsToWin) return 0;
  // Approximate: P(P1 wins a set) ≈ 60-65% for favorite (simplified)
  const pSet = setWinProb(pS1, pS2, 0, 0, 1, setsToWin);
  // Remaining sets needed
  const need1 = setsToWin - s1;
  const need2 = setsToWin - s2;
  const total = need1 + need2 - 1;
  // Binomial: P(win need1 or more out of total)
  let prob = 0;
  for (let w = need1; w <= total; w++) {
    prob += binomial(total, w) * Math.pow(pSet, w) * Math.pow(1 - pSet, total - w);
  }
  return prob;
}

function binomial(n: number, k: number): number {
  if (k > n) return 0;
  let r = 1;
  for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1);
  return r;
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Point Tracker Component
   ═══════════════════════════════════════════════════════════════════════════ */

interface PointTrackerProps {
  match: ScheduledMatch;
}

export default function PointTracker({ match }: PointTrackerProps) {
  const [state, setState] = useState<MatchState>(() => initState(match.best_of));
  const [history, setHistory] = useState<PointEntry[]>([]);
  const [pServe, setPServe] = useState(0.64); // avg serve point win %

  // Derive serve probabilities from base win prob
  const pServe1 = useMemo(() => {
    // Rough calibration: if P1 win prob is p, back-derive serve %
    // Typical: favorite has ~65% serve pts, underdog ~60%
    const base = match.p1_win_prob;
    return Math.max(0.50, Math.min(0.75, 0.55 + (base - 0.5) * 0.3));
  }, [match.p1_win_prob]);

  const pServe2 = useMemo(() => {
    const base = match.p2_win_prob;
    return Math.max(0.50, Math.min(0.75, 0.55 + (base - 0.5) * 0.3));
  }, [match.p2_win_prob]);

  const currentP1WinProb = useMemo(() => {
    return matchWinProb(pServe1, pServe2, state);
  }, [pServe1, pServe2, state]);

  const logPoint = useCallback((winner: 1 | 2) => {
    if (state.matchOver) return;
    const scoreBefore = formatScore(state);
    const isBreakOpportunity = state.server !== winner;
    const newState = advancePoint(state, winner);
    const scoreAfter = formatScore(newState);

    // Check if this point ended a game, set, or match
    const gameEnded = newState.currentGame.p1 === 0 && newState.currentGame.p2 === 0
      && (scoreBefore !== scoreAfter);
    const isBreak = isBreakOpportunity && gameEnded && !state.tiebreak;

    const newP1Prob = matchWinProb(pServe1, pServe2, newState);

    const entry: PointEntry = {
      id: history.length + 1,
      winner,
      server: state.server,
      scoreBefore,
      scoreAfter,
      p1WinProb: newP1Prob,
      timestamp: Date.now(),
      isBreak,
      isSetPoint: false,
      isMatchPoint: newState.matchOver,
    };

    setHistory(prev => [...prev, entry]);
    setState(newState);
  }, [state, history, pServe1, pServe2]);

  const undo = useCallback(() => {
    if (history.length === 0) return;
    const prev = history.slice(0, -1);
    // Rebuild state from scratch
    let s = initState(match.best_of);
    for (const entry of prev) {
      s = advancePoint(s, entry.winner);
    }
    setHistory(prev);
    setState(s);
  }, [history, match.best_of]);

  const reset = useCallback(() => {
    setState(initState(match.best_of));
    setHistory([]);
  }, [match.best_of]);

  // Stats
  const p1PtsWon = history.filter(p => p.winner === 1).length;
  const p2PtsWon = history.filter(p => p.winner === 2).length;
  const p1ServeWon = history.filter(p => p.server === 1 && p.winner === 1).length;
  const p1ServeTotal = history.filter(p => p.server === 1).length;
  const p2ServeWon = history.filter(p => p.server === 2 && p.winner === 2).length;
  const p2ServeTotal = history.filter(p => p.server === 2).length;
  const breaks1 = history.filter(p => p.isBreak && p.winner === 1).length;
  const breaks2 = history.filter(p => p.isBreak && p.winner === 2).length;

  // Mini probability chart (last 30 points)
  const probHistory = useMemo(() => {
    const pts: { id: number; p: number }[] = [{ id: 0, p: match.p1_win_prob }];
    history.forEach(h => pts.push({ id: h.id, p: h.p1WinProb }));
    return pts.slice(-30);
  }, [history, match.p1_win_prob]);

  // Momentum: last 5 points
  const last5 = history.slice(-5);
  const momentum1 = last5.filter(p => p.winner === 1).length;
  const momentum2 = last5.filter(p => p.winner === 2).length;

  // Current odds
  const odds1 = probToOdds(currentP1WinProb);
  const odds2 = probToOdds(1 - currentP1WinProb);

  return (
    <div className="p-3 space-y-3 text-[11px]">
      {/* Header */}
      <div className="text-center">
        <div className="text-xs font-bold text-terminal-cyan mb-1">🎾 POINT TRACKER</div>
        <div className="text-slate-200 font-medium">{match.player1} vs {match.player2}</div>
        <div className="text-[10px] text-terminal-muted">{match.tournament} · {match.surface} · Bo{match.best_of}</div>
      </div>

      {/* Scoreboard */}
      <div className="border border-terminal-border rounded overflow-hidden">
        <div className="bg-terminal-panel/80 px-3 py-2">
          {/* Set scores row */}
          <div className="grid grid-cols-[1fr_repeat(5,28px)_40px] gap-1 items-center text-[11px] font-mono">
            <div />
            {[0, 1, 2, 3, 4].map(i => (
              <div key={i} className="text-center text-[8px] text-terminal-muted">
                {i < state.sets.length ? `S${i + 1}` : i === state.sets.length ? "S*" : ""}
              </div>
            ))}
            <div className="text-center text-[8px] text-terminal-muted">PTS</div>
          </div>

          {/* P1 */}
          <div className="grid grid-cols-[1fr_repeat(5,28px)_40px] gap-1 items-center text-[11px] font-mono mt-1">
            <div className="flex items-center gap-1 truncate">
              {state.server === 1 && <span className="text-terminal-yellow text-[8px]">●</span>}
              <span className={`truncate ${state.winner === 1 ? "text-terminal-green font-bold" : "text-slate-200"}`}>
                {match.player1.split(" ").pop()}
              </span>
            </div>
            {[0, 1, 2, 3, 4].map(i => (
              <div key={i} className="text-center font-bold">
                {i < state.sets.length
                  ? <span className={state.sets[i].p1 > state.sets[i].p2 ? "text-terminal-green" : "text-slate-400"}>{state.sets[i].p1}</span>
                  : i === state.sets.length
                    ? <span className="text-slate-200">{state.currentSet.p1}</span>
                    : ""}
              </div>
            ))}
            <div className="text-center text-terminal-yellow font-bold">
              {state.tiebreak ? state.currentGame.p1 : (state.currentGame.p1 <= 3 ? PT_LABELS[state.currentGame.p1] : "AD")}
            </div>
          </div>

          {/* P2 */}
          <div className="grid grid-cols-[1fr_repeat(5,28px)_40px] gap-1 items-center text-[11px] font-mono mt-0.5">
            <div className="flex items-center gap-1 truncate">
              {state.server === 2 && <span className="text-terminal-yellow text-[8px]">●</span>}
              <span className={`truncate ${state.winner === 2 ? "text-terminal-green font-bold" : "text-slate-200"}`}>
                {match.player2.split(" ").pop()}
              </span>
            </div>
            {[0, 1, 2, 3, 4].map(i => (
              <div key={i} className="text-center font-bold">
                {i < state.sets.length
                  ? <span className={state.sets[i].p2 > state.sets[i].p1 ? "text-terminal-green" : "text-slate-400"}>{state.sets[i].p2}</span>
                  : i === state.sets.length
                    ? <span className="text-slate-200">{state.currentSet.p2}</span>
                    : ""}
              </div>
            ))}
            <div className="text-center text-terminal-yellow font-bold">
              {state.tiebreak ? state.currentGame.p2 : (state.currentGame.p2 <= 3 ? PT_LABELS[state.currentGame.p2] : "AD")}
            </div>
          </div>
        </div>

        {/* Serving indicator */}
        <div className="px-3 py-1 border-t border-terminal-border bg-terminal-bg text-[9px] text-terminal-muted text-center">
          {state.matchOver
            ? <span className="text-terminal-green font-bold">MATCH OVER — {state.winner === 1 ? match.player1 : match.player2} wins!</span>
            : <>{state.tiebreak ? "TIEBREAK · " : ""}{state.server === 1 ? match.player1 : match.player2} serving</>
          }
        </div>
      </div>

      {/* Point input buttons */}
      {!state.matchOver && (
        <div className="grid grid-cols-2 gap-2">
          <button onClick={() => logPoint(1)}
            className="py-2.5 rounded border border-terminal-green/40 bg-terminal-green/10 hover:bg-terminal-green/20 text-terminal-green font-bold text-[11px] transition active:scale-95">
            {state.server === 1 ? "🟢" : "🔴"} {match.player1.split(" ").pop()} wins pt
          </button>
          <button onClick={() => logPoint(2)}
            className="py-2.5 rounded border border-terminal-cyan/40 bg-terminal-cyan/10 hover:bg-terminal-cyan/20 text-terminal-cyan font-bold text-[11px] transition active:scale-95">
            {state.server === 2 ? "🟢" : "🔴"} {match.player2.split(" ").pop()} wins pt
          </button>
        </div>
      )}

      {/* Undo / Reset */}
      <div className="flex gap-2">
        <button onClick={undo} disabled={history.length === 0}
          className="flex-1 text-[9px] py-1 rounded border border-terminal-border text-terminal-muted hover:text-slate-300 disabled:opacity-30">
          ↩ Undo
        </button>
        <button onClick={reset} disabled={history.length === 0}
          className="flex-1 text-[9px] py-1 rounded border border-terminal-border text-terminal-muted hover:text-slate-300 disabled:opacity-30">
          ⟲ Reset
        </button>
      </div>

      {/* Live Win Probability */}
      <Section title="LIVE WIN PROBABILITY">
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className={currentP1WinProb >= 0.5 ? "text-terminal-green font-bold" : "text-slate-300"}>{match.player1}</span>
            <span className={currentP1WinProb >= 0.5 ? "text-terminal-green font-bold" : "text-slate-400"}>{pct(currentP1WinProb)}</span>
          </div>
          <div className="h-3 bg-terminal-border rounded-full overflow-hidden flex">
            <div className="h-full bg-terminal-green transition-all duration-300" style={{ width: pct(currentP1WinProb) }} />
            <div className="h-full bg-terminal-cyan transition-all duration-300" style={{ width: pct(1 - currentP1WinProb) }} />
          </div>
          <div className="flex items-center justify-between">
            <span className={currentP1WinProb < 0.5 ? "text-terminal-cyan font-bold" : "text-slate-300"}>{match.player2}</span>
            <span className={currentP1WinProb < 0.5 ? "text-terminal-cyan font-bold" : "text-slate-400"}>{pct(1 - currentP1WinProb)}</span>
          </div>
        </div>
        <div className="text-[9px] text-terminal-muted text-center mt-1">
          Fair odds: {odds1.toFixed(2)} / {odds2.toFixed(2)} · Pre-match: {pct(match.p1_win_prob)} / {pct(match.p2_win_prob)}
        </div>
      </Section>

      {/* Probability chart (ASCII sparkline) */}
      {probHistory.length > 1 && (
        <Section title="PROBABILITY TIMELINE">
          <MiniChart data={probHistory} p1Name={match.player1} />
        </Section>
      )}

      {/* Momentum */}
      {history.length >= 3 && (
        <Section title="MOMENTUM">
          <div className="flex items-center gap-3">
            <div className="flex-1 text-center">
              <div className="text-[14px] font-bold text-terminal-green">{momentum1}</div>
              <div className="text-[8px] text-terminal-muted">Last 5</div>
            </div>
            <div className="flex gap-0.5">
              {last5.map((p, i) => (
                <div key={i} className={`w-3 h-3 rounded-full ${p.winner === 1 ? "bg-terminal-green" : "bg-terminal-cyan"}`} />
              ))}
            </div>
            <div className="flex-1 text-center">
              <div className="text-[14px] font-bold text-terminal-cyan">{momentum2}</div>
              <div className="text-[8px] text-terminal-muted">Last 5</div>
            </div>
          </div>
        </Section>
      )}

      {/* Match Statistics */}
      {history.length > 0 && (
        <Section title="MATCH STATISTICS">
          <div className="grid grid-cols-3 gap-y-1 text-[10px] text-center">
            <div className="text-terminal-green font-bold">{p1PtsWon}</div>
            <div className="text-terminal-muted">Points Won</div>
            <div className="text-terminal-cyan font-bold">{p2PtsWon}</div>

            <div className="text-terminal-green font-bold">{p1ServeTotal > 0 ? `${Math.round(p1ServeWon / p1ServeTotal * 100)}%` : "—"}</div>
            <div className="text-terminal-muted">Serve Pts Won</div>
            <div className="text-terminal-cyan font-bold">{p2ServeTotal > 0 ? `${Math.round(p2ServeWon / p2ServeTotal * 100)}%` : "—"}</div>

            <div className="text-terminal-green font-bold">{breaks1}</div>
            <div className="text-terminal-muted">Breaks</div>
            <div className="text-terminal-cyan font-bold">{breaks2}</div>

            <div className="text-terminal-green font-bold">{p1ServeTotal > 0 ? `${p1ServeWon}/${p1ServeTotal}` : "—"}</div>
            <div className="text-terminal-muted">Serve W/L</div>
            <div className="text-terminal-cyan font-bold">{p2ServeTotal > 0 ? `${p2ServeWon}/${p2ServeTotal}` : "—"}</div>
          </div>
        </Section>
      )}

      {/* Point-by-point log (last 15) */}
      {history.length > 0 && (
        <Section title={`POINT LOG (${history.length} pts)`}>
          <div className="max-h-[180px] overflow-y-auto space-y-0.5">
            {history.slice(-15).reverse().map(p => (
              <div key={p.id} className={`flex items-center gap-1.5 text-[9px] py-0.5 px-1 rounded ${
                p.isBreak ? "bg-terminal-red/10" : p.isMatchPoint ? "bg-terminal-yellow/10" : ""
              }`}>
                <span className="text-terminal-muted w-[18px] shrink-0">#{p.id}</span>
                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${p.winner === 1 ? "bg-terminal-green" : "bg-terminal-cyan"}`} />
                <span className={`font-medium ${p.winner === 1 ? "text-terminal-green" : "text-terminal-cyan"}`}>
                  {p.winner === 1 ? match.player1.split(" ").pop() : match.player2.split(" ").pop()}
                </span>
                {p.server !== p.winner && <span className="text-terminal-red text-[7px]">BRK</span>}
                {p.isBreak && <span className="text-terminal-red text-[7px] font-bold">BREAK!</span>}
                {p.isMatchPoint && <span className="text-terminal-yellow text-[7px] font-bold">MATCH!</span>}
                <span className="text-terminal-muted ml-auto shrink-0 font-mono">{pct(p.p1WinProb)}</span>
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

/* ── Mini Chart ── */

function MiniChart({ data, p1Name }: { data: { id: number; p: number }[]; p1Name: string }) {
  const h = 50;
  const w = 240;
  const padX = 2;
  const padY = 4;
  const innerW = w - padX * 2;
  const innerH = h - padY * 2;

  const points = data.map((d, i) => {
    const x = padX + (i / Math.max(data.length - 1, 1)) * innerW;
    const y = padY + (1 - d.p) * innerH;
    return `${x},${y}`;
  });

  const lastP = data[data.length - 1]?.p ?? 0.5;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[50px]">
        {/* 50% line */}
        <line x1={padX} y1={h / 2} x2={w - padX} y2={h / 2} stroke="rgba(255,255,255,0.1)" strokeDasharray="2,2" />
        {/* Probability line */}
        <polyline fill="none" stroke={lastP >= 0.5 ? "#4ade80" : "#22d3ee"} strokeWidth="1.5" points={points.join(" ")} />
        {/* End dot */}
        <circle cx={parseFloat(points[points.length - 1]?.split(",")[0] || "0")}
                cy={parseFloat(points[points.length - 1]?.split(",")[1] || "25")}
                r="3" fill={lastP >= 0.5 ? "#4ade80" : "#22d3ee"} />
      </svg>
      <div className="flex justify-between text-[8px] text-terminal-muted mt-0.5">
        <span>Start</span>
        <span>{pct(lastP)} {p1Name.split(" ").pop()}</span>
      </div>
    </div>
  );
}

/* ── Shared ── */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
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
