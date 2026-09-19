/**
 * Real match telemetry, derived from the score.
 *
 * WHY THIS EXISTS
 *   Three panels — momentum, fatigue, break opportunity — were reporting
 *   numbers nobody had measured:
 *
 *     momentum  ewmaFromStats read `stats.p1_firstServePercent || 50`. With an
 *               unreported stat that is 0, so the `|| 50` fired, and every
 *               match showed +20.0% for BOTH players. The number was the
 *               fallback, arithmetic'd.
 *     fatigue   totalPoints fell back to `games × 4.5`, deuce games to
 *               `games × 0.3`, and p1Load and p2Load were assigned the same
 *               value — two identical bars implying a per-player measurement
 *               that never happened.
 *     momentum  and it was not a moving average of anything. EWMA needs a
 *               SEQUENCE; that function had one snapshot and took a difference
 *               between two ratios of it.
 *
 * WHAT IS ACTUALLY AVAILABLE
 *   The feed polls a score. Two consecutive scores differ by the point that was
 *   just played, so the point WINNER is recoverable by differencing — no serve
 *   stats and no server required. That gives a real point sequence, and from a
 *   real sequence a real moving average, a real point count and a real deuce
 *   count all follow.
 *
 * WHAT IS NOT, AND IS NOT FAKED
 *   Polling misses points in fast games. Every count here is therefore a FLOOR
 *   — "at least this many" — and `gaps` records how often a jump was too large
 *   to attribute. The UI is expected to show `observedPoints`, not to imply it
 *   saw the whole match. Anything needing the server (break opportunity, any
 *   per-player fatigue split) is not derivable from this and returns null
 *   rather than a plausible-looking number.
 */

/** Exponential weights. Fast reacts within a game; slow is the match baseline. */
const ALPHA_FAST = 0.15;
const ALPHA_SLOW = 0.04;

export interface TelemetryState {
  /** Points we actually saw played, by differencing the score. A floor. */
  observedPoints: number;
  p1Points: number;
  p2Points: number;
  /** Transitions too large to attribute — the points we know we missed. */
  gaps: number;
  /** Times the game score stood at 40-40. Counted, not estimated. */
  deuceGames: number;
  tiebreaks: number;
  gamesPlayed: number;
  setsPlayed: number;
  /** EWMA of "P1 won the point", fast and slow. Only meaningful once seeded. */
  fastP1: number;
  slowP1: number;
  /** The last few point winners, newest last — for a form dot strip. */
  recent: (1 | 2)[];
  /**
   * Winner of each COMPLETED game, in order. Real and unambiguous — the game
   * score tells you who won a game with no attribution needed — but it is NOT
   * enough on its own to say who held serve and who broke: that needs to know
   * who was SERVING each game, and serve strictly alternates from an anchor
   * this feed does not send. Kept here so that the moment a server reading
   * does arrive (even once, even briefly — see holdsAndBreaks in
   * breakHoldEngine.ts), every game already played can be retroactively
   * labelled hold/break by counting back from it via alternation, instead of
   * only the games that happen to follow the reading.
   */
  gameLog: (1 | 2)[];
  // internals
  lastPts: [number, number] | null;
  lastGames: [number, number] | null;
  lastSet: number;
  inDeuceThisGame: boolean;
}

export interface TelemetrySnapshot {
  matchId: string;
  /** Point indices for P1 and P2: 0/15/30/40/A -> 0/1/2/3/4. */
  p1Pts: number;
  p2Pts: number;
  gamesP1: number;
  gamesP2: number;
  setIndex: number;
  isTiebreak?: boolean;
}

const blank = (): TelemetryState => ({
  observedPoints: 0, p1Points: 0, p2Points: 0, gaps: 0,
  deuceGames: 0, tiebreaks: 0, gamesPlayed: 0, setsPlayed: 1,
  fastP1: 0.5, slowP1: 0.5, recent: [], gameLog: [],
  lastPts: null, lastGames: null, lastSet: 1, inDeuceThisGame: false,
});

const memory = new Map<string, TelemetryState>();

/**
 * Who won the point between two scores in the SAME game?
 *
 * Returns null when the transition is not a single point — a missed point, a
 * feed correction, or a jump we cannot attribute. Guessing here would inject
 * fabricated points into the very sequence this exists to measure.
 */
function pointWinner(
  [a0, b0]: [number, number],
  [a1, b1]: [number, number],
): 1 | 2 | null {
  // Ordinary progression: exactly one side advanced by exactly one.
  if (a1 === a0 + 1 && b1 === b0) return 1;
  if (b1 === b0 + 1 && a1 === a0) return 2;
  // Advantage lost: 4-3 -> 3-3 means the OTHER player took the point.
  if (a0 === 4 && b0 === 3 && a1 === 3 && b1 === 3) return 2;
  if (b0 === 4 && a0 === 3 && b1 === 3 && a1 === 3) return 1;
  return null;
}

/**
 * Fold one polled score into the match's telemetry.
 *
 * Idempotent on a repeated identical score — the board polls far faster than
 * points are played, and counting each poll would inflate every figure here.
 */
export function observeTelemetry(s: TelemetrySnapshot): TelemetryState {
  const prev = memory.get(s.matchId) ?? blank();
  const t: TelemetryState = { ...prev, recent: [...prev.recent], gameLog: [...prev.gameLog] };
  const pts: [number, number] = [s.p1Pts, s.p2Pts];
  const games: [number, number] = [s.gamesP1, s.gamesP2];

  if (s.setIndex !== t.lastSet) {
    t.setsPlayed = s.setIndex;
    t.lastSet = s.setIndex;
    t.lastGames = null;
    t.lastPts = null;
    t.inDeuceThisGame = false;
  }

  // ── game boundary ──
  //
  // A game score that goes DOWN inside a set is the feed correcting itself or
  // delivering a stale poll out of order — not a game. Replaying the recorded
  // corpus, scores bounce (1-0 -> 1-1 -> 1-0), and counting every transition
  // produced 123 games in a match that can hold about forty. Regressions are
  // dropped, so a bounce costs nothing rather than inventing a game each way.
  if (t.lastGames && (games[0] < t.lastGames[0] || games[1] < t.lastGames[1])) {
    memory.set(s.matchId, t);
    return t;
  }
  if (t.lastGames && (games[0] !== t.lastGames[0] || games[1] !== t.lastGames[1])) {
    const d0 = games[0] - t.lastGames[0], d1 = games[1] - t.lastGames[1];
    if ((d0 === 1 && d1 === 0) || (d0 === 0 && d1 === 1)) {
      t.gamesPlayed += 1;
      // The game's last point went to whoever won the game. This is the one
      // point a naive differ always loses, because the score resets to 0-0.
      const w: 1 | 2 = d0 === 1 ? 1 : 2;
      record(t, w);
      t.gameLog.push(w);
      if (s.isTiebreak || (t.lastGames[0] === 6 && t.lastGames[1] === 6)) t.tiebreaks += 1;
    } else {
      t.gaps += 1;                      // games appeared; we cannot attribute them
    }
    t.inDeuceThisGame = false;
    t.lastPts = null;                   // new game — do not difference across it
  }
  t.lastGames = games;

  // ── deuce, counted rather than assumed ──
  if (!s.isTiebreak && pts[0] >= 3 && pts[1] >= 3 && pts[0] === pts[1] && !t.inDeuceThisGame) {
    t.deuceGames += 1;
    t.inDeuceThisGame = true;
  }

  // ── the point itself ──
  if (t.lastPts && (pts[0] !== t.lastPts[0] || pts[1] !== t.lastPts[1])) {
    const w = s.isTiebreak
      ? (pts[0] === t.lastPts[0] + 1 ? 1 : pts[1] === t.lastPts[1] + 1 ? 2 : null)
      : pointWinner(t.lastPts, pts);
    if (w) record(t, w); else t.gaps += 1;
  }
  t.lastPts = pts;

  memory.set(s.matchId, t);
  return t;
}

/** Add one observed point to the counts and both moving averages. */
function record(t: TelemetryState, winner: 1 | 2): void {
  const y = winner === 1 ? 1 : 0;
  t.observedPoints += 1;
  if (winner === 1) t.p1Points += 1; else t.p2Points += 1;
  t.fastP1 = t.fastP1 + ALPHA_FAST * (y - t.fastP1);
  t.slowP1 = t.slowP1 + ALPHA_SLOW * (y - t.slowP1);
  t.recent.push(winner);
  if (t.recent.length > 12) t.recent.shift();
}

/**
 * How many points must be seen before the fast average means anything.
 *
 * With α=0.15 the average still carries most of its 0.5 seed for the first
 * several points, so reporting momentum immediately would be reporting the
 * seed. This is the same mistake as `|| 50`, arrived at more slowly.
 */
export const MIN_POINTS_FOR_MOMENTUM = 12;

export interface Momentum {
  /** Positive = P1 is outperforming their own match baseline, and by how much. */
  p1: number;
  p2: number;
  /** Recent share of points won by P1, 0–1. */
  fastP1: number;
  observedPoints: number;
  recent: (1 | 2)[];
}

/** Real momentum: recent form against this match's own baseline. Null until seeded. */
export function momentum(t: TelemetryState): Momentum | null {
  if (t.observedPoints < MIN_POINTS_FOR_MOMENTUM) return null;
  const d = t.fastP1 - t.slowP1;
  return { p1: d, p2: -d, fastP1: t.fastP1, observedPoints: t.observedPoints, recent: t.recent };
}

export interface Workload {
  points: number;
  deuceGames: number;
  tiebreaks: number;
  games: number;
  /** 0–1 load estimate for BOTH players — see the note on symmetry. */
  load: number;
  /** True when polling gaps mean the counts understate the real match. */
  incomplete: boolean;
}

/**
 * Match workload.
 *
 * DELIBERATELY ONE FIGURE, NOT TWO. Both players contest every point, so a load
 * derived from the score is identical for both by construction — which is
 * exactly what the old panel showed, two bars at 19%, while implying it had
 * measured them separately. Splitting them honestly needs who served each game
 * (long service games tire the server more), and the live feed does not report
 * the server. So this reports the match, and says so.
 */
export function workload(t: TelemetryState): Workload | null {
  if (t.observedPoints < 4) return null;
  const load = Math.min(1, Math.max(0,
    t.observedPoints * 0.003 + t.deuceGames * 0.015 + t.tiebreaks * 0.04 - (t.setsPlayed - 1) * 0.1));
  return {
    points: t.observedPoints,
    deuceGames: t.deuceGames,
    tiebreaks: t.tiebreaks,
    games: t.gamesPlayed,
    load,
    incomplete: t.gaps > 0,
  };
}

export function forgetTelemetry(matchId: string): void { memory.delete(matchId); }
export function resetTelemetry(): void { memory.clear(); }
export function peekTelemetry(matchId: string): TelemetryState | undefined { return memory.get(matchId); }
