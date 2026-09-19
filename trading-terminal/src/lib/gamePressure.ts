/**
 * GAME PRESSURE INDEX — one score per match, not six independent bets.
 *
 * The point of this file is the thing that was missing everywhere else: MEMORY.
 * breakHoldEngine reads a single snapshot and can tell you "this is a break
 * point". It cannot tell you it is the THIRD break point of the game, that the
 * previous two were saved, or that the server was broken in the last game —
 * and that sequence is the entire signal. `breakPointCount` there is
 * `isBreakPoint ? 1 : 0`, which is a per-snapshot flag wearing a counter's name.
 *
 * WHY ONE SCORE AND NOT SIX SIGNALS
 *   A deuce, a break point, a saved break point and a previous break are not
 *   independent observations — they are four views of the same player being
 *   under pressure. Emitting them as separate signals invites acting on the
 *   same evidence four times, which is the portfolio error the whole design is
 *   meant to prevent. They accumulate into one number instead.
 *
 * THE FEED GIVES SNAPSHOTS, NOT EVENTS
 *   The live board polls; it does not receive a point stream. So every event
 *   here is inferred by comparing consecutive snapshots, and the inference is
 *   deliberately conservative: a transition that cannot be read unambiguously
 *   is dropped rather than guessed. Polling at a few seconds will miss points
 *   in fast games, so these counts are a FLOOR — "at least this many break
 *   points" — and the code says so rather than implying it saw every point.
 */

import { gameTree, gameStateKey as stateKey } from "./gameTree";

/* ── what the engine is told, once per poll ───────────────────────────────── */

export interface PressureObservation {
  /** Stable id for the match — the memory is keyed on it. */
  matchId: string;
  /** 1 or 2 — which player is serving this game. */
  server: 1 | 2;
  /** Point index for the SERVER: 0/15/30/40/A -> 0/1/2/3/4. */
  srvPts: number;
  /** Point index for the returner. */
  retPts: number;
  /** Games won in the current set. */
  gamesP1: number;
  gamesP2: number;
  /** Which set (1-based) — a change resets the per-set counters. */
  setIndex: number;
  isTiebreak?: boolean;
}

/* ── what it remembers ────────────────────────────────────────────────────── */

export type PressureRegime = "NORMAL" | "ELEVATED" | "DOMINANCE";

export interface PressureState {
  /** 0–100. Decayed across games, accumulated within one. */
  gpi: number;
  regime: PressureRegime;

  // ── current game ──
  /** Break points seen in this game. A floor: polling can miss points. */
  bpThisGame: number;
  /** Break points the server has come back from, this game. */
  bpSavedThisGame: number;
  /** Times this game has stood at deuce. */
  deuceThisGame: number;
  /** True while the returner is holding a break point right now. */
  atBreakPoint: boolean;

  // ── current set ──
  bpThisSet: number;
  breaksThisSet: number;
  /** The game immediately before this one ended in a break. */
  prevGameWasBreak: boolean;

  /** Who is under pressure — the player who is SERVING and may be broken. */
  underPressure: 1 | 2;

  // internal bookkeeping, exposed for debugging and replay
  lastKey: string;
  lastGames: [number, number];
  lastSet: number;
}

/* ── the event weights, straight from the strategy table ──────────────────── */

const W = {
  deuce: 1,
  firstBp: 2,
  bpAfterDeuce: 3,
  bpSaved: 2,
  secondBpSameGame: 3,
  thirdBpSameGame: 5,
  prevGameWasBreak: 3,
  prevBreakAndBp: 5,
  secondBreakInSet: 5,
} as const;

/**
 * Decay applied at each GAME boundary, not each point.
 *
 * Pressure is a property of the last few games, so it should fade on the same
 * clock that games are played on. Decaying per poll would make the score depend
 * on how often we happened to sample, which is an artefact of our
 * infrastructure rather than anything happening on court.
 */
const DECAY = 0.8;

/** GPI is reported 0–100; this is the raw score treated as "maximum pressure". */
const GPI_FULL_SCALE = 25;

const blank = (server: 1 | 2): PressureState => ({
  gpi: 0, regime: "NORMAL",
  bpThisGame: 0, bpSavedThisGame: 0, deuceThisGame: 0, atBreakPoint: false,
  bpThisSet: 0, breaksThisSet: 0, prevGameWasBreak: false,
  underPressure: server,
  lastKey: "", lastGames: [0, 0], lastSet: 1,
});

const memory = new Map<string, PressureState>();

/** Is the returner one point from the break? */
const isBp = (srvPts: number, retPts: number, isTiebreak: boolean) =>
  !isTiebreak && retPts >= 3 && retPts > srvPts;

/**
 * Fold one polled observation into the match's pressure state.
 *
 * Idempotent on a repeated identical snapshot: the board polls faster than
 * points are played, so the same score arrives many times and must not be
 * counted many times.
 */
export function observePressure(obs: PressureObservation): PressureState {
  const prev = memory.get(obs.matchId) ?? blank(obs.server);
  const s: PressureState = { ...prev, underPressure: obs.server };
  const key = stateKey(obs.srvPts, obs.retPts, !!obs.isTiebreak);

  // ── set boundary: the per-set counters are about THIS set ──
  if (obs.setIndex !== s.lastSet) {
    s.bpThisSet = 0;
    s.breaksThisSet = 0;
    s.prevGameWasBreak = false;
    s.lastSet = obs.setIndex;
    // Game-level counters go too, since a new set starts a new game.
    s.bpThisGame = 0; s.bpSavedThisGame = 0; s.deuceThisGame = 0;
  }

  // ── game boundary: detected by the game score changing ──
  const [pg1, pg2] = s.lastGames;
  const gamesChanged = obs.gamesP1 !== pg1 || obs.gamesP2 !== pg2;
  if (gamesChanged) {
    const d1 = obs.gamesP1 - pg1, d2 = obs.gamesP2 - pg2;
    // Exactly one game, to exactly one player. Anything else means we missed
    // games between polls and cannot say who won what — so we record nothing
    // rather than inventing a break.
    let broke = false, resolved = false;
    if ((d1 === 1 && d2 === 0) || (d1 === 0 && d2 === 1)) {
      resolved = true;
      const winner: 1 | 2 = d1 === 1 ? 1 : 2;
      // The server of the game that just ENDED is whoever was serving in the
      // previous snapshot — s.underPressure still holds it at this point,
      // because it is only overwritten below.
      broke = winner !== prev.underPressure;
    }
    if (resolved && broke) {
      s.breaksThisSet += 1;
      s.prevGameWasBreak = true;
      if (s.breaksThisSet === 2) s.gpi += W.secondBreakInSet;
    } else if (resolved) {
      s.prevGameWasBreak = false;
    }
    // New game: decay what came before, clear the per-game counters.
    s.gpi *= DECAY;
    s.bpThisGame = 0; s.bpSavedThisGame = 0; s.deuceThisGame = 0;
    s.lastGames = [obs.gamesP1, obs.gamesP2];
    // A break in the previous game is itself pressure on the next server.
    if (s.prevGameWasBreak) s.gpi += W.prevGameWasBreak;
  }

  // ── within-game events, only on an actual state change ──
  if (key !== s.lastKey) {
    const wasBp = isBp2(s.lastKey);
    const nowBp = isBp(obs.srvPts, obs.retPts, !!obs.isTiebreak);

    if (key === "DEUCE") {
      s.deuceThisGame += 1;
      // Plain deuce is the weakest evidence there is — it says the game is
      // close, not who is winning it. Weight 1, and never an entry on its own.
      s.gpi += W.deuce;
      // A break point that was just saved returns the game to deuce.
      if (wasBp) { s.bpSavedThisGame += 1; s.gpi += W.bpSaved; }
    }

    if (nowBp && !wasBp) {
      s.bpThisGame += 1;
      s.bpThisSet += 1;
      s.gpi += s.bpThisGame === 1 ? W.firstBp
        : s.bpThisGame === 2 ? W.secondBpSameGame
        : W.thirdBpSameGame;
      // A break point arrived at THROUGH deuce is worth more than one reached
      // from 0-40: the server has already had to hold the game together once.
      if (s.lastKey === "DEUCE") s.gpi += W.bpAfterDeuce;
      // The compound state: the last game was a break and the next server is
      // already facing one. This is the strongest cross-game evidence there is.
      if (s.prevGameWasBreak) s.gpi += W.prevBreakAndBp;
    }

    // Server escaped a break point without reaching deuce (e.g. 30-40 -> 40-40
    // is handled above; 30-40 -> game is handled at the game boundary).
    if (wasBp && !nowBp && key !== "DEUCE" && !gamesChanged) {
      s.bpSavedThisGame += 1;
      s.gpi += W.bpSaved;
    }

    s.atBreakPoint = nowBp;
    s.lastKey = key;
  }

  s.gpi = Math.max(0, s.gpi);
  s.regime = s.breaksThisSet >= 2 ? "DOMINANCE"
    : s.breaksThisSet === 1 || s.gpi >= 8 ? "ELEVATED"
    : "NORMAL";

  memory.set(obs.matchId, s);
  return s;
}

/** Was the PREVIOUS state key a break point? Read back off the key. */
function isBp2(key: string): boolean {
  if (key === "AD-OUT") return true;
  const m = key.match(/^(\d+)-(\d+)$/);
  if (!m) return false;
  const idx: Record<string, number> = { "0": 0, "15": 1, "30": 2, "40": 3 };
  const s = idx[m[1]], r = idx[m[2]];
  return s !== undefined && r !== undefined && r >= 3 && r > s;
}

/* ── the signal ───────────────────────────────────────────────────────────── */

export type SignalTier = "S1_WATCH" | "S2_EARLY" | "S3_STRONG" | "S4_VERY_STRONG" | "S5_EXTREME" | "S6_REGIME" | "NONE";

export interface PressureSignal {
  tier: SignalTier;
  /** 0–100, for a meter. */
  score: number;
  /** One line, in plain English, naming the evidence rather than the maths. */
  headline: string;
  /** The sequence that produced it, so the number is auditable on screen. */
  evidence: string[];
  /** The model's probability the SERVER is broken this game. */
  breakProb: number;
  /** True when the model's own calibration says this number is trustworthy. */
  trustworthy: boolean;
  regime: PressureRegime;
}

/**
 * The honest band.
 *
 * Measured on 91 held-out matches: after isotonic calibration the game model's
 * predictions only survive between roughly 28% and 64%. Outside that range the
 * raw tree was overconfident by up to 40 points, so a number there is not a
 * probability, it is an artefact. Signals still fire — the SEQUENCE is real
 * evidence — but the probability is flagged untrustworthy rather than quietly
 * shown next to a market price as if it were tradeable.
 */
const TRUST_LO = 0.28, TRUST_HI = 0.64;

/**
 * Rank the current state on the entry hierarchy.
 *
 * Ordered by information content, not by how dramatic the event looks on a
 * scoreboard. A plain deuce is the most visible thing that happens in a game
 * and the least informative, so it sits at the bottom.
 */
export function pressureSignal(s: PressureState, pointWinProb: number, srvPts: number, retPts: number): PressureSignal {
  const tree = gameTree(pointWinProb, srvPts, retPts);
  const breakProb = tree.pReturner;
  const evidence: string[] = [];

  if (s.breaksThisSet) evidence.push(`${s.breaksThisSet} break${s.breaksThisSet > 1 ? "s" : ""} this set`);
  if (s.prevGameWasBreak) evidence.push("previous game was a break");
  if (s.bpThisGame) evidence.push(`${s.bpThisGame} break point${s.bpThisGame > 1 ? "s" : ""} this game`);
  if (s.bpSavedThisGame) evidence.push(`${s.bpSavedThisGame} saved`);
  if (s.deuceThisGame) evidence.push(`deuce ×${s.deuceThisGame}`);

  let tier: SignalTier = "NONE";
  let headline = "Nothing unusual yet";

  if (s.prevGameWasBreak && s.atBreakPoint) {
    tier = "S5_EXTREME";
    headline = "Break point straight after a break — the server is not settling";
  } else if (s.breaksThisSet >= 2) {
    tier = "S6_REGIME";
    headline = "Two breaks this set — wait for confirmation before backing another";
  } else if (s.prevGameWasBreak && (s.bpThisGame > 0 || s.deuceThisGame > 0)) {
    tier = "S4_VERY_STRONG";
    headline = "Pressure carrying over from the break in the last game";
  } else if (s.bpThisGame >= 2) {
    tier = "S3_STRONG";
    headline = `${s.bpThisGame} break points in this game — the pressure is not going away`;
  } else if (s.bpThisGame === 1 && s.deuceThisGame >= 1) {
    tier = "S2_EARLY";
    headline = "Break point out of deuce — worth watching, not yet worth backing";
  } else if (s.deuceThisGame >= 1 || s.atBreakPoint) {
    tier = "S1_WATCH";
    headline = s.atBreakPoint ? "Break point" : "Deuce — close game, no edge on its own";
  }

  return {
    tier,
    score: Math.round(Math.min(100, (s.gpi / GPI_FULL_SCALE) * 100)),
    headline,
    evidence,
    breakProb,
    trustworthy: breakProb >= TRUST_LO && breakProb <= TRUST_HI,
    regime: s.regime,
  };
}

/** Forget a match — call when it finishes, so the map does not grow forever. */
export function forgetPressure(matchId: string): void {
  memory.delete(matchId);
}

/** Test seam: wipe all memory. */
export function resetPressure(): void {
  memory.clear();
}

/** Read the current state without recording an observation. */
export function peekPressure(matchId: string): PressureState | undefined {
  return memory.get(matchId);
}
