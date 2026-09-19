/**
 * The joint game-state probability engine — PRD §13–§15.
 *
 * ONE recursion, four outcomes. Everything the terminal prices about a game
 * (winner, deuce, break, path) is read off a single joint distribution rather
 * than computed by a separate predictor per market. That is the PRD's core
 * engineering principle, and it is what stops "A Game", "Deuce" and
 * "A after Deuce" from being counted as three independent edges when they are
 * three views of one tennis outcome.
 *
 * WHY THE INVARIANT IS STRUCTURAL, NOT ASSERTED
 *   §15 requires P(A,D) + P(A,¬D) + P(B,D) + P(B,¬D) = 1, and §37 makes it an
 *   acceptance criterion. Rather than compute four numbers and check they sum
 *   to one, the recursion splits a single unit of probability at every node and
 *   never creates or destroys any. The invariant then holds by construction, to
 *   floating-point, and cannot be broken by a later edit that forgets to
 *   renormalise. The tests assert it anyway — as a tripwire on the arithmetic,
 *   not as the mechanism.
 *
 * RELATIONSHIP TO THE EXISTING CODE
 *   breakHoldEngine.gameWinProb already computes the winner leg of this exact
 *   recursion, and momentumEngine.gameWinProb already computes the three terms
 *   of the 0-0 decomposition (pre / deuce / winDeuce) before collapsing them to
 *   a scalar and discarding the split. This file keeps the split. It is
 *   deliberately a superset of both, so they can be retired onto it rather than
 *   maintained alongside it — two implementations of one tennis fact is how the
 *   numbers start disagreeing between the board and the backtester.
 */

/** Points as integers: 0/15/30/40 -> 0/1/2/3, advantage -> 4. */
export const POINT_INDEX: Record<string, number> = {
  "0": 0, "15": 1, "30": 2, "40": 3, A: 4, AD: 4, ADV: 4,
};

/** The reverse of POINT_INDEX, for display: 0/1/2/3 -> "0"/"15"/"30"/"40". */
export const SCORE_LABEL: Record<number, string> = { 0: "0", 1: "15", 2: "30", 3: "40" };

/**
 * Canonical, server-relative label for a game score: "0-0" .. "40-40", or
 * "DEUCE" / "AD-IN" / "AD-OUT" once both players have reached 40.
 *
 * Consolidated from THREE near-identical copies (breakHoldEngine's hedge
 * memory, gamePressure's state tracking, and this one) that had drifted into
 * existing side by side — the same two-implementations-of-one-fact risk this
 * file's own docstring warns about for gameWinProb. Every caller now imports
 * this one.
 */
export function gameStateKey(srvPts: number, retPts: number, isTiebreak = false): string {
  if (isTiebreak) return "TIEBREAK";
  if (srvPts >= 3 && retPts >= 3) {
    if (srvPts === retPts) return "DEUCE";
    return srvPts > retPts ? "AD-IN" : "AD-OUT";
  }
  return `${SCORE_LABEL[srvPts] ?? "40"}-${SCORE_LABEL[retPts] ?? "40"}`;
}

/**
 * The four terminal outcomes of a game, from the SERVER's point of view.
 *
 * "D" means the game reached deuce (40-40) at any point — including before the
 * state this was called from. A game standing at Ad-In has already reached
 * deuce, so its mass sits entirely in aD/bD: a "will this game go to deuce"
 * market is settled YES the moment 40-40 appears, and the engine has to agree
 * with the market's settlement rule or every edge measured against it is noise.
 */
export interface JointOutcomes {
  /** Server wins, game never reached deuce. */
  aNoD: number;
  /** Server wins, game passed through deuce. */
  aD: number;
  /** Returner wins (a break, on a service game), no deuce. */
  bNoD: number;
  /** Returner wins, game passed through deuce. */
  bD: number;
}

export interface GameTree extends JointOutcomes {
  /** P(server holds) = aNoD + aD. */
  pServer: number;
  /** P(returner wins the game) = bNoD + bD. On a service game this is the break. */
  pReturner: number;
  /** P(the game reaches — or has reached — deuce) = aD + bD. */
  pDeuce: number;
  /** Equal to pReturner. Named separately because the PRD prices it as its own market. */
  pBreak: number;
  /** The point-win probability this tree was built from, carried for provenance. */
  p: number;
}

/** Guard p away from the degenerate ends, where the recursion stops meaning anything. */
const clampP = (p: number) => Math.min(Math.max(p, 0.01), 0.99);

/**
 * P(server wins from deuce) — the classic closed form.
 *
 * From 40-40 the server must win two in a row before the returner does; the
 * geometric series collapses to p²/(p²+q²). This terminates the recursion,
 * which would otherwise not terminate at all: deuce can repeat forever.
 */
function fromDeuce(p: number): number {
  const q = 1 - p;
  return (p * p) / (p * p + q * q);
}

/**
 * The joint distribution over (winner × reached-deuce) from an arbitrary score.
 *
 * `srvPts`/`retPts` are point indices (see POINT_INDEX), NOT tennis score
 * strings — 30-40 is (2, 3).
 *
 * `seenDeuce` records whether the path that reached this state already passed
 * through 40-40. Callers pass it implicitly: any state with both players on 3+
 * IS at or past deuce, so it is derived rather than tracked, and a caller
 * dropping into the middle of a live game gets the same answer as one that
 * recursed there from 0-0.
 */
function jointFrom(p: number, srvPts: number, retPts: number): JointOutcomes {
  const q = 1 - p;

  // Terminal: the game is already decided. Whether deuce happened is settled by
  // whether both players ever reached 3 — at 5-3 in points (Ad converted) the
  // loser stood on 3, so deuce was reached.
  if (srvPts >= 4 && srvPts - retPts >= 2) {
    return retPts >= 3
      ? { aNoD: 0, aD: 1, bNoD: 0, bD: 0 }
      : { aNoD: 1, aD: 0, bNoD: 0, bD: 0 };
  }
  if (retPts >= 4 && retPts - srvPts >= 2) {
    return srvPts >= 3
      ? { aNoD: 0, aD: 0, bNoD: 0, bD: 1 }
      : { aNoD: 0, aD: 0, bNoD: 1, bD: 0 };
  }

  // At or past deuce: deuce has occurred, so ALL remaining mass is in the
  // "D" columns whatever happens next. Closed form rather than recursion,
  // because from here the state can cycle indefinitely.
  if (srvPts >= 3 && retPts >= 3) {
    const d = srvPts - retPts;
    const wd = fromDeuce(p);
    // d === 0 -> deuce; +1 -> advantage server; -1 -> advantage returner.
    const pA = d === 0 ? wd
      : d === 1 ? p + q * wd
      : p * wd;
    return { aNoD: 0, aD: pA, bNoD: 0, bD: 1 - pA };
  }

  // Ordinary play: split this node's probability down the two branches. Nothing
  // is created or lost here, which is what makes the §15 invariant structural.
  const win = jointFrom(p, srvPts + 1, retPts);
  const lose = jointFrom(p, srvPts, retPts + 1);
  return {
    aNoD: p * win.aNoD + q * lose.aNoD,
    aD: p * win.aD + q * lose.aD,
    bNoD: p * win.bNoD + q * lose.bNoD,
    bD: p * win.bD + q * lose.bD,
  };
}

/**
 * The full game tree from a live state. This is the object §41 calls the single
 * source of truth for the terminal, the API and the backtester.
 *
 * @param p       P(server wins the next point) — the one scalar the whole tree
 *                is a function of. Every probability below inherits its error,
 *                and the deuce leg amplifies it (20p³q³ is cubic in p), so the
 *                calibration of `p` matters more to the path markets than to
 *                the game winner.
 * @param srvPts  Server's points as an index: 0/15/30/40/A -> 0/1/2/3/4.
 * @param retPts  Returner's points, same encoding.
 */
export function gameTree(p: number, srvPts = 0, retPts = 0): GameTree {
  const pc = clampP(p);
  const j = jointFrom(pc, srvPts, retPts);
  const pServer = j.aNoD + j.aD;
  const pReturner = j.bNoD + j.bD;
  return {
    ...j,
    pServer,
    pReturner,
    pDeuce: j.aD + j.bD,
    pBreak: pReturner,
    p: pc,
  };
}

/** Parse a tennis score pair ("30", "40") into the index pair the tree wants. */
export function pointIndices(
  srv: string | number | undefined,
  ret: string | number | undefined,
): { srvPts: number; retPts: number } | null {
  const one = (v: string | number | undefined): number | null => {
    if (v === undefined || v === null) return null;
    const k = String(v).toUpperCase().trim();
    const i = POINT_INDEX[k];
    if (i !== undefined) return i;
    // Tiebreak scores arrive as plain integers and are NOT game points — the
    // caller must route those to a tiebreak model, so refuse rather than
    // silently treating "7" as an advantage.
    return null;
  };
  const s = one(srv), r = one(ret);
  if (s === null || r === null) return null;
  // Both on advantage, or advantage without the opponent on 40, is not a state
  // tennis can produce; it means the feed is mid-update.
  if (s === 4 && r === 4) return null;
  if (s === 4 && r !== 3) return null;
  if (r === 4 && s !== 3) return null;
  return { srvPts: s, retPts: r };
}

/**
 * The two states one point away, with the probability of each — PRD §19, and
 * the data behind the §33 path panel.
 *
 * Returned as trees rather than scores so the panel can show what each branch
 * is worth, which is the whole point of a path view: 40-30 and 30-40 are one
 * point apart and value the same position very differently.
 */
export function nextStates(p: number, srvPts: number, retPts: number): {
  onServerPoint: { prob: number; tree: GameTree };
  onReturnerPoint: { prob: number; tree: GameTree };
} {
  const pc = clampP(p);
  return {
    onServerPoint: { prob: pc, tree: gameTree(pc, srvPts + 1, retPts) },
    onReturnerPoint: { prob: 1 - pc, tree: gameTree(pc, srvPts, retPts + 1) },
  };
}

/**
 * Edge against a two-sided market, de-vigged — PRD §17–§18.
 *
 * Takes BOTH sides' raw implied probabilities because a one-sided price cannot
 * be de-vigged, and edge measured against a vigged price is overstated by
 * roughly half the overround on every single market. Returns null rather than
 * guessing when the pair is not a coherent two-way book.
 */
export function marketEdge(
  modelProb: number,
  rawImpliedA: number,
  rawImpliedB: number,
): { fairA: number; fairB: number; edge: number; overround: number } | null {
  if (!(rawImpliedA > 0) || !(rawImpliedB > 0)) return null;
  const total = rawImpliedA + rawImpliedB;
  // A two-way book prices at 100% plus vig. Far from that and this is not the
  // pair of sides it was assumed to be — most often one leg of a different market.
  if (total < 0.9 || total > 1.5) return null;
  const fairA = rawImpliedA / total;
  return { fairA, fairB: rawImpliedB / total, edge: modelProb - fairA, overround: total - 1 };
}
