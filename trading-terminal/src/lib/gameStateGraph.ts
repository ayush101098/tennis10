/**
 * The full game-state decision graph, built once from a single point-win
 * probability — every legal score from 0-0 to game over, with the exact
 * server/deuce/win probabilities at each one.
 *
 * WHY A GRAPH AND NOT A TREE
 * A literal tree — one node per point sequence — never terminates once the
 * score can return to deuce: 40-40, 40-40 again after two more points, and so
 * on forever. This is a GRAPH instead, on the same insight lib/gameTree
 * already relies on: DEUCE, AD-IN and AD-OUT are each a single reusable state,
 * not a new one per visit, because the probabilities from (3,3) and (5,5) are
 * identical (gameTree's own closed form depends only on d = srvPts - retPts
 * once both players have reached 3, never on the absolute count). So the
 * deuce loop is three nodes with edges back into themselves, exactly matching
 * how the game actually behaves, and the graph is finite and small — about
 * twenty states total, enumerated once by walking outward from 0-0.
 *
 * WHAT LIVES HERE
 * Pure state and probability — no React, no rendering, no live-match reading.
 * lib/gameTree supplies the numbers (already tested against the acceptance
 * criteria this session established); this file is only responsible for
 * knowing which states exist and how they connect.
 */

import { gameTree, gameStateKey, SCORE_LABEL, type GameTree } from "./gameTree";

export type NodeKind = "normal" | "deuce" | "ad-in" | "ad-out" | "game-server" | "game-returner";

export interface GameNode {
  key: string;              // canonical: "0-0" .. "40-40", "DEUCE", "AD-IN", "AD-OUT", "GAME-SERVER", "GAME-RETURNER"
  kind: NodeKind;
  /** A representative (srvPts, retPts) pair for this node's probabilities —
   *  for deuce/ad nodes this is the smallest pair with that shape (3,3 / 4,3 /
   *  3,4), since every deeper repetition of the loop computes identically. */
  srvPts: number;
  retPts: number;
  label: string;            // "30-30", "Deuce", "Ad-In", "Server wins", ...
  /** How many points into the game this node first becomes reachable —
   *  used purely for laying the graph out left to right, not a probability. */
  depth: number;
  stats: GameTree | null;   // null only for the two GAME-* terminal nodes
}

export interface GameEdge {
  from: string;
  to: string;
  /** Whose point sends the game from `from` to `to`. */
  on: "server" | "returner";
}

export interface GameStateGraph {
  p: number;
  nodes: Map<string, GameNode>;
  edges: GameEdge[];
  /** Nodes grouped by depth, in the order a left-to-right layout would want. */
  byDepth: GameNode[][];
}

function isTerminal(srvPts: number, retPts: number): "server" | "returner" | null {
  if (srvPts >= 4 && srvPts - retPts >= 2) return "server";
  if (retPts >= 4 && retPts - srvPts >= 2) return "returner";
  return null;
}

/** The next (srvPts, retPts) on a server point / returner point, BEFORE
 *  canonicalising into the deuce-loop representative. */
function rawNext(srvPts: number, retPts: number, on: "server" | "returner"): [number, number] {
  return on === "server" ? [srvPts + 1, retPts] : [srvPts, retPts + 1];
}

/** Collapse any (srvPts, retPts) at or past deuce onto its canonical
 *  representative pair, so the loop reuses one node instead of growing. */
function canonicalise(srvPts: number, retPts: number): [number, number] {
  if (srvPts >= 3 && retPts >= 3) {
    const d = srvPts - retPts;
    if (d === 0) return [3, 3];
    if (d > 0) return [4, 3];
    return [3, 4];
  }
  return [srvPts, retPts];
}

function nodeKind(srvPts: number, retPts: number, terminal: "server" | "returner" | null): NodeKind {
  if (terminal === "server") return "game-server";
  if (terminal === "returner") return "game-returner";
  const key = gameStateKey(srvPts, retPts);
  if (key === "DEUCE") return "deuce";
  if (key === "AD-IN") return "ad-in";
  if (key === "AD-OUT") return "ad-out";
  return "normal";
}

function nodeLabel(kind: NodeKind, srvPts: number, retPts: number): string {
  switch (kind) {
    case "deuce": return "Deuce";
    case "ad-in": return "Ad-In";
    case "ad-out": return "Ad-Out";
    case "game-server": return "Server wins";
    case "game-returner": return "Receiver wins";
    default: return `${SCORE_LABEL[srvPts] ?? srvPts}-${SCORE_LABEL[retPts] ?? retPts}`;
  }
}

/**
 * The canonical key AND representative (srvPts, retPts) for a raw score,
 * handling terminal states as a case of their own.
 *
 * This is where the two real bugs in the first version of this function lived.
 * Both `gameStateKey` and `canonicalise` were written for — and only ever
 * previously called on — a game still IN PROGRESS, where "both players have
 * reached 3" always means deuce/advantage. Neither was designed for a
 * DECIDED game: gameStateKey(5, 3) returns "AD-IN" (its `d>0` branch does not
 * check how large d is), and canonicalise(5, 3) collapses it onto (4, 3) —
 * both silently treating a server's WIN from Ad-In as if the game were still
 * at Ad-In. The three original call sites never triggered this, because none
 * of them ever fed in an already-terminal score; this graph builder is the
 * first caller that walks a state one point PAST where the game ends, so it
 * is the first place the gap actually mattered. Checking termination first,
 * before either function runs, is the fix.
 */
function resolve(srvPts: number, retPts: number): { key: string; terminal: "server" | "returner" | null; srvPts: number; retPts: number } {
  const terminal = isTerminal(srvPts, retPts);
  if (terminal === "server") return { key: "GAME-SERVER", terminal, srvPts, retPts };
  if (terminal === "returner") return { key: "GAME-RETURNER", terminal, srvPts, retPts };
  const [cs, cr] = canonicalise(srvPts, retPts);
  return { key: gameStateKey(cs, cr), terminal: null, srvPts: cs, retPts: cr };
}

/**
 * Build the whole graph for one point-win probability, by breadth-first
 * walking outward from 0-0. Small and cheap — about twenty states — so this
 * is meant to be called once per render and memoised on `p`, not optimised
 * further.
 */
export function buildGameStateGraph(p: number): GameStateGraph {
  const nodes = new Map<string, GameNode>();
  const edges: GameEdge[] = [];
  const byDepth: GameNode[][] = [];

  const queue: Array<{ srvPts: number; retPts: number; depth: number }> = [{ srvPts: 0, retPts: 0, depth: 0 }];
  const seen = new Set<string>();

  while (queue.length) {
    const { srvPts, retPts, depth } = queue.shift()!;
    const { key, terminal } = resolve(srvPts, retPts);
    if (seen.has(key)) continue;
    seen.add(key);

    const kind = nodeKind(srvPts, retPts, terminal);
    const stats = terminal ? null : gameTree(p, srvPts, retPts);

    const node: GameNode = { key, kind, srvPts, retPts, label: nodeLabel(kind, srvPts, retPts), depth, stats };
    nodes.set(key, node);
    (byDepth[depth] ??= []).push(node);

    if (terminal) continue; // no outgoing edges from a decided game

    for (const on of ["server", "returner"] as const) {
      const [rawS, rawR] = rawNext(srvPts, retPts, on);
      const next = resolve(rawS, rawR);
      edges.push({ from: key, to: next.key, on });
      if (!seen.has(next.key)) queue.push({ srvPts: next.srvPts, retPts: next.retPts, depth: depth + 1 });
    }
  }

  return { p, nodes, edges, byDepth };
}

/** The two edges leaving a node, if it has any (a terminal node has none). */
export function outgoingEdges(graph: GameStateGraph, key: string): { onServer?: GameEdge; onReturner?: GameEdge } {
  const out: { onServer?: GameEdge; onReturner?: GameEdge } = {};
  for (const e of graph.edges) {
    if (e.from !== key) continue;
    if (e.on === "server") out.onServer = e; else out.onReturner = e;
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  HOLD / HEDGE / STOP — a position-management read at every node
//
//  There is no per-game market (see this file's module docstring), so this is
//  NOT "your position's live P&L" the way the hedge engine in breakHoldEngine
//  is for a match-level bet against real odds. It answers a narrower, honest
//  question: for someone who backed a side when the game started at 0-0, has
//  this state moved FOR or AGAINST them enough to matter? That is derivable
//  from the model alone — it needs no price — so it is shown for both sides
//  at every node, not just the one the "favourite" happens to be.
// ═══════════════════════════════════════════════════════════════════════════

export type PositionAction = "HOLD" | "HEDGE" | "STOP";

export interface PositionDecision {
  action: PositionAction;
  /** Percentage points moved since 0-0, signed: negative is against this side. */
  deltaPp: number;
  reason: string;
}

/** How far the model can move against a side before it stops being "hold and
 *  wait" — same two-tier shape as the live hedge engine's LOW/HIGH/IMMEDIATE
 *  urgency, expressed in probability points instead of an odds move, since a
 *  game has no odds to move. */
const HEDGE_AT_PP = 12;
const STOP_AT_PP = 25;
/** Below this win probability, continuing to hold is priced against you
 *  regardless of how you got here — the AD-OUT case: one point from broken. */
const STOP_FLOOR = 0.20;

/**
 * Position read for one side (the server's backer, or the receiver's) at one
 * node, relative to the SAME side's probability at 0-0.
 *
 * @param pNow  This side's win-the-game probability at the node in question.
 * @param pAt00 The same side's win-the-game probability at 0-0 — the "entry".
 */
export function positionDecision(pNow: number, pAt00: number): PositionDecision {
  const deltaPp = Math.round((pNow - pAt00) * 100);

  if (pNow <= STOP_FLOOR) {
    return { action: "STOP", deltaPp, reason: `Down to ${Math.round(pNow * 100)}% — one bad point from losing the game outright.` };
  }
  if (deltaPp <= -STOP_AT_PP) {
    return { action: "STOP", deltaPp, reason: `${Math.abs(deltaPp)} points worse than at 0-0 — the game has turned.` };
  }
  if (deltaPp <= -HEDGE_AT_PP) {
    return { action: "HEDGE", deltaPp, reason: `${Math.abs(deltaPp)} points worse than at 0-0 — worth covering some of it.` };
  }
  return { action: "HOLD", deltaPp, reason: deltaPp >= 0 ? "At or ahead of where this side started." : "Within normal swing of the score — nothing has actually turned yet." };
}
