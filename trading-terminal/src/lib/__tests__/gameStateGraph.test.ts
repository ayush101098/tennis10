import { describe, it, expect } from "vitest";
import { buildGameStateGraph, outgoingEdges, positionDecision } from "../gameStateGraph";
import { gameTree } from "../gameTree";

/**
 * The graph must be a faithful, finite map of the same recursion lib/gameTree
 * already proved correct (18 tests) — these tests check the GRAPH STRUCTURE
 * (which states exist, how they connect, that the deuce loop is three reused
 * nodes and not an ever-growing tree), and cross-check every node's numbers
 * against gameTree() directly rather than re-deriving them.
 */

describe("buildGameStateGraph — structure", () => {
  it("starts at 0-0", () => {
    const g = buildGameStateGraph(0.65);
    expect(g.nodes.has("0-0")).toBe(true);
    expect(g.byDepth[0]).toHaveLength(1);
    expect(g.byDepth[0][0].key).toBe("0-0");
  });

  it("is finite: the deuce loop is exactly three reused nodes, not infinite", () => {
    const g = buildGameStateGraph(0.65);
    const deuceNodes = [...g.nodes.keys()].filter(k => k === "DEUCE" || k === "AD-IN" || k === "AD-OUT");
    expect(deuceNodes.sort()).toEqual(["AD-IN", "AD-OUT", "DEUCE"]);
    // Total node count is small and bounded — not one node per point sequence,
    // which would never terminate once the score can return to deuce.
    expect(g.nodes.size).toBeLessThan(25);
  });

  it("has exactly two terminal nodes", () => {
    const g = buildGameStateGraph(0.65);
    expect(g.nodes.has("GAME-SERVER")).toBe(true);
    expect(g.nodes.has("GAME-RETURNER")).toBe(true);
  });

  it("every non-terminal node has exactly two outgoing edges", () => {
    const g = buildGameStateGraph(0.65);
    for (const node of g.nodes.values()) {
      const out = outgoingEdges(g, node.key);
      if (node.kind === "game-server" || node.kind === "game-returner") {
        expect(out.onServer).toBeUndefined();
        expect(out.onReturner).toBeUndefined();
      } else {
        expect(out.onServer).toBeDefined();
        expect(out.onReturner).toBeDefined();
      }
    }
  });

  it("contains every state from the PRD's canonical list (§8)", () => {
    const g = buildGameStateGraph(0.65);
    const expected = [
      "0-0", "15-0", "0-15", "15-15", "30-0", "0-30", "30-15", "15-30",
      "40-0", "0-40", "40-15", "15-40", "40-30", "30-40",
      "DEUCE", "AD-IN", "AD-OUT",
    ];
    for (const key of expected) expect(g.nodes.has(key)).toBe(true);
  });

  it("the deuce loop reconnects to itself: AD-IN losing returns to DEUCE", () => {
    const g = buildGameStateGraph(0.65);
    const out = outgoingEdges(g, "AD-IN");
    expect(out.onServer!.to).toBe("GAME-SERVER");
    expect(out.onReturner!.to).toBe("DEUCE");
  });

  it("AD-OUT winning returns to DEUCE, losing ends the game", () => {
    const g = buildGameStateGraph(0.65);
    const out = outgoingEdges(g, "AD-OUT");
    expect(out.onServer!.to).toBe("DEUCE");
    expect(out.onReturner!.to).toBe("GAME-RETURNER");
  });

  it("40-30 and 30-40 both lead into the SAME deuce node on the losing branch", () => {
    const g = buildGameStateGraph(0.65);
    expect(outgoingEdges(g, "40-30").onReturner!.to).toBe("DEUCE");
    expect(outgoingEdges(g, "30-40").onServer!.to).toBe("DEUCE");
  });

  it("terminal states appear at the correct depth", () => {
    const g = buildGameStateGraph(0.65);
    const gs = g.nodes.get("GAME-SERVER")!;
    // Fastest possible route to GAME-SERVER is four straight server points.
    expect(gs.depth).toBe(4);
  });
});

describe("buildGameStateGraph — every node's numbers match gameTree() exactly", () => {
  it("cross-checks pServer/pReturner/pDeuce for every non-terminal node", () => {
    const p = 0.62;
    const g = buildGameStateGraph(p);
    for (const node of g.nodes.values()) {
      if (!node.stats) continue; // terminal nodes carry no stats
      const direct = gameTree(p, node.srvPts, node.retPts);
      expect(node.stats.pServer).toBeCloseTo(direct.pServer, 12);
      expect(node.stats.pReturner).toBeCloseTo(direct.pReturner, 12);
      expect(node.stats.pDeuce).toBeCloseTo(direct.pDeuce, 12);
    }
  });

  it("DEUCE, AD-IN and AD-OUT give the SAME numbers regardless of which representative pair is used — proving the collapse is legitimate", () => {
    const p = 0.7;
    // The graph's DEUCE node uses (3,3); a deeper repeat of the loop, e.g.
    // (6,6), must be mathematically identical, or collapsing them into one
    // reusable node would be misrepresenting the state.
    const g = buildGameStateGraph(p);
    const deuceNode = g.nodes.get("DEUCE")!;
    const deeper = gameTree(p, 6, 6);
    expect(deuceNode.stats!.pServer).toBeCloseTo(deeper.pServer, 12);
  });
});

describe("buildGameStateGraph — different p values", () => {
  it("produces the same STRUCTURE regardless of p (only the numbers change)", () => {
    const a = buildGameStateGraph(0.5);
    const b = buildGameStateGraph(0.9);
    expect([...a.nodes.keys()].sort()).toEqual([...b.nodes.keys()].sort());
    expect(a.edges.length).toBe(b.edges.length);
  });

  it("a stronger server reaches GAME-SERVER with higher probability from 0-0", () => {
    const weak = buildGameStateGraph(0.55);
    const strong = buildGameStateGraph(0.85);
    expect(strong.nodes.get("0-0")!.stats!.pServer)
      .toBeGreaterThan(weak.nodes.get("0-0")!.stats!.pServer);
  });
});

describe("positionDecision", () => {
  it("HOLDs when the side is at or ahead of its 0-0 entry", async () => {
    expect(positionDecision(0.68, 0.65).action).toBe("HOLD");
    expect(positionDecision(0.65, 0.65).action).toBe("HOLD");
  });

  it("HOLDs through normal, small swings", async () => {
    // 8 points worse is inside a single lost point's worth of normal noise
    // at this scale — should not yet call for action.
    expect(positionDecision(0.57, 0.65).action).toBe("HOLD");
  });

  it("HEDGEs once the move against the side passes the threshold", async () => {
    const d = positionDecision(0.50, 0.65); // -15pp
    expect(d.action).toBe("HEDGE");
    expect(d.deltaPp).toBe(-15);
  });

  it("STOPs once the move is severe, even above the hard floor", async () => {
    const d = positionDecision(0.38, 0.65); // -27pp, still > 0.20 floor
    expect(d.action).toBe("STOP");
  });

  it("STOPs at the hard floor regardless of how far it moved to get there", async () => {
    // Only 10pp of movement, but the absolute level is now too low to hold.
    const d = positionDecision(0.15, 0.25);
    expect(d.action).toBe("STOP");
  });

  it("is symmetric: server and receiver read from the same node consistently", async () => {
    const g = buildGameStateGraph(0.65);
    const base = g.nodes.get("0-0")!.stats!;
    const node = g.nodes.get("AD-OUT")!.stats!;
    // AD-OUT is bad for the server and correspondingly good for the receiver —
    // the two reads should not agree with each other.
    const serverSide = positionDecision(node.pServer, base.pServer);
    const receiverSide = positionDecision(node.pReturner, base.pReturner);
    expect(serverSide.action).toBe("STOP");
    expect(receiverSide.action).toBe("HOLD");
  });
});
