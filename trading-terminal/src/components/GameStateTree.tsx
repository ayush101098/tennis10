"use client";

import { useMemo, useState } from "react";
import { buildGameStateGraph, outgoingEdges, positionDecision, type GameNode, type PositionAction } from "@/lib/gameStateGraph";
import { gameStateKey } from "@/lib/gameTree";
import type { ScheduledMatch } from "@/lib/scheduleService";
import { displayName } from "@/lib/scheduleService";

/**
 * The game-level decision tree: every legal score from 0-0 to game over, with
 * the server win / receiver win / deuce probability at each one.
 *
 * WHAT THIS IS BUILT FROM
 * lib/gameStateGraph enumerates the states and lib/gameTree supplies every
 * number — both already tested (18 + 13 tests) against the invariant that
 * matters most here: the four outcomes at every node sum to exactly 1, so
 * nothing shown below is a rounding artefact. Deuce, Ad-In and Ad-Out are ONE
 * node each, not one per visit — a literal tree never terminates once the
 * score can return to deuce, and the three states really are identical no
 * matter how many times the loop has repeated (see gameStateGraph's docs).
 *
 * WHAT IS DELIBERATELY NOT HERE
 * There is no per-state market price or edge. Nothing in this product's data
 * feed prices an individual game — Polymarket carries match and set markets
 * only (established this session, in the calibration work) — so a "Deuce
 * edge +15%" figure would be invented from nothing. Every node instead shows
 * only what the model actually knows: the three probabilities, and the
 * evidence when the node is the one actually being played right now.
 *
 * SCOPE, STATED PLAINLY
 * This is a compact, honest map you can click through, not a pannable/
 * zoomable canvas — the state space is ~20 nodes, small enough that a
 * scrollable flex layout shows the whole thing without needing a graph
 * library this app does not otherwise depend on.
 */

const NODE_ORDER: Record<string, number> = {
  "0-0": 0, "15-0": 1, "0-15": 2, "30-0": 3, "15-15": 4, "0-30": 5,
  "40-0": 6, "30-15": 7, "15-30": 8, "0-40": 9,
  "40-15": 10, "30-30": 11, "15-40": 12,
  "40-30": 13, "30-40": 14,
  "DEUCE": 15, "AD-IN": 16, "AD-OUT": 17,
  "GAME-SERVER": 18, "GAME-RETURNER": 19,
};

export default function GameStateTree({ match: m }: { match: ScheduledMatch }) {
  const ls = m.liveScore;
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  // A point-win rate to build the tree from. Prefers the server breakdown
  // when the feed has one; falls back to the pre-match prior converted the
  // same way computeLiveMatchProbNoServer does elsewhere, so this tree and
  // the rest of the panel are never disagreeing about the same match.
  const p = useMemo(() => {
    const b = ls?.breakHoldSignals?.serveBreakdown;
    if (b) {
      const f = b.firstServeIn / 100;
      if (f > 0 && f < 1) {
        const q = f * (b.firstServeWon / 100) + (1 - f) * (b.secondServeWon / 100);
        if (q > 0 && q < 1) return q;
      }
    }
    return 0.55 + (m.p1_win_prob - 0.5) * 0.3;
  }, [ls, m.p1_win_prob]);

  const graph = useMemo(() => buildGameStateGraph(p), [p]);

  // The LIVE current node, if this match has a readable point score.
  const liveKey = useMemo(() => {
    if (!ls?.pointScore) return null;
    const idx: Record<string, number> = { "0": 0, "15": 1, "30": 2, "40": 3, A: 4, AD: 4 };
    const srv = ls.server === 2 ? idx[String(ls.pointScore.p2).toUpperCase()] : idx[String(ls.pointScore.p1).toUpperCase()];
    const ret = ls.server === 2 ? idx[String(ls.pointScore.p1).toUpperCase()] : idx[String(ls.pointScore.p2).toUpperCase()];
    if (srv === undefined || ret === undefined) return null;
    return gameStateKey(srv, ret, !!ls.tiebreakScore);
  }, [ls]);

  const activeKey = selectedKey ?? liveKey;
  const active = activeKey ? graph.nodes.get(activeKey) : null;

  const serverName = ls?.server === 1 ? m.player1 : ls?.server === 2 ? m.player2 : null;
  const serverShort = serverName ? displayName(serverName) : "Server";
  const returnerShort = ls?.server === 1 ? displayName(m.player2) : ls?.server === 2 ? displayName(m.player1) : "Receiver";

  // Deuce loop drawn on its own, since it is a loop and the main flow reads
  // left to right — folding it into the same row would either draw an arrow
  // backward across the whole tree or hide that AD-IN/AD-OUT return to DEUCE.
  const mainRows = graph.byDepth
    .map(row => row.filter(n => n.kind !== "deuce" && n.kind !== "ad-in" && n.kind !== "ad-out"))
    .filter(row => row.length > 0);
  const deuceTrio = ["DEUCE", "AD-IN", "AD-OUT"]
    .map(k => graph.nodes.get(k))
    .filter((n): n is GameNode => !!n);

  return (
    <div className="rounded border border-terminal-border bg-terminal-bg/40 p-2.5 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-[8px] font-bold text-terminal-muted tracking-wider">
          GAME MAP{ls?.server ? ` — ${serverShort} SERVING` : ""}
        </div>
        {liveKey && (
          <button onClick={() => setSelectedKey(null)}
            className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
              selectedKey === null ? "bg-terminal-green/20 text-terminal-green" : "text-terminal-muted hover:text-slate-300"
            }`}>
            ● jump to live
          </button>
        )}
      </div>

      <p className="text-[9px] text-terminal-muted leading-relaxed">
        Every path this game can take, from 0-0. Upper branch on every node is{" "}
        <span className="text-terminal-green font-bold">{serverShort} wins the point</span>, lower is{" "}
        <span className="text-terminal-cyan font-bold">{returnerShort} wins the point</span>. Click any state to see its numbers.
      </p>

      {/* ── main flow, depth by depth ── */}
      <div className="overflow-x-auto">
        <div className="flex flex-col gap-1.5 min-w-max pb-1">
          {mainRows.map((row, i) => (
            <div key={i} className="flex gap-1.5">
              {[...row].sort((a, b) => (NODE_ORDER[a.key] ?? 99) - (NODE_ORDER[b.key] ?? 99)).map(n => (
                <Node key={n.key} node={n} active={n.key === activeKey} live={n.key === liveKey}
                  onClick={() => setSelectedKey(n.key)} />
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* ── the deuce loop, drawn separately since it is a loop ── */}
      <div>
        <div className="text-[8px] font-bold text-terminal-muted tracking-wider mb-1">DEUCE LOOP</div>
        <div className="flex items-center gap-2 flex-wrap">
          {deuceTrio.map((n, i) => (
            <div key={n.key} className="flex items-center gap-2">
              <Node node={n} active={n.key === activeKey} live={n.key === liveKey}
                onClick={() => setSelectedKey(n.key)} />
              {i < deuceTrio.length - 1 && <span className="text-terminal-muted text-xs">⇄</span>}
            </div>
          ))}
          <span className="text-[9px] text-terminal-muted ml-1">
            (loops back into itself — same three states, however many times it repeats)
          </span>
        </div>
      </div>

      {/* ── inspector for whichever node is active ── */}
      {active && <Inspector graph={graph} node={active} serverShort={serverShort} returnerShort={returnerShort}
        isLive={active.key === liveKey} />}
    </div>
  );
}

function Node({ node, active, live, onClick }: {
  node: GameNode; active: boolean; live: boolean; onClick: () => void;
}) {
  const terminal = node.kind === "game-server" || node.kind === "game-returner";
  const deuceish = node.kind === "deuce" || node.kind === "ad-in" || node.kind === "ad-out";
  return (
    <button onClick={onClick}
      className={`shrink-0 rounded border px-2 py-1 text-center transition ${
        active ? "border-terminal-cyan bg-terminal-cyan/15"
        : live ? "border-terminal-green/60 bg-terminal-green/10"
        : terminal ? "border-terminal-border bg-terminal-panel/40"
        : deuceish ? "border-terminal-yellow/40 bg-terminal-yellow/5"
        : "border-terminal-border bg-terminal-panel/20 hover:border-terminal-cyan/40"
      }`}>
      <div className="text-[10px] font-mono font-bold text-slate-200 whitespace-nowrap">
        {live && !active ? "● " : ""}{node.label}
      </div>
      {node.stats && (
        <div className="text-[8px] text-terminal-muted font-mono">{Math.round(node.stats.pServer * 100)}%</div>
      )}
    </button>
  );
}

function Inspector({ graph, node, serverShort, returnerShort, isLive }: {
  graph: ReturnType<typeof buildGameStateGraph>; node: GameNode; serverShort: string; returnerShort: string; isLive: boolean;
}) {
  const out = outgoingEdges(graph, node.key);
  const terminal = node.kind === "game-server" || node.kind === "game-returner";

  return (
    <div className="rounded border border-terminal-cyan/30 bg-terminal-cyan/[0.04] p-2.5 space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-[11px] font-bold text-slate-100">
          {node.label}{isLive && <span className="text-terminal-green ml-1.5">● live now</span>}
        </div>
      </div>

      {terminal ? (
        <div className="text-[10px] text-terminal-muted">
          {node.kind === "game-server" ? `${serverShort} wins the game.` : `${returnerShort} wins the game.`} No further points to play.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-2">
            <Stat label={`${serverShort.toUpperCase()} WINS GAME`} value={node.stats!.pServer} tone="green" />
            <Stat label={`${returnerShort.toUpperCase()} WINS GAME`} value={node.stats!.pReturner} tone="cyan" />
            <Stat label="REACHES DEUCE" value={node.stats!.pDeuce} tone="yellow" />
          </div>

          <div className="text-[9px] text-terminal-muted">
            From here, on the next point:
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {out.onServer && (
              <div className="rounded border border-terminal-border px-2 py-1">
                <div className="text-[8px] text-terminal-green font-bold">IF {serverShort.toUpperCase()} WINS THE POINT</div>
                <div className="text-[10px] text-slate-300">→ {graph.nodes.get(out.onServer.to)?.label}</div>
              </div>
            )}
            {out.onReturner && (
              <div className="rounded border border-terminal-border px-2 py-1">
                <div className="text-[8px] text-terminal-cyan font-bold">IF {returnerShort.toUpperCase()} WINS THE POINT</div>
                <div className="text-[10px] text-slate-300">→ {graph.nodes.get(out.onReturner.to)?.label}</div>
              </div>
            )}
          </div>

          {/* ── HOLD / HEDGE / STOP, for whichever side you backed ──
              Relative to that side's OWN probability at 0-0 — the "entry" —
              not a live market position, since none exists at game level (see
              the module docstring). Shown for BOTH sides at every node,
              because a state that says HOLD for the server's backer usually
              says STOP for the receiver's, and either could be who is asking. */}
          <div className="grid grid-cols-2 gap-1.5">
            <PositionRow label={serverShort} decision={positionDecision(node.stats!.pServer, graph.nodes.get("0-0")!.stats!.pServer)} />
            <PositionRow label={returnerShort} decision={positionDecision(node.stats!.pReturner, graph.nodes.get("0-0")!.stats!.pReturner)} />
          </div>

          {/* Deliberately no market/edge row: nothing prices an individual
              game, so a number here would be invented. Said plainly instead
              of just leaving a blank space that looks like it forgot to load. */}
          <div className="text-[8px] text-terminal-muted border-t border-terminal-border/60 pt-1.5">
            No market prices an individual game, so there is no edge to show here — this is the model&apos;s
            view only, not a trade.
          </div>
        </>
      )}
    </div>
  );
}

const ACTION_STYLE: Record<PositionAction, string> = {
  HOLD: "border-terminal-green/40 bg-terminal-green/5 text-terminal-green",
  HEDGE: "border-terminal-yellow/40 bg-terminal-yellow/5 text-terminal-yellow",
  STOP: "border-terminal-red/40 bg-terminal-red/5 text-terminal-red",
};

/** One side's HOLD / HEDGE / STOP read at the current node, if you backed them. */
function PositionRow({ label, decision }: { label: string; decision: ReturnType<typeof positionDecision> }) {
  return (
    <div className={`rounded border px-2 py-1 ${ACTION_STYLE[decision.action]}`}>
      <div className="flex items-center justify-between">
        <span className="text-[8px] font-bold opacity-80">IF YOU BACKED {label.toUpperCase()}</span>
        <span className="text-[10px] font-bold">{decision.action}</span>
      </div>
      <div className="text-[8px] text-terminal-muted mt-0.5 leading-snug">{decision.reason}</div>
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: number; tone: "green" | "cyan" | "yellow" }) {
  const cls = { green: "text-terminal-green", cyan: "text-terminal-cyan", yellow: "text-terminal-yellow" }[tone];
  return (
    <div className="text-center">
      <div className={`text-sm font-bold font-mono ${cls}`}>{Math.round(value * 100)}%</div>
      <div className="text-[7px] text-terminal-muted">{label}</div>
    </div>
  );
}
