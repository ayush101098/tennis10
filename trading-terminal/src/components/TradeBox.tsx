"use client";

import type { TradeBoxFrame, Action } from "@/lib/types";
import clsx from "clsx";

const ACTION_COLORS: Record<Action, string> = {
  ENTER: "bg-terminal-green/20 border-terminal-green text-terminal-green",
  SCALP: "bg-terminal-green/10 border-terminal-green/60 text-terminal-green",
  HOLD: "bg-terminal-yellow/20 border-terminal-yellow text-terminal-yellow",
  HEDGE: "bg-terminal-red/20 border-terminal-red text-terminal-red",
  EXIT: "bg-terminal-red/30 border-terminal-red text-terminal-red",
  EMERGENCY_EXIT: "bg-red-900/40 border-red-500 text-red-400",
  SKIP: "bg-terminal-panel border-terminal-border text-terminal-muted",
};

export default function TradeBox({ frame }: { frame: TradeBoxFrame }) {
  const { match, probability: prob, signal, hedge, position, next_action } = frame;
  const score = match.score;
  const serverName =
    match.info.server === 1 ? match.info.player1_name : match.info.player2_name;
  const receiverName =
    match.info.server === 1 ? match.info.player2_name : match.info.player1_name;

  return (
    <div className="flex flex-col gap-2 h-full">
      {/* ── MATCH STATE ── */}
      <Section title="MATCH STATE">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-[10px] text-terminal-muted uppercase">Sets</div>
            <div className="text-lg font-bold">
              {score.server_sets}–{score.receiver_sets}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-terminal-muted uppercase">Games</div>
            <div className="text-lg font-bold">
              {score.server_games}–{score.receiver_games}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-terminal-muted uppercase">Points</div>
            <div className="text-2xl font-bold text-terminal-cyan">
              {score.point_display || "0-0"}
            </div>
          </div>
        </div>
        <div className="flex justify-between mt-2 text-xs">
          <span>
            🎾 <span className="text-terminal-green">{serverName}</span>{" "}
            <span className="text-terminal-muted">serving</span>
          </span>
          <span className="text-terminal-muted">
            vs {receiverName}
          </span>
          {match.last_point_winner && (
            <span className="text-terminal-yellow text-[10px]">
              Last: {match.last_point_winner}
            </span>
          )}
        </div>
      </Section>

      {/* ── PROBABILITY ── */}
      <Section title="PROBABILITY">
        <div className="grid grid-cols-3 gap-2 text-center">
          <ProbCell label="TRUE P" value={prob.true_probability} fmt="pct" />
          <ProbCell label="MARKET P" value={prob.market_probability} fmt="pct" />
          <ProbCell
            label="EDGE"
            value={prob.edge_pct}
            fmt="pct"
            color={prob.edge_pct > 0 ? "text-terminal-green" : "text-terminal-red"}
          />
        </div>
        {/* edge bar */}
        <div className="mt-2 h-1.5 bg-terminal-border rounded-full overflow-hidden">
          <div
            className={clsx(
              "h-full rounded-full transition-all duration-300",
              prob.edge_pct > 0 ? "bg-terminal-green" : "bg-terminal-red"
            )}
            style={{ width: `${Math.min(Math.abs(prob.edge_pct) * 500, 100)}%` }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-terminal-muted mt-1">
          <span>EV: {(prob.ev * 100).toFixed(2)}%</span>
          <span>Conf: {prob.confidence}/100</span>
        </div>
      </Section>

      {/* ── DECISION ENGINE ── */}
      <Section title="DECISION ENGINE">
        <div
          className={clsx(
            "rounded-md border px-3 py-2 text-center font-bold text-lg signal-pulse",
            ACTION_COLORS[signal.action]
          )}
        >
          {signal.action}
        </div>
        <div className="flex justify-between text-xs mt-1">
          <span>Side: {signal.side}</span>
          <span>Size: {signal.bet_size_pct.toFixed(2)}%</span>
          <span>Conf: {signal.confidence}</span>
        </div>
        <div className="text-[10px] text-terminal-muted mt-1 truncate">
          {signal.reason}
        </div>
      </Section>

      {/* ── POSITION ── */}
      <Section title="POSITION">
        {position.open_trades.length === 0 ? (
          <div className="text-xs text-terminal-muted text-center py-1">FLAT</div>
        ) : (
          <>
            <div className="grid grid-cols-4 gap-2 text-center text-xs">
              <div>
                <div className="text-[10px] text-terminal-muted">Type</div>
                <div className="font-bold">{position.current_type}</div>
              </div>
              <div>
                <div className="text-[10px] text-terminal-muted">Entry</div>
                <div className="font-bold">{position.entry_odds.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-[10px] text-terminal-muted">Current</div>
                <div className="font-bold">{position.current_odds.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-[10px] text-terminal-muted">PnL</div>
                <div
                  className={clsx(
                    "font-bold",
                    position.pnl >= 0 ? "text-terminal-green" : "text-terminal-red"
                  )}
                >
                  {position.pnl >= 0 ? "+" : ""}
                  {position.pnl.toFixed(2)}
                </div>
              </div>
            </div>
          </>
        )}
      </Section>

      {/* ── NEXT ACTION ── */}
      <Section title="NEXT POINT →">
        <div className="grid grid-cols-2 gap-2 text-xs text-center">
          <div className="bg-terminal-green/10 rounded px-2 py-1.5 border border-terminal-green/30">
            <div className="text-[10px] text-terminal-muted">IF WIN</div>
            <div className="font-bold text-terminal-green">{next_action.if_point_won}</div>
          </div>
          <div className="bg-terminal-red/10 rounded px-2 py-1.5 border border-terminal-red/30">
            <div className="text-[10px] text-terminal-muted">IF LOSE</div>
            <div className="font-bold text-terminal-red">{next_action.if_point_lost}</div>
            {next_action.hedge_size_if_lost > 0 && (
              <div className="text-[10px] text-terminal-muted">
                hedge: ${next_action.hedge_size_if_lost.toFixed(0)}
              </div>
            )}
          </div>
        </div>
      </Section>

      {/* ── HEDGE ALERT ── */}
      {hedge.should_hedge && (
        <div className="rounded border border-terminal-red bg-terminal-red/10 px-3 py-2 text-xs">
          <div className="font-bold text-terminal-red">⚠ HEDGE SIGNAL</div>
          <div className="text-terminal-muted mt-0.5">{hedge.reason}</div>
          <div className="flex justify-between mt-1">
            <span>Size: ${hedge.hedge_size.toFixed(0)}</span>
            <span>Urgency: {hedge.urgency}</span>
            <span>Neutral PnL: ${hedge.expected_neutral_pnl.toFixed(2)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Helpers ── */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-terminal-panel border border-terminal-border rounded-md px-3 py-2">
      <div className="text-[10px] font-bold text-terminal-muted uppercase tracking-wider mb-1.5">
        {title}
      </div>
      {children}
    </div>
  );
}

function ProbCell({
  label,
  value,
  fmt,
  color,
}: {
  label: string;
  value: number;
  fmt: "pct" | "dec";
  color?: string;
}) {
  const display = fmt === "pct" ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);
  return (
    <div>
      <div className="text-[10px] text-terminal-muted">{label}</div>
      <div className={clsx("font-bold text-sm", color)}>{display}</div>
    </div>
  );
}
