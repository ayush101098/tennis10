"use client";

import type { TradeBoxFrame } from "@/lib/types";
import clsx from "clsx";

/**
 * Position Tracker + Dual Account balances + Risk panel.
 */
export default function PositionTracker({ frame }: { frame: TradeBoxFrame }) {
  const { position, risk, deuce_loop, execution } = frame;

  return (
    <div className="flex flex-col gap-2 h-full overflow-y-auto px-2 py-2">
      {/* ── Dual Account ── */}
      <Panel title="DUAL ACCOUNTS">
        <div className="grid grid-cols-2 gap-2 text-xs text-center">
          <div className="bg-terminal-green/5 rounded px-2 py-1 border border-terminal-green/20">
            <div className="text-[9px] text-terminal-muted">ACCOUNT A (Entries)</div>
            <div className="font-bold text-terminal-green">
              ${position.account_a_balance.toFixed(0)}
            </div>
          </div>
          <div className="bg-terminal-red/5 rounded px-2 py-1 border border-terminal-red/20">
            <div className="text-[9px] text-terminal-muted">ACCOUNT B (Hedges)</div>
            <div className="font-bold text-terminal-cyan">
              ${position.account_b_balance.toFixed(0)}
            </div>
          </div>
        </div>
        <div className="text-[10px] text-terminal-muted text-center mt-1">
          Net Exposure: ${position.combined_exposure.toFixed(0)}
        </div>
      </Panel>

      {/* ── Risk ── */}
      <Panel title="RISK">
        <div className="flex items-center justify-between text-xs">
          <span className="text-terminal-muted">Level</span>
          <span
            className={clsx(
              "font-bold px-1.5 py-0.5 rounded text-[10px]",
              risk.risk_level === "LOW" && "bg-terminal-green/20 text-terminal-green",
              risk.risk_level === "MEDIUM" && "bg-terminal-yellow/20 text-terminal-yellow",
              risk.risk_level === "HIGH" && "bg-terminal-red/20 text-terminal-red",
              risk.risk_level === "CRITICAL" && "bg-red-900/40 text-red-400"
            )}
          >
            {risk.risk_level}
          </span>
        </div>
        <div className="flex items-center justify-between text-xs mt-1">
          <span className="text-terminal-muted">Exposure</span>
          <span>{risk.current_exposure_pct.toFixed(1)}%</span>
        </div>
        <div className="flex items-center justify-between text-xs mt-1">
          <span className="text-terminal-muted">Consec. Losses</span>
          <span className={risk.consecutive_losses >= 2 ? "text-terminal-red" : ""}>
            {risk.consecutive_losses}
          </span>
        </div>
        <div className="flex items-center justify-between text-xs mt-1">
          <span className="text-terminal-muted">Trading</span>
          <span className={risk.is_trading_enabled ? "text-terminal-green" : "text-terminal-red"}>
            {risk.is_trading_enabled ? "ENABLED" : "DISABLED"}
          </span>
        </div>
        {risk.stop_reason && (
          <div className="text-[10px] text-terminal-red mt-1">⛔ {risk.stop_reason}</div>
        )}
        {/* Exposure bar */}
        <div className="mt-1.5 h-1 bg-terminal-border rounded-full overflow-hidden">
          <div
            className={clsx(
              "h-full rounded-full transition-all",
              risk.current_exposure_pct < 4 && "bg-terminal-green",
              risk.current_exposure_pct >= 4 && risk.current_exposure_pct < 7 && "bg-terminal-yellow",
              risk.current_exposure_pct >= 7 && "bg-terminal-red"
            )}
            style={{ width: `${Math.min(risk.current_exposure_pct * 10, 100)}%` }}
          />
        </div>
      </Panel>

      {/* ── Deuce Loop ── */}
      <Panel title="DEUCE LOOP">
        <div className="flex items-center justify-between text-xs">
          <span className="text-terminal-muted">Active</span>
          <span className={deuce_loop.is_active ? "text-terminal-green" : "text-terminal-muted"}>
            {deuce_loop.is_active ? "YES" : "NO"}
          </span>
        </div>
        <div className="flex items-center justify-between text-xs mt-1">
          <span className="text-terminal-muted">Cycles</span>
          <span>
            {deuce_loop.cycle_count}/{deuce_loop.max_cycles}
          </span>
        </div>
        <div className="flex items-center justify-between text-xs mt-1">
          <span className="text-terminal-muted">Net Profit</span>
          <span
            className={clsx(
              "font-bold",
              deuce_loop.net_profit >= 0 ? "text-terminal-green" : "text-terminal-red"
            )}
          >
            ${deuce_loop.net_profit.toFixed(2)}
          </span>
        </div>
      </Panel>

      {/* ── Last Execution ── */}
      {execution && (
        <Panel title="LAST EXECUTION">
          <div className="grid grid-cols-2 gap-1 text-[10px]">
            <span className="text-terminal-muted">Requested</span>
            <span className="text-right">{execution.requested_price.toFixed(2)}</span>
            <span className="text-terminal-muted">Filled</span>
            <span
              className={clsx(
                "text-right font-bold",
                execution.filled ? "text-terminal-green" : "text-terminal-red"
              )}
            >
              {execution.filled ? execution.fill_price.toFixed(2) : "MISS"}
            </span>
            <span className="text-terminal-muted">Slippage</span>
            <span className="text-right">{execution.slippage_ticks} ticks</span>
            <span className="text-terminal-muted">Delay</span>
            <span className="text-right">{execution.delay_ms}ms</span>
          </div>
        </Panel>
      )}
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-terminal-panel border border-terminal-border rounded-md px-3 py-2">
      <div className="text-[10px] font-bold text-terminal-muted uppercase tracking-wider mb-1.5">
        {title}
      </div>
      {children}
    </div>
  );
}
