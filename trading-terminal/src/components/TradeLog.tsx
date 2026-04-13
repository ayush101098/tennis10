"use client";

import type { TradeEntry, StatePerformanceRow } from "@/lib/types";
import clsx from "clsx";

/* ── Trade Log ── */
export function TradeLog({ trades }: { trades: TradeEntry[] }) {
  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-1.5 border-b border-terminal-border">
        <span className="text-[10px] font-bold text-terminal-muted uppercase tracking-wider">
          TRADE LOG
        </span>
      </div>
      <div className="flex-1 overflow-y-auto">
        <table className="w-full text-[10px]">
          <thead className="sticky top-0 bg-terminal-panel">
            <tr className="text-terminal-muted border-b border-terminal-border">
              <th className="text-left px-2 py-1">ID</th>
              <th className="px-2 py-1">ACCT</th>
              <th className="px-2 py-1">SIDE</th>
              <th className="px-2 py-1">ENTRY</th>
              <th className="px-2 py-1">CUR</th>
              <th className="px-2 py-1">STAKE</th>
              <th className="text-right px-2 py-1">PNL</th>
              <th className="px-2 py-1">STATE</th>
            </tr>
          </thead>
          <tbody>
            {trades
              .slice()
              .reverse()
              .map((t) => (
                <tr
                  key={t.id}
                  className="border-b border-terminal-border/30 hover:bg-terminal-panel/50"
                >
                  <td className="px-2 py-0.5 text-terminal-muted font-mono">{t.id}</td>
                  <td className="px-2 py-0.5 text-center">
                    <span
                      className={
                        t.account === "A"
                          ? "text-terminal-green"
                          : "text-terminal-cyan"
                      }
                    >
                      {t.account}
                    </span>
                  </td>
                  <td className="px-2 py-0.5 text-center">{t.side}</td>
                  <td className="px-2 py-0.5 text-center font-mono">
                    {t.entry_odds.toFixed(2)}
                  </td>
                  <td className="px-2 py-0.5 text-center font-mono">
                    {t.current_odds.toFixed(2)}
                  </td>
                  <td className="px-2 py-0.5 text-center">${t.stake.toFixed(0)}</td>
                  <td
                    className={clsx(
                      "px-2 py-0.5 text-right font-bold",
                      t.pnl >= 0 ? "text-terminal-green" : "text-terminal-red"
                    )}
                  >
                    {t.pnl >= 0 ? "+" : ""}
                    {t.pnl.toFixed(2)}
                  </td>
                  <td className="px-2 py-0.5 text-center text-terminal-muted">
                    {t.state_at_entry}
                  </td>
                </tr>
              ))}
            {trades.length === 0 && (
              <tr>
                <td colSpan={8} className="text-center py-3 text-terminal-muted">
                  No trades yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── State Performance Panel ── */
export function StatePerformance({ rows }: { rows: StatePerformanceRow[] }) {
  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-1.5 border-b border-terminal-border">
        <span className="text-[10px] font-bold text-terminal-muted uppercase tracking-wider">
          PERFORMANCE BY STATE
        </span>
      </div>
      <div className="flex-1 overflow-y-auto">
        <table className="w-full text-[10px]">
          <thead className="sticky top-0 bg-terminal-panel">
            <tr className="text-terminal-muted border-b border-terminal-border">
              <th className="text-left px-2 py-1">STATE</th>
              <th className="text-center px-2 py-1">TRADES</th>
              <th className="text-center px-2 py-1">WIN %</th>
              <th className="text-center px-2 py-1">AVG EV</th>
              <th className="text-right px-2 py-1">PNL</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.state}
                className="border-b border-terminal-border/30 hover:bg-terminal-panel/50"
              >
                <td className="px-2 py-0.5 font-bold">{r.state}</td>
                <td className="px-2 py-0.5 text-center">{r.trades}</td>
                <td className="px-2 py-0.5 text-center">
                  {(r.win_rate * 100).toFixed(0)}%
                </td>
                <td
                  className={clsx(
                    "px-2 py-0.5 text-center",
                    r.avg_ev >= 0 ? "text-terminal-green" : "text-terminal-red"
                  )}
                >
                  {(r.avg_ev * 100).toFixed(1)}%
                </td>
                <td
                  className={clsx(
                    "px-2 py-0.5 text-right font-bold",
                    r.total_pnl >= 0 ? "text-terminal-green" : "text-terminal-red"
                  )}
                >
                  ${r.total_pnl.toFixed(2)}
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={5} className="text-center py-3 text-terminal-muted">
                  No data
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
