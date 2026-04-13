"use client";

import type { TradeBoxFrame } from "@/lib/types";
import clsx from "clsx";

/**
 * Odds Ladder — shows live odds movement and the current book depth.
 * Simulates a Betfair-style ladder.
 */
export default function OddsLadder({ frame }: { frame: TradeBoxFrame }) {
  const { probability: prob, position } = frame;
  const marketOdds = 1 / prob.market_probability;
  const trueOdds = 1 / prob.true_probability;

  // Build a small ladder around the current market odds
  const ticks = generateLadder(marketOdds, 8);

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-2 border-b border-terminal-border">
        <span className="text-[10px] font-bold text-terminal-muted uppercase tracking-wider">
          ODDS LADDER
        </span>
      </div>
      <div className="flex-1 overflow-y-auto">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-terminal-muted">
              <th className="text-left px-2 py-1">BACK</th>
              <th className="text-center px-2 py-1">ODDS</th>
              <th className="text-right px-2 py-1">LAY</th>
            </tr>
          </thead>
          <tbody>
            {ticks.map((tick) => {
              const isMarket = Math.abs(tick - marketOdds) < 0.03;
              const isTrue = Math.abs(tick - trueOdds) < 0.03;
              const isEntry = position.open_trades.some(
                (t) => Math.abs(t.entry_odds - tick) < 0.03
              );
              return (
                <tr
                  key={tick}
                  className={clsx(
                    "border-b border-terminal-border/50 transition-colors",
                    isMarket && "bg-terminal-blue/10",
                    isTrue && "bg-terminal-green/5",
                    isEntry && "bg-terminal-yellow/10"
                  )}
                >
                  <td className="px-2 py-0.5">
                    <div
                      className="bg-terminal-blue/20 text-terminal-blue rounded text-center"
                      style={{ width: `${Math.max(20, 100 - (tick - marketOdds) * 30)}%` }}
                    >
                      {tick < marketOdds ? "▓" : "░"}
                    </div>
                  </td>
                  <td className="text-center px-2 py-0.5 font-mono font-bold">
                    <span
                      className={clsx(
                        isMarket && "text-terminal-blue",
                        isTrue && "text-terminal-green",
                        isEntry && "text-terminal-yellow"
                      )}
                    >
                      {tick.toFixed(2)}
                    </span>
                    {isMarket && <span className="ml-1 text-[8px] text-terminal-blue">MKT</span>}
                    {isTrue && <span className="ml-1 text-[8px] text-terminal-green">TRUE</span>}
                    {isEntry && <span className="ml-1 text-[8px] text-terminal-yellow">ENT</span>}
                  </td>
                  <td className="px-2 py-0.5 text-right">
                    <div
                      className="bg-terminal-red/20 text-terminal-red rounded text-center ml-auto"
                      style={{ width: `${Math.max(20, 100 - (marketOdds - tick) * 30)}%` }}
                    >
                      {tick > marketOdds ? "▓" : "░"}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function generateLadder(center: number, halfRange: number): number[] {
  const step = 0.02;
  const ticks: number[] = [];
  for (let i = -halfRange; i <= halfRange; i++) {
    ticks.push(Math.round((center + i * step * 5) * 100) / 100);
  }
  return ticks.filter((t) => t >= 1.01).sort((a, b) => a - b);
}
