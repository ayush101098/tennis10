"use client";

import { useState } from "react";
import { sendPoint, sendOdds } from "@/lib/api";
import type { PointStatsInput } from "@/lib/types";
import clsx from "clsx";

/**
 * Point-by-point input bar with live stats toggles.
 * Sits at the very top of the CENTER panel.
 */
export default function PointInput({ matchId }: { matchId: string }) {
  const [oddsServer, setOddsServer] = useState(1.8);
  const [oddsReceiver, setOddsReceiver] = useState(2.1);
  const [breakOdds, setBreakOdds] = useState(3.0);
  const [holdOdds, setHoldOdds] = useState(1.4);

  // Stat toggles (reset after each point)
  const [isAce, setIsAce] = useState(false);
  const [isDf, setIsDf] = useState(false);
  const [isFirstIn, setIsFirstIn] = useState(true);
  const [isWinner, setIsWinner] = useState(false);
  const [isUe, setIsUe] = useState(false);
  const [isBp, setIsBp] = useState(false);

  const resetStats = () => {
    setIsAce(false);
    setIsDf(false);
    setIsFirstIn(true);
    setIsWinner(false);
    setIsUe(false);
    setIsBp(false);
  };

  const fire = async (winner: "SERVER" | "RECEIVER") => {
    const stats: PointStatsInput = {
      is_ace: isAce,
      is_double_fault: isDf,
      is_first_serve_in: isFirstIn,
      is_winner: isWinner,
      is_unforced_error: isUe,
      is_break_point: isBp,
    };
    await sendPoint(matchId, winner, oddsServer, oddsReceiver, stats);
    resetStats();
  };

  const pushOdds = async () => {
    await sendOdds(matchId, breakOdds, holdOdds);
  };

  return (
    <div className="bg-terminal-panel border border-terminal-border rounded-md px-3 py-1.5 space-y-1.5">
      {/* Row 1: Point buttons + odds */}
      <div className="flex items-center gap-3 flex-wrap">
        <button
          onClick={() => fire("SERVER")}
          className="bg-terminal-green/20 text-terminal-green border border-terminal-green/40 rounded px-3 py-1 text-xs font-bold hover:bg-terminal-green/30 active:scale-95 transition"
        >
          🎾 SERVER WINS
        </button>
        <button
          onClick={() => fire("RECEIVER")}
          className="bg-terminal-red/20 text-terminal-red border border-terminal-red/40 rounded px-3 py-1 text-xs font-bold hover:bg-terminal-red/30 active:scale-95 transition"
        >
          RECEIVER WINS
        </button>

        <div className="w-px h-6 bg-terminal-border" />

        {/* Odds inputs */}
        <div className="flex items-center gap-1.5 text-[10px]">
          <span className="text-terminal-muted">SRV</span>
          <input
            type="number"
            step="0.01"
            value={oddsServer}
            onChange={(e) => setOddsServer(Number(e.target.value))}
            className="w-14 bg-terminal-bg border border-terminal-border rounded px-1.5 py-0.5 text-xs text-slate-200 text-center"
          />
          <span className="text-terminal-muted">RCV</span>
          <input
            type="number"
            step="0.01"
            value={oddsReceiver}
            onChange={(e) => setOddsReceiver(Number(e.target.value))}
            className="w-14 bg-terminal-bg border border-terminal-border rounded px-1.5 py-0.5 text-xs text-slate-200 text-center"
          />
        </div>

        <div className="w-px h-6 bg-terminal-border" />

        {/* Break/Hold odds */}
        <div className="flex items-center gap-1.5 text-[10px]">
          <span className="text-terminal-muted">BRK</span>
          <input
            type="number"
            step="0.01"
            value={breakOdds}
            onChange={(e) => setBreakOdds(Number(e.target.value))}
            className="w-14 bg-terminal-bg border border-terminal-border rounded px-1.5 py-0.5 text-xs text-slate-200 text-center"
          />
          <span className="text-terminal-muted">HLD</span>
          <input
            type="number"
            step="0.01"
            value={holdOdds}
            onChange={(e) => setHoldOdds(Number(e.target.value))}
            className="w-14 bg-terminal-bg border border-terminal-border rounded px-1.5 py-0.5 text-xs text-slate-200 text-center"
          />
          <button
            onClick={pushOdds}
            className="bg-terminal-blue/20 text-terminal-blue border border-terminal-blue/40 rounded px-2 py-0.5 text-[10px] hover:bg-terminal-blue/30"
          >
            PUSH
          </button>
        </div>

        <div className="ml-auto text-[9px] text-terminal-muted hidden xl:block">
          S = server &nbsp; R = receiver
        </div>
      </div>

      {/* Row 2: Stat toggles */}
      <div className="flex items-center gap-1.5">
        <span className="text-[9px] text-terminal-muted uppercase mr-1">Stats:</span>
        <Toggle label="1st In" active={isFirstIn} onChange={setIsFirstIn} />
        <Toggle
          label="ACE"
          active={isAce}
          onChange={(v) => {
            setIsAce(v);
            if (v) { setIsFirstIn(true); setIsDf(false); }
          }}
          color="green"
        />
        <Toggle
          label="DF"
          active={isDf}
          onChange={(v) => {
            setIsDf(v);
            if (v) { setIsFirstIn(false); setIsAce(false); }
          }}
          color="red"
        />
        <Toggle label="WINNER" active={isWinner} onChange={setIsWinner} color="green" />
        <Toggle label="UE" active={isUe} onChange={setIsUe} color="red" />
        <Toggle label="BP" active={isBp} onChange={setIsBp} color="yellow" />
      </div>
    </div>
  );
}

function Toggle({
  label,
  active,
  onChange,
  color = "cyan",
}: {
  label: string;
  active: boolean;
  onChange: (v: boolean) => void;
  color?: "cyan" | "green" | "red" | "yellow";
}) {
  const colors = {
    cyan: "bg-terminal-cyan/20 border-terminal-cyan/50 text-terminal-cyan",
    green: "bg-terminal-green/20 border-terminal-green/50 text-terminal-green",
    red: "bg-terminal-red/20 border-terminal-red/50 text-terminal-red",
    yellow: "bg-terminal-yellow/20 border-terminal-yellow/50 text-terminal-yellow",
  };

  return (
    <button
      onClick={() => onChange(!active)}
      className={clsx(
        "rounded px-2 py-0.5 text-[10px] font-bold border transition",
        active ? colors[color] : "bg-transparent border-terminal-border text-terminal-muted"
      )}
    >
      {label}
    </button>
  );
}
