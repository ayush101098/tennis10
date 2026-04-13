"use client";

import { useState, useEffect, useCallback } from "react";
import { useTradeStream } from "@/hooks/useTradeStream";
import { fetchMatches, sendPoint, setupMatch } from "@/lib/api";
import type { MatchListItem, TradeBoxFrame } from "@/lib/types";

import MatchList from "@/components/MatchList";
import LiveMatchPanel from "@/components/LiveMatchPanel";
import SchedulePanel from "@/components/SchedulePanel";
import TradeBox from "@/components/TradeBox";
import OddsLadder from "@/components/OddsLadder";
import PositionTracker from "@/components/PositionTracker";
import PointInput from "@/components/PointInput";
import LiveStatsPanel from "@/components/LiveStatsPanel";
import EnsemblePanel from "@/components/EnsemblePanel";
import { TradeLog, StatePerformance } from "@/components/TradeLog";

export default function TerminalPage() {
  const [matches, setMatches] = useState<MatchListItem[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [leftTab, setLeftTab] = useState<"matches" | "live" | "schedule">("schedule");
  const { frame, connected } = useTradeStream(activeId);

  // Auto-setup from schedule panel
  const handleScheduleSetup = useCallback(
    async (player1: string, player2: string, tournament: string, surface: string, bestOf: number) => {
      try {
        const res = await setupMatch({
          player1_name: player1,
          player2_name: player2,
          surface,
          best_of: bestOf,
          tournament,
          tournament_level: "",
          initial_server: 1,
          p1_serve_pct: 63,
          p2_serve_pct: 63,
          p1_return_pct: 35,
          p2_return_pct: 35,
          p1_rank: 50,
          p2_rank: 50,
          p1_ranking_points: 1000,
          p2_ranking_points: 1000,
          p1_first_serve_pct: 62,
          p2_first_serve_pct: 62,
          p1_first_serve_win_pct: 72,
          p2_first_serve_win_pct: 72,
          p1_second_serve_win_pct: 52,
          p2_second_serve_win_pct: 52,
          p1_bp_save_pct: 62,
          p2_bp_save_pct: 62,
          p1_win_rate: 50,
          p2_win_rate: 50,
          bankroll: 10000,
        });
        if (res.match_id) {
          setActiveId(res.match_id);
          setLeftTab("matches");
          fetchMatches().then(setMatches).catch(() => {});
        }
      } catch { /* ignore */ }
    },
    []
  );

  // Poll match list
  useEffect(() => {
    const load = async () => {
      try {
        const list = await fetchMatches();
        setMatches(list);
      } catch {
        /* server not up yet */
      }
    };
    load();
    const iv = setInterval(load, 5000);
    return () => clearInterval(iv);
  }, []);

  // Keyboard shortcuts: S = server point, R = receiver point
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!activeId) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.key.toLowerCase() === "s") sendPoint(activeId, "SERVER", 1.8, 2.1);
      if (e.key.toLowerCase() === "r") sendPoint(activeId, "RECEIVER", 1.8, 2.1);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [activeId]);

  const onMatchCreated = useCallback(
    (id: string) => {
      setActiveId(id);
      fetchMatches().then(setMatches).catch(() => {});
    },
    []
  );

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden">
      {/* ── Header Bar ── */}
      <header className="flex items-center justify-between px-4 py-1.5 border-b border-terminal-border bg-terminal-panel shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-terminal-green font-bold text-sm">◉ TENNIS TERMINAL</span>
          <span className="text-[10px] text-terminal-muted">v1.0</span>
        </div>
        <div className="flex items-center gap-3 text-[10px]">
          <span className={connected ? "text-terminal-green" : "text-terminal-red"}>
            {connected ? "● CONNECTED" : "○ DISCONNECTED"}
          </span>
          {frame && (
            <span className="text-terminal-muted">
              Latency: {Math.round((Date.now() / 1000 - frame.server_ts) * 1000)}ms
            </span>
          )}
        </div>
      </header>

      {/* ── Main Grid ── */}
      <div className="flex-1 grid grid-cols-[340px_1fr_280px] grid-rows-[1fr_220px] min-h-0">
        {/* LEFT — Match List + Live Matches (tabbed) */}
        <aside className="row-span-2 border-r border-terminal-border overflow-hidden flex flex-col">
          {/* Tab bar */}
          <div className="flex border-b border-terminal-border shrink-0">
            <button
              onClick={() => setLeftTab("matches")}
              className={`flex-1 text-[10px] font-bold uppercase tracking-wider py-1.5 transition ${
                leftTab === "matches"
                  ? "text-terminal-green border-b-2 border-terminal-green bg-terminal-green/5"
                  : "text-terminal-muted hover:text-slate-300"
              }`}
            >
              MATCHES
            </button>
            <button
              onClick={() => setLeftTab("schedule")}
              className={`flex-1 text-[10px] font-bold uppercase tracking-wider py-1.5 transition ${
                leftTab === "schedule"
                  ? "text-terminal-yellow border-b-2 border-terminal-yellow bg-terminal-yellow/5"
                  : "text-terminal-muted hover:text-slate-300"
              }`}
            >
              📅
            </button>
            <button
              onClick={() => setLeftTab("live")}
              className={`flex-1 text-[10px] font-bold uppercase tracking-wider py-1.5 transition ${
                leftTab === "live"
                  ? "text-terminal-cyan border-b-2 border-terminal-cyan bg-terminal-cyan/5"
                  : "text-terminal-muted hover:text-slate-300"
              }`}
            >
              📡
            </button>
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-hidden">
            {leftTab === "matches" ? (
              <MatchList
                matches={matches}
                activeId={activeId}
                onSelect={setActiveId}
                onMatchCreated={onMatchCreated}
              />
            ) : leftTab === "schedule" ? (
              <SchedulePanel onAutoSetup={handleScheduleSetup} />
            ) : (
              <LiveMatchPanel activeMatchId={activeId} />
            )}
          </div>
        </aside>

        {/* CENTER — Trade Box + Point Input */}
        <main className="flex flex-col gap-2 p-2 overflow-y-auto min-h-0">
          {activeId && <PointInput matchId={activeId} />}

          {frame ? (
            <TradeBox frame={frame} />
          ) : (
            <div className="flex-1 flex items-center justify-center text-terminal-muted text-sm">
              {activeId ? "Connecting..." : "← Select or create a match"}
            </div>
          )}
        </main>

        {/* RIGHT — Live Stats + Ensemble + Odds + Position */}
        <aside className="row-span-2 border-l border-terminal-border flex flex-col min-h-0">
          {frame ? (
            <>
              <div className="flex-[2] overflow-hidden border-b border-terminal-border">
                <LiveStatsPanel stats={frame.live_stats} />
              </div>
              <div className="flex-[1.5] overflow-hidden border-b border-terminal-border">
                <EnsemblePanel ensemble={frame.ensemble} />
              </div>
              <div className="flex-1 overflow-hidden border-b border-terminal-border">
                <OddsLadder frame={frame} />
              </div>
              <div className="flex-1 overflow-hidden">
                <PositionTracker frame={frame} />
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-full text-terminal-muted text-[10px]">
              No data
            </div>
          )}
        </aside>

        {/* BOTTOM — Trade Log + State Performance */}
        <div className="col-start-2 border-t border-terminal-border grid grid-cols-2 min-h-0">
          <div className="border-r border-terminal-border overflow-hidden">
            {frame ? (
              <TradeLog trades={frame.trade_log} />
            ) : (
              <div className="flex items-center justify-center h-full text-terminal-muted text-[10px]">
                –
              </div>
            )}
          </div>
          <div className="overflow-hidden">
            {frame ? (
              <StatePerformance rows={frame.state_performance} />
            ) : (
              <div className="flex items-center justify-center h-full text-terminal-muted text-[10px]">
                –
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
