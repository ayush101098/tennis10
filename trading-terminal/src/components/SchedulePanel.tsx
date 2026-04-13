"use client";

import { useEffect, useState, useCallback } from "react";
import { fetchSchedule } from "@/lib/api";
import type { ScheduledMatch, ScheduleResponse } from "@/lib/api";

interface Props {
  onAutoSetup?: (player1: string, player2: string, tournament: string, surface: string, bestOf: number) => void;
}

export default function SchedulePanel({ onAutoSetup }: Props) {
  const [schedule, setSchedule] = useState<ScheduleResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeDay, setActiveDay] = useState<"today" | "tomorrow">("today");
  const [tourFilter, setTourFilter] = useState<string>("ALL");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchSchedule();
      setSchedule(data);
    } catch {
      setError("Failed to fetch schedule");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 120_000); // refresh every 2 min
    return () => clearInterval(iv);
  }, [refresh]);

  const matches = schedule
    ? activeDay === "today"
      ? schedule.today
      : schedule.tomorrow
    : [];

  // Collect unique tours for filter
  const allTours = Array.from(new Set(matches.map((m) => m.tour))).sort();

  const filtered = tourFilter === "ALL" ? matches : matches.filter((m) => m.tour === tourFilter);

  // Group by tournament
  const grouped = filtered.reduce<Record<string, ScheduledMatch[]>>((acc, m) => {
    const key = m.tournament || "Unknown";
    if (!acc[key]) acc[key] = [];
    acc[key].push(m);
    return acc;
  }, {});

  const dateLabel = schedule
    ? activeDay === "today"
      ? schedule.today_date
      : schedule.tomorrow_date
    : "";

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border shrink-0">
        <span className="text-[10px] font-bold text-terminal-muted uppercase tracking-wider">
          📅 SCHEDULE
        </span>
        <div className="flex items-center gap-2">
          <span className="text-[9px] text-terminal-muted">
            {filtered.length} matches
          </span>
          <button
            onClick={refresh}
            disabled={loading}
            className="text-[10px] bg-terminal-blue/20 text-terminal-blue border border-terminal-blue/40 rounded px-2 py-0.5 hover:bg-terminal-blue/30 disabled:opacity-50"
          >
            {loading ? "..." : "↻"}
          </button>
        </div>
      </div>

      {/* Day toggle + Tour filter */}
      <div className="flex items-center gap-1 px-3 py-1.5 border-b border-terminal-border shrink-0">
        <button
          onClick={() => setActiveDay("today")}
          className={`text-[10px] font-bold px-2 py-0.5 rounded transition ${
            activeDay === "today"
              ? "bg-terminal-green/20 text-terminal-green border border-terminal-green/40"
              : "text-terminal-muted hover:text-slate-300"
          }`}
        >
          TODAY
        </button>
        <button
          onClick={() => setActiveDay("tomorrow")}
          className={`text-[10px] font-bold px-2 py-0.5 rounded transition ${
            activeDay === "tomorrow"
              ? "bg-terminal-cyan/20 text-terminal-cyan border border-terminal-cyan/40"
              : "text-terminal-muted hover:text-slate-300"
          }`}
        >
          TOMORROW
        </button>
        <span className="text-[9px] text-terminal-muted ml-1">{dateLabel}</span>

        <div className="ml-auto flex gap-1">
          <button
            onClick={() => setTourFilter("ALL")}
            className={`text-[8px] px-1.5 py-0.5 rounded ${
              tourFilter === "ALL" ? "bg-terminal-blue/20 text-terminal-blue" : "text-terminal-muted"
            }`}
          >
            ALL
          </button>
          {allTours.map((t) => (
            <button
              key={t}
              onClick={() => setTourFilter(t)}
              className={`text-[8px] px-1.5 py-0.5 rounded ${
                tourFilter === t ? "bg-terminal-blue/20 text-terminal-blue" : "text-terminal-muted"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="px-3 py-1.5 bg-terminal-red/10 text-terminal-red text-[10px]">{error}</div>
      )}

      {/* Match list grouped by tournament */}
      <div className="flex-1 overflow-y-auto">
        {Object.entries(grouped).map(([tournament, tMatches]) => (
          <div key={tournament}>
            {/* Tournament header */}
            <div className="px-3 py-1 bg-terminal-panel/60 border-b border-terminal-border sticky top-0 z-10">
              <div className="flex items-center gap-2">
                <SurfaceDot surface={tMatches[0]?.surface || "Hard"} />
                <span className="text-[10px] font-bold text-slate-300 truncate">{tournament}</span>
                <TourBadge tour={tMatches[0]?.tour || "ATP"} />
                <span className="text-[8px] text-terminal-muted">{tMatches[0]?.surface}</span>
              </div>
            </div>

            {/* Matches */}
            {tMatches.map((m) => (
              <MatchRow key={m.id} match={m} onSetup={onAutoSetup} />
            ))}
          </div>
        ))}

        {!loading && filtered.length === 0 && (
          <div className="text-terminal-muted text-[10px] text-center py-8">
            No matches found for {activeDay}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Match Row ── */

function MatchRow({
  match: m,
  onSetup,
}: {
  match: ScheduledMatch;
  onSetup?: (p1: string, p2: string, tournament: string, surface: string, bestOf: number) => void;
}) {
  const isFav1 = m.p1_win_prob > 0.5;
  const probKnown = m.prob_method !== "unknown";

  return (
    <div className="px-3 py-1.5 border-b border-terminal-border hover:bg-terminal-panel/40 transition flex items-center gap-2">
      {/* Time */}
      <div className="w-[36px] shrink-0 text-[9px] text-terminal-muted text-center">
        {m.status === "live" ? (
          <span className="text-terminal-green font-bold flex items-center gap-0.5 justify-center">
            <span className="w-1 h-1 rounded-full bg-terminal-green live-dot" />
            LIVE
          </span>
        ) : (
          m.start_time || "TBD"
        )}
      </div>

      {/* Players + ranks */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <PlayerName
            name={m.player1}
            rank={m.p1_rank}
            seed={m.p1_seed}
            isFav={isFav1 && probKnown}
          />
        </div>
        <div className="flex items-center gap-1.5 mt-0.5">
          <PlayerName
            name={m.player2}
            rank={m.p2_rank}
            seed={m.p2_seed}
            isFav={!isFav1 && probKnown}
          />
        </div>
      </div>

      {/* Probability bars */}
      <div className="w-[80px] shrink-0">
        {probKnown ? (
          <ProbBar p1={m.p1_win_prob} p2={m.p2_win_prob} />
        ) : (
          <span className="text-[8px] text-terminal-muted text-center block">—</span>
        )}
      </div>

      {/* Round */}
      <div className="w-[50px] shrink-0 text-[8px] text-terminal-muted text-right truncate">
        {m.round || ""}
      </div>

      {/* Setup button */}
      {onSetup && m.status !== "finished" && (
        <button
          onClick={() => onSetup(m.player1, m.player2, m.tournament, m.surface, m.best_of)}
          className="text-[8px] bg-terminal-cyan/20 text-terminal-cyan border border-terminal-cyan/40 rounded px-1.5 py-0.5 hover:bg-terminal-cyan/30 whitespace-nowrap shrink-0"
        >
          TRADE
        </button>
      )}
    </div>
  );
}

/* ── Player Name ── */

function PlayerName({
  name,
  rank,
  seed,
  isFav,
}: {
  name: string;
  rank: number;
  seed: number;
  isFav: boolean;
}) {
  return (
    <div className="flex items-center gap-1 min-w-0">
      {isFav && <span className="text-[7px] text-terminal-green">▶</span>}
      <span className={`text-[11px] truncate ${isFav ? "text-slate-100 font-medium" : "text-slate-300"}`}>
        {name}
      </span>
      {rank > 0 && (
        <span className="text-[8px] text-terminal-muted shrink-0">#{rank}</span>
      )}
      {seed > 0 && (
        <span className="text-[8px] text-terminal-cyan/80 shrink-0">[{seed}]</span>
      )}
    </div>
  );
}

/* ── Probability Bar ── */

function ProbBar({ p1, p2 }: { p1: number; p2: number }) {
  const p1Pct = Math.round(p1 * 100);
  const p2Pct = Math.round(p2 * 100);
  const favIsP1 = p1 >= p2;

  return (
    <div className="space-y-0.5">
      <div className="flex items-center gap-1">
        <div className="flex-1 h-[4px] rounded-full bg-terminal-border overflow-hidden">
          <div
            className={`h-full rounded-full ${favIsP1 ? "bg-terminal-green" : "bg-terminal-blue"}`}
            style={{ width: `${p1Pct}%` }}
          />
        </div>
        <span className={`text-[9px] font-mono w-[28px] text-right ${favIsP1 ? "text-terminal-green" : "text-slate-400"}`}>
          {p1Pct}%
        </span>
      </div>
      <div className="flex items-center gap-1">
        <div className="flex-1 h-[4px] rounded-full bg-terminal-border overflow-hidden">
          <div
            className={`h-full rounded-full ${!favIsP1 ? "bg-terminal-green" : "bg-terminal-blue"}`}
            style={{ width: `${p2Pct}%` }}
          />
        </div>
        <span className={`text-[9px] font-mono w-[28px] text-right ${!favIsP1 ? "text-terminal-green" : "text-slate-400"}`}>
          {p2Pct}%
        </span>
      </div>
    </div>
  );
}

/* ── Tour Badge ── */

const TOUR_COLORS: Record<string, string> = {
  ATP: "text-blue-400 bg-blue-400/10",
  WTA: "text-pink-400 bg-pink-400/10",
  "ITF-M": "text-orange-400 bg-orange-400/10",
  "ITF-W": "text-orange-300 bg-orange-300/10",
  Challenger: "text-yellow-400 bg-yellow-400/10",
};

function TourBadge({ tour }: { tour: string }) {
  const color = TOUR_COLORS[tour] || "text-terminal-muted bg-terminal-muted/10";
  return (
    <span className={`text-[7px] font-bold px-1 rounded ${color}`}>{tour}</span>
  );
}

/* ── Surface Dot ── */

const SURFACE_COLORS: Record<string, string> = {
  Hard: "bg-blue-500",
  Clay: "bg-orange-500",
  Grass: "bg-green-500",
};

function SurfaceDot({ surface }: { surface: string }) {
  return <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${SURFACE_COLORS[surface] || "bg-slate-500"}`} />;
}
