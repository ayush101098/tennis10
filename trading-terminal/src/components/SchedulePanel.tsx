"use client";

import { useEffect, useState, useCallback } from "react";
import { fetchScheduleClient } from "@/lib/scheduleService";
import type { ScheduledMatch, ScheduleData } from "@/lib/scheduleService";

interface Props {
  onAutoSetup?: (
    player1: string,
    player2: string,
    tournament: string,
    surface: string,
    bestOf: number,
  ) => void;
}

export default function SchedulePanel({ onAutoSetup }: Props) {
  const [schedule, setSchedule] = useState<ScheduleData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeDay, setActiveDay] = useState<"today" | "tomorrow">("today");
  const [tourFilter, setTourFilter] = useState<string>("ALL");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchScheduleClient();
      setSchedule(data);
    } catch {
      setError("Failed to load matches");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 90_000);
    return () => clearInterval(iv);
  }, [refresh]);

  const matches = schedule
    ? activeDay === "today"
      ? schedule.today
      : schedule.tomorrow
    : [];

  const allTours = Array.from(new Set(matches.map((m) => m.tour))).sort();
  const afterTour =
    tourFilter === "ALL" ? matches : matches.filter((m) => m.tour === tourFilter);

  const filtered =
    statusFilter === "ALL"
      ? afterTour
      : afterTour.filter((m) => m.status === statusFilter);

  const statusCounts = matches.reduce<Record<string, number>>((acc, m) => {
    acc[m.status] = (acc[m.status] || 0) + 1;
    return acc;
  }, {});

  const grouped = filtered.reduce<Record<string, ScheduledMatch[]>>(
    (acc, m) => {
      const key = m.tournament || "Unknown";
      if (!acc[key]) acc[key] = [];
      acc[key].push(m);
      return acc;
    },
    {},
  );

  const dateLabel = schedule
    ? activeDay === "today"
      ? schedule.today_date
      : schedule.tomorrow_date
    : "";

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border shrink-0">
        <span className="text-[10px] font-bold text-terminal-yellow uppercase tracking-wider">
          📅 MATCH SCHEDULE
        </span>
        <div className="flex items-center gap-2">
          <span className="text-[9px] text-terminal-muted">
            {filtered.length} match{filtered.length !== 1 ? "es" : ""}
          </span>
          <button
            onClick={refresh}
            disabled={loading}
            className="text-[10px] bg-terminal-blue/20 text-terminal-blue border border-terminal-blue/40 rounded px-2 py-0.5 hover:bg-terminal-blue/30 disabled:opacity-50"
          >
            {loading ? "⟳" : "↻"}
          </button>
        </div>
      </div>

      {/* Day toggle */}
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
      </div>

      {/* Filters row */}
      <div className="flex items-center gap-1 px-3 py-1 border-b border-terminal-border shrink-0 flex-wrap">
        <FilterPill active={tourFilter === "ALL"} onClick={() => setTourFilter("ALL")} label="ALL" />
        {allTours.map((t) => (
          <FilterPill key={t} active={tourFilter === t} onClick={() => setTourFilter(t)} label={t} />
        ))}
        <span className="text-terminal-border mx-1">│</span>
        <FilterPill
          active={statusFilter === "ALL"}
          onClick={() => setStatusFilter("ALL")}
          label={`ALL (${matches.length})`}
        />
        {statusCounts.live ? (
          <FilterPill
            active={statusFilter === "live"}
            onClick={() => setStatusFilter("live")}
            label={`🔴 LIVE (${statusCounts.live})`}
            color="green"
          />
        ) : null}
        {statusCounts.scheduled ? (
          <FilterPill
            active={statusFilter === "scheduled"}
            onClick={() => setStatusFilter("scheduled")}
            label={`📋 SCHED (${statusCounts.scheduled})`}
          />
        ) : null}
        {statusCounts.finished ? (
          <FilterPill
            active={statusFilter === "finished"}
            onClick={() => setStatusFilter("finished")}
            label={`✓ DONE (${statusCounts.finished})`}
            color="muted"
          />
        ) : null}
      </div>

      {error && (
        <div className="px-3 py-1.5 bg-terminal-red/10 text-terminal-red text-[10px]">
          {error}
        </div>
      )}

      {loading && !schedule && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-terminal-muted text-[11px] animate-pulse">
            Loading matches from ESPN...
          </div>
        </div>
      )}

      {/* Match list grouped by tournament */}
      <div className="flex-1 overflow-y-auto">
        {Object.entries(grouped).map(([tournament, tMatches]) => (
          <div key={tournament}>
            <div className="px-3 py-1 bg-terminal-panel/60 border-b border-terminal-border sticky top-0 z-10">
              <div className="flex items-center gap-2">
                <SurfaceDot surface={tMatches[0]?.surface || "Hard"} />
                <span className="text-[10px] font-bold text-slate-300 truncate">
                  {tournament}
                </span>
                <TourBadge tour={tMatches[0]?.tour || "ATP"} />
                <span className="text-[8px] text-terminal-muted">
                  {tMatches[0]?.surface}
                </span>
                <span className="text-[8px] text-terminal-muted ml-auto">
                  {tMatches.length} match{tMatches.length > 1 ? "es" : ""}
                </span>
              </div>
            </div>
            {tMatches.map((m) => (
              <MatchRow key={m.id} match={m} onSetup={onAutoSetup} />
            ))}
          </div>
        ))}

        {!loading && filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center py-10 gap-2">
            <span className="text-terminal-muted text-[11px]">
              No {statusFilter !== "ALL" ? statusFilter : ""} matches for {activeDay}
            </span>
            <span className="text-terminal-muted text-[9px]">
              Try changing filters or check tomorrow
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Filter Pill ── */

function FilterPill({
  active,
  onClick,
  label,
  color,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  color?: string;
}) {
  const base =
    color === "green"
      ? "bg-terminal-green/20 text-terminal-green"
      : color === "muted"
      ? "bg-terminal-muted/20 text-terminal-muted"
      : "bg-terminal-blue/20 text-terminal-blue";

  return (
    <button
      onClick={onClick}
      className={`text-[8px] px-1.5 py-0.5 rounded font-bold transition ${
        active ? base : "text-terminal-muted hover:text-slate-400"
      }`}
    >
      {label}
    </button>
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
  const isFinished = m.status === "finished";
  const isLive = m.status === "live";

  return (
    <div
      className={`px-3 py-1.5 border-b border-terminal-border transition flex items-center gap-2 ${
        isLive
          ? "bg-terminal-green/5 hover:bg-terminal-green/10"
          : isFinished
          ? "opacity-60 hover:opacity-80"
          : "hover:bg-terminal-panel/40"
      }`}
    >
      <div className="w-[40px] shrink-0 text-center">
        {isLive ? (
          <span className="text-[9px] text-terminal-green font-bold flex items-center gap-0.5 justify-center">
            <span className="w-1.5 h-1.5 rounded-full bg-terminal-green animate-pulse" />
            LIVE
          </span>
        ) : isFinished ? (
          <span className="text-[9px] text-terminal-muted">FIN</span>
        ) : (
          <span className="text-[9px] text-terminal-muted">{m.start_time || "TBD"}</span>
        )}
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <PlayerName
            name={m.player1}
            rank={m.p1_rank}
            seed={m.p1_seed}
            isFav={isFav1 && probKnown}
            isWinner={m.score?.winner === 1}
          />
          {m.score && <ScoreLine sets={m.score.p1_sets} won={m.score.winner === 1} />}
        </div>
        <div className="flex items-center gap-1.5 mt-0.5">
          <PlayerName
            name={m.player2}
            rank={m.p2_rank}
            seed={m.p2_seed}
            isFav={!isFav1 && probKnown}
            isWinner={m.score?.winner === 2}
          />
          {m.score && <ScoreLine sets={m.score.p2_sets} won={m.score.winner === 2} />}
        </div>
      </div>

      <div className="w-[80px] shrink-0">
        {probKnown ? (
          <ProbBar p1={m.p1_win_prob} p2={m.p2_win_prob} />
        ) : (
          <span className="text-[8px] text-terminal-muted text-center block">50/50</span>
        )}
      </div>

      <div className="w-[55px] shrink-0 text-[8px] text-terminal-muted text-right truncate">
        {m.round || ""}
      </div>

      {onSetup && !isFinished && (
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

function ScoreLine({ sets, won }: { sets: number[]; won: boolean }) {
  return (
    <div className="flex items-center gap-0.5 ml-auto shrink-0">
      {sets.map((s, i) => (
        <span
          key={i}
          className={`text-[9px] font-mono w-[14px] text-center ${
            won ? "text-terminal-green" : "text-terminal-muted"
          }`}
        >
          {s}
        </span>
      ))}
    </div>
  );
}

function PlayerName({
  name,
  rank,
  seed,
  isFav,
  isWinner,
}: {
  name: string;
  rank: number;
  seed: number;
  isFav: boolean;
  isWinner?: boolean;
}) {
  return (
    <div className="flex items-center gap-1 min-w-0">
      {isFav && <span className="text-[7px] text-terminal-green shrink-0">▶</span>}
      {isWinner && <span className="text-[7px] text-terminal-green shrink-0">✓</span>}
      <span
        className={`text-[11px] truncate ${
          isWinner
            ? "text-terminal-green font-medium"
            : isFav
            ? "text-slate-100 font-medium"
            : "text-slate-300"
        }`}
      >
        {name}
      </span>
      {rank > 0 && <span className="text-[8px] text-terminal-muted shrink-0">#{rank}</span>}
      {seed > 0 && <span className="text-[8px] text-terminal-cyan/80 shrink-0">[{seed}]</span>}
    </div>
  );
}

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

const TOUR_COLORS: Record<string, string> = {
  ATP: "text-blue-400 bg-blue-400/10",
  WTA: "text-pink-400 bg-pink-400/10",
  "ITF-M": "text-orange-400 bg-orange-400/10",
  "ITF-W": "text-orange-300 bg-orange-300/10",
  Challenger: "text-yellow-400 bg-yellow-400/10",
};

function TourBadge({ tour }: { tour: string }) {
  const color = TOUR_COLORS[tour] || "text-terminal-muted bg-terminal-muted/10";
  return <span className={`text-[7px] font-bold px-1 rounded ${color}`}>{tour}</span>;
}

const SURFACE_COLORS: Record<string, string> = {
  Hard: "bg-blue-500",
  Clay: "bg-orange-500",
  Grass: "bg-green-500",
};

function SurfaceDot({ surface }: { surface: string }) {
  return <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${SURFACE_COLORS[surface] || "bg-slate-500"}`} />;
}
