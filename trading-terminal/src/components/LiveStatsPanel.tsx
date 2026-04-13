"use client";

import type { LiveStatsSnapshot, PlayerStatsSnapshot } from "@/lib/types";

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function StatRow({
  label,
  p1,
  p2,
  fmt = "pct",
}: {
  label: string;
  p1: number;
  p2: number;
  fmt?: "pct" | "int" | "dec";
}) {
  const format = (v: number) => {
    if (fmt === "pct") return pct(v);
    if (fmt === "int") return String(v);
    return v.toFixed(2);
  };

  const better1 = p1 > p2;
  const better2 = p2 > p1;

  return (
    <div className="grid grid-cols-[60px_1fr_60px] gap-1 text-xs py-0.5">
      <span className={better1 ? "text-terminal-green font-semibold text-right" : "text-slate-300 text-right"}>
        {format(p1)}
      </span>
      <span className="text-center text-terminal-muted text-[10px]">{label}</span>
      <span className={better2 ? "text-terminal-green font-semibold" : "text-slate-300"}>
        {format(p2)}
      </span>
    </div>
  );
}

export default function LiveStatsPanel({ stats }: { stats: LiveStatsSnapshot | null }) {
  if (!stats || stats.points_played === 0) {
    return (
      <div className="h-full flex items-center justify-center text-terminal-muted text-[10px]">
        Live stats will appear after first point
      </div>
    );
  }

  const p1 = stats.player1;
  const p2 = stats.player2;

  return (
    <div className="h-full overflow-y-auto px-2 py-1.5">
      <div className="text-[10px] font-bold text-terminal-muted uppercase tracking-wider mb-1">
        LIVE MATCH STATS ({stats.points_played} pts)
      </div>

      {/* Serve stats */}
      <div className="text-[9px] text-terminal-cyan font-bold uppercase mb-0.5 mt-1">SERVE</div>
      <StatRow label="1st Srv %" p1={p1.first_serve_pct} p2={p2.first_serve_pct} />
      <StatRow label="1st Win %" p1={p1.first_serve_win_pct} p2={p2.first_serve_win_pct} />
      <StatRow label="2nd Win %" p1={p1.second_serve_win_pct} p2={p2.second_serve_win_pct} />
      <StatRow label="SPW %" p1={p1.serve_points_won_pct} p2={p2.serve_points_won_pct} />
      <StatRow label="Aces" p1={p1.aces} p2={p2.aces} fmt="int" />
      <StatRow label="DFs" p1={p1.double_faults} p2={p2.double_faults} fmt="int" />

      {/* Return stats */}
      <div className="text-[9px] text-terminal-cyan font-bold uppercase mb-0.5 mt-2">RETURN</div>
      <StatRow label="RPW %" p1={p1.return_points_won_pct} p2={p2.return_points_won_pct} />

      {/* Break points */}
      <div className="text-[9px] text-terminal-cyan font-bold uppercase mb-0.5 mt-2">BREAK PTS</div>
      <StatRow label="BP Saved %" p1={p1.break_point_save_pct} p2={p2.break_point_save_pct} />
      <StatRow label="BP Faced" p1={p1.break_points_faced} p2={p2.break_points_faced} fmt="int" />

      {/* General */}
      <div className="text-[9px] text-terminal-cyan font-bold uppercase mb-0.5 mt-2">SHOT QUALITY</div>
      <StatRow label="Winners" p1={p1.winners} p2={p2.winners} fmt="int" />
      <StatRow label="UE" p1={p1.unforced_errors} p2={p2.unforced_errors} fmt="int" />

      {/* Points won */}
      <div className="text-[9px] text-terminal-cyan font-bold uppercase mb-0.5 mt-2">OVERALL</div>
      <StatRow label="Win Rate" p1={p1.win_rate} p2={p2.win_rate} />
      <StatRow label="Pts Won" p1={p1.total_points_won} p2={p2.total_points_won} fmt="int" />
    </div>
  );
}
