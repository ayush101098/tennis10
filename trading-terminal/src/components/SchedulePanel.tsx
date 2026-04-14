"use client";

import { useEffect, useState, useCallback } from "react";
import { fetchScheduleClient, probToOdds, kellyFraction } from "@/lib/scheduleService";
import type { ScheduledMatch, ScheduleData } from "@/lib/scheduleService";
import PointTracker from "@/components/PointTracker";

interface Props {
  onSelectMatch?: (match: ScheduledMatch) => void;
}

export default function SchedulePanel({ onSelectMatch }: Props) {
  const [data, setData] = useState<ScheduleData | null>(null);
  const [loading, setLoading] = useState(true);
  const [day, setDay] = useState<"today" | "tomorrow">("today");
  const [tourFilter, setTourFilter] = useState("ALL");
  const [statusFilter, setStatusFilter] = useState("ALL");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [rightTab, setRightTab] = useState<"edge" | "tracker">("edge");

  const refresh = useCallback(async () => {
    setLoading(true);
    try { setData(await fetchScheduleClient()); } catch { /* */ }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 15_000); // 15s — fast schedule refresh
    return () => clearInterval(iv);
  }, [refresh]);

  const raw = data ? (day === "today" ? data.today : data.tomorrow) : [];
  const tours = Array.from(new Set(raw.map(m => m.tour))).sort();
  const afterTour = tourFilter === "ALL" ? raw : raw.filter(m => m.tour === tourFilter);
  const matches = statusFilter === "ALL" ? afterTour : afterTour.filter(m => m.status === statusFilter);

  const counts: Record<string, number> = {};
  raw.forEach(m => { counts[m.status] = (counts[m.status] || 0) + 1; });

  const grouped: Record<string, ScheduledMatch[]> = {};
  matches.forEach(m => {
    const k = m.tournament || "Unknown";
    if (!grouped[k]) grouped[k] = [];
    grouped[k].push(m);
  });

  const selected = matches.find(m => m.id === selectedId) || null;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border shrink-0">
        <span className="text-xs font-bold text-terminal-yellow tracking-wider">📅 MATCH CENTRE</span>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-terminal-muted">{matches.length} matches</span>
          <button onClick={refresh} disabled={loading}
            className="text-[10px] bg-terminal-blue/20 text-terminal-blue border border-terminal-blue/40 rounded px-2 py-0.5 hover:bg-terminal-blue/30 disabled:opacity-50">
            {loading ? "⟳" : "↻"}
          </button>
        </div>
      </div>

      {/* Day toggle */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-terminal-border shrink-0">
        <Pill active={day === "today"} onClick={() => setDay("today")} color="green">
          TODAY {data ? data.today_date.slice(5) : ""}
        </Pill>
        <Pill active={day === "tomorrow"} onClick={() => setDay("tomorrow")} color="cyan">
          TOMORROW {data ? data.tomorrow_date.slice(5) : ""}
        </Pill>
      </div>

      {/* Tour + Status filters */}
      <div className="flex items-center gap-1 px-3 py-1 border-b border-terminal-border shrink-0 flex-wrap">
        <Pill active={tourFilter === "ALL"} onClick={() => setTourFilter("ALL")}>ALL</Pill>
        {tours.map(t => <Pill key={t} active={tourFilter === t} onClick={() => setTourFilter(t)}>{t}</Pill>)}
        <span className="text-terminal-border mx-0.5">│</span>
        <Pill active={statusFilter === "ALL"} onClick={() => setStatusFilter("ALL")}>ALL ({raw.length})</Pill>
        {counts.live ? <Pill active={statusFilter === "live"} onClick={() => setStatusFilter("live")} color="green">🔴 LIVE ({counts.live})</Pill> : null}
        {counts.scheduled ? <Pill active={statusFilter === "scheduled"} onClick={() => setStatusFilter("scheduled")}>📋 SCHED ({counts.scheduled})</Pill> : null}
        {counts.finished ? <Pill active={statusFilter === "finished"} onClick={() => setStatusFilter("finished")} color="muted">✓ DONE ({counts.finished})</Pill> : null}
      </div>

      {/* Two-pane: match list + edge panel */}
      <div className="flex-1 flex min-h-0">
        {/* Left: scrollable match list */}
        <div className={`overflow-y-auto border-r border-terminal-border ${selected ? "w-1/2" : "w-full"} transition-all`}>
          {loading && !data && (
            <div className="flex items-center justify-center h-full text-terminal-muted text-xs animate-pulse">Loading ESPN data…</div>
          )}
          {Object.entries(grouped).map(([t, ms]) => (
            <div key={t}>
              <div className="px-3 py-1 bg-terminal-panel/50 border-b border-terminal-border sticky top-0 z-10 flex items-center gap-2">
                <SurfaceDot s={ms[0]?.surface} />
                <span className="text-[11px] font-bold text-slate-300 truncate">{t}</span>
                <TourBadge t={ms[0]?.tour} />
                <span className="text-[9px] text-terminal-muted">{ms[0]?.surface}</span>
                <span className="text-[9px] text-terminal-muted ml-auto">{ms.length}</span>
              </div>
              {ms.map(m => (
                <MatchRow key={m.id} m={m} active={m.id === selectedId} onClick={() => {
                  setSelectedId(m.id === selectedId ? null : m.id);
                  onSelectMatch?.(m);
                }} />
              ))}
            </div>
          ))}
          {!loading && matches.length === 0 && (
            <div className="text-terminal-muted text-xs text-center py-8">No matches for {day}. Try changing filters.</div>
          )}
        </div>

        {/* Right: edge + tracker panel */}
        {selected && (
          <div className="w-1/2 flex flex-col min-h-0">
            {/* Tab bar */}
            <div className="flex border-b border-terminal-border shrink-0">
              <button onClick={() => setRightTab("edge")}
                className={`flex-1 text-[10px] font-bold uppercase tracking-wider py-1.5 transition ${
                  rightTab === "edge"
                    ? "text-terminal-yellow border-b-2 border-terminal-yellow bg-terminal-yellow/5"
                    : "text-terminal-muted hover:text-slate-300"
                }`}>
                ⚡ Edge
              </button>
              <button onClick={() => setRightTab("tracker")}
                className={`flex-1 text-[10px] font-bold uppercase tracking-wider py-1.5 transition ${
                  rightTab === "tracker"
                    ? "text-terminal-cyan border-b-2 border-terminal-cyan bg-terminal-cyan/5"
                    : "text-terminal-muted hover:text-slate-300"
                }`}>
                🎾 Tracker
              </button>
            </div>
            {/* Tab content */}
            <div className="flex-1 overflow-y-auto">
              {rightTab === "edge" ? <EdgePanel match={selected} /> : <PointTracker match={selected} />}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
   EDGE PANEL — Full bookmaker-style analysis for the selected match
   ═══════════════════════════════════════════════════════════════════════════ */

function EdgePanel({ match: m }: { match: ScheduledMatch }) {
  const [odds1, setOdds1] = useState(() => probToOdds(m.p1_win_prob));
  const [odds2, setOdds2] = useState(() => probToOdds(m.p2_win_prob));
  const [bankroll, setBankroll] = useState(1000);

  // Recalc when match changes
  useEffect(() => {
    setOdds1(probToOdds(m.p1_win_prob));
    setOdds2(probToOdds(m.p2_win_prob));
  }, [m.id, m.p1_win_prob, m.p2_win_prob]);

  const imp1 = odds1 > 0 ? 1 / odds1 : 0;
  const imp2 = odds2 > 0 ? 1 / odds2 : 0;
  const edge1 = m.p1_win_prob - imp1;
  const edge2 = m.p2_win_prob - imp2;
  const kelly1 = kellyFraction(m.p1_win_prob, odds1);
  const kelly2 = kellyFraction(m.p2_win_prob, odds2);
  const stake1 = Math.round(bankroll * kelly1 * 0.25); // quarter Kelly
  const stake2 = Math.round(bankroll * kelly2 * 0.25);
  const vig = imp1 + imp2 - 1;

  const isFav1 = m.p1_win_prob >= m.p2_win_prob;

  return (
    <div className="p-3 space-y-3 text-[11px]">
      {/* Title */}
      <div className="text-center">
        <div className="text-xs font-bold text-terminal-yellow mb-1">⚡ EDGE ANALYSIS</div>
        <div className="text-slate-200 font-medium">{m.player1} vs {m.player2}</div>
        <div className="text-[10px] text-terminal-muted">{m.tournament} · {m.round} · {m.surface} · Bo{m.best_of}</div>
      </div>

      {/* Model probability */}
      <Section title="MODEL PROBABILITY">
        <div className="flex items-center gap-3 mb-2">
          <div className="flex-1">
            <div className="flex items-center justify-between mb-0.5">
              <span className={isFav1 ? "text-terminal-green font-bold" : "text-slate-300"}>{m.player1}</span>
              <span className={isFav1 ? "text-terminal-green font-bold" : "text-slate-400"}>{pct(m.p1_win_prob)}</span>
            </div>
            <div className="h-2 bg-terminal-border rounded-full overflow-hidden">
              <div className={`h-full rounded-full ${isFav1 ? "bg-terminal-green" : "bg-terminal-blue"}`} style={{ width: pct(m.p1_win_prob) }} />
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <div className="flex items-center justify-between mb-0.5">
              <span className={!isFav1 ? "text-terminal-green font-bold" : "text-slate-300"}>{m.player2}</span>
              <span className={!isFav1 ? "text-terminal-green font-bold" : "text-slate-400"}>{pct(m.p2_win_prob)}</span>
            </div>
            <div className="h-2 bg-terminal-border rounded-full overflow-hidden">
              <div className={`h-full rounded-full ${!isFav1 ? "bg-terminal-green" : "bg-terminal-blue"}`} style={{ width: pct(m.p2_win_prob) }} />
            </div>
          </div>
        </div>
        <div className="text-[9px] text-terminal-muted mt-1 text-center">
          Method: {m.prob_method} · Fair odds: {probToOdds(m.p1_win_prob).toFixed(2)} / {probToOdds(m.p2_win_prob).toFixed(2)}
          {m.p1_rank > 0 && m.p2_rank > 0 && ` · Rank #{m.p1_rank} vs #${m.p2_rank}`}
        </div>
      </Section>

      {/* Odds input + edge calculator */}
      <Section title="BOOKMAKER ODDS & EDGE">
        <div className="grid grid-cols-2 gap-2 mb-2">
          <div>
            <label className="text-[9px] text-terminal-muted block mb-0.5">{m.player1} odds</label>
            <input type="number" step="0.01" min="1.01" value={odds1}
              onChange={e => setOdds1(parseFloat(e.target.value) || 1.01)}
              className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[11px] text-slate-200 focus:border-terminal-cyan outline-none" />
          </div>
          <div>
            <label className="text-[9px] text-terminal-muted block mb-0.5">{m.player2} odds</label>
            <input type="number" step="0.01" min="1.01" value={odds2}
              onChange={e => setOdds2(parseFloat(e.target.value) || 1.01)}
              className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[11px] text-slate-200 focus:border-terminal-cyan outline-none" />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[10px]">
          <KV label="Implied P1" value={pct(imp1)} />
          <KV label="Implied P2" value={pct(imp2)} />
          <KV label="Edge P1" value={edgeFmt(edge1)} color={edge1 > 0 ? "green" : edge1 < -0.02 ? "red" : "muted"} />
          <KV label="Edge P2" value={edgeFmt(edge2)} color={edge2 > 0 ? "green" : edge2 < -0.02 ? "red" : "muted"} />
          <KV label="Overround" value={pct(vig)} color={vig > 0.08 ? "red" : "muted"} />
          <KV label="Margin" value={`${(vig * 100).toFixed(1)}%`} />
        </div>
      </Section>

      {/* Value bet signals */}
      <Section title="VALUE BET SIGNALS">
        <ValueSignal label={`${m.player1} ML`} edge={edge1} odds={odds1} prob={m.p1_win_prob} />
        <ValueSignal label={`${m.player2} ML`} edge={edge2} odds={odds2} prob={m.p2_win_prob} />
      </Section>

      {/* Kelly staking */}
      <Section title="KELLY STAKING">
        <div className="mb-2">
          <label className="text-[9px] text-terminal-muted block mb-0.5">Bankroll ($)</label>
          <input type="number" value={bankroll} onChange={e => setBankroll(parseInt(e.target.value) || 100)}
            className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[11px] text-slate-200 focus:border-terminal-cyan outline-none" />
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          <KV label="Full Kelly P1" value={pct(kelly1)} />
          <KV label="Full Kelly P2" value={pct(kelly2)} />
          <KV label="¼ Kelly P1" value={`$${stake1}`} color={stake1 > 0 ? "green" : "muted"} />
          <KV label="¼ Kelly P2" value={`$${stake2}`} color={stake2 > 0 ? "green" : "muted"} />
        </div>

        {(kelly1 > 0 || kelly2 > 0) && (
          <div className={`mt-2 p-2 rounded border ${
            kelly1 > kelly2
              ? "border-terminal-green/40 bg-terminal-green/5"
              : "border-terminal-cyan/40 bg-terminal-cyan/5"
          }`}>
            <div className="text-[10px] font-bold text-terminal-green">
              💎 RECOMMENDED: {kelly1 > kelly2 ? m.player1 : m.player2} @ {kelly1 > kelly2 ? odds1.toFixed(2) : odds2.toFixed(2)}
            </div>
            <div className="text-[9px] text-terminal-muted">
              Stake ${kelly1 > kelly2 ? stake1 : stake2} (¼ Kelly) · Edge {edgeFmt(kelly1 > kelly2 ? edge1 : edge2)} · EV ${
                ((kelly1 > kelly2 ? edge1 : edge2) * (kelly1 > kelly2 ? stake1 : stake2) * (kelly1 > kelly2 ? odds1 : odds2)).toFixed(0)
              }
            </div>
          </div>
        )}
      </Section>

      {/* Match context */}
      <Section title="MATCH CONTEXT">
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px]">
          <KV label="Surface" value={m.surface} />
          <KV label="Best of" value={`${m.best_of}`} />
          <KV label="P1 Rank" value={m.p1_rank > 0 ? `#${m.p1_rank}` : "—"} />
          <KV label="P2 Rank" value={m.p2_rank > 0 ? `#${m.p2_rank}` : "—"} />
          <KV label="P1 Seed" value={m.p1_seed > 0 ? `[${m.p1_seed}]` : "—"} />
          <KV label="P2 Seed" value={m.p2_seed > 0 ? `[${m.p2_seed}]` : "—"} />
          <KV label="Round" value={m.round || "—"} />
          <KV label="Time" value={m.start_time || "TBD"} />
          {m.venue && <KV label="Venue" value={m.venue} />}
          <KV label="Status" value={m.status.toUpperCase()} color={m.status === "live" ? "green" : "muted"} />
        </div>
      </Section>

      {/* Score if live/finished */}
      {m.score && (
        <Section title={m.status === "live" ? "LIVE SCORE" : "FINAL SCORE"}>
          <div className="font-mono text-[12px] text-center space-y-0.5">
            <div className={m.score.winner === 1 ? "text-terminal-green font-bold" : "text-slate-300"}>
              {m.player1}  {m.score.p1_sets.join("  ")}
            </div>
            <div className={m.score.winner === 2 ? "text-terminal-green font-bold" : "text-slate-300"}>
              {m.player2}  {m.score.p2_sets.join("  ")}
            </div>
          </div>
        </Section>
      )}
    </div>
  );
}

/* ── Value Signal ── */

function ValueSignal({ label, edge, odds, prob }: { label: string; edge: number; odds: number; prob: number }) {
  const hasEdge = edge > 0.02;
  const strongEdge = edge > 0.05;
  const noEdge = edge < 0;

  return (
    <div className={`flex items-center gap-2 py-1 px-2 rounded mb-1 ${
      strongEdge ? "bg-terminal-green/10 border border-terminal-green/30" :
      hasEdge ? "bg-terminal-yellow/10 border border-terminal-yellow/30" :
      "bg-terminal-panel/30"
    }`}>
      <span className={`text-[10px] font-bold ${
        strongEdge ? "text-terminal-green" :
        hasEdge ? "text-terminal-yellow" :
        "text-terminal-muted"
      }`}>
        {strongEdge ? "🔥" : hasEdge ? "⚡" : noEdge ? "✗" : "—"}
      </span>
      <span className="text-[10px] flex-1 text-slate-300">{label}</span>
      <span className="text-[10px] text-terminal-muted">@{odds.toFixed(2)}</span>
      <span className={`text-[10px] font-bold ${
        strongEdge ? "text-terminal-green" :
        hasEdge ? "text-terminal-yellow" :
        noEdge ? "text-terminal-red" : "text-terminal-muted"
      }`}>
        {edge > 0 ? "+" : ""}{(edge * 100).toFixed(1)}%
      </span>
      <span className="text-[9px] text-terminal-muted">
        ({pct(prob)} vs {pct(1 / odds)})
      </span>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
   Sub-components
   ═══════════════════════════════════════════════════════════════════════════ */

function MatchRow({ m, active, onClick }: { m: ScheduledMatch; active: boolean; onClick: () => void }) {
  const fav1 = m.p1_win_prob > 0.5;
  const known = m.prob_method !== "unknown";
  const live = m.status === "live";
  const fin = m.status === "finished";

  return (
    <button onClick={onClick} className={`w-full text-left px-3 py-2 border-b border-terminal-border transition ${
      active ? "bg-terminal-cyan/10 border-l-2 border-l-terminal-cyan" :
      live ? "bg-terminal-green/5 hover:bg-terminal-green/10" :
      fin ? "opacity-50 hover:opacity-70" : "hover:bg-terminal-panel/40"
    }`}>
      <div className="flex items-center gap-2">
        {/* Time/status */}
        <div className="w-[38px] shrink-0 text-center">
          {live ? (
            <span className="text-[9px] text-terminal-green font-bold flex items-center gap-0.5 justify-center">
              <span className="w-1.5 h-1.5 rounded-full bg-terminal-green animate-pulse" />LIVE
            </span>
          ) : fin ? (
            <span className="text-[9px] text-terminal-muted">FIN</span>
          ) : (
            <span className="text-[10px] text-terminal-muted">{m.start_time || "TBD"}</span>
          )}
        </div>

        {/* Players */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1">
            {fav1 && known && <span className="text-[7px] text-terminal-green">▶</span>}
            {m.score?.winner === 1 && <span className="text-[7px] text-terminal-green">✓</span>}
            <span className={`text-[11px] truncate ${fav1 && known ? "text-slate-100 font-medium" : "text-slate-300"}`}>{m.player1}</span>
            {m.p1_rank > 0 && <span className="text-[8px] text-terminal-muted shrink-0">#{m.p1_rank}</span>}
            {m.p1_seed > 0 && <span className="text-[8px] text-terminal-cyan/70 shrink-0">[{m.p1_seed}]</span>}
            {m.score && <span className="text-[9px] text-terminal-muted ml-auto shrink-0 font-mono">{m.score.p1_sets.join(" ")}</span>}
          </div>
          <div className="flex items-center gap-1 mt-0.5">
            {!fav1 && known && <span className="text-[7px] text-terminal-green">▶</span>}
            {m.score?.winner === 2 && <span className="text-[7px] text-terminal-green">✓</span>}
            <span className={`text-[11px] truncate ${!fav1 && known ? "text-slate-100 font-medium" : "text-slate-300"}`}>{m.player2}</span>
            {m.p2_rank > 0 && <span className="text-[8px] text-terminal-muted shrink-0">#{m.p2_rank}</span>}
            {m.p2_seed > 0 && <span className="text-[8px] text-terminal-cyan/70 shrink-0">[{m.p2_seed}]</span>}
            {m.score && <span className="text-[9px] text-terminal-muted ml-auto shrink-0 font-mono">{m.score.p2_sets.join(" ")}</span>}
          </div>
        </div>

        {/* Prob column */}
        {known && (
          <div className="w-[55px] shrink-0 text-right">
            <div className={`text-[10px] font-mono ${fav1 ? "text-terminal-green font-bold" : "text-slate-400"}`}>{pct(m.p1_win_prob)}</div>
            <div className={`text-[10px] font-mono ${!fav1 ? "text-terminal-green font-bold" : "text-slate-400"}`}>{pct(m.p2_win_prob)}</div>
          </div>
        )}

        {/* Round */}
        <div className="w-[50px] shrink-0 text-[8px] text-terminal-muted text-right truncate">{m.round}</div>
      </div>
    </button>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border border-terminal-border rounded overflow-hidden">
      <div className="px-2 py-1 bg-terminal-panel/50 border-b border-terminal-border">
        <span className="text-[9px] font-bold text-terminal-muted uppercase tracking-wider">{title}</span>
      </div>
      <div className="p-2">{children}</div>
    </div>
  );
}

function KV({ label, value, color }: { label: string; value: string; color?: string }) {
  const c = color === "green" ? "text-terminal-green" : color === "red" ? "text-terminal-red" : color === "muted" ? "text-terminal-muted" : "text-slate-200";
  return (
    <div className="flex justify-between">
      <span className="text-terminal-muted">{label}</span>
      <span className={`font-mono ${c}`}>{value}</span>
    </div>
  );
}

function Pill({ active, onClick, children, color }: {
  active: boolean; onClick: () => void; children: React.ReactNode; color?: string;
}) {
  const base =
    color === "green" ? "bg-terminal-green/20 text-terminal-green" :
    color === "cyan" ? "bg-terminal-cyan/20 text-terminal-cyan" :
    color === "muted" ? "bg-terminal-muted/20 text-terminal-muted" :
    "bg-terminal-blue/20 text-terminal-blue";
  return (
    <button onClick={onClick}
      className={`text-[9px] px-2 py-0.5 rounded font-bold transition ${active ? base : "text-terminal-muted hover:text-slate-400"}`}>
      {children}
    </button>
  );
}

function TourBadge({ t }: { t?: string }) {
  const c: Record<string, string> = {
    ATP: "text-blue-400 bg-blue-400/10",
    WTA: "text-pink-400 bg-pink-400/10",
    "ITF M": "text-emerald-400 bg-emerald-400/10",
    "ITF W": "text-rose-300 bg-rose-300/10",
    CHAL: "text-amber-400 bg-amber-400/10",
    W125: "text-fuchsia-400 bg-fuchsia-400/10",
  };
  return <span className={`text-[7px] font-bold px-1 rounded ${c[t || ""] || "text-terminal-muted bg-terminal-muted/10"}`}>{t}</span>;
}

function SurfaceDot({ s }: { s?: string }) {
  const c: Record<string, string> = { Hard: "bg-blue-500", Clay: "bg-orange-500", Grass: "bg-green-500" };
  return <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${c[s || ""] || "bg-slate-500"}`} />;
}

/* ── Helpers ── */

function pct(v: number): string { return `${Math.round(v * 100)}%`; }
function edgeFmt(e: number): string { return `${e > 0 ? "+" : ""}${(e * 100).toFixed(1)}%`; }
