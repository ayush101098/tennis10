"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { fetchScheduleClient, probToOdds, kellyFraction } from "@/lib/scheduleService";
import type { ScheduledMatch, ScheduleData, BreakHoldSignals } from "@/lib/scheduleService";
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
  const refreshingRef = useRef(false);

  const refresh = useCallback(async () => {
    // Guard: skip if a refresh is already in progress
    if (refreshingRef.current) return;
    refreshingRef.current = true;
    setLoading(true);
    try {
      const result = await fetchScheduleClient();
      console.log("[SchedulePanel] loaded:", result?.today?.length, "today,", result?.tomorrow?.length, "tomorrow");
      setData(result);
    } catch (err) {
      console.error("[SchedulePanel] fetchScheduleClient FAILED:", err);
    } finally {
      setLoading(false);
      refreshingRef.current = false;
    }
  }, []);

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 30_000); // 30s — avoids flooding proxy
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
  const bookOdds = m.liveScore?.bookmakerOdds;
  const [odds1, setOdds1] = useState(() => bookOdds?.p1 || probToOdds(m.p1_win_prob));
  const [odds2, setOdds2] = useState(() => bookOdds?.p2 || probToOdds(m.p2_win_prob));
  const [bankroll, setBankroll] = useState(1000);

  // Recalc when match changes — prefer real bookmaker odds
  useEffect(() => {
    const bk = m.liveScore?.bookmakerOdds;
    setOdds1(bk?.p1 || probToOdds(m.p1_win_prob));
    setOdds2(bk?.p2 || probToOdds(m.p2_win_prob));
  }, [m.id, m.p1_win_prob, m.p2_win_prob, m.liveScore?.bookmakerOdds]);

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
          {bookOdds && <span className="text-terminal-yellow"> · Book: {bookOdds.p1.toFixed(2)} / {bookOdds.p2.toFixed(2)}</span>}
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
            <div className={`flex items-center justify-center gap-2 ${m.score.winner === 1 ? "text-terminal-green font-bold" : "text-slate-300"}`}>
              {m.status === "live" && m.liveScore?.server === 1 && <span className="text-[8px]">🎾</span>}
              <span className="w-[120px] text-right truncate">{m.player1}</span>
              <span>{m.score.p1_sets.join("  ")}</span>
              {m.status === "live" && m.liveScore?.pointScore && (
                <span className="text-terminal-yellow font-bold w-[20px]">{m.liveScore.pointScore.p1}</span>
              )}
            </div>
            <div className={`flex items-center justify-center gap-2 ${m.score.winner === 2 ? "text-terminal-green font-bold" : "text-slate-300"}`}>
              {m.status === "live" && m.liveScore?.server === 2 && <span className="text-[8px]">🎾</span>}
              <span className="w-[120px] text-right truncate">{m.player2}</span>
              <span>{m.score.p2_sets.join("  ")}</span>
              {m.status === "live" && m.liveScore?.pointScore && (
                <span className="text-terminal-yellow font-bold w-[20px]">{m.liveScore.pointScore.p2}</span>
              )}
            </div>
          </div>
          {m.status === "live" && m.liveScore?.tiebreakScore && (
            <div className="text-[9px] text-terminal-cyan text-center mt-1">Tiebreak: {m.liveScore.tiebreakScore.p1}-{m.liveScore.tiebreakScore.p2}</div>
          )}
          {m.status === "live" && m.liveScore?.statusDetail && (
            <div className="text-[9px] text-terminal-muted text-center mt-1">{m.liveScore.statusDetail}</div>
          )}
        </Section>
      )}

      {/* Live match statistics */}
      {m.status === "live" && m.liveScore?.stats && (
        <Section title="📊 LIVE STATS">
          <div className="grid grid-cols-3 gap-y-1 text-[10px] text-center">
            <span className="text-terminal-muted text-left">Stat</span>
            <span className="text-slate-300 font-bold">{m.player1.split(' ').pop()}</span>
            <span className="text-slate-300 font-bold">{m.player2.split(' ').pop()}</span>

            <span className="text-terminal-muted text-left">Aces</span>
            <StatVal v={m.liveScore.stats.p1_aces} o={m.liveScore.stats.p2_aces} />
            <StatVal v={m.liveScore.stats.p2_aces} o={m.liveScore.stats.p1_aces} />

            <span className="text-terminal-muted text-left">Double Faults</span>
            <StatVal v={m.liveScore.stats.p1_doubleFaults} o={m.liveScore.stats.p2_doubleFaults} lower />
            <StatVal v={m.liveScore.stats.p2_doubleFaults} o={m.liveScore.stats.p1_doubleFaults} lower />

            <span className="text-terminal-muted text-left">1st Serve %</span>
            <StatVal v={m.liveScore.stats.p1_firstServePercent} o={m.liveScore.stats.p2_firstServePercent} pct />
            <StatVal v={m.liveScore.stats.p2_firstServePercent} o={m.liveScore.stats.p1_firstServePercent} pct />

            <span className="text-terminal-muted text-left">1st Serve Won</span>
            <StatVal v={m.liveScore.stats.p1_firstServeWon} o={m.liveScore.stats.p2_firstServeWon} pct />
            <StatVal v={m.liveScore.stats.p2_firstServeWon} o={m.liveScore.stats.p1_firstServeWon} pct />

            <span className="text-terminal-muted text-left">2nd Serve Won</span>
            <StatVal v={m.liveScore.stats.p1_secondServeWon} o={m.liveScore.stats.p2_secondServeWon} pct />
            <StatVal v={m.liveScore.stats.p2_secondServeWon} o={m.liveScore.stats.p1_secondServeWon} pct />

            <span className="text-terminal-muted text-left">Break Points</span>
            <span className="text-slate-200 font-mono">{m.liveScore.stats.p1_breakPointsConverted}</span>
            <span className="text-slate-200 font-mono">{m.liveScore.stats.p2_breakPointsConverted}</span>

            <span className="text-terminal-muted text-left">Points Won</span>
            <StatVal v={m.liveScore.stats.p1_totalPointsWon} o={m.liveScore.stats.p2_totalPointsWon} />
            <StatVal v={m.liveScore.stats.p2_totalPointsWon} o={m.liveScore.stats.p1_totalPointsWon} />
          </div>
          {(() => {
            const s = m.liveScore!.stats!;
            const tot = (s.p1_totalPointsWon + s.p2_totalPointsWon) || 1;
            const dominance = s.p1_totalPointsWon / tot;
            const srvGap = s.p1_firstServePercent - s.p2_firstServePercent;
            return (
              <div className="mt-2 pt-2 border-t border-terminal-border space-y-1">
                <div className="flex justify-between text-[9px]">
                  <span className="text-terminal-muted">Point Dominance</span>
                  <span className={dominance > 0.52 ? "text-terminal-green font-bold" : dominance < 0.48 ? "text-terminal-red font-bold" : "text-slate-300"}>
                    {Math.round(dominance * 100)}% - {Math.round((1 - dominance) * 100)}%
                  </span>
                </div>
                <div className="h-1.5 bg-terminal-border rounded-full overflow-hidden">
                  <div className="h-full bg-terminal-green rounded-full" style={{ width: `${dominance * 100}%` }} />
                </div>
                {Math.abs(srvGap) > 10 && (
                  <div className="text-[9px] text-terminal-yellow">
                    ⚡ {srvGap > 0 ? m.player1.split(' ').pop() : m.player2.split(' ').pop()} serve dominance (+{Math.abs(Math.round(srvGap))}% 1st serve)
                  </div>
                )}
                {(s.p1_doubleFaults >= 4 || s.p2_doubleFaults >= 4) && (
                  <div className="text-[9px] text-terminal-red">
                    ⚠ {s.p1_doubleFaults >= s.p2_doubleFaults ? m.player1.split(' ').pop() : m.player2.split(' ').pop()} DF crisis ({Math.max(s.p1_doubleFaults, s.p2_doubleFaults)} DFs)
                  </div>
                )}
              </div>
            );
          })()}
        </Section>
      )}

      {/* ═══ BREAK/HOLD SIGNAL ENGINE ═══ */}
      {m.status === "live" && m.liveScore?.breakHoldSignals && (
        <BreakHoldPanel signals={m.liveScore.breakHoldSignals} m={m} />
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
   BREAK/HOLD SIGNAL PANEL — Industry-grade serve analysis
   ═══════════════════════════════════════════════════════════════════════════ */

function BreakHoldPanel({ signals: bh, m }: { signals: BreakHoldSignals; m: ScheduledMatch }) {
  const srvName = bh.server === 1 ? m.player1.split(" ").pop() : m.player2.split(" ").pop();
  const retName = bh.server === 1 ? m.player2.split(" ").pop() : m.player1.split(" ").pop();

  const dangerColors: Record<string, string> = {
    SAFE: "text-terminal-green",
    ALERT: "text-terminal-cyan",
    WARNING: "text-terminal-yellow",
    DANGER: "text-orange-400",
    CRITICAL: "text-terminal-red",
  };
  const dangerBg: Record<string, string> = {
    SAFE: "bg-terminal-green/10 border-terminal-green/30",
    ALERT: "bg-terminal-cyan/10 border-terminal-cyan/30",
    WARNING: "bg-terminal-yellow/10 border-terminal-yellow/30",
    DANGER: "bg-orange-400/10 border-orange-400/30",
    CRITICAL: "bg-terminal-red/10 border-terminal-red/30",
  };
  const dangerEmoji: Record<string, string> = {
    SAFE: "🟢", ALERT: "🔵", WARNING: "🟡", DANGER: "🟠", CRITICAL: "🔴",
  };

  const serTierColors: Record<string, string> = {
    ELITE: "text-terminal-green", STRONG: "text-terminal-cyan",
    AVERAGE: "text-slate-300", WEAK: "text-terminal-yellow", CRISIS: "text-terminal-red",
  };
  const rpiTierColors: Record<string, string> = {
    DOMINANT: "text-terminal-red", AGGRESSIVE: "text-orange-400",
    NEUTRAL: "text-slate-300", PASSIVE: "text-terminal-cyan", ABSENT: "text-terminal-green",
  };

  return (
    <>
      {/* Primary Signal Banner */}
      <div className={`p-2 rounded border ${dangerBg[bh.dangerLevel] || dangerBg.SAFE}`}>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm">{bh.primarySignal.emoji}</span>
          <span className={`text-[11px] font-bold flex-1 ${dangerColors[bh.dangerLevel]}`}>
            {bh.primarySignal.label}
          </span>
          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${dangerBg[bh.dangerLevel]}`}>
            {dangerEmoji[bh.dangerLevel]} {bh.dangerLevel}
          </span>
        </div>
        <div className="flex items-center gap-3 text-[9px] text-terminal-muted">
          <span>Phase: <b className="text-slate-300">{bh.gamePhase.replace(/_/g, " ")}</b></span>
          <span>Server: <b className="text-slate-300">{srvName}</b> 🎾</span>
          {bh.pointsToBreakPoint === 0 && <span className="text-terminal-red font-bold">● BREAK POINT</span>}
          {bh.pointsToBreakPoint > 0 && bh.pointsToBreakPoint <= 2 && (
            <span className="text-terminal-yellow">{bh.pointsToBreakPoint}pt to BP</span>
          )}
        </div>
      </div>

      {/* Hold / Break Probability Gauges */}
      <Section title="🎯 HOLD / BREAK PROBABILITY">
        <div className="space-y-2">
          {/* Hold gauge */}
          <div>
            <div className="flex items-center justify-between text-[10px] mb-0.5">
              <span className="text-terminal-muted">🛡 {srvName} Hold</span>
              <span className={`font-bold font-mono ${bh.holdProb > 0.75 ? "text-terminal-green" : bh.holdProb > 0.55 ? "text-terminal-yellow" : "text-terminal-red"}`}>
                {Math.round(bh.holdProb * 100)}%
              </span>
            </div>
            <div className="h-2.5 bg-terminal-border rounded-full overflow-hidden relative">
              <div className={`h-full rounded-full transition-all ${bh.holdProb > 0.75 ? "bg-terminal-green" : bh.holdProb > 0.55 ? "bg-terminal-yellow" : "bg-terminal-red"}`}
                style={{ width: `${bh.holdProb * 100}%` }} />
              {/* Tour average marker */}
              <div className="absolute top-0 h-full w-px bg-slate-400" style={{ left: "82%" }} title="Tour avg hold: 82%" />
            </div>
          </div>
          {/* Break gauge */}
          <div>
            <div className="flex items-center justify-between text-[10px] mb-0.5">
              <span className="text-terminal-muted">⚔️ {retName} Break</span>
              <span className={`font-bold font-mono ${bh.breakProb > 0.40 ? "text-terminal-red" : bh.breakProb > 0.25 ? "text-terminal-yellow" : "text-terminal-green"}`}>
                {Math.round(bh.breakProb * 100)}%
              </span>
            </div>
            <div className="h-2.5 bg-terminal-border rounded-full overflow-hidden relative">
              <div className={`h-full rounded-full transition-all ${bh.breakProb > 0.40 ? "bg-terminal-red" : bh.breakProb > 0.25 ? "bg-terminal-yellow" : "bg-terminal-green/50"}`}
                style={{ width: `${bh.breakProb * 100}%` }} />
              <div className="absolute top-0 h-full w-px bg-slate-400" style={{ left: "18%" }} title="Tour avg break: 18%" />
            </div>
          </div>
          {/* Pressure bar */}
          <div className="flex items-center gap-2 text-[9px]">
            <span className="text-terminal-muted shrink-0">Pressure</span>
            <div className="flex-1 h-1.5 bg-terminal-border rounded-full overflow-hidden">
              <div className={`h-full rounded-full ${bh.serverPressure > 60 ? "bg-terminal-red" : bh.serverPressure > 35 ? "bg-terminal-yellow" : "bg-terminal-green"}`}
                style={{ width: `${bh.serverPressure}%` }} />
            </div>
            <span className={`font-mono ${bh.serverPressure > 60 ? "text-terminal-red" : bh.serverPressure > 35 ? "text-terminal-yellow" : "text-terminal-green"}`}>
              {Math.round(bh.serverPressure)}
            </span>
          </div>
        </div>
      </Section>

      {/* Serve Efficiency Rating */}
      <Section title="📡 SERVE EFFICIENCY RATING (SER)">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-terminal-muted">{srvName}</span>
            <span className={`text-[11px] font-bold font-mono ${serTierColors[bh.serverSERTier]}`}>
              {bh.serverSER}/100
            </span>
          </div>
          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
            bh.serverSERTier === "ELITE" ? "bg-terminal-green/15 text-terminal-green" :
            bh.serverSERTier === "STRONG" ? "bg-terminal-cyan/15 text-terminal-cyan" :
            bh.serverSERTier === "CRISIS" ? "bg-terminal-red/15 text-terminal-red" :
            bh.serverSERTier === "WEAK" ? "bg-terminal-yellow/15 text-terminal-yellow" :
            "bg-slate-700/50 text-slate-300"
          }`}>
            {bh.serverSERTier}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[9px]">
          <KV label="1st Serve In" value={`${bh.serveBreakdown.firstServeIn}%`} color={bh.serveBreakdown.firstServeIn >= 60 ? "green" : bh.serveBreakdown.firstServeIn < 50 ? "red" : undefined} />
          <KV label="1st Serve Won" value={`${bh.serveBreakdown.firstServeWon}%`} color={bh.serveBreakdown.firstServeWon >= 70 ? "green" : bh.serveBreakdown.firstServeWon < 60 ? "red" : undefined} />
          <KV label="2nd Serve Won" value={`${bh.serveBreakdown.secondServeWon}%`} color={bh.serveBreakdown.secondServeWon >= 50 ? "green" : bh.serveBreakdown.secondServeWon < 40 ? "red" : undefined} />
          <KV label="DF/Game" value={bh.serveBreakdown.doubleFaultRate.toFixed(1)} color={bh.serveBreakdown.doubleFaultRate > 0.8 ? "red" : bh.serveBreakdown.doubleFaultRate < 0.3 ? "green" : undefined} />
          <KV label="Ace/Game" value={bh.serveBreakdown.aceRate.toFixed(1)} color={bh.serveBreakdown.aceRate > 1 ? "green" : undefined} />
          <KV label="Serve Trend" value={bh.serveTrend > 0 ? `↑ +${(bh.serveTrend * 100).toFixed(0)}%` : `↓ ${(bh.serveTrend * 100).toFixed(0)}%`}
            color={bh.serveTrend > 0.05 ? "green" : bh.serveTrend < -0.05 ? "red" : "muted"} />
        </div>
      </Section>

      {/* Return Pressure Index */}
      <Section title="🎯 RETURN PRESSURE INDEX (RPI)">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-terminal-muted">{retName}</span>
            <span className={`text-[11px] font-bold font-mono ${rpiTierColors[bh.returnerRPITier]}`}>
              {bh.returnerRPI}/100
            </span>
          </div>
          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
            bh.returnerRPITier === "DOMINANT" ? "bg-terminal-red/15 text-terminal-red" :
            bh.returnerRPITier === "AGGRESSIVE" ? "bg-orange-400/15 text-orange-400" :
            bh.returnerRPITier === "ABSENT" ? "bg-terminal-green/15 text-terminal-green" :
            bh.returnerRPITier === "PASSIVE" ? "bg-terminal-cyan/15 text-terminal-cyan" :
            "bg-slate-700/50 text-slate-300"
          }`}>
            {bh.returnerRPITier}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[9px]">
          <KV label="Ret Pts Won" value={`${bh.returnBreakdown.returnPointsWon}%`} color={bh.returnBreakdown.returnPointsWon >= 42 ? "red" : bh.returnBreakdown.returnPointsWon < 30 ? "green" : undefined} />
          <KV label="BP Conversion" value={`${bh.returnBreakdown.breakPointConversion}%`} color={bh.returnBreakdown.breakPointConversion >= 50 ? "red" : bh.returnBreakdown.breakPointConversion < 30 ? "green" : undefined} />
          <KV label="2nd Srv Return" value={`${bh.returnBreakdown.secondServeReturnWon}%`} color={bh.returnBreakdown.secondServeReturnWon >= 55 ? "red" : undefined} />
          <KV label="Return Trend" value={bh.returnTrend > 0 ? `↑ +${(bh.returnTrend * 100).toFixed(0)}%` : `↓ ${(bh.returnTrend * 100).toFixed(0)}%`}
            color={bh.returnTrend > 0.05 ? "red" : bh.returnTrend < -0.05 ? "green" : "muted"} />
        </div>
      </Section>

      {/* Active Signals */}
      {bh.signals.length > 0 && (
        <Section title="⚡ LIVE TRADE SIGNALS">
          <div className="space-y-1">
            {bh.signals.slice(0, 6).map((sig, i) => (
              <div key={i} className={`flex items-center gap-2 p-1.5 rounded text-[10px] ${
                sig.actionable ? "bg-terminal-yellow/5 border border-terminal-yellow/20" : "bg-terminal-panel/30 border border-terminal-border"
              }`}>
                <span className="text-sm shrink-0">{sig.emoji}</span>
                <span className={`flex-1 ${sig.actionable ? "text-slate-200 font-medium" : "text-terminal-muted"}`}>
                  {sig.label}
                </span>
                <span className={`text-[8px] font-bold px-1 py-0.5 rounded ${
                  sig.strength >= 70 ? "bg-terminal-red/20 text-terminal-red" :
                  sig.strength >= 40 ? "bg-terminal-yellow/20 text-terminal-yellow" :
                  "bg-terminal-panel text-terminal-muted"
                }`}>
                  {sig.strength}
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}
    </>
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
  const bh = m.liveScore?.breakHoldSignals;

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
            {live && m.liveScore?.server === 1 && <span className="text-[7px] text-terminal-green">🎾</span>}
            {!live && fav1 && known && <span className="text-[7px] text-terminal-green">▶</span>}
            {m.score?.winner === 1 && <span className="text-[7px] text-terminal-green">✓</span>}
            <span className={`text-[11px] truncate ${fav1 && known ? "text-slate-100 font-medium" : "text-slate-300"}`}>{m.player1}</span>
            {m.p1_rank > 0 && <span className="text-[8px] text-terminal-muted shrink-0">#{m.p1_rank}</span>}
            {m.p1_seed > 0 && <span className="text-[8px] text-terminal-cyan/70 shrink-0">[{m.p1_seed}]</span>}
            {m.score && <span className="text-[9px] text-terminal-muted ml-auto shrink-0 font-mono">{m.score.p1_sets.join(" ")}</span>}
            {live && m.liveScore?.pointScore && <span className="text-[9px] text-terminal-yellow font-bold shrink-0 font-mono w-[18px] text-right">{m.liveScore.pointScore.p1}</span>}
          </div>
          <div className="flex items-center gap-1 mt-0.5">
            {live && m.liveScore?.server === 2 && <span className="text-[7px] text-terminal-green">🎾</span>}
            {!live && !fav1 && known && <span className="text-[7px] text-terminal-green">▶</span>}
            {m.score?.winner === 2 && <span className="text-[7px] text-terminal-green">✓</span>}
            <span className={`text-[11px] truncate ${!fav1 && known ? "text-slate-100 font-medium" : "text-slate-300"}`}>{m.player2}</span>
            {m.p2_rank > 0 && <span className="text-[8px] text-terminal-muted shrink-0">#{m.p2_rank}</span>}
            {m.p2_seed > 0 && <span className="text-[8px] text-terminal-cyan/70 shrink-0">[{m.p2_seed}]</span>}
            {m.score && <span className="text-[9px] text-terminal-muted ml-auto shrink-0 font-mono">{m.score.p2_sets.join(" ")}</span>}
            {live && m.liveScore?.pointScore && <span className="text-[9px] text-terminal-yellow font-bold shrink-0 font-mono w-[18px] text-right">{m.liveScore.pointScore.p2}</span>}
          </div>
        </div>

        {/* Break/Hold badges */}
        {live && bh && (
          <div className="w-[40px] shrink-0 flex flex-col items-center gap-0.5">
            {bh.isBreakPoint && (
              <span className="text-[7px] font-bold px-1 py-0 rounded bg-terminal-red/20 text-terminal-red animate-pulse">BP!</span>
            )}
            {!bh.isBreakPoint && bh.dangerLevel === "CRITICAL" && (
              <span className="text-[7px] font-bold px-1 py-0 rounded bg-orange-400/20 text-orange-400">⚠</span>
            )}
            {!bh.isBreakPoint && bh.dangerLevel === "DANGER" && (
              <span className="text-[7px] font-bold px-1 py-0 rounded bg-terminal-yellow/20 text-terminal-yellow">⚡</span>
            )}
            {bh.holdProb > 0.85 && bh.dangerLevel === "SAFE" && (
              <span className="text-[7px] font-bold px-1 py-0 rounded bg-terminal-green/20 text-terminal-green">🛡</span>
            )}
            <span className={`text-[7px] font-mono ${
              bh.breakProb > 0.40 ? "text-terminal-red" :
              bh.breakProb > 0.25 ? "text-terminal-yellow" :
              "text-terminal-muted"
            }`}>
              {Math.round(bh.breakProb * 100)}%
            </span>
          </div>
        )}

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

/** Stat value cell — highlights who has the better stat (green = better, red = worse) */
function StatVal({ v, o, lower, pct: isPct }: { v: number; o: number; lower?: boolean; pct?: boolean }) {
  const better = lower ? v < o : v > o;
  const worse = lower ? v > o : v < o;
  const color = better ? "text-terminal-green font-bold" : worse ? "text-terminal-red" : "text-slate-200";
  return <span className={`font-mono ${color}`}>{v}{isPct ? "%" : ""}</span>;
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
