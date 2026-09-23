"use client";

/**
 * The main terminal view — replaces the old tabbed match-centre.
 *
 * Operator instruction, 2026-09-20: no tabs, no tour/status filters. One
 * screen, a grid of matches, click one to see it. This is that screen.
 *
 * WHAT WAS REMOVED AND WHY
 *   MATCH CENTRE / VALUE BOARD toggle, TODAY / TOMORROW toggle, the ALL / ITF /
 *   ATP / WTA / CHALLENGER filter row, the ALL / LIVE / SCHED / DONE status row,
 *   and the EDGE / TRACKER tab pair on the detail pane — six controls a visitor
 *   had to learn before finding a match. A grid you scan and a card you click
 *   needs none of them: live matches sort first by construction, and a card
 *   already shows its tour and surface.
 *
 * WHAT REPLACED THE EDGE PANEL
 *   The old detail pane led with editable odds inputs, implied probability,
 *   overround and ¼-Kelly staking — bookmaker-terminal language for a page that
 *   is supposed to read as "who is winning and why". The expanded card instead
 *   leads with LiveSignal (win chance, before/live, the pressure signal) and
 *   the real per-match numbers from lib/gamePressure and lib/liveTelemetry —
 *   the same score-derived engines built this session, with no fabricated
 *   figures and no "0 seen so far" dressed up as data.
 *
 * Finished matches do not appear here at all — "happening" is the point of
 * this screen, and a completed match has nothing left to click into.
 */

import { useEffect, useState, useCallback, useRef } from "react";
import { fetchScheduleClient, refreshLiveMatches, tourRank } from "@/lib/scheduleService";
import type { ScheduledMatch, ScheduleData } from "@/lib/scheduleService";
import { displayName } from "@/lib/scheduleService";
import LiveSignal from "@/components/LiveSignal";
import { FEED_STALE_MS, ODDS_STALE_MS } from "@/lib/scheduleService";

// push_sofa.py refreshes the score cache roughly every 15s, so polling faster
// than that can't surface new data — 8s just means most cycles find nothing
// changed. Kept snappier than the cache interval so a fresh push is picked up
// quickly rather than waiting out a matching 15s window.
const LIVE_POLL_MS = 8_000;
const FULL_POLL_MS = 45_000;

export default function LiveGrid() {
  const [data, setData] = useState<ScheduleData | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const refreshingRef = useRef(false);
  const dataRef = useRef<ScheduleData | null>(null);
  dataRef.current = data;

  const refresh = useCallback(async () => {
    if (refreshingRef.current) return;
    refreshingRef.current = true;
    try {
      const result = await fetchScheduleClient(partial => { setData(partial); setLoading(false); });
      setData(result);
    } catch (err) {
      console.error("[LiveGrid] load failed:", err);
    } finally {
      setLoading(false);
      refreshingRef.current = false;
    }
  }, []);

  useEffect(() => {
    refresh();
    const iv = setInterval(() => { if (document.visibilityState === "visible") refresh(); }, FULL_POLL_MS);
    return () => clearInterval(iv);
  }, [refresh]);

  // Fast path: re-price live matches between full rebuilds, same as before.
  useEffect(() => {
    const iv = setInterval(async () => {
      if (document.visibilityState !== "visible") return;
      const prev = dataRef.current;
      if (!prev || !prev.today.some(m => m.status === "live")) return;
      const changed = await refreshLiveMatches([...prev.today, ...prev.tomorrow]);
      if (changed) setData({ ...prev, today: [...prev.today], tomorrow: [...prev.tomorrow] });
    }, LIVE_POLL_MS);
    return () => clearInterval(iv);
  }, []);

  // Everything happening today, live matches first — no day toggle, no filter.
  // Tomorrow's card is not shown here: "matches happening" is today's matches,
  // and a match that has not started yet has no signal to click into.
  const order = { live: 0, scheduled: 1, finished: 2, cancelled: 3 } as const;
  const all = (data?.today ?? []).slice().sort((a, b) =>
    ((order[a.status] ?? 2) - (order[b.status] ?? 2)) ||
    (tourRank(a.tour) - tourRank(b.tour)) ||
    ((a.start_timestamp || 9e9) - (b.start_timestamp || 9e9)));

  const live = all.filter(m => m.status === "live");
  const upcoming = all.filter(m => m.status === "scheduled");
  // Finished matches are not "happening" — the instruction this screen follows
  // is explicit about that word. A day can finish 100+ matches across every
  // tour, and a wall of them under a "live matches" screen is exactly the
  // clutter this redesign removed everywhere else. Left off entirely rather
  // than capped-with-a-count, which just relocates the same clutter.

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border shrink-0">
        <div className="flex items-center gap-2 text-xs font-bold text-slate-200">
          <span className="w-1.5 h-1.5 rounded-full bg-terminal-green animate-pulse" />
          {live.length} live now
        </div>
        <button onClick={refresh} disabled={loading}
          className="text-[10px] bg-terminal-blue/20 text-terminal-blue border border-terminal-blue/40 rounded px-2 py-0.5 hover:bg-terminal-blue/30 disabled:opacity-50">
          {loading ? "⟳" : "↻ refresh"}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        <StaleBanner ageMs={data?.feedAgeMs} fallback={data?.fallbackActive}
          oddsAgeMs={data?.oddsAgeMs} oddsDown={data?.oddsDown} />

        {loading && !data && (
          <div className="flex items-center justify-center h-40 text-terminal-muted text-xs animate-pulse">
            Loading matches…
          </div>
        )}

        {!loading && live.length === 0 && (
          <div className="text-center py-10 text-terminal-muted text-xs">
            {data?.sourcesDown
              ? "⚠ The match feed isn't responding — this is on our side, it reconnects on its own."
              : "Nothing live right now. Check back when a match starts, or see what's coming up below."}
          </div>
        )}

        {live.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3 mb-6">
            {live.map(m => (
              <MatchCard key={m.id} m={m}
                expanded={expandedId === m.id}
                onToggle={() => setExpandedId(expandedId === m.id ? null : m.id)} />
            ))}
          </div>
        )}

        {upcoming.length > 0 && (
          <>
            <div className="text-[10px] font-bold text-terminal-muted tracking-wider mb-2">
              UPCOMING TODAY — {upcoming.length}
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-2 mb-6">
              {upcoming.map(m => <UpcomingCard key={m.id} m={m} />)}
            </div>
          </>
        )}


      </div>
    </div>
  );
}

/**
 * A live match. Collapsed: enough to scan a grid of them. Expanded: the whole
 * LiveSignal panel inline, in place — no navigation, no side pane.
 */
function MatchCard({ m, expanded, onToggle }: { m: ScheduledMatch; expanded: boolean; onToggle: () => void }) {
  const ls = m.liveScore;
  const liveP1 = ls?.trueProbabilities?.p1MatchProb;
  const p1 = liveP1 ?? m.p1_win_prob;
  const fav1 = p1 >= 0.5;
  const bh = ls?.breakHoldSignals;
  const lm = ls?.liveMomentum;

  // Pre-match vs live, for the SAME (favoured) player — the point of showing
  // both is the movement between them, and comparing two different players'
  // numbers would not show that.
  const prePct = Math.round((fav1 ? m.p1_win_prob : 1 - m.p1_win_prob) * 100);
  const livePct = Math.round((fav1 ? p1 : 1 - p1) * 100);
  const delta = livePct - prePct;
  const favShort = displayName(fav1 ? m.player1 : m.player2);

  return (
    <div className={`rounded-lg border transition-all ${
      expanded ? "sm:col-span-2 xl:col-span-3 border-terminal-cyan/50 bg-terminal-panel/60" : "border-terminal-border bg-terminal-panel/30 hover:border-terminal-cyan/30"
    }`}>
      <button onClick={onToggle} className="w-full text-left p-3">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-terminal-green animate-pulse" />
            <span className="text-[9px] font-bold text-terminal-green">LIVE</span>
            <TourBadge t={m.tour} />
          </div>
          <span className="text-[9px] text-terminal-muted truncate max-w-[120px]">{m.tournament}</span>
        </div>

        <div className="flex items-center justify-between gap-2">
          <span className={`text-[13px] truncate ${fav1 ? "text-slate-100 font-bold" : "text-slate-300"}`}>{m.player1}</span>
          <span className="text-[11px] font-mono text-terminal-muted shrink-0">{m.score?.p1_sets.join(" ")}</span>
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className={`text-[13px] truncate ${!fav1 ? "text-slate-100 font-bold" : "text-slate-300"}`}>{m.player2}</span>
          <span className="text-[11px] font-mono text-terminal-muted shrink-0">{m.score?.p2_sets.join(" ")}</span>
        </div>

        {/* Pre-match vs live, always both, right on the card — no click needed
            to see what the match has done to the forecast. */}
        <div className="flex items-center gap-2 mt-2 pt-2 border-t border-terminal-border/60">
          <div title="Before a ball was struck">
            <div className="text-[8px] text-terminal-muted">PRE-MATCH</div>
            <div className="text-sm font-mono text-slate-400">{prePct}%</div>
          </div>
          <div className="text-terminal-muted text-xs">→</div>
          <div title="Re-priced on the current score">
            <div className="text-[8px] text-terminal-green font-bold">LIVE NOW</div>
            <div className="flex items-baseline gap-1">
              <span className="text-lg font-bold font-mono text-terminal-green">{livePct}%</span>
              {Math.abs(delta) >= 1 && (
                <span className={`text-[10px] font-bold font-mono ${delta > 0 ? "text-terminal-green" : "text-terminal-red"}`}>
                  {delta > 0 ? "▲" : "▼"}{Math.abs(delta)}
                </span>
              )}
            </div>
          </div>
          <span className="text-[10px] text-terminal-muted ml-0.5">{favShort} to win</span>
          {bh?.isBreakPoint && (
            <span className="ml-auto text-[9px] font-bold px-1.5 py-0.5 rounded bg-terminal-red/20 text-terminal-red animate-pulse">BP!</span>
          )}
        </div>

        {/* Momentum right on the card too, when there is enough of a point
            sequence to say something real — see lib/liveTelemetry. */}
        {lm && (
          <div className="text-[9px] text-terminal-muted mt-1">
            momentum: {Math.abs(lm.p1) < 0.02 ? "even" : `${lm.p1 > 0 ? displayName(m.player1) : displayName(m.player2)} +${Math.round(Math.abs(lm.p1) * 100)}%`}
            <span className="opacity-70"> · {lm.observedPoints} pts seen</span>
          </div>
        )}

        <div className="text-[9px] text-terminal-muted mt-1">{expanded ? "▲ tap to close" : "▼ tap for signals & momentum"}</div>
      </button>

      {expanded && (
        <div className="border-t border-terminal-border p-3">
          <LiveSignal match={m} />
        </div>
      )}
    </div>
  );
}

function UpcomingCard({ m }: { m: ScheduledMatch }) {
  const fav1 = m.p1_win_prob >= 0.5;
  return (
    <div className="rounded border border-terminal-border bg-terminal-panel/20 p-2.5">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[9px] text-terminal-muted">{m.start_time || "TBD"}</span>
        <TourBadge t={m.tour} />
      </div>
      <div className={`text-[11px] truncate ${fav1 ? "text-slate-200 font-medium" : "text-slate-400"}`}>{m.player1}</div>
      <div className={`text-[11px] truncate ${!fav1 ? "text-slate-200 font-medium" : "text-slate-400"}`}>{m.player2}</div>
    </div>
  );
}

function TourBadge({ t }: { t?: string }) {
  const c: Record<string, string> = {
    ATP: "text-blue-400 bg-blue-400/10",
    WTA: "text-pink-400 bg-pink-400/10",
    "ITF M": "text-emerald-400 bg-emerald-400/10",
    "ITF W": "text-rose-300 bg-rose-300/10",
    CHALLENGER: "text-amber-400 bg-amber-400/10",
    W125: "text-fuchsia-400 bg-fuchsia-400/10",
    "DAVIS CUP": "text-sky-400 bg-sky-400/10",
  };
  const shortLabel: Record<string, string> = { CHALLENGER: "CHAL", "DAVIS CUP": "DC" };
  return <span className={`text-[8px] font-bold px-1 rounded shrink-0 ${c[t || ""] || "text-terminal-muted bg-terminal-muted/10"}`}>{shortLabel[t || ""] || t}</span>;
}

/** Same banner as before — the feed-honesty work stays, only its home moves. */
function StaleBanner({ ageMs, fallback, oddsAgeMs, oddsDown }: {
  ageMs?: number; fallback?: boolean; oddsAgeMs?: number; oddsDown?: boolean;
}) {
  const fmt = (ms: number) => {
    const mins = Math.round(ms / 60000);
    return mins < 90 ? `${mins} min` : `${Math.round(mins / 60)} h`;
  };
  const scoresStale = !!ageMs && ageMs >= FEED_STALE_MS;
  const oddsStale = (!!oddsAgeMs && oddsAgeMs >= ODDS_STALE_MS) || !!oddsDown;
  if (!scoresStale && !oddsStale) return null;

  if (!scoresStale) {
    return (
      <div className="mb-3 px-3 py-2 rounded border border-terminal-yellow/40 bg-terminal-yellow/10">
        <div className="text-[11px] font-bold text-terminal-yellow">
          {oddsDown ? "⚠ Bookmaker feed down — pricing from Polymarket only"
            : `⚠ No live market — bookmaker odds are ${fmt(oddsAgeMs!)} old`}
        </div>
        <div className="text-[10px] text-terminal-muted mt-0.5">
          Scores and win chances are live. Prices are withheld rather than shown at their last value.
        </div>
      </div>
    );
  }

  const age = fmt(ageMs!);
  return (
    <div className="mb-3 px-3 py-2 rounded border border-terminal-yellow/40 bg-terminal-yellow/10">
      <div className="text-[11px] font-bold text-terminal-yellow">
        {fallback ? "⚠ Primary feed down — ATP & WTA on backup source" : `⚠ Feed delayed — scores are ${age} old`}
      </div>
      <div className="text-[10px] text-terminal-muted mt-0.5">
        Don&apos;t rely on scores or signals below until this clears.
      </div>
    </div>
  );
}
