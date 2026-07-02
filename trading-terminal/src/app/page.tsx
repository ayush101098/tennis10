"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { fetchScheduleClient } from "@/lib/scheduleService";
import type { ScheduledMatch, ScheduleData } from "@/lib/scheduleService";
import { EdgePanel } from "@/components/SchedulePanel";
import PricingModal from "@/components/PricingModal";
import { useTier, claimPublicAnalysis, getPublicAnalysisId } from "@/lib/auth";

/**
 * Landing page — the public storefront.
 * Anyone can see today's live + upcoming matches across every professional
 * tour, and open the FULL analysis for exactly one match. The second click
 * opens the pricing modal (Free = pre-match probabilities, $99 Pro = the
 * complete trading terminal).
 */
export default function LandingPage() {
  const { session, tier, refresh } = useTier();
  const [data, setData] = useState<ScheduleData | null>(null);
  const [selected, setSelected] = useState<ScheduledMatch | null>(null);
  const [pricingOpen, setPricingOpen] = useState(false);
  const isPro = tier === "pro";

  useEffect(() => {
    const load = () => fetchScheduleClient().then(setData).catch(() => {});
    load();
    const iv = setInterval(load, 30_000);
    return () => clearInterval(iv);
  }, []);

  const matches = useMemo(() => {
    if (!data) return [];
    const order = { live: 0, scheduled: 1, finished: 2, cancelled: 3 } as const;
    return [...data.today]
      .filter(m => m.status === "live" || m.status === "scheduled")
      .sort((a, b) => (order[a.status] - order[b.status]) || (a.start_timestamp - b.start_timestamp));
  }, [data]);

  const liveCount = matches.filter(m => m.status === "live").length;
  const tours = useMemo(() => Array.from(new Set(matches.map(m => m.tour))), [matches]);

  const onPick = useCallback((m: ScheduledMatch) => {
    // Signed-in users analyse freely (depth still gated by tier inside EdgePanel).
    if (session) { setSelected(m); return; }
    // Public visitors: exactly one match per day.
    if (claimPublicAnalysis(m.id)) setSelected(m);
    else setPricingOpen(true);
  }, [session]);

  return (
    <div className="min-h-screen bg-terminal-bg text-slate-200">
      {/* ── Nav ── */}
      <nav className="sticky top-0 z-40 flex items-center justify-between px-6 py-3 border-b border-terminal-border bg-terminal-bg/95 backdrop-blur">
        <span className="text-terminal-green font-bold text-sm">◉ TENNIS INTELLIGENCE TERMINAL</span>
        <div className="flex items-center gap-3 text-[11px]">
          {session ? (
            <>
              <span className={`font-bold px-1.5 py-0.5 rounded ${session.isAdmin ? "bg-terminal-red/20 text-terminal-red" : isPro ? "bg-terminal-green/20 text-terminal-green" : "bg-terminal-border text-slate-300"}`}>
                {session.isAdmin ? "ADMIN" : isPro ? "PRO" : "FREE"}
              </span>
              <Link href="/terminal" className="font-bold px-3 py-1.5 rounded bg-terminal-green text-black hover:opacity-90">
                LAUNCH TERMINAL →
              </Link>
            </>
          ) : (
            <>
              <button onClick={() => setPricingOpen(true)} className="text-terminal-muted hover:text-slate-200">Pricing</button>
              <button onClick={() => setPricingOpen(true)} className="font-bold px-3 py-1.5 rounded bg-terminal-green text-black hover:opacity-90">
                GET ACCESS
              </button>
            </>
          )}
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="px-6 pt-14 pb-10 text-center max-w-[860px] mx-auto">
        <h1 className="text-3xl md:text-4xl font-bold text-slate-100 leading-tight">
          True probabilities for <span className="text-terminal-green">every professional tennis match.</span>
        </h1>
        <p className="mt-4 text-[13px] text-slate-400 leading-relaxed max-w-[640px] mx-auto">
          A neural network trained on 41,750 tour matches, fused with a score-conditioned Markov engine,
          priced against live bookmaker odds — with ¼-Kelly staking and hedge-timing discipline built in.
          ATP · WTA · Challenger · W125 · ITF, every day.
        </p>
        <div className="mt-6 flex items-center justify-center gap-3">
          <button onClick={() => setPricingOpen(true)}
            className="px-5 py-2.5 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90">
            GET FULL ACCESS — $99
          </button>
          <a href="#matches" className="px-5 py-2.5 rounded border border-terminal-border text-xs font-bold text-slate-200 hover:bg-terminal-panel">
            SEE TODAY&apos;S MATCHES ↓
          </a>
        </div>
        {/* Stat chips */}
        <div className="mt-8 flex flex-wrap items-center justify-center gap-2 text-[10px]">
          <Chip label={`${matches.length || "—"} matches today`} />
          <Chip label={`${liveCount} live now`} tone="green" pulse={liveCount > 0} />
          <Chip label={`${tours.length || "—"} tours incl. ITF`} />
          <Chip label="41,750-match neural network" />
          <Chip label="Markov live engine" />
          <Chip label="¼-Kelly staking" />
        </div>
      </section>

      {/* ── Live board + analysis ── */}
      <section id="matches" className="px-6 pb-14 max-w-[1180px] mx-auto">
        <div className="border border-terminal-border rounded-lg overflow-hidden bg-terminal-panel/30">
          <div className="flex items-center justify-between px-4 py-2 border-b border-terminal-border bg-terminal-panel/60">
            <span className="text-[11px] font-bold text-terminal-yellow tracking-wider">📅 TODAY — LIVE &amp; UPCOMING</span>
            <span className="text-[10px] text-terminal-muted">
              {session ? "click any match to analyse" : "analyse 1 match free — no account needed"}
            </span>
          </div>
          <div className="grid md:grid-cols-[1fr_420px]" style={{ minHeight: 420, maxHeight: 560 }}>
            {/* Match list */}
            <div className="overflow-y-auto border-r border-terminal-border" style={{ maxHeight: 560 }}>
              {!data && <div className="p-8 text-center text-terminal-muted text-xs animate-pulse">Loading live schedule…</div>}
              {matches.slice(0, 120).map(m => (
                <PublicRow key={m.id} m={m} active={selected?.id === m.id}
                  freeSlot={!session && getPublicAnalysisId() === m.id}
                  showProb={tier !== "public"}
                  onClick={() => onPick(m)} />
              ))}
              {data && matches.length === 0 && (
                <div className="p-8 text-center text-terminal-muted text-xs">No more matches today — check back tomorrow.</div>
              )}
            </div>
            {/* Analysis pane */}
            <div className="overflow-y-auto bg-terminal-bg" style={{ maxHeight: 560 }}>
              {selected ? (
                <EdgePanel
                  match={selected}
                  tier={session ? tier : "pro" /* the one free public analysis shows the full stack */}
                  onUpgrade={() => setPricingOpen(true)}
                />
              ) : (
                <div className="h-full flex flex-col items-center justify-center gap-2 text-center p-8">
                  <div className="text-2xl">⚡</div>
                  <div className="text-[12px] font-bold text-slate-200">Pick a match to see the full analysis</div>
                  <div className="text-[10px] text-terminal-muted max-w-[280px]">
                    Model probability, bookmaker edge, Kelly stake, live break/hold signals and hedge timing
                    {session ? "" : " — one match free, then choose a plan"}.
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section className="px-6 pb-14 max-w-[1000px] mx-auto">
        <h2 className="text-center text-lg font-bold text-slate-100 mb-6">The full process, end to end</h2>
        <div className="grid md:grid-cols-4 gap-3">
          <Feature n="01" title="TRUE P" body="A Platt-calibrated neural network (41,750 tour matches) sets the pre-match prior; a tour-aware Markov engine re-prices the match on every game of the live score." />
          <Feature n="02" title="EDGE" body="True P is compared against de-vigged bookmaker odds — live prices for live matches, never stale ones. Edges over 20% are quarantined as data errors, not bets." />
          <Feature n="03" title="STAKE" body="¼-Kelly staking capped at 5% of bankroll, with a hard 2% edge floor. The discipline is the product: no edge, no bet." />
          <Feature n="04" title="HEDGE" body="Trend-break, adverse-move and deuce-loss triggers tell you when to hedge a live position — protecting profit beats chasing it." />
        </div>
      </section>

      {/* ── Pricing ── */}
      <section className="px-6 pb-16 max-w-[720px] mx-auto text-center">
        <h2 className="text-lg font-bold text-slate-100 mb-2">Simple pricing</h2>
        <p className="text-[11px] text-terminal-muted mb-6">Free gets you the model&apos;s pre-match probabilities on every match. Pro gets you the terminal.</p>
        <div className="flex items-center justify-center gap-4">
          <button onClick={() => setPricingOpen(true)}
            className="px-6 py-3 rounded border border-terminal-border text-xs font-bold text-slate-200 hover:bg-terminal-panel">
            START FREE
          </button>
          <button onClick={() => setPricingOpen(true)}
            className="px-6 py-3 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90">
            GO PRO — $99 ONE-TIME
          </button>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="px-6 py-6 border-t border-terminal-border text-center text-[9px] text-terminal-muted leading-relaxed">
        Model outputs are calibrated probabilities, not guarantees. Sports betting involves risk — bet only what you can afford to lose.
        Staking discipline (¼ Kelly, 5% cap, 2% edge floor) is enforced in the product for a reason.
        <br />© {new Date().getFullYear()} Tennis Intelligence Terminal
      </footer>

      <PricingModal open={pricingOpen} onClose={() => setPricingOpen(false)} onDone={refresh} />
    </div>
  );
}

/* ── Sub-components ── */

function Chip({ label, tone, pulse }: { label: string; tone?: "green"; pulse?: boolean }) {
  return (
    <span className={`px-2.5 py-1 rounded-full border ${tone === "green" ? "border-terminal-green/40 text-terminal-green" : "border-terminal-border text-slate-400"}`}>
      {pulse && <span className="inline-block w-1.5 h-1.5 rounded-full bg-terminal-green animate-pulse mr-1.5 align-middle" />}
      {label}
    </span>
  );
}

function Feature({ n, title, body }: { n: string; title: string; body: string }) {
  return (
    <div className="border border-terminal-border rounded-lg p-4 bg-terminal-panel/30">
      <div className="text-[10px] text-terminal-muted font-bold">{n}</div>
      <div className="text-terminal-green font-bold text-sm mt-1 mb-2">{title}</div>
      <div className="text-[10px] text-slate-400 leading-relaxed">{body}</div>
    </div>
  );
}

function PublicRow({ m, active, freeSlot, showProb, onClick }: {
  m: ScheduledMatch; active: boolean; freeSlot: boolean; showProb: boolean; onClick: () => void;
}) {
  const live = m.status === "live";
  return (
    <button onClick={onClick}
      className={`w-full flex items-center gap-2 px-3 py-2 border-b border-terminal-border text-left transition ${
        active ? "bg-terminal-cyan/10 border-l-2 border-l-terminal-cyan" : live ? "bg-terminal-green/5 hover:bg-terminal-green/10" : "hover:bg-terminal-panel/40"
      }`}>
      <span className="w-[42px] shrink-0 text-center">
        {live ? (
          <span className="text-[9px] text-terminal-green font-bold">● LIVE</span>
        ) : (
          <span className="text-[9px] text-terminal-muted">{m.start_time || "TBD"}</span>
        )}
      </span>
      <span className="w-[46px] shrink-0 text-[8px] font-bold text-terminal-cyan/80">{m.tour}</span>
      <span className="flex-1 min-w-0">
        <span className="block text-[11px] text-slate-200 truncate">{m.player1}</span>
        <span className="block text-[11px] text-slate-400 truncate">{m.player2}</span>
      </span>
      {live && m.score && (
        <span className="shrink-0 font-mono text-[10px] text-terminal-yellow text-right">
          <span className="block">{m.score.p1_sets.join(" ")}</span>
          <span className="block">{m.score.p2_sets.join(" ")}</span>
        </span>
      )}
      {showProb && m.prob_method !== "unknown" && (
        <span className="shrink-0 w-[44px] font-mono text-[10px] text-right">
          <span className={`block ${m.p1_win_prob >= 0.5 ? "text-terminal-green font-bold" : "text-slate-500"}`}>{Math.round(m.p1_win_prob * 100)}%</span>
          <span className={`block ${m.p1_win_prob < 0.5 ? "text-terminal-green font-bold" : "text-slate-500"}`}>{Math.round(m.p2_win_prob * 100)}%</span>
        </span>
      )}
      <span className="shrink-0 w-[52px] text-right">
        {freeSlot
          ? <span className="text-[8px] font-bold text-terminal-yellow">YOUR PICK</span>
          : <span className="text-[8px] text-terminal-muted">analyse ›</span>}
      </span>
    </button>
  );
}
