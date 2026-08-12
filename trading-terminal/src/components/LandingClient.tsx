"use client";

// 5s was chosen for point-by-point immediacy. Games take ~45s, so the board
// cannot change faster than that — 15s shows every score change and cuts the
// request rate by two thirds.
const LIVE_POLL_MS = 15_000;
/** Matches a free visitor sees on the homepage. */
const FREE_MATCH_LIMIT = 3;

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { SsrMatch } from "@/components/SsrMatchList";
import SsrMatchList from "@/components/SsrMatchList";
import Link from "next/link";
import Wordmark from "@/components/Wordmark";
import Socials from "@/components/Socials";
import TrialBanner from "@/components/TrialBanner";
import Faq from "@/components/Faq";
import SiteFooter from "@/components/SiteFooter";
import { DonatePrompt } from "@/components/Donate";
import { SoftwareApplicationLd } from "@/components/JsonLd";
import { fetchScheduleClient, refreshLiveMatches, tourRank } from "@/lib/scheduleService";
import type { ScheduledMatch, ScheduleData } from "@/lib/scheduleService";
import { EdgePanel } from "@/components/SchedulePanel";
import EmailCapture from "@/components/EmailCapture";
import { useTier, TRIAL_DAYS } from "@/lib/auth";
import PricingModal from "@/components/PricingModal";

/**
 * Landing page — the public storefront.
 * Anyone can see today's live + upcoming matches across every professional
 * tour, and open the FULL analysis for exactly one match. The second click
 * opens the pricing modal (Free = pre-match probabilities, $99 Pro = the
 * complete trading terminal).
 */
export default function LandingClient({ initialMatches = [] }: { initialMatches?: SsrMatch[] }) {
  const { session, tier } = useTier();
  const [data, setData] = useState<ScheduleData | null>(null);
  const [selected, setSelected] = useState<ScheduledMatch | null>(null);
  const isPro = tier === "pro";
  const dataRef = useRef<ScheduleData | null>(null);
  dataRef.current = data;

  useEffect(() => {
    // Full schedule rebuild (ESPN + SofaScore + rankings + odds for ~450
    // matches) is expensive — 45s is often enough since the match LIST
    // rarely changes mid-cycle; live scores are kept fresh separately below.
    // setData twice on purpose: today's rows as soon as they land, then the
    // complete result. Waiting for both days plus live odds cost ~2s of blank board.
    const load = () => fetchScheduleClient(setData).then(setData).catch(() => {});
    load();
    // A hidden tab must not poll. Background tabs left open for hours were
    // costing as much as active readers.
    const iv = setInterval(() => { if (document.visibilityState === "visible") load(); }, 45_000);
    return () => clearInterval(iv);
  }, []);

  // Lightweight live-score refresh — ONE bulk request updates every live
  // match's score/point and recomputes True P, so the board tracks points
  // as they happen instead of waiting on the next full 45s rebuild.
  useEffect(() => {
    const iv = setInterval(async () => {
      // Only while a live match is actually on screen AND the tab is visible.
      // This ran every 5s regardless — on a page with no live matches, in a
      // background tab, forever — which is most of the traffic that blew
      // through the hosting limits.
      if (document.visibilityState !== "visible") return;
      const prev = dataRef.current;
      if (!prev) return;
      if (!prev.today.some(m => m.status === "live")) return;   // nothing live to refresh
      const changed = await refreshLiveMatches([...prev.today, ...prev.tomorrow]);
      if (changed) setData({ ...prev, today: [...prev.today], tomorrow: [...prev.tomorrow] });
    }, LIVE_POLL_MS);
    return () => clearInterval(iv);
  }, []);

  const matches = useMemo(() => {
    if (!data) return [];
    const order = { live: 0, scheduled: 1, finished: 2, cancelled: 3 } as const;
    return [...data.today]
      .filter(m => m.status === "live" || m.status === "scheduled")
      // live, then tour tier, then time — see tourRank
      .sort((a, b) =>
        (order[a.status] - order[b.status]) ||
        (tourRank(a.tour) - tourRank(b.tour)) ||
        (a.start_timestamp - b.start_timestamp));
  }, [data]);

  const liveCount = matches.filter(m => m.status === "live").length;
  const tours = useMemo(() => Array.from(new Set(matches.map(m => m.tour))), [matches]);

  // Every match opens for everyone. The gate is on the ACTIONABLE layer, not on
  // access: edge, Kelly stakes and the trade signals render blurred for
  // non-subscribers, so a visitor sees the shape of the answer and what it is
  // worth. Locking whole matches instead hid the product from the people it
  // needs to convince — and from search engines, which index this page.
  const [pricingOpen, setPricingOpen] = useState(false);
  const onPick = useCallback((m: ScheduledMatch) => setSelected(m), []);

  return (
    <div className="min-h-screen bg-terminal-bg text-slate-200">
      <SoftwareApplicationLd />
      {/* ── Nav ── */}
      <nav className="sticky top-0 z-40 flex items-center justify-between gap-2 px-3 sm:px-6 py-3 border-b border-terminal-border bg-terminal-bg/95 backdrop-blur">
        <Wordmark size={17} />
        <div className="flex items-center gap-2 sm:gap-3 text-[11px] shrink-0">
          <Socials />
          <Link href="/manual" className="hidden sm:inline text-terminal-muted hover:text-slate-200">Manual</Link>
          {session ? (
            <>
              <span className={`font-bold px-1.5 py-0.5 rounded ${session.isAdmin ? "bg-terminal-red/20 text-terminal-red" : isPro ? "bg-terminal-green/20 text-terminal-green" : "bg-terminal-border text-slate-300"}`}>
                {session.isAdmin ? "ADMIN" : isPro ? "PRO" : "FREE"}
              </span>
              <Link href="/terminal" className="inline-flex items-center justify-center min-h-[40px] font-bold px-3 rounded bg-terminal-green text-black hover:opacity-90">
                LAUNCH TERMINAL →
              </Link>
            </>
          ) : (
            <Link href="/terminal" className="inline-flex items-center justify-center min-h-[40px] font-bold px-3 rounded bg-terminal-green text-black hover:opacity-90">
              OPEN TERMINAL →
            </Link>
          )}
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="px-4 sm:px-6 pt-10 sm:pt-14 pb-10 text-center max-w-[860px] mx-auto">
        <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-slate-100 leading-tight">
          True probabilities for <span className="text-terminal-green">every professional tennis match.</span>
        </h1>
        <p className="mt-4 text-[13px] text-slate-400 leading-relaxed max-w-[640px] mx-auto">
          A neural network trained on 41,750 tour matches, fused with a score-conditioned Markov engine,
          priced against live bookmaker odds — with ¼-Kelly staking and hedge-timing discipline built in.
          ATP · WTA · Challenger · W125 · ITF, every day.
        </p>
        <div className="mt-6 flex items-center justify-center gap-3">
          <Link href="/terminal"
            className="inline-flex items-center justify-center min-h-[44px] px-5 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90">
            OPEN THE TERMINAL →
          </Link>
          <a href="#matches" className="inline-flex items-center justify-center min-h-[44px] px-5 rounded border border-terminal-border text-xs font-bold text-slate-200 hover:bg-terminal-panel">
            SEE TODAY&apos;S MATCHES ↓
          </a>
        </div>
        <div className="mt-6 max-w-[560px] mx-auto">
          <TrialBanner onStart={() => setPricingOpen(true)} />
        </div>

        {/* Stat chips */}
        <div className="mt-8 flex flex-wrap items-center justify-center gap-2 text-[10px]">
          <Chip label={`${matches.length || "—"} matches today`} />
          <Chip label={`${liveCount} matches live now`} tone="green" pulse={liveCount > 0} />
          <Chip label={`${tours.length || "—"} tours incl. ITF`} />
          <Chip label="41,750-match neural network" />
          <Chip label="Markov live engine" />
          <Chip label="¼-Kelly staking" />
        </div>
      </section>

      {/* ── Live board + analysis ── */}
      <section id="matches" className="px-4 sm:px-6 pb-14 max-w-[1180px] mx-auto">
        <div className="border border-terminal-border rounded-lg overflow-hidden bg-terminal-panel/30">
          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-3 sm:px-4 py-2 border-b border-terminal-border bg-terminal-panel/60">
            <span className="text-[11px] font-bold text-terminal-yellow tracking-wider">📅 TODAY — LIVE &amp; UPCOMING</span>
            <span className="hidden xs:block text-[10px] text-terminal-muted">
              {isPro ? "every match unlocked — analyse anything"
                : "open any match free — signals unlock with Pro"}
            </span>
          </div>
          {/* On mobile: list caps at ~55vh (scrolls), analysis flows in the
              page below it — no nested scroll trap. On md+: fixed 560px with
              both panes side-by-side, each scrolling internally. */}
          <div className="grid md:grid-cols-[1fr_420px] md:h-[560px]">
            {/* Match list */}
            <div className="overflow-y-auto border-b md:border-b-0 md:border-r border-terminal-border max-h-[55vh] [max-height:55dvh] md:max-h-none md:[max-height:none]">
              {/* Build-time rows, present in the HTML a crawler receives. The
                  live fetch replaces them on hydration; until then this is the
                  page's actual content rather than a spinner. */}
              {!data && initialMatches.length > 0 && <SsrMatchList matches={initialMatches} />}
              {!data && initialMatches.length === 0 && (
                <div className="p-8 text-center text-terminal-muted text-xs animate-pulse">Loading live schedule…</div>
              )}
              {/* Free visitors see THREE matches. Not a teaser for its own
                  sake: every row on screen is polled, so an ungated board was
                  serving the full request cost of a paying customer to people
                  who are not one. Trial and Pro see the full board. */}
              {matches.slice(0, isPro ? 250 : FREE_MATCH_LIMIT).map(m => (
                <PublicRow key={m.id} m={m} active={selected?.id === m.id}
                  freeSlot={false}
                  showProb
                  onClick={() => onPick(m)} />
              ))}
              {!isPro && matches.length > FREE_MATCH_LIMIT && (
                <button onClick={() => setPricingOpen(true)}
                  className="w-full px-4 py-4 text-center border-t border-terminal-border hover:bg-terminal-panel/40 transition">
                  <div className="text-[12px] font-bold text-terminal-green">
                    +{matches.length - FREE_MATCH_LIMIT} more matches today
                  </div>
                  <div className="text-[10px] text-terminal-muted mt-0.5">
                    Start your {TRIAL_DAYS}-day free trial to open the full board and the terminal
                  </div>
                </button>
              )}
              {data && matches.length === 0 && (
                data.sourcesDown ? (
                  // An outage must never masquerade as a quiet day — saying
                  // "no matches" when the feed is down teaches people the
                  // product is empty rather than that it is briefly unwell.
                  <div className="p-8 text-center text-xs">
                    <div className="text-terminal-yellow font-bold mb-1">⚠ Live data temporarily unavailable</div>
                    <div className="text-terminal-muted">The match feed isn&apos;t responding. This is on our side — it reconnects on its own.</div>
                  </div>
                ) : (
                  <div className="p-8 text-center text-terminal-muted text-xs">No more matches today — check back tomorrow.</div>
                )
              )}
            </div>
            {/* Analysis pane */}
            <div className="md:overflow-y-auto bg-terminal-bg min-h-[360px]">
              {selected ? (
                <EdgePanel match={selected} tier={isPro ? "pro" : "preview"}
                  onUpgrade={() => setPricingOpen(true)} />
              ) : (
                <div className="h-full flex flex-col items-center justify-center gap-2 text-center p-8">
                  <div className="text-2xl">⚡</div>
                  <div className="text-[12px] font-bold text-slate-200">Pick a match to see the full analysis</div>
                  <div className="text-[10px] text-terminal-muted max-w-[280px]">
                    Model probability, bookmaker edge, Kelly stake, live break/hold signals and hedge timing.
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section className="px-4 sm:px-6 pb-14 max-w-[1000px] mx-auto">
        <h2 className="text-center text-lg font-bold text-slate-100 mb-6">The full process, end to end</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
          <Feature n="01" title="TRUE P" body="A Platt-calibrated neural network (41,750 tour matches) sets the pre-match prior; a tour-aware Markov engine re-prices the match on every game of the live score." />
          <Feature n="02" title="EDGE" body="True P is compared against de-vigged bookmaker odds — live prices for live matches, never stale ones. Edges over 20% are quarantined as data errors, not bets." />
          <Feature n="03" title="STAKE" body="¼-Kelly staking capped at 5% of bankroll, with a hard 2% edge floor. The discipline is the product: no edge, no bet." />
          <Feature n="04" title="HEDGE" body="Trend-break, adverse-move and deuce-loss triggers tell you when to hedge a live position — protecting profit beats chasing it." />
        </div>
      </section>

      {/* ── Video manual ──
          youtube-nocookie so a visitor who never presses play is not handed a
          tracking cookie; lazy so the embed costs nothing until scrolled to. */}
      <section id="manual" className="px-4 sm:px-6 pb-14 max-w-[900px] mx-auto">
        <h2 className="text-center text-lg font-bold text-slate-100 mb-1">Watch the manual</h2>
        <p className="text-center text-[11px] text-terminal-muted mb-5 max-w-[560px] mx-auto leading-relaxed">
          A live walkthrough of the terminal — reading True P, spotting an edge against the
          book, sizing the stake and timing the hedge.
        </p>
        <div className="relative w-full rounded-lg overflow-hidden border border-terminal-border bg-black"
          style={{ aspectRatio: "16 / 9" }}>
          <iframe
            className="absolute inset-0 w-full h-full"
            src="https://www.youtube-nocookie.com/embed/75hhhVIWVNM?rel=0"
            title="How to Trade Tennis Prediction Markets Like a Pro (Live Terminal Demo)"
            loading="lazy"
            allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            referrerPolicy="strict-origin-when-cross-origin"
            allowFullScreen
          />
        </div>
        <p className="text-center text-[10px] text-terminal-muted mt-3">
          Prefer to read?{" "}
          <Link href="/manual" className="text-terminal-green hover:underline">
            The written trading manual
          </Link>{" "}
          covers the same ground in detail.
        </p>
      </section>

      {/* ── Get in touch ── */}
      <section className="px-4 sm:px-6 pb-16 max-w-[720px] mx-auto text-center">
        <div className="rounded-lg border border-terminal-green/30 bg-terminal-green/[0.06] px-4 sm:px-6 py-7">
          <h2 className="text-lg font-bold text-slate-100 mb-2">Interested?</h2>
          <p className="text-[12px] text-slate-400 mb-5 max-w-[420px] mx-auto leading-relaxed">
            Drop your email for early access, or reach out about a partnership.
          </p>
          <div className="flex justify-center">
            <EmailCapture source="landing-cta" cta="Get early access" />
          </div>
        </div>
        <p className="text-[10px] text-terminal-muted mt-4">
          or email{" "}
          <a href="mailto:jessefuture10@gmail.com" className="text-terminal-green hover:underline">
            jessefuture10@gmail.com
          </a>
        </p>
      </section>

      <PricingModal open={pricingOpen} onClose={() => setPricingOpen(false)} />
      <DonatePrompt />

      <Faq />

      <SiteFooter />

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
