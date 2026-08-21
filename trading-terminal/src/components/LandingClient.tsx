"use client";

// 5s was chosen for point-by-point immediacy. Games take ~45s, so the board
// cannot change faster than that — 15s shows every score change and cuts the
// request rate by two thirds.
const LIVE_POLL_MS = 15_000;
/**
 * Matches a free visitor sees on the homepage.
 *
 * One, since 2026-08-14: the terminal is members-only and this is the whole
 * free product — a single glance at what the model does, not a usable board.
 * Every row on screen is polled, so this is also the request-cost floor.
 */
const FREE_MATCH_LIMIT = 1;

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
import { useTier } from "@/lib/auth";
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
    // When the primary feed is stale, a match it calls "live" is not live — it
    // is a score frozen hours ago. Those sorted to the top of the free board,
    // so the first three matches a visitor ever saw were dead scoreboards
    // labelled LIVE. Anything from the fresh standby leads instead.
    const stale = !!data.fallbackActive;
    const trust = (m: ScheduledMatch) => (stale && m.source !== "espn" ? 1 : 0);
    return [...data.today]
      .filter(m => m.status === "live" || m.status === "scheduled")
      // live, then tour tier, then time — see tourRank
      .sort((a, b) =>
        (trust(a) - trust(b)) ||
        (order[a.status] - order[b.status]) ||
        (tourRank(a.tour) - tourRank(b.tour)) ||
        (a.start_timestamp - b.start_timestamp));
  }, [data]);

  // Only matches from a feed we currently trust — a frozen scoreboard is not
  // a live match, and counting it inflates the headline the whole page leads on.
  const liveCount = matches.filter(
    m => m.status === "live" && !(data?.fallbackActive && m.source !== "espn")).length;
  const tours = useMemo(() => Array.from(new Set(matches.map(m => m.tour))), [matches]);

  // Every match opens for everyone. The gate is on the ACTIONABLE layer, not on
  // access: edge, Kelly stakes and the trade signals render blurred for
  // non-subscribers, so a visitor sees the shape of the answer and what it is
  // worth. Locking whole matches instead hid the product from the people it
  // needs to convince — and from search engines, which index this page.
  const [pricingOpen, setPricingOpen] = useState(false);
  const onPick = useCallback((m: ScheduledMatch) => setSelected(m), []);

  return (
    <div className="marketing min-h-screen bg-terminal-bg text-slate-200">
      <SoftwareApplicationLd />
      {/* ── Nav ── */}
      <nav className="sticky top-0 z-40 flex items-center justify-between gap-2 px-3 sm:px-6 py-3 border-b border-terminal-border bg-terminal-bg/95 backdrop-blur">
        <Wordmark size={17} />
        <div className="flex items-center gap-2 sm:gap-3 text-[11px] shrink-0">
          <Socials />
          <Link href="/manual" className="hidden sm:inline text-terminal-muted hover:text-slate-200">Manual</Link>
          {/* Signed-in users keep their way in — someone who already signed up,
              or is paying, must never be sent to a waitlist for access they
              already hold. The waitlist CTA is for new visitors only. */}
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
            <a href="#waitlist" className="inline-flex items-center justify-center min-h-[40px] font-bold px-3 rounded bg-terminal-green text-black hover:opacity-90">
              JOIN WAITLIST →
            </a>
          )}
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="px-4 sm:px-6 pt-10 sm:pt-14 pb-10 text-center max-w-[860px] mx-auto">
        <span className="eyebrow">Live model · ATP · WTA · Challenger · W125 · ITF</span>
        <h1 className="text-slate-100">
          Market intelligence for <span className="text-terminal-green">serious tennis traders.</span>
        </h1>
        <p className="mt-5 text-slate-400 max-w-[620px] mx-auto">
          Live point-by-point pricing across every professional tour, fused with a
          score-conditioned Markov engine and compared against real exchange odds —
          with edge confidence, ¼-Kelly staking and hedge-timing discipline built in.
          ATP · WTA · Challenger · W125 · ITF, every day.
        </p>

        {/* Waitlist is the primary action. The email lands in the Netlify Blobs
            `leads` store via /api/subscribe, same path the rest of the site uses,
            and is mirrored to the waitlist sheet — so nothing new to maintain. */}
        <div id="waitlist" className="mt-7 scroll-mt-24 flex flex-col items-center gap-2">
          <div className="w-full max-w-md flex justify-center">
            <EmailCapture source="waitlist-hero" cta="Join the waitlist" variant="waitlist" />
          </div>
          <p className="text-[11px] text-terminal-muted">
            Free while in beta · no card required · unsubscribe anytime
          </p>
          <a href="#matches" className="mt-2 inline-flex items-center justify-center min-h-[40px] px-5 rounded border border-terminal-border text-xs font-bold text-slate-200 hover:bg-terminal-panel">
            SEE TODAY&apos;S MATCHES ↓
          </a>
        </div>
        {/* Only for signed-in users. Asking a new visitor to join a waitlist and
            to subscribe in the same eyeful gives them two different next steps
            and so no clear one; the waitlist is the ask on this page now. */}
        {session && (
          <div className="mt-6 max-w-[560px] mx-auto">
            <TrialBanner onStart={() => setPricingOpen(true)} />
          </div>
        )}

        {/* A stat row rather than pills: the same facts, but laid out as
            figures, which reads as a product with numbers behind it instead of
            a list of tags. Values are live where they can be. */}
        <div className="mt-10 grid grid-cols-2 sm:grid-cols-4 border-t border-terminal-border">
          <Stat n={matches.length || "—"} l="Matches today" />
          <Stat n={liveCount} l="Live right now" tone={liveCount > 0 ? "green" : undefined} />
          {/* Was "41,750 matches in training set" — a training-set size is not a
              result, and the model it referred to could not be separated from a
              coin flip out of sample. Tours covered is a fact about the product. */}
          <Stat n="5" l="Pro tours covered" />
          <Stat n="2%" l="Hard edge floor" />
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
              {data?.fallbackActive && (
                <div className="px-4 py-2 border-b border-terminal-yellow/40 bg-terminal-yellow/10">
                  <div className="text-[11px] font-bold text-terminal-yellow">
                    ⚠ Primary feed down — ATP &amp; WTA on backup source
                  </div>
                  <div className="text-[10px] text-terminal-muted mt-0.5">
                    Tour matches are current. Challenger and ITF scores are delayed.
                  </div>
                </div>
              )}
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
                    Subscribe to open the full board and the terminal
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
        <div className="text-center mb-8">
          <span className="eyebrow">The process</span>
          <h2 className="text-slate-100">From true probability to a sized, hedged position.</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
          <Feature n="01" title="TRUE P" body="A tour-aware Markov engine prices the match from the live score — point to game to set — re-pricing on every game rather than resting on a pre-match number." />
          <Feature n="02" title="EDGE" body="True P is compared against de-vigged bookmaker odds — live prices for live matches, never stale ones. Edges over 20% are quarantined as data errors, not bets." />
          <Feature n="03" title="STAKE" body="¼-Kelly staking capped at 5% of bankroll, with a hard 2% edge floor. The discipline is the product: no edge, no bet." />
          <Feature n="04" title="HEDGE" body="Trend-break, adverse-move and deuce-loss triggers tell you when to hedge a live position — protecting profit beats chasing it." />
        </div>
      </section>

      {/* ── The three tools ──
          One full section per tool rather than a grid of cards. A card says a
          feature exists; a section has room to show what it puts on screen,
          which is what a trader is actually deciding about. Panels are built
          from live design tokens rather than screenshots so they cannot drift
          out of date with the product. */}
      <section id="tools" className="px-4 sm:px-6 pb-4 max-w-[1080px] mx-auto scroll-mt-20">
        <div className="text-center mb-10">
          <span className="eyebrow">The toolkit</span>
          <h2 className="text-slate-100">Three questions, every match, in one place.</h2>
          <p className="mt-4 text-slate-400 max-w-[600px] mx-auto text-sm">
            Is it mispriced, is it moving, and am I actually any good at this?
            Everything on the terminal answers one of those.
          </p>
        </div>
      </section>

      <Pillar
        tag="EDGE"
        title="Find the matches the market has mispriced."
        body="Every fixture is priced by a score-conditioned Markov engine and compared against de-vigged exchange odds. But a raw edge is not a signal — an edge on a number we do not trust is noise. Each opportunity is divided by how much our independent estimates disagree, so a clean 5% beats a shaky 9%."
        points={[
          "EdgeScore = edge ÷ uncertainty, not edge alone",
          "De-vigged two-sided exchange prices, never a stale line",
          "Edges over 20% quarantined as data errors, not bets",
          "Hard 2% edge floor — no edge, no bet",
        ]}
        panel={
          <div className="text-[10px] mono">
            {[
              { m: "Sinner / Alcaraz", e: "+9.0%", s: "3.1", g: "green" },
              { m: "Rybakina / Sabalenka", e: "+6.2%", s: "2.2", g: "green" },
              { m: "Fritz / Nakashima", e: "+9.4%", s: "0.8", g: "red" },
              { m: "Bergs / Sakellaridis", e: "+3.1%", s: "1.4", g: "amber" },
            ].map((r) => (
              <div key={r.m} className="flex items-center gap-2 px-3 py-2 border-b border-terminal-border last:border-0">
                <span className="flex-1 truncate text-slate-300">{r.m}</span>
                <span className="w-[52px] text-right text-slate-400">{r.e}</span>
                <span className={`w-[36px] text-right font-bold ${
                  r.g === "green" ? "text-terminal-green" : r.g === "amber" ? "text-terminal-yellow" : "text-terminal-red"
                }`}>{r.s}</span>
                <span className={`w-[14px] text-right ${
                  r.g === "green" ? "text-terminal-green" : r.g === "amber" ? "text-terminal-yellow" : "text-terminal-red"
                }`}>●</span>
              </div>
            ))}
            <div className="px-3 py-2 text-[9px] text-terminal-muted">
              Same 9% edge, opposite verdicts — the difference is confidence.
            </div>
          </div>
        }
      />

      <Pillar
        reverse
        tag="PULSE"
        title="Read the match while it is still moving."
        body="Sets and games are the last thing to change. A live momentum engine tracks serve regression and break pressure point by point, so a break coming is visible before it lands — and so is a lead that has stopped meaning anything."
        points={[
          "Break probability on the current service game",
          "Momentum weighted toward the most recent games",
          "Rally profiles: first-strike vs grinder, per player",
          "Hedge triggers on trend-break and adverse moves",
        ]}
        panel={
          <div className="p-3 space-y-3">
            <div>
              <div className="flex justify-between text-[10px] mono mb-1">
                <span className="text-slate-300">Break probability · current game</span>
                <span className="text-terminal-yellow font-bold">72%</span>
              </div>
              <div className="h-1.5 rounded bg-terminal-border overflow-hidden">
                <div className="h-full bg-terminal-yellow" style={{ width: "72%" }} />
              </div>
              <div className="text-[9px] text-terminal-muted mt-1">returner leads 15–40 on serve</div>
            </div>
            <div>
              <div className="flex justify-between text-[10px] mono mb-1">
                <span className="text-slate-300">Momentum</span>
                <span className="text-terminal-green font-bold">P1 +0.18</span>
              </div>
              <div className="h-1.5 rounded bg-terminal-border overflow-hidden flex">
                <div className="h-full bg-terminal-border" style={{ width: "41%" }} />
                <div className="h-full bg-terminal-green" style={{ width: "18%" }} />
              </div>
            </div>
            <div className="flex gap-1.5 pt-1">
              {["W", "W", "L", "W", "W", "W", "L", "W"].map((p, i) => (
                <span key={i} className={`flex-1 h-5 rounded text-[9px] mono flex items-center justify-center ${
                  p === "W" ? "bg-terminal-green/20 text-terminal-green" : "bg-terminal-border text-terminal-muted"
                }`}>{p}</span>
              ))}
            </div>
            <div className="text-[9px] text-terminal-muted">last 8 points on serve</div>
          </div>
        }
      />

      <Pillar
        tag="LEDGER"
        title="Find out whether the model is actually right."
        body="Every intended and placed trade is one row in a journal that settles itself against the market. The calibration report then bins predicted probability against what actually happened — which is the only thing that can tell a sharp model from a merely confident one."
        points={[
          "Auto-settled P&L per bet, per source",
          "Calibration curve: predicted vs actual, per bucket",
          "Brier score against the 0.25 coin-flip line",
          "ROI and drawdown by signal source",
        ]}
        panel={
          <div className="p-3">
            <div className="text-[9px] text-terminal-muted mb-2 mono">PREDICTED vs ACTUAL — by bucket</div>
            {[
              { b: "50–60%", p: 55, a: 53 },
              { b: "60–70%", p: 65, a: 58 },
              { b: "70–80%", p: 75, a: 61 },
              { b: "80–90%", p: 85, a: 62 },
            ].map((r) => (
              <div key={r.b} className="flex items-center gap-2 mb-1.5">
                <span className="w-[52px] text-[9px] mono text-terminal-muted">{r.b}</span>
                <div className="flex-1 h-3 rounded bg-terminal-border/60 relative overflow-hidden">
                  <div className="absolute inset-y-0 left-0 bg-slate-600" style={{ width: `${r.p}%` }} />
                  <div className="absolute inset-y-0 left-0 bg-terminal-green/70" style={{ width: `${r.a}%` }} />
                </div>
                <span className="w-[62px] text-right text-[9px] mono text-slate-400">{r.p}→{r.a}%</span>
              </div>
            ))}
            <div className="text-[9px] text-terminal-muted mt-2 leading-relaxed">
              Grey = predicted, green = actual. Gaps like these are why staking is
              scaled by confidence instead of run at full Kelly.
            </div>
          </div>
        }
      />

      {/* ── Coverage ── */}
      <section className="px-4 sm:px-6 py-14 max-w-[1000px] mx-auto">
        <div className="text-center mb-8">
          <span className="eyebrow">Coverage</span>
          <h2 className="text-slate-100">Every professional tour, not just the ones on TV.</h2>
          <p className="mt-4 text-slate-400 max-w-[600px] mx-auto text-sm">
            The mispricing is rarely in the Sunday final. It is in the Challenger
            second round that three people are watching.
          </p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {["ATP", "WTA", "CHALLENGER", "W125", "ITF"].map((t) => (
            <div key={t} className="border border-terminal-border rounded-lg py-4 text-center bg-terminal-panel/30">
              <div className="text-terminal-green font-bold text-sm mono">{t}</div>
              <div className="text-[10px] text-terminal-muted mt-1">singles</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Honesty panel ──
          Says plainly what the product does not do. A page that only claims
          upside reads like every tout site the audience has already been burned
          by; the disclaimer is the credibility, not a legal afterthought. */}
      <section className="px-4 sm:px-6 pb-14 max-w-[820px] mx-auto">
        <div className="border border-terminal-border rounded-lg p-5 sm:p-6 bg-terminal-panel/30">
          <div className="text-terminal-yellow font-bold text-sm mb-3">What this is not</div>
          <ul className="space-y-2 text-[12px] text-slate-400 leading-relaxed">
            <li>· <span className="text-slate-300">Not a tipping service.</span> No one hands you a slip to copy. It prices matches and shows you the working.</li>
            <li>· <span className="text-slate-300">Not a guarantee.</span> A positive edge is a claim about the long run, and the long run contains losing months.</li>
            <li>· <span className="text-slate-300">Not a substitute for your own judgement.</span> Every number ships with the uncertainty attached so you can disagree with it.</li>
            <li>· <span className="text-slate-300">Not for money you need.</span> Staking is capped and fractional for a reason. Bet accordingly, or do not bet.</li>
          </ul>
        </div>
      </section>

      {/* ── Video manual ──
          youtube-nocookie so a visitor who never presses play is not handed a
          tracking cookie; lazy so the embed costs nothing until scrolled to. */}
      <section id="manual" className="px-4 sm:px-6 pb-14 max-w-[900px] mx-auto">
        <span className="eyebrow text-center">Watch the manual</span>
        <h2 className="text-center text-slate-100 mb-2">A live walkthrough of the terminal.</h2>
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
          <h2 className="text-slate-100 mb-3">Interested?</h2>
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

function Stat({ n, l, tone }: { n: React.ReactNode; l: string; tone?: "green" }) {
  return (
    <div className="py-5 px-3 border-r border-terminal-border last:border-r-0 text-center sm:text-left">
      <div className={`mono text-xl font-bold ${tone === "green" ? "text-terminal-green" : "text-slate-100"}`}>{n}</div>
      <div className="text-[11.5px] text-terminal-muted mt-1">{l}</div>
    </div>
  );
}

function Chip({ label, tone, pulse }: { label: string; tone?: "green"; pulse?: boolean }) {
  return (
    <span className={`px-2.5 py-1 rounded-full border ${tone === "green" ? "border-terminal-green/40 text-terminal-green" : "border-terminal-border text-slate-400"}`}>
      {pulse && <span className="inline-block w-1.5 h-1.5 rounded-full bg-terminal-green animate-pulse mr-1.5 align-middle" />}
      {label}
    </span>
  );
}

/**
 * One tool, given a full band: copy on one side, a live-token mock of what it
 * puts on screen on the other. `reverse` alternates the sides so a run of them
 * does not read as a stack of identical rows.
 *
 * The mock is built from the same design tokens as the real terminal rather
 * than a screenshot, so it cannot quietly drift out of date with the product —
 * and it stays legible at any width, which a scaled-down screenshot does not.
 */
function Pillar({ tag, title, body, points, panel, reverse }: {
  tag: string; title: string; body: string; points: string[];
  panel: React.ReactNode; reverse?: boolean;
}) {
  return (
    <section className="px-4 sm:px-6 py-10 max-w-[1080px] mx-auto">
      <div className={`flex flex-col gap-8 md:gap-12 md:items-center ${reverse ? "md:flex-row-reverse" : "md:flex-row"}`}>
        <div className="flex-1 min-w-0">
          <span className="inline-block text-[10px] font-bold tracking-[0.14em] text-terminal-green border border-terminal-green/40 rounded-full px-2.5 py-1">
            {tag}
          </span>
          <h3 className="mt-4 text-slate-100 text-[22px] sm:text-[26px] leading-tight font-bold">{title}</h3>
          <p className="mt-3 text-slate-400 text-sm leading-relaxed">{body}</p>
          <ul className="mt-5 space-y-2">
            {points.map((p) => (
              <li key={p} className="flex gap-2.5 text-[12.5px] text-slate-300">
                <span className="text-terminal-green shrink-0" aria-hidden>▸</span>
                <span>{p}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="flex-1 min-w-0 w-full">
          <div className="border border-terminal-border rounded-lg bg-terminal-panel/40 overflow-hidden">
            <div className="flex items-center gap-1.5 px-3 py-2 border-b border-terminal-border bg-terminal-panel/60">
              <span className="w-2 h-2 rounded-full bg-terminal-red/60" />
              <span className="w-2 h-2 rounded-full bg-terminal-yellow/60" />
              <span className="w-2 h-2 rounded-full bg-terminal-green/60" />
              <span className="ml-1.5 text-[9px] mono text-terminal-muted tracking-wider">{tag}</span>
            </div>
            {panel}
          </div>
        </div>
      </div>
    </section>
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
