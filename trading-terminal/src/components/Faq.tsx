"use client";

import { useState } from "react";

/**
 * FAQ accordion.
 *
 * Every answer here is one the product actually keeps — the 2% floor, the 5%
 * cap and the tier prices are read from the same constants the terminal
 * enforces, so a price change cannot leave the FAQ quietly lying. Where the
 * honest answer is unflattering (no, we are not a bookmaker; no, this is not a
 * guarantee; yes, the model can be wrong) it says so, because the audience is
 * quantitative and a page that oversells loses them at the first check.
 *
 * Mirrored as FAQPage structured data on /manual.
 */

interface QA { q: string; a: React.ReactNode }

const FAQS: QA[] = [
  {
    q: "What is Tennis Alpha?",
    a: <>A live win chance for every professional tennis match, updated as the points are
      played, alongside the moments that decide matches — repeated break points, a player
      who has just been broken, a set getting away from someone. Where a market exists we
      show what it is charging, so you can see when the two disagree. It covers ATP, WTA, Challenger, W125
      and ITF — men&apos;s and women&apos;s — every match day.</>,
  },
  {
    q: "Is Tennis Alpha a bookmaker?",
    a: <>No. You cannot place a bet here and we never hold your money. Tennis Alpha is an
      analytics terminal: it tells you what it thinks a match is worth and what the market is
      charging. You place the bet wherever you already do.</>,
  },
  {
    q: "How is the win probability actually calculated?",
    a: <>Two layers. Pre-match, a Platt-calibrated neural network reads ranking, form, surface
      and head-to-head. In play, a Markov chain re-prices from the live score — game, set and
      point state — so the number reflects where the match stands rather than where it started.
      A tennis match is a sequence of independent-ish service points, which is exactly the
      structure a Markov model handles well.</>,
  },
  {
    q: "What is &ldquo;edge&rdquo;, and why de-vig the odds?",
    a: <>Edge is the model&apos;s probability minus the bookmaker&apos;s implied probability
      once the margin is removed. De-vigging matters: raw prices sum to more than 100%, so
      comparing against them overstates your edge on every single bet. Skipping that step is
      the most common way a betting model appears profitable and is not.</>,
  },
  {
    q: "How much should I stake?",
    a: <>Quarter Kelly, capped at 5% of bankroll, with a hard 2% edge floor. Below that floor
      the terminal recommends nothing at all — an edge smaller than the model&apos;s own error
      is not an edge. The calculator will also show you how a bankroll compounds at a given
      edge, and how long reaching a target actually takes.</>,
  },
  {
    q: "Does this guarantee profit?",
    a: <>No, and anyone who tells you otherwise is selling something. The model produces
      calibrated probabilities, not certainties; variance at ¼ Kelly is wide and losing runs
      are ordinary inside a winning strategy. The discipline — sizing, the edge floor, hedge
      timing — is what the product is for. Bet only what you can afford to lose.</>,
  },
  {
    q: "What does it cost?",
    a: <>Nothing. The terminal is free — the whole of it, with no plan, no card and
      no trial that runs out. Leave an email and it opens.</>,
  },
  {
    q: "What is the email for, then?",
    a: <>It is your sign-in, and it is how your bet journal follows you from your
      laptop to your phone. It also tells us who is actually using this, which is
      the only thing we ask for in return. It is not sold or passed on, and one
      line to the address in the footer removes it.</>,
  },
  {
    q: "What do I get without signing in?",
    a: <>Today&apos;s board — every live and upcoming match across ATP, WTA, Challenger
      and ITF, with the model&apos;s probability on each. What stays blurred is the
      actionable layer: edge against the market, ¼-Kelly stakes, live re-pricing,
      hedge timing and the bet journal. An email lifts it.</>,
  },
  {
    q: "Can I use one account on several devices?",
    a: <>Yes. Sign in with the same address anywhere — the account follows the email,
      and your bet journal comes with it.</>,
  },
  {
    q: "Is sports betting legal where I am?",
    a: <>That depends entirely on your jurisdiction and it is your responsibility to know.
      Tennis Alpha is an analytics tool and is not available as, and does not constitute,
      betting advice or a betting service. If betting is restricted where you live, use this
      as a model, not as an instruction.</>,
  },
];

function Item({ qa, open, onToggle }: { qa: QA; open: boolean; onToggle: () => void }) {
  return (
    <div className={`border-b border-terminal-border ${open ? "bg-terminal-panel/40" : ""}`}>
      <button
        onClick={onToggle}
        aria-expanded={open}
        className="w-full flex items-center justify-between gap-4 text-left px-4 py-4 min-h-[56px] hover:bg-terminal-panel/30 transition">
        <span className="text-[13px] font-bold text-slate-100">
          {qa.q.replace(/&ldquo;|&rdquo;/g, '"')}
        </span>
        <span className={`shrink-0 w-5 h-5 rounded-full border flex items-center justify-center text-[13px] leading-none transition ${
          open ? "border-terminal-green text-terminal-green" : "border-terminal-border text-terminal-muted"
        }`} aria-hidden="true">
          {open ? "−" : "+"}
        </span>
      </button>
      {open && (
        <div className="px-4 pb-4 -mt-1">
          <div className="border-t border-terminal-border/60 pt-3 text-[12px] text-slate-400 leading-relaxed">
            {qa.a}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Faq() {
  // First one open, as in most FAQ patterns — it shows the interaction without
  // the reader having to guess that the rows expand.
  const [open, setOpen] = useState<number | null>(0);

  return (
    <section id="faq" className="marketing px-4 sm:px-6 pb-14 max-w-[820px] mx-auto">
      <h2 className="text-center text-lg font-bold text-slate-100 mb-1">Frequently asked questions</h2>
      <p className="text-center text-[11px] text-terminal-muted mb-6">
        Methodology, access and the limits of what a model can tell you.
      </p>
      <div className="border border-terminal-border rounded-lg overflow-hidden bg-terminal-panel/20">
        {FAQS.map((qa, i) => (
          <Item key={qa.q} qa={qa} open={open === i} onToggle={() => setOpen(open === i ? null : i)} />
        ))}
      </div>
    </section>
  );
}

/** The same questions, as plain strings, for the FAQPage structured data. */
export const FAQ_PLAIN = FAQS.map(f => f.q.replace(/&ldquo;|&rdquo;/g, '"'));
