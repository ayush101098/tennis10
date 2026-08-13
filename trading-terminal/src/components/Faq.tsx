"use client";

import { useState } from "react";
import { PLANS, planById } from "@/lib/plans";
import { TRIAL_LENGTH, UPI_ID, USD_INR } from "@/lib/auth";

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

const day = planById("day"), month = planById("month"), year = planById("year");

const FAQS: QA[] = [
  {
    q: "What is Tennis Alpha?",
    a: <>A live win-probability model for professional tennis. A neural network trained on
      41,750 tour matches sets the pre-match prior; a score-conditioned Markov engine re-prices
      the match as the score changes. That probability is compared against de-vigged bookmaker
      odds to find edge, and turned into a ¼-Kelly stake. It covers ATP, WTA, Challenger, W125
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
    a: <>{PLANS.map((p, i) => (
      <span key={p.id}>{i > 0 ? " · " : ""}<b>${p.usd}</b> {p.label.toLowerCase()}</span>
    ))}. No auto-charging — access simply lapses unless you pay again. Every new account starts
      with <b>{TRIAL_LENGTH} free</b>, no card required.</>,
  },
  {
    q: `Is there a free trial?`,
    a: <>Yes — {TRIAL_LENGTH} of the full terminal when you sign up, with an email or a
      Google account. No payment details are asked for, and nothing charges you when it ends;
      you simply drop back to the free board.</>,
  },
  {
    q: "What do I get for free?",
    a: <>Today&apos;s matches with the model&apos;s probability on the home page. The
      actionable layer — edge against the book, ¼-Kelly stakes, live re-pricing and hedge
      timing — is what the subscription opens.</>,
  },
  {
    q: "How do I pay?",
    a: <>Three ways: <b>UPI</b> (scan the QR or pay <span className="mono">{UPI_ID}</span> from
      any UPI app), <b>PayPal</b>, or <b>crypto</b> (ETH, USDC, USDT or DAI on Ethereum
      mainnet). Crypto unlocks automatically once the transaction confirms on-chain. UPI and
      PayPal are confirmed by hand, so message us on X or Telegram after paying and access goes
      on the same day.</>,
  },
  {
    q: "Can I pay with UPI from India?",
    a: <>Yes. The payment screen shows a UPI QR with the amount already filled in, and the ID
      <span className="mono"> {UPI_ID}</span> if you would rather type it — GPay, PhonePe, Paytm
      or any UPI app works. The rupee amount is converted at ₹{USD_INR} to the dollar, which is
      a display rate rather than a live one, so it may sit slightly above the day&apos;s market
      rate.</>,
  },
  {
    q: "How long until my access is switched on after paying?",
    a: <>Crypto is automatic — paste the transaction hash and the terminal unlocks once it
      confirms on-chain. UPI and PayPal have no callback we can verify, so they are switched on
      by hand: message us on X or Telegram with the name you paid under and it is done the same
      day. If you want access the instant you pay, use crypto.</>,
  },
  {
    q: "Can I use one account on several devices?",
    a: <>One subscription covers one device. Signing in somewhere else moves your access to
      that device and signs the previous one out — so changing phone is fine, but sharing an
      account with a friend means the two of you keep logging each other out.</>,
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
        Methodology, pricing and the limits of what a model can tell you.
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
