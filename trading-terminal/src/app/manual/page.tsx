import fs from "node:fs";
import path from "node:path";
import type { Metadata } from "next";
import Link from "next/link";
import Wordmark from "@/components/Wordmark";
import Socials from "@/components/Socials";
import { BreadcrumbLd, FaqLd } from "@/components/JsonLd";
import { LEGAL_NAME } from "@/lib/brand";
import { markdownToHtml } from "@/lib/markdown";

export const metadata: Metadata = {
  title: "Trading Execution Manual — Tennis Alpha",
  description:
    "The complete in-play tennis trading playbook: reading the edge panel, entry rules, the signal → action → size state machine, hedge math and non-negotiable risk rules.",
  // Explicit: without this the root layout's canonical:"/" is inherited and the
  // page declares itself a duplicate of the homepage.
  alternates: { canonical: "/manual" },
  openGraph: { title: "Tennis Trading Execution Manual", url: "/manual" },
};

/**
 * Static guide page. The manual lives as markdown in src/content so it stays
 * editable as prose; we read + render it at build time (this is a server
 * component, so it runs during `next build` and works with `output: export`).
 */
export default function ManualPage() {
  const md = fs.readFileSync(
    path.join(process.cwd(), "src/content/trading-manual.md"),
    "utf8",
  );
  const html = markdownToHtml(md);

  return (
    <div className="min-h-screen bg-terminal-bg text-slate-200">
      <BreadcrumbLd trail={[{ name: "Trading Manual", path: "/manual" }]} />
      {/* Answers drawn from what the product actually does — every claim here
          is one the manual itself makes and the code enforces. */}
      <FaqLd qa={[
        {
          q: "What is Tennis Alpha?",
          a: "A live win-probability model for professional tennis. A neural network trained on 41,750 tour matches sets the pre-match prior; a score-conditioned Markov engine re-prices the match as the score changes. That probability is compared against de-vigged bookmaker odds to find edge, and turned into a quarter-Kelly stake. Covers ATP, WTA, Challenger, W125 and ITF.",
        },
        {
          q: "Is Tennis Alpha a bookmaker?",
          a: "No. You cannot place a bet through Tennis Alpha and it never holds your money. It is an analytics terminal: it tells you what a match is worth and what the market is charging. You place the bet wherever you already do.",
        },
        {
          q: "How is the tennis win probability calculated?",
          a: "A Platt-calibrated neural network reads ranking, form, surface and head-to-head for the pre-match prior. In play, a Markov chain re-prices from the live score — game, set and point state — so the number reflects where the match stands rather than where it started.",
        },
        {
          q: "What is edge in tennis betting, and why de-vig the odds?",
          a: "Edge is the model's probability minus the bookmaker's implied probability after the margin is removed. De-vigging matters because raw prices sum to more than 100%, so comparing against them overstates your edge on every bet.",
        },
        {
          q: "How much should I stake on a tennis bet?",
          a: "Quarter Kelly, capped at 5% of bankroll, with a hard 2% edge floor. Below that floor Tennis Alpha recommends no bet at all — an edge smaller than the model's own error is not an edge.",
        },
        {
          q: "Does Tennis Alpha guarantee profit?",
          a: "No. The model produces calibrated probabilities, not certainties. Variance at quarter Kelly is wide and losing runs are ordinary inside a winning strategy. Bet only what you can afford to lose.",
        },
        {
          q: "What does Tennis Alpha cost, and is there a free trial?",
          a: "$19 for a day pass, $99 monthly, $999 yearly, with no auto-charging — access lapses unless you pay again. Every new account starts with 24 hours of the full terminal free, with no card required.",
        },
        {
          q: "Can I use one account on several devices?",
          a: "One subscription covers one device. Signing in elsewhere moves access to that device and signs the previous one out, so changing phone is fine but sharing an account is not practical.",
        },
      ]} />
      {/* ── Nav ── */}
      <nav className="sticky top-0 z-40 flex items-center justify-between gap-2 px-3 sm:px-6 py-3 border-b border-terminal-border bg-terminal-bg/95 backdrop-blur">
        <Link href="/" className="hover:opacity-80">
          <Wordmark size={16} />
        </Link>
        <div className="flex items-center gap-2 sm:gap-3 text-[11px] shrink-0">
          <Socials />
          <Link href="/" className="text-terminal-muted hover:text-slate-200">Home</Link>
          <Link
            href="/terminal"
            className="inline-flex items-center min-h-[40px] font-bold px-3 rounded bg-terminal-green text-black hover:opacity-90"
          >
            LAUNCH TERMINAL →
          </Link>
        </div>
      </nav>

      <main className="px-4 sm:px-6 py-8 sm:py-12 max-w-[900px] mx-auto">
        <div className="mb-6 text-[10px] uppercase tracking-wider text-terminal-cyan font-bold">
          Playbook · v1.0
        </div>
        <article
          className="manual-content"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </main>

      <footer className="px-6 py-6 border-t border-terminal-border text-center text-[9px] text-terminal-muted leading-relaxed">
        Model outputs are calibrated probabilities, not guarantees. Sports betting involves risk — bet only what you can afford to lose.
        <div className="mt-3 flex items-center justify-center"><Socials variant="footer" /></div>
        <div className="mt-2">© {new Date().getFullYear()} {LEGAL_NAME}</div>
      </footer>
    </div>
  );
}
