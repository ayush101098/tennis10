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
          q: "How is the tennis win probability calculated?",
          a: "A neural network trained on 41,750 professional matches sets the pre-match prior from ranking, form, surface and head-to-head. Once play starts, a score-conditioned Markov engine re-prices the match from the live score — game, set and point state — so the probability reflects where the match actually stands rather than where it started.",
        },
        {
          q: "What is edge in tennis betting?",
          a: "Edge is the model's true probability minus the bookmaker's implied probability after the vig is removed. De-vigging matters: raw bookmaker prices sum to more than 100%, so comparing against them overstates your edge on every single bet.",
        },
        {
          q: "How much should I stake on a tennis bet?",
          a: "Tennis Alpha uses quarter Kelly, capped at 5% of bankroll, with a hard 2% edge floor. Below that floor no bet is recommended at all — an edge smaller than the model's own error is not an edge.",
        },
        {
          q: "When should you hedge a live tennis position?",
          a: "On trend break, adverse move, or a deuce-game loss against the position. The terminal flags these as they happen; the discipline is to protect a profit rather than chase a bigger one.",
        },
        {
          q: "Which tours does Tennis Alpha cover?",
          a: "ATP, WTA, Challenger, W125 and ITF — men's and women's — every match day, including the lower tours where bookmaker pricing is loosest.",
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
