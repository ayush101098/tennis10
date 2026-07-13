import fs from "node:fs";
import path from "node:path";
import type { Metadata } from "next";
import Link from "next/link";
import { markdownToHtml } from "@/lib/markdown";

export const metadata: Metadata = {
  title: "Trading Execution Manual — Tennis Intelligence Terminal",
  description:
    "The complete in-play tennis trading playbook: reading the edge panel, entry rules, the signal → action → size state machine, hedge math and non-negotiable risk rules.",
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
      {/* ── Nav ── */}
      <nav className="sticky top-0 z-40 flex items-center justify-between gap-2 px-3 sm:px-6 py-3 border-b border-terminal-border bg-terminal-bg/95 backdrop-blur">
        <Link href="/" className="text-terminal-green font-bold text-xs sm:text-sm hover:opacity-80">
          ◉ <span className="hidden sm:inline">TENNIS INTELLIGENCE TERMINAL</span>
          <span className="sm:hidden">TENNIS T.</span>
        </Link>
        <div className="flex items-center gap-2 sm:gap-3 text-[11px] shrink-0">
          <Link href="/" className="text-terminal-muted hover:text-slate-200">Home</Link>
          <Link
            href="/terminal"
            className="font-bold px-3 py-1.5 rounded bg-terminal-green text-black hover:opacity-90"
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
        <br />© {new Date().getFullYear()} Tennis Intelligence Terminal
      </footer>
    </div>
  );
}
