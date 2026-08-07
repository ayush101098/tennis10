import fs from "node:fs";
import path from "node:path";
import type { Metadata } from "next";
import Link from "next/link";
import Wordmark from "@/components/Wordmark";
import Socials from "@/components/Socials";
import { LEGAL_NAME } from "@/lib/brand";
import { markdownToHtml } from "@/lib/markdown";

export const metadata: Metadata = {
  title: "TT Trading Guide — Tennis Alpha",
  description:
    "How to trade the table-tennis terminal: reading live True P and character residuals, where factory-league edges come from, sizing, and the journal discipline.",
};

/**
 * Static guide page — same pattern as /manual: markdown in src/content,
 * rendered at build time so it survives `output: export`.
 */
export default function TtManualPage() {
  const md = fs.readFileSync(
    path.join(process.cwd(), "src/content/tt-trading-guide.md"),
    "utf8",
  );
  const html = markdownToHtml(md);

  return (
    <div className="min-h-screen bg-terminal-bg text-slate-200">
      {/* ── Nav ── */}
      <nav className="sticky top-0 z-40 flex items-center justify-between gap-2 px-3 sm:px-6 py-3 border-b border-terminal-border bg-terminal-bg/95 backdrop-blur">
        <Link href="/tt" className="flex items-center gap-1.5 hover:opacity-80">
          <span>🏓</span><Wordmark size={16} mark={false} />
        </Link>
        <div className="flex items-center gap-2 sm:gap-3 text-[11px] shrink-0">
          <Socials />
          <Link href="/manual" className="text-terminal-muted hover:text-slate-200">Tennis manual</Link>
          <Link
            href="/tt"
            className="inline-flex items-center min-h-[40px] font-bold px-3 rounded bg-terminal-green text-black hover:opacity-90"
          >
            LAUNCH TT TERMINAL →
          </Link>
        </div>
      </nav>

      <main className="px-4 sm:px-6 py-8 sm:py-12 max-w-[900px] mx-auto">
        <div className="mb-6 text-[10px] uppercase tracking-wider text-terminal-cyan font-bold">
          Playbook · TT v1.0
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
