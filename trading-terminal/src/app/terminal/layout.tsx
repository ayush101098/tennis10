import type { Metadata } from "next";

/**
 * The terminal page is a client component, so its metadata lives here — a
 * "use client" module cannot export `metadata`, and without this every route
 * inherited the root title and Google saw one title across the whole site.
 */
export const metadata: Metadata = {
  title: "Live Tennis Trading Terminal — True P, Edge & Kelly Stakes | Tennis Alpha",
  description:
    "Live win probability for every ATP, WTA, Challenger, W125 and ITF match, priced against de-vigged bookmaker odds. Edge, ¼-Kelly stake and hedge timing in one terminal.",
  alternates: { canonical: "/terminal" },
  openGraph: {
    title: "Live Tennis Trading Terminal — Tennis Alpha",
    description:
      "Score-conditioned Markov re-pricing on every point, edge against the book, and Kelly-disciplined stakes.",
    url: "/terminal",
  },
};

export default function TerminalLayout({ children }: { children: React.ReactNode }) {
  return children;
}
