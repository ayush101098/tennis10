import type { Metadata } from "next";

/**
 * Targets "Kelly criterion for tennis" / "tennis Kelly staking calculator" —
 * the generic "Kelly criterion calculator" SERP belongs to high-authority
 * betting-tool sites, but the tennis-specific long tail is unowned.
 */
export const metadata: Metadata = {
  title: "Tennis Kelly Criterion Calculator — Stake Sizing for Tennis Betting | Tennis Alpha",
  description:
    "Free Kelly criterion calculator built for tennis. Enter your true win probability and the bookmaker's price to get a ¼-Kelly stake, capped at 5% of bankroll, with the 2% edge floor applied.",
  alternates: { canonical: "/calculator" },
  openGraph: {
    title: "Tennis Kelly Criterion Calculator",
    description: "¼-Kelly stake sizing for tennis, with a 5% bankroll cap and a 2% edge floor.",
    url: "/calculator",
  },
};

export default function CalculatorLayout({ children }: { children: React.ReactNode }) {
  return children;
}
