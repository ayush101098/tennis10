import type { Metadata } from "next";
import "./globals.css";
import { TierProvider } from "@/lib/auth";

export const metadata: Metadata = {
  title: "Tennis Intelligence Terminal",
  description: "NN + Markov true probabilities, bookmaker edge, Kelly staking and hedge timing for every professional tennis match — ATP, WTA, Challenger, ITF.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-terminal-bg text-slate-200 font-mono">
        <TierProvider>{children}</TierProvider>
      </body>
    </html>
  );
}
