import type { Metadata } from "next";
import "./globals.css";
import { TierProvider } from "@/lib/auth";
import Analytics from "@/components/Analytics";
import Prefetch from "@/components/Prefetch";
import PageviewTracker from "@/components/PageviewTracker";

// The canonical home. Set NEXT_PUBLIC_SITE_URL in the Netlify env to move it
// without a code change; the default is the live domain, so relative OG/canonical
// URLs resolve correctly instead of silently pointing at localhost.
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://tennisalpha.in";
const DESCRIPTION =
  "NN + Markov true probabilities, bookmaker edge, Kelly staking and hedge timing for every professional tennis match — ATP, WTA, Challenger, ITF.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: "Tennis Alpha",
  description: DESCRIPTION,
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    url: SITE_URL,
    siteName: "Tennis Alpha",
    title: "Tennis Alpha",
    description: DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
    title: "Tennis Alpha",
    description: DESCRIPTION,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Starts the board's requests before the bundle is parsed — without
            this nothing is asked for until React hydrates, ~1.2s in. */}
        <Prefetch />
        <link rel="preload" as="fetch" href="/rankings.json" crossOrigin="anonymous" />
      </head>
      <body className="bg-terminal-bg text-slate-200 font-mono">
        <Analytics />
        <PageviewTracker />
        <TierProvider>{children}</TierProvider>
      </body>
    </html>
  );
}
