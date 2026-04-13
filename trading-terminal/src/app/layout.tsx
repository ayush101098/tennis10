import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tennis Trading Terminal",
  description: "Real-time in-play tennis betting decision engine",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-terminal-bg text-slate-200 font-mono">{children}</body>
    </html>
  );
}
