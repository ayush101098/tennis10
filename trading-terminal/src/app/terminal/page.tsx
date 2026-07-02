"use client";

import { useState } from "react";
import Link from "next/link";
import SchedulePanel from "@/components/SchedulePanel";
import BetTracker from "@/components/BetTracker";
import PricingModal from "@/components/PricingModal";
import { useTier, signOut } from "@/lib/auth";

/**
 * The terminal — gated by tier.
 *   free:  every match + pre-match model probabilities; pro features locked in place
 *   pro:   full intelligence stack + bet tracker
 *   admin: pro forever
 */
export default function TerminalPage() {
  const { session, tier, refresh } = useTier();
  const [tab, setTab] = useState<"centre" | "tracker">("centre");
  const [pricingOpen, setPricingOpen] = useState(false);
  const isPro = tier === "pro";

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden">
      {/* ── Header Bar ── */}
      <header className="flex items-center justify-between px-4 py-1.5 border-b border-terminal-border bg-terminal-panel shrink-0">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-terminal-green font-bold text-sm hover:opacity-80">◉ TENNIS INTELLIGENCE TERMINAL</Link>
          <button onClick={() => setTab("centre")}
            className={`text-[10px] font-bold px-2 py-0.5 rounded ${tab === "centre" ? "text-terminal-yellow bg-terminal-yellow/10" : "text-terminal-muted hover:text-slate-300"}`}>
            MATCH CENTRE
          </button>
          <button onClick={() => (isPro ? setTab("tracker") : setPricingOpen(true))}
            className={`text-[10px] font-bold px-2 py-0.5 rounded ${tab === "tracker" ? "text-terminal-cyan bg-terminal-cyan/10" : "text-terminal-muted hover:text-slate-300"}`}>
            {isPro ? "BET TRACKER" : "🔒 BET TRACKER"}
          </button>
        </div>
        <div className="flex items-center gap-3 text-[10px]">
          {session ? (
            <>
              <span className="text-terminal-muted">{session.email}</span>
              <span className={`font-bold px-1.5 py-0.5 rounded ${
                session.isAdmin ? "bg-terminal-red/20 text-terminal-red"
                  : isPro ? "bg-terminal-green/20 text-terminal-green"
                  : "bg-terminal-border text-slate-300"
              }`}>
                {session.isAdmin ? "ADMIN" : isPro ? "PRO" : "FREE"}
              </span>
              {!isPro && (
                <button onClick={() => setPricingOpen(true)}
                  className="font-bold px-2 py-0.5 rounded bg-terminal-green text-black hover:opacity-90">
                  UPGRADE $99
                </button>
              )}
              <button onClick={() => { signOut(); refresh(); }} className="text-terminal-muted hover:text-slate-300">
                sign out
              </button>
            </>
          ) : (
            <button onClick={() => setPricingOpen(true)}
              className="font-bold px-2 py-0.5 rounded bg-terminal-green text-black hover:opacity-90">
              SIGN IN
            </button>
          )}
        </div>
      </header>

      {/* ── Body ── */}
      <div className="flex-1 min-h-0">
        {!session ? (
          <div className="h-full flex flex-col items-center justify-center gap-3 text-center px-6">
            <div className="text-3xl">🎾</div>
            <div className="text-sm font-bold text-slate-100">Sign in to open the terminal</div>
            <div className="text-[11px] text-terminal-muted max-w-[380px]">
              Free accounts see pre-match model probabilities on every ATP, WTA, Challenger and ITF match.
              Pro unlocks the full trading intelligence stack.
            </div>
            <button onClick={() => setPricingOpen(true)}
              className="mt-2 px-4 py-2 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90">
              SIGN IN / GET ACCESS
            </button>
          </div>
        ) : tab === "tracker" && isPro ? (
          <BetTracker />
        ) : (
          <SchedulePanel tier={tier} onUpgrade={() => setPricingOpen(true)} />
        )}
      </div>

      <PricingModal open={pricingOpen} onClose={() => setPricingOpen(false)} onDone={refresh} />
    </div>
  );
}
