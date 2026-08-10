"use client";

import { useEffect, useState } from "react";
import { useTier, subActive, signIn, TRIAL_DAYS, type Session } from "@/lib/auth";

/**
 * The trial prompt, and the trial countdown.
 *
 * Three states, because one banner for all of them would lie to two thirds of
 * the audience:
 *   signed out / free  -> offer the trial
 *   on trial           -> how long is left, and how to keep it
 *   paid               -> nothing at all
 *
 * A subscriber must never be shown a "start your free trial" bar; that is the
 * fastest way to make someone who already paid feel like they shouldn't have.
 */
const DAY = 86400000;

function onTrial(s: Session | null): boolean {
  // A trial is a grant with no payment behind it. paidUntil covers both, so
  // the distinguishing fact is whether they ever paid.
  return !!s && subActive(s) && !s.txHash && !s.isAdmin;
}

export default function TrialBanner({ onStart }: { onStart: () => void }) {
  const { session } = useTier();
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const iv = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(iv);
  }, []);

  const paid = subActive(session);
  const until = session?.paidUntil || 0;
  const trialing = onTrial(session) && until > now;

  if (paid && !trialing) return null;   // real subscriber — say nothing

  if (trialing) {
    const left = until - now;
    const days = Math.floor(left / DAY);
    const hours = Math.floor((left % DAY) / 3600000);
    const remaining = days > 0 ? `${days}d ${hours}h` : `${hours}h`;
    const urgent = left < DAY;
    return (
      <div className={`flex flex-wrap items-center justify-between gap-2 px-4 py-2 rounded border ${
        urgent ? "border-terminal-yellow/50 bg-terminal-yellow/10" : "border-terminal-green/40 bg-terminal-green/[0.07]"
      }`}>
        <span className={`text-[12px] ${urgent ? "text-terminal-yellow" : "text-slate-200"}`}>
          {urgent ? "⏳" : "✓"} Free trial — <b>{remaining} left</b> of full access.
        </span>
        <button onClick={onStart}
          className="inline-flex items-center min-h-[36px] px-4 rounded bg-terminal-green text-black text-[11px] font-bold hover:opacity-90">
          KEEP FULL ACCESS
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2 rounded border border-terminal-green/40 bg-terminal-green/[0.07]">
      <span className="text-[12px] text-slate-200">
        Unlock full access — your first {TRIAL_DAYS} days are free.
      </span>
      <button
        onClick={() => {
          // An email is all the trial needs; the pricing modal collects it.
          onStart();
        }}
        className="inline-flex items-center min-h-[36px] px-4 rounded bg-terminal-green text-black text-[11px] font-bold hover:opacity-90">
        START YOUR {TRIAL_DAYS}-DAY FREE TRIAL
      </button>
    </div>
  );
}

export { signIn };
