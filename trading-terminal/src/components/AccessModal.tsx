"use client";

import { useEffect, useRef, useState } from "react";
import { signIn, useTier, subActive, loadSession, TERMINAL_FREE } from "@/lib/auth";
import GoogleSignIn, { GOOGLE_ENABLED } from "@/components/GoogleSignIn";

/**
 * The way into a free terminal.
 *
 * This replaces PricingModal at every entry point. That component still exists,
 * untouched, and still contains the whole paid flow — plans, Stripe, PayPal,
 * UPI, the crypto verifier — because the paywall came down by operator decision
 * (see TERMINAL_FREE in lib/auth) and not because the product stopped being
 * sellable. Putting the free door in its own file means neither has to be bent
 * around the other, and restoring the paid one is a two-line swap.
 *
 * Two presentations, one door:
 *
 *   "welcome"  the landing page. Someone who has never heard of this needs to
 *              know what it is BEFORE being asked for anything, so the product
 *              comes first and the field is the last thing on screen.
 *
 *   "signin"   the terminal. They already clicked through; they want the board,
 *              not a pitch. One line, one field.
 *
 * Neither ever mentions a price, because there isn't one.
 */

export type AccessVariant = "welcome" | "signin";

interface Props {
  open: boolean;
  onClose: () => void;
  /** Fired once a session exists — the caller re-reads it and re-renders. */
  onDone?: () => void;
  variant?: AccessVariant;
  /** Where the lead came from, recorded on the account. */
  source?: string;
}

/** What the product is, in the fewest lines that are still true. */
const WHAT_IT_IS = [
  {
    k: "A live win chance",
    v: "Every match gets a number saying how likely each player is to win — and it moves as the points are played, rather than being set before the match and left alone. We always label which one you are looking at.",
  },
  {
    k: "The moments that decide matches",
    v: "Repeated break points. A player who has just been broken and is already in trouble again. A set getting away from someone. We watch the sequence, not just the current score.",
  },
  {
    k: "What the market is charging",
    v: "Where a market exists for a match, we show its price next to ours, so you can see when the two disagree — and we show nothing when there is no market to trade into.",
  },
  {
    k: "ATP · WTA · Challenger · ITF",
    v: "Every match on the calendar, live scores included — not just the tournaments that make television.",
  },
];

export default function AccessModal({
  open, onClose, onDone, variant = "signin", source,
}: Props) {
  const { refresh } = useTier();
  const [email, setEmail] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus the one field that matters. Not on the welcome variant: stealing the
  // caret before someone has read a word makes the page feel like a form.
  useEffect(() => {
    if (open && variant === "signin") inputRef.current?.focus();
  }, [open, variant]);

  // Escape closes. A modal you cannot dismiss with the key everyone tries is
  // a modal people close by leaving the site.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const valid = /\S+@\S+\.\S+/.test(email);

  const enter = (addr: string) => {
    const s = signIn(addr.trim().toLowerCase(), source);
    refresh();
    setMsg({ ok: true, text: s.isAdmin ? "Welcome back — admin access." : "You're in. Opening the terminal…" });
    onDone?.();
    // Straight to the board. Landing someone back on the page they just signed
    // up from is where most of them stop.
    setTimeout(() => {
      onClose();
      if (typeof window !== "undefined" && !window.location.pathname.startsWith("/terminal")) {
        window.location.href = "/terminal";
      }
    }, 700);
  };

  const submit = () => {
    if (!valid) { setMsg({ ok: false, text: "That doesn't look like an email address." }); return; }
    enter(email);
  };

  // Someone who is already signed in has no business being shown the door.
  if (subActive(loadSession())) {
    return (
      <Shell onClose={onClose} wide={false}>
        <div className="p-6 text-center space-y-3">
          <div className="text-sm font-bold text-terminal-green">You already have full access.</div>
          <a href="/terminal"
            className="inline-flex items-center justify-center min-h-[44px] px-5 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90">
            OPEN THE TERMINAL →
          </a>
        </div>
      </Shell>
    );
  }

  const form = (
    <div className="space-y-3">
      {GOOGLE_ENABLED && (
        <>
          <GoogleSignIn onEmail={enter} onError={(reason) => setMsg({ ok: false, text: reason })} />
          <div className="flex items-center gap-3">
            <div className="flex-1 h-px bg-terminal-border" />
            <span className="text-[9px] text-terminal-muted">OR USE EMAIL</span>
            <div className="flex-1 h-px bg-terminal-border" />
          </div>
        </>
      )}

      <div>
        <label htmlFor="access-email" className="block text-[11px] text-terminal-muted mb-1">
          Your email
        </label>
        <input
          id="access-email" ref={inputRef} type="email" autoComplete="email"
          value={email} onChange={e => setEmail(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter") submit(); }}
          placeholder="you@example.com"
          className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2.5 text-sm text-slate-200 focus:border-terminal-cyan outline-none"
        />
      </div>

      <button onClick={submit} disabled={!valid}
        className="w-full min-h-[46px] rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90 disabled:opacity-40 transition">
        OPEN THE TERMINAL →
      </button>

      {/* The honest version of what the address is for. Saying "no spam" and
          leaving it there is what every list says; saying what it is actually
          used for is the part people can check. */}
      <p className="text-[10px] text-terminal-muted leading-relaxed">
        Free — no card, no plan, no trial that runs out. The address is your
        sign-in and how your bet journal follows you between devices. It is not
        sold or passed on, and one line to the address in the footer removes it.
      </p>

      {msg && (
        <div className={`text-[11px] font-bold ${msg.ok ? "text-terminal-green" : "text-terminal-red"}`}>
          {msg.text}
        </div>
      )}
    </div>
  );

  if (variant === "signin") {
    return (
      <Shell onClose={onClose} wide={false}>
        <div className="p-5">
          <div className="text-[11px] font-bold text-terminal-green mb-1">● OPEN THE TERMINAL</div>
          <p className="text-[11px] text-terminal-muted mb-4">
            Live win chances, pressure signals and your bet journal.
            One field and you&apos;re in.
          </p>
          {form}
        </div>
      </Shell>
    );
  }

  return (
    <Shell onClose={onClose} wide>
      <div className="p-5 sm:p-6">
        <div className="text-[11px] font-bold text-terminal-green mb-1">● WHAT THIS IS</div>
        <h2 className="text-lg sm:text-xl font-bold text-slate-100 leading-snug mb-1">
          Market intelligence for tennis traders.
        </h2>
        <p className="text-[12px] text-terminal-muted mb-5">
          {TERMINAL_FREE
            ? "The whole terminal is free. Leave an email and it opens."
            : "Leave an email to get started."}
        </p>

        <div className="grid gap-3 sm:grid-cols-2 mb-5">
          {WHAT_IT_IS.map(({ k, v }) => (
            <div key={k} className="border border-terminal-border rounded-lg p-3">
              <div className="text-[11px] font-bold text-slate-100 mb-1">{k}</div>
              <div className="text-[11px] text-terminal-muted leading-relaxed">{v}</div>
            </div>
          ))}
        </div>

        <div className="border-t border-terminal-border pt-4">{form}</div>

        <p className="text-[10px] text-terminal-muted mt-4 leading-relaxed">
          These are probabilities, not guarantees, and we are open about where we
          are not accurate enough to be useful. Bet only what you can afford to lose.
        </p>
      </div>
    </Shell>
  );
}

/** Dialog chrome: backdrop, escape hatch, scroll containment. */
function Shell({ children, onClose, wide }: {
  children: React.ReactNode; onClose: () => void; wide: boolean;
}) {
  return (
    <div
      role="dialog" aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
      // Only a click that both starts AND ends on the backdrop dismisses —
      // otherwise a text selection that drags past the panel edge closes the
      // dialog and throws away what was typed.
      onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className={`relative w-full ${wide ? "max-w-2xl" : "max-w-md"} max-h-[90vh] overflow-y-auto
          bg-terminal-panel border border-terminal-border rounded-lg shadow-2xl`}
      >
        <button onClick={onClose} aria-label="Close"
          className="absolute top-2 right-3 text-terminal-muted hover:text-slate-200 text-lg leading-none">
          ×
        </button>
        {children}
      </div>
    </div>
  );
}
