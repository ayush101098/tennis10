"use client";

import { useState } from "react";
import { captureLead } from "@/lib/subscribe";
import { queueLead } from "@/lib/leadQueue";
import { signIn } from "@/lib/auth";
import { trackEvent } from "@/components/Analytics";

/**
 * Email-capture CTA. Stores the address via /api/subscribe (Netlify Blobs in
 * production). Shows inline success / error — no dead ends.
 *
 * `variant="waitlist"` is for the top-of-page CTA, where the terminal is not
 * being offered. The default variant's confirmation hands the visitor a
 * "OPEN THE TERMINAL" button, which would flatly contradict having just asked
 * them to wait — so the waitlist confirmation tells them what actually happens
 * next instead of pointing at the thing they were not given.
 */
export default function EmailCapture({
  source = "landing",
  cta = "Get early access",
  variant = "default",
  autoFocus = false,
}: {
  source?: string;
  cta?: string;
  variant?: "default" | "waitlist";
  autoFocus?: boolean;
}) {
  const [email, setEmail] = useState("");
  const [state, setState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [msg, setMsg] = useState("");

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (state === "loading") return;
    setState("loading");
    const res = await captureLead(email.trim(), source);
    // ok:true does not mean SAFE — subscribe.js answers ok:true even when it
    // fell back to per-container memory because the blob write failed (see
    // lib/leadQueue's docstring: confirmed happening in production right now,
    // store suspended, every write 403ing). Queue it so a later retry —
    // automatic, on next page load — lands it durably instead of the address
    // quietly evaporating with the container.
    if (res.ok && res.stored !== "blobs") queueLead(email.trim(), source);
    if (res.ok) {
      // Creates the account — the leads store, the sheet mirror — AND opens the
      // terminal, because the terminal is free now (TERMINAL_FREE). There is no
      // waitlist and no spot to wait for, so neither message may say there is:
      // copy that holds someone back from access they already have is the
      // fastest way to lose them at the one moment they were interested.
      signIn(email.trim().toLowerCase(), source);
      setState("done");
      setMsg("You're in — the terminal is open.");
      trackEvent("Signup", { source });
    } else {
      setState("error");
      setMsg(res.error || "Something went wrong.");
    }
  }

  if (state === "done") {
    return (
      <div className="flex flex-wrap items-center justify-center gap-3 text-sm text-terminal-green" role="status">
        <span><span aria-hidden>✓</span> {msg}</span>
        {/* One destination either way. The waitlist variant used to offer a
            look at today's matches instead, which was the right consolation
            when the terminal was behind a paywall and the wrong one now. */}
        <a href="/terminal"
          className="inline-flex items-center min-h-[36px] px-3 rounded bg-terminal-green text-black text-[11px] font-bold hover:opacity-90">
          OPEN THE TERMINAL →
        </a>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="flex flex-col sm:flex-row gap-2 w-full max-w-md">
      <label htmlFor={`email-${source}`} className="sr-only">Email address</label>
      <input
        id={`email-${source}`}
        type="email"
        required
        autoComplete="email"
        value={email}
        onChange={(e) => { setEmail(e.target.value); if (state === "error") setState("idle"); }}
        // eslint-disable-next-line jsx-a11y/no-autofocus -- opt-in only; the
        // hero CTA is the page's primary action, and it is off by default so no
        // other call site steals focus on load.
        autoFocus={autoFocus}
        placeholder="you@email.com"
        className="flex-1 min-h-[44px] px-3 py-2.5 rounded bg-terminal-panel border border-terminal-border text-sm text-slate-100 placeholder:text-terminal-muted focus:outline-none focus:ring-2 focus:ring-terminal-green/60"
      />
      <button
        type="submit"
        disabled={state === "loading"}
        className="min-h-[44px] px-5 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90 disabled:opacity-60 whitespace-nowrap"
      >
        {state === "loading" ? "Adding…" : cta}
      </button>
      {state === "error" && (
        <p className="text-xs text-red-400 sm:hidden" role="alert">{msg}</p>
      )}
      {state === "error" && (
        <p className="hidden sm:block text-xs text-red-400 self-center" role="alert">{msg}</p>
      )}
    </form>
  );
}
