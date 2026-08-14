"use client";

import { useState } from "react";
import { captureLead } from "@/lib/subscribe";
import { signIn } from "@/lib/auth";
import { trackEvent } from "@/components/Analytics";

/**
 * Email-capture CTA. Stores the address via /api/subscribe (Netlify Blobs in
 * production). Shows inline success / error — no dead ends.
 */
export default function EmailCapture({ source = "landing", cta = "Get early access" }:
  { source?: string; cta?: string }) {
  const [email, setEmail] = useState("");
  const [state, setState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [msg, setMsg] = useState("");

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (state === "loading") return;
    setState("loading");
    const res = await captureLead(email.trim(), source);
    if (res.ok) {
      // Still creates the account: this is what puts the address in the leads
      // store and the sheet mirror. It no longer unlocks anything — trials are
      // off (see TRIALS_ENABLED) — so the confirmation must not imply it does.
      signIn(email.trim().toLowerCase());
      setState("done");
      setMsg("You're on the list — we'll be in touch.");
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
