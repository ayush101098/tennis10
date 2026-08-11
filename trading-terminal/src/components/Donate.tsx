"use client";

import { useEffect, useState } from "react";
import { PAYMENT_ADDRESS, subActive, useTier } from "@/lib/auth";
import QrCode from "@/components/QrCode";

/**
 * Support-the-project ask.
 *
 * Two surfaces:
 *   DonateLink   — a quiet footer link, always available
 *   DonatePrompt — appears once someone has spent DWELL_MINUTES actually using
 *                  the site, then never again unless they ask for it
 *
 * Deliberate restraint, because a donation ask is easy to get wrong:
 *   - Dwell counts only while the tab is VISIBLE. Ten minutes of a background
 *     tab is not ten minutes of use, and asking on that basis is asking a
 *     stranger.
 *   - Subscribers never see the prompt. Asking someone who already pays $99 to
 *     also donate reads as ingratitude.
 *   - Dismissal is permanent (localStorage). One ask, one answer.
 */

const ADDRESS = PAYMENT_ADDRESS;
const DWELL_KEY = "tt_site_dwell_v1";
const DISMISS_KEY = "tt_donate_dismissed_v1";
const DWELL_MINUTES = 10;

/** Seconds of visible time on the site, accumulated across visits. */
function useVisibleDwell(): number {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    const read = () => parseInt(localStorage.getItem(DWELL_KEY) || "0", 10) || 0;
    setSecs(read());
    const iv = setInterval(() => {
      if (document.visibilityState !== "visible") return;
      const next = read() + 1;
      localStorage.setItem(DWELL_KEY, String(next));
      setSecs(next);
    }, 1000);
    return () => clearInterval(iv);
  }, []);
  return secs;
}

function AddressBlock({ onCopied }: { onCopied?: () => void }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="flex items-center gap-2">
      <code className="flex-1 bg-terminal-bg border border-terminal-border rounded px-2 py-1.5 text-[10px] text-terminal-cyan break-all select-all">
        {ADDRESS}
      </code>
      <button
        onClick={() => {
          navigator.clipboard?.writeText(ADDRESS);
          setCopied(true);
          onCopied?.();
          setTimeout(() => setCopied(false), 2000);
        }}
        className="shrink-0 min-h-[34px] px-2 rounded border border-terminal-border text-[10px] font-bold text-slate-300 hover:bg-terminal-bg">
        {copied ? "COPIED" : "COPY"}
      </button>
    </div>
  );
}

/** Quiet, permanent link. */
export function DonateLink() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button onClick={() => setOpen(true)}
        className="text-terminal-muted hover:text-slate-300 underline underline-offset-2">
        support the project
      </button>
      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setOpen(false)}>
          <div className="w-full max-w-[380px] rounded-lg border border-terminal-border bg-terminal-panel p-4"
            onClick={e => e.stopPropagation()}>
            <div className="flex items-start justify-between mb-2">
              <span className="text-[12px] font-bold text-slate-100">Support Tennis Alpha</span>
              <button onClick={() => setOpen(false)} className="text-terminal-muted hover:text-slate-200 px-1">×</button>
            </div>
            <p className="text-[10px] text-terminal-muted leading-relaxed mb-3">
              The model and the live feed run on a machine someone pays for. ETH or any
              ERC-20 on Ethereum mainnet, any amount.
            </p>
            <div className="flex justify-center mb-3">
              <QrCode value={ADDRESS} size={120} label="Scan to send ETH" />
            </div>
            <AddressBlock />
            <p className="mt-3 text-[9px] text-terminal-muted">
              A donation is not a subscription — it grants no access and expects nothing back.
            </p>
          </div>
        </div>
      )}
    </>
  );
}

/** The dwell-triggered ask. Renders nothing until it has earned the right to. */
export function DonatePrompt() {
  const { session } = useTier();
  const secs = useVisibleDwell();
  const [dismissed, setDismissed] = useState(true);   // assume dismissed until localStorage says otherwise

  useEffect(() => {
    try { setDismissed(!!localStorage.getItem(DISMISS_KEY)); } catch { setDismissed(true); }
  }, []);

  const close = () => {
    try { localStorage.setItem(DISMISS_KEY, String(Date.now())); } catch { /* fine */ }
    setDismissed(true);
  };

  // Never to a paying customer, never before the dwell threshold, never twice.
  if (subActive(session)) return null;
  if (dismissed) return null;
  if (secs < DWELL_MINUTES * 60) return null;

  return (
    <div className="fixed bottom-4 right-4 z-40 w-[300px] rounded-lg border border-terminal-green/40 bg-terminal-panel shadow-xl p-3">
      <div className="flex items-start justify-between gap-2 mb-1">
        <span className="text-[11px] font-bold text-slate-100">Getting value from this?</span>
        <button onClick={close} aria-label="Dismiss"
          className="text-terminal-muted hover:text-slate-200 leading-none px-1">×</button>
      </div>
      <p className="text-[10px] text-terminal-muted leading-relaxed mb-2">
        You&apos;ve been here {Math.floor(secs / 60)} minutes. The feed and the model run on
        hardware someone pays for — ETH on mainnet, any amount, no strings.
      </p>
      <AddressBlock onCopied={close} />
      <p className="mt-2 text-[9px] text-terminal-muted">
        Grants no access — that&apos;s what the subscription is for.
      </p>
    </div>
  );
}
