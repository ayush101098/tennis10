"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import SchedulePanel from "@/components/SchedulePanel";
import BetTracker from "@/components/BetTracker";
import PricingModal from "@/components/PricingModal";
import LiveUsers from "@/components/LiveUsers";
import { TtMatchCentre, TtBetTracker } from "@/components/TtPanel";
import { useTier, signIn, signOut, subActive, grantPro, PRO_PRICE_USD } from "@/lib/auth";
import { confirmStripeSession, capturePaypal } from "@/lib/entitlement";
import { disconnectPolymarket, loadPmConnection, PM_CHANGED_EVENT, type PmConnection } from "@/lib/pmTrading";

/**
 * The unified terminal — 🎾 tennis and 🏓 table tennis in one shell.
 *
 * Access model: everyone gets the FULL terminal immediately, no paywall on
 * entry. A cumulative dwell meter (localStorage, ticks only while the tab is
 * visible) grants FREE_PREVIEW_SECONDS of real usage; when it runs out,
 * non-subscribers get the pro-subscription ask. Paid/admin: never gated.
 */

const FREE_PREVIEW_SECONDS = 180;
const DWELL_KEY = "tt_dwell_v1";

type Sport = "tennis" | "tt";
type View = "centre" | "tracker";

/** Seconds of free preview remaining; null = unlimited (subscriber/admin). */
function useDwellGate(paid: boolean): number | null {
  const [remaining, setRemaining] = useState<number | null>(paid ? null : FREE_PREVIEW_SECONDS);
  useEffect(() => {
    if (paid) { setRemaining(null); return; }
    const used = () => parseInt(localStorage.getItem(DWELL_KEY) || "0", 10) || 0;
    setRemaining(Math.max(FREE_PREVIEW_SECONDS - used(), 0));
    const iv = setInterval(() => {
      if (document.visibilityState !== "visible") return;
      const u = used() + 1;
      localStorage.setItem(DWELL_KEY, String(u));
      setRemaining(Math.max(FREE_PREVIEW_SECONDS - u, 0));
    }, 1000);
    return () => clearInterval(iv);
  }, [paid]);
  return remaining;
}

export default function TerminalPage() {
  const { session, refresh } = useTier();
  const [sport, setSport] = useState<Sport>("tennis");
  const [view, setView] = useState<View>("centre");
  const [pricingOpen, setPricingOpen] = useState(false);
  const paid = subActive(session);
  const email = session?.email || "guest";

  const remaining = useDwellGate(paid);
  const expired = remaining === 0;

  // deep links: /terminal?tab=tt (read once — useSearchParams needs a Suspense
  // boundary under static export, window.location does not)
  useEffect(() => {
    if (new URLSearchParams(window.location.search).get("tab") === "tt") setSport("tt");
  }, []);

  // Return from Stripe Checkout: confirm the session server-side and unlock.
  // The server asks Stripe directly, so this does not wait on the webhook —
  // a customer who has just paid must never land back on a paywall.
  const [checkoutMsg, setCheckoutMsg] = useState<string | null>(null);
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    if (q.get("checkout") !== "success") return;
    const sid = q.get("session_id");
    if (!sid) return;
    setCheckoutMsg("Confirming your payment…");
    confirmStripeSession(sid).then(r => {
      if (r.ok && r.paidUntil) {
        if (r.email) signIn(r.email);
        grantPro(`stripe:${sid}`, r.paidUntil, r.amountUsd);
        refresh();
        setCheckoutMsg(`${r.reason} Full terminal unlocked — welcome.`);
      } else {
        setCheckoutMsg(r.reason);
      }
      // drop the query so a refresh doesn't re-run this
      window.history.replaceState({}, "", window.location.pathname);
      setTimeout(() => setCheckoutMsg(null), 6000);
    });
  }, [refresh]);

  // Return from PayPal approval: capture server-side and unlock. PayPal sends
  // the order back as ?token=<orderId>; the money is only actually taken when
  // we capture, so this step is mandatory, not cosmetic.
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    if (q.get("paypal") !== "success") return;
    const orderId = q.get("token");
    if (!orderId) return;
    const who = session?.email || localStorage.getItem("tt_last_email") || "";
    setCheckoutMsg("Completing your PayPal payment…");
    capturePaypal(orderId, who).then(r => {
      if (r.ok && r.paidUntil) {
        if (r.email) signIn(r.email);
        grantPro(`paypal:${orderId}`, r.paidUntil, r.amountUsd);
        refresh();
        setCheckoutMsg(`${r.reason} Full terminal unlocked — welcome.`);
      } else {
        setCheckoutMsg(r.reason);
      }
      window.history.replaceState({}, "", window.location.pathname);
      setTimeout(() => setCheckoutMsg(null), 6000);
    });
  }, [refresh, session?.email]);

  // the moment the preview runs out, put the subscription ask in front of them
  useEffect(() => { if (expired) setPricingOpen(true); }, [expired]);

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden">
      {/* ── Header Bar ── */}
      <header className="flex items-center justify-between px-4 py-1.5 border-b border-terminal-border bg-terminal-panel shrink-0">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-terminal-green font-bold text-sm hover:opacity-80">◉ INTELLIGENCE TERMINAL</Link>
          <button onClick={() => { setSport("tennis"); setView("centre"); }}
            className={`text-[10px] font-bold px-2 py-0.5 rounded ${sport === "tennis" && view === "centre" ? "text-terminal-yellow bg-terminal-yellow/10" : "text-terminal-muted hover:text-slate-300"}`}>
            🎾 TENNIS
          </button>
          <button onClick={() => { setSport("tt"); setView("centre"); }}
            className={`text-[10px] font-bold px-2 py-0.5 rounded ${sport === "tt" && view === "centre" ? "text-terminal-yellow bg-terminal-yellow/10" : "text-terminal-muted hover:text-slate-300"}`}>
            🏓 TABLE TENNIS
          </button>
          <button onClick={() => setView("tracker")}
            className={`text-[10px] font-bold px-2 py-0.5 rounded ${view === "tracker" ? "text-terminal-cyan bg-terminal-cyan/10" : "text-terminal-muted hover:text-slate-300"}`}>
            📒 BET TRACKER
          </button>
          <Link href={sport === "tt" ? "/tt/manual" : "/manual"}
            className="text-[10px] font-bold px-2 py-0.5 rounded text-terminal-muted hover:text-slate-300">
            📘 MANUAL
          </Link>
        </div>
        <div className="flex items-center gap-3 text-[10px]">
          <LiveUsers />
          {session && <PmStatus email={session.email} />}
          {!paid && remaining !== null && (
            <span className={`font-mono font-bold px-1.5 py-0.5 rounded ${remaining <= 30 ? "bg-terminal-red/20 text-terminal-red" : "bg-terminal-border text-slate-300"}`}
              title="Free preview time remaining — subscribe for unlimited access">
              ⏱ {Math.floor(remaining / 60)}:{String(remaining % 60).padStart(2, "0")}
            </span>
          )}
          {session ? (
            <>
              <span className="text-terminal-muted">{session.email}</span>
              <span className={`font-bold px-1.5 py-0.5 rounded ${
                session.isAdmin ? "bg-terminal-red/20 text-terminal-red"
                  : paid ? "bg-terminal-green/20 text-terminal-green"
                  : "bg-terminal-border text-slate-300"
              }`}>
                {session.isAdmin ? "ADMIN" : paid ? "PRO" : "PREVIEW"}
              </span>
              {!paid && (
                <button onClick={() => setPricingOpen(true)}
                  className="font-bold px-2 py-0.5 rounded bg-terminal-green text-black hover:opacity-90">
                  GO PRO
                </button>
              )}
              <button onClick={() => { signOut(); refresh(); }} className="text-terminal-muted hover:text-slate-300">
                sign out
              </button>
            </>
          ) : (
            <button onClick={() => setPricingOpen(true)}
              className="font-bold px-2 py-0.5 rounded border border-terminal-green/50 text-terminal-green hover:bg-terminal-green/10">
              SIGN IN
            </button>
          )}
        </div>
      </header>

      {checkoutMsg && (
        <div className="px-4 py-1.5 text-[11px] font-bold text-center bg-terminal-green/15 text-terminal-green border-b border-terminal-green/40 shrink-0">
          {checkoutMsg}
        </div>
      )}

      {/* ── Body ── */}
      <div className="flex-1 min-h-0 relative">
        {view === "tracker" ? (
          <div className="h-full flex flex-col">
            <div className="flex items-center gap-2 px-3 py-1.5 border-b border-terminal-border shrink-0 text-[10px]">
              <button onClick={() => setSport("tennis")}
                className={`px-2 py-0.5 rounded font-bold ${sport === "tennis" ? "bg-terminal-green/20 text-terminal-green" : "text-terminal-muted hover:text-slate-300"}`}>
                🎾 TENNIS BETS
              </button>
              <button onClick={() => setSport("tt")}
                className={`px-2 py-0.5 rounded font-bold ${sport === "tt" ? "bg-terminal-green/20 text-terminal-green" : "text-terminal-muted hover:text-slate-300"}`}>
                🏓 TT BETS
              </button>
            </div>
            <div className="flex-1 min-h-0">
              {sport === "tt" ? <TtBetTracker email={email} /> : <BetTracker />}
            </div>
          </div>
        ) : sport === "tt" ? (
          <TtMatchCentre email={email} />
        ) : (
          <SchedulePanel tier="pro" onUpgrade={() => setPricingOpen(true)} />
        )}

        {/* ── Preview expired: lock overlay + subscription ask ── */}
        {expired && !paid && (
          <div className="absolute inset-0 z-30 bg-terminal-bg/90 backdrop-blur-sm flex flex-col items-center justify-center gap-3 text-center px-6">
            <div className="text-3xl">⏱</div>
            <div className="text-sm font-bold text-slate-100">Your free preview is up</div>
            <div className="text-[11px] text-terminal-muted max-w-[420px]">
              You&apos;ve had {FREE_PREVIEW_SECONDS / 60} minutes with the full terminal — live True P for
              tennis and table tennis, edge boards, trade tickets and both bet journals. Keep it running
              with Pro: <b className="text-slate-200">${PRO_PRICE_USD}/month</b>, unlimited access.
            </div>
            <button onClick={() => setPricingOpen(true)}
              className="mt-2 px-5 py-2.5 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90">
              GO PRO — ${PRO_PRICE_USD}/MONTH
            </button>
          </div>
        )}
      </div>

      <PricingModal open={pricingOpen} onClose={() => setPricingOpen(false)} onDone={refresh} />
    </div>
  );
}

/** Polymarket wallet status — connect happens inside any trade ticket. */
function PmStatus({ email }: { email: string }) {
  const [conn, setConn] = useState<PmConnection | null>(null);
  useEffect(() => {
    const load = () => setConn(loadPmConnection(email));
    load();
    window.addEventListener(PM_CHANGED_EVENT, load);
    return () => window.removeEventListener(PM_CHANGED_EVENT, load);
  }, [email]);

  return conn ? (
    <button onClick={() => disconnectPolymarket(email)}
      title={`Polymarket connected — orders sign with ${conn.address}. Click to disconnect.`}
      className="font-bold px-1.5 py-0.5 rounded bg-terminal-green/15 text-terminal-green border border-terminal-green/40 hover:bg-terminal-red/15 hover:text-terminal-red hover:border-terminal-red/40 font-mono">
      ⬢ PM {conn.address.slice(0, 6)}…{conn.address.slice(-4)}
    </button>
  ) : (
    <span title="Not connected to Polymarket — trades log as paper. Connect from any ⚡ TRADE ticket."
      className="font-bold px-1.5 py-0.5 rounded bg-terminal-border text-terminal-muted">
      ⬢ PM PAPER MODE
    </span>
  );
}
