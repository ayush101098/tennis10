"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import Wordmark from "@/components/Wordmark";
import LiveGrid from "@/components/LiveGrid";
import BetTracker from "@/components/BetTracker";
import AccessModal from "@/components/AccessModal";
import { DonatePrompt } from "@/components/Donate";
import LiveUsers from "@/components/LiveUsers";
import { useTier, signIn, signOut, subActive, grantPro } from "@/lib/auth";
import { confirmStripeSession, capturePaypal } from "@/lib/entitlement";
import { disconnectPolymarket, loadPmConnection, PM_CHANGED_EVENT, type PmConnection } from "@/lib/pmTrading";

/**
 * The trading terminal.
 *
 * Table tennis was removed: it polled /api/tt every 8s for every terminal
 * user, for a product this does not sell. The panel, routes, feed function
 * and Python pipeline are all gone as of 2026-09-02.
 */


type View = "centre" | "tracker";

export default function TerminalPage() {
  const { session, refresh } = useTier();
  const [view, setView] = useState<View>("centre");
  const [pricingOpen, setPricingOpen] = useState(false);
  const paid = subActive(session);
  const email = session?.email || "guest";

  // The terminal is the expensive page — it polls the board continuously — so
  // it is still mounted only for a session. That session now costs nothing: the
  // paywall is gone (TERMINAL_FREE) and `paid` means no more than "signed in".
  const expired = !paid;

  // Return from Stripe Checkout: confirm the session server-side and unlock.
  // The server asks Stripe directly, so this does not wait on the webhook —
  // a customer who has just paid must never land back on a paywall.
  const [checkoutMsg, setCheckoutMsg] = useState<string | null>(null);
  // Set by TierProvider when this device lost the seat to another sign-in.
  // Being silently signed out reads like a bug; say what happened.
  const [evicted, setEvicted] = useState(false);
  useEffect(() => {
    const read = () => { try { if (localStorage.getItem("tt_evicted")) setEvicted(true); } catch {} };
    read();
    const iv = setInterval(read, 5000);
    return () => clearInterval(iv);
  }, []);
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
  //
  // …but take it back down once access is confirmed. The session is read from
  // localStorage in an effect, so the first render of every page load has no
  // session and looks exactly like an expired one: an admin or a subscriber
  // landing here got the paywall thrown over their own terminal a frame before
  // their session loaded, and nothing ever closed it again.
  //
  // Only an ASK this effect raised is withdrawn — autoOpened. A trial user
  // counts as paid, so a blanket "close whenever paid" would slam the modal
  // shut the instant they clicked KEEP FULL ACCESS.
  const autoOpened = useRef(false);
  useEffect(() => {
    if (expired) { autoOpened.current = true; setPricingOpen(true); }
    else if (autoOpened.current) { autoOpened.current = false; setPricingOpen(false); }
  }, [expired]);

  return (
    // 100dvh, not 100vh: on iOS the URL bar is counted in vh, so a vh-sized
    // shell is always taller than the visible area and the bottom row hides
    // under the browser chrome. h-screen stays as the fallback for old engines.
    <div className="h-screen [height:100dvh] w-full flex flex-col overflow-hidden safe-x">
      {/* ── Header Bar ──
          The row scrolls sideways rather than wrapping: at 375px the nav used
          to break "BET TRACKER" onto a second line and crush
          the account cluster out of reach. */}
      <header className="safe-top flex items-center justify-between gap-2 px-3 sm:px-4 py-1.5 border-b border-terminal-border bg-terminal-panel shrink-0">
        <div className="flex items-center gap-1.5 sm:gap-3 min-w-0 overflow-x-auto scroll-touch">
          <Link href="/" className="hover:opacity-80 whitespace-nowrap shrink-0">
            <Wordmark size={15} />
          </Link>
          <button onClick={() => setView("centre")}
            className={`nav-tab ${view === "centre" ? "text-terminal-yellow bg-terminal-yellow/10" : "text-terminal-muted hover:text-slate-300"}`}>
            🎾<span className="hidden xs:inline"> TENNIS</span>
          </button>
          <button onClick={() => setView("tracker")}
            className={`nav-tab ${view === "tracker" ? "text-terminal-cyan bg-terminal-cyan/10" : "text-terminal-muted hover:text-slate-300"}`}>
            📒<span className="hidden xs:inline"> BET TRACKER</span>
          </button>
          <Link href="/manual"
            className="nav-tab text-terminal-muted hover:text-slate-300">
            📘<span className="hidden xs:inline"> MANUAL</span>
          </Link>
        </div>
        <div className="flex items-center gap-2 sm:gap-3 text-[10px] shrink-0">
          {/* Capacity note. States the cap only — no seats-remaining counter,
              since we do not enforce a live count and inventing one would be
              manufactured scarcity. */}
          <span className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-terminal-yellow/40 text-terminal-yellow whitespace-nowrap"
            title="This terminal is kept to 50 members. The edge is thin and shared — a bigger room would move the prices we trade into.">
            🔒 CAPPED AT 50 MEMBERS
          </span>
          <LiveUsers />
          {session && <PmStatus email={session.email} />}
          {session ? (
            <>
              <span className="text-terminal-muted">{session.email}</span>
              {/* No PREVIEW state and nothing to upgrade to: signed in IS full
                  access. ADMIN stays because it opens /admin, which PRO does
                  not — that distinction is still real. */}
              <span className={`font-bold px-1.5 py-0.5 rounded ${
                session.isAdmin ? "bg-terminal-red/20 text-terminal-red"
                  : "bg-terminal-green/20 text-terminal-green"
              }`}>
                {session.isAdmin ? "ADMIN" : "FULL ACCESS"}
              </span>
              <button onClick={() => { signOut(); refresh(); }} className="inline-flex items-center min-h-[36px] px-1 text-terminal-muted hover:text-slate-300">
                sign out
              </button>
            </>
          ) : (
            <button onClick={() => setPricingOpen(true)}
              className="inline-flex items-center min-h-[36px] font-bold px-2.5 rounded border border-terminal-green/50 text-terminal-green hover:bg-terminal-green/10">
              SIGN IN
            </button>
          )}
        </div>
      </header>

      {evicted && (
        <div className="px-4 py-1.5 text-[11px] font-bold text-center bg-terminal-yellow/15 text-terminal-yellow border-b border-terminal-yellow/40 shrink-0">
          Signed out — this account is tied to one device. One account, one device.
          <button
            onClick={() => { try { localStorage.removeItem("tt_evicted"); localStorage.removeItem("tt_device_locked"); } catch {} setEvicted(false); setPricingOpen(true); }}
            className="ml-2 underline hover:opacity-80">
            sign back in
          </button>
        </div>
      )}

      {checkoutMsg && (
        <div className="px-4 py-1.5 text-[11px] font-bold text-center bg-terminal-green/15 text-terminal-green border-b border-terminal-green/40 shrink-0">
          {checkoutMsg}
        </div>
      )}

      {/* ── Body ── */}
      <div className="flex-1 min-h-0 relative">
        {view === "tracker" ? (
          <BetTracker />
        ) : (
          // Still not mounted for a signed-out visitor: the board polls
          // continuously, and an overlay over a polling board makes every one
          // of those requests for someone who cannot see it.
          paid ? <LiveGrid /> : <div />
        )}

        {/* ── Signed out: the door, not a paywall ──
            There is no price to state and nothing to upgrade to; the terminal
            is free (TERMINAL_FREE). All that is left is the email, and this
            says so rather than dressing a one-field form as a purchase. */}
        {!paid && (
          <div className="absolute inset-0 z-30 bg-terminal-bg/90 backdrop-blur-sm flex flex-col items-center justify-center gap-3 text-center px-6">
            <div className="text-3xl">🎾</div>
            <div className="text-sm font-bold text-slate-100">Leave an email and the terminal opens</div>
            <div className="text-[11px] text-terminal-muted max-w-[420px]">
              Live True P, the edge board, ¼-Kelly staking, hedge signals and the
              bet journal — all of it, free. The address is your sign-in and how
              your bet journal follows you between devices.
            </div>
            <button onClick={() => setPricingOpen(true)}
              className="mt-2 px-5 py-2.5 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90">
              OPEN THE TERMINAL →
            </button>
          </div>
        )}
      </div>

      <AccessModal open={pricingOpen} onClose={() => setPricingOpen(false)} onDone={refresh}
        variant="signin" source="terminal" />
      <DonatePrompt />
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
