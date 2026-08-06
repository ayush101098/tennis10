"use client";

import { useState } from "react";
import {
  PAYMENT_ADDRESS, PRO_PRICE_USD, PAYPAL_ME_URL, PAYPAL_ID,
  signIn, grantPro, useTier,
} from "@/lib/auth";
import { serverVerifyPayment } from "@/lib/entitlement";
import QrCode from "@/components/QrCode";

interface Props {
  open: boolean;
  onClose: () => void;
  /** Called after a successful sign-in or upgrade */
  onDone?: () => void;
}

export default function PricingModal({ open, onClose, onDone }: Props) {
  const { session, refresh } = useTier();
  const [email, setEmail] = useState(session?.email || "");
  const [step, setStep] = useState<"plans" | "pay">("plans");
  const [txHash, setTxHash] = useState("");
  // which action is in flight — distinguishes the card and crypto buttons so
  // only the one that was clicked shows a spinner
  const [busy, setBusy] = useState<false | "crypto" | "paypalme" | "intent">(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  // PayPal.me: the follow-up form only appears once they have actually opened
  // the link, so the modal is not cluttered for everyone else
  const [paypalMeOpened, setPaypalMeOpened] = useState(false);
  const [paypalNote, setPaypalNote] = useState("");
  // email banked before the PayPal link is handed over
  const [intentSaved, setIntentSaved] = useState(false);

  if (!open) return null;

  const validEmail = /\S+@\S+\.\S+/.test(email);

  /**
   * Record a PayPal.me payment claim.
   *
   * Deliberately does NOT call grantPro: nothing here has been verified, and a
   * client-side unlock on an unverifiable claim is just a free-access button.
   * The claim lands in the account database as pending and is switched on by
   * hand from /admin once the payment is matched in PayPal.
   */
  /**
   * Bank the email before handing over the PayPal link.
   *
   * PayPal.me transfers arrive carrying only a display name, so without an
   * address captured up front there is nothing to match a payment against.
   * This writes the lead (and mirrors it to the sheet with a timestamp), then
   * releases the link.
   */
  const savePaypalIntent = async () => {
    if (!validEmail) { setMsg({ ok: false, text: "Enter a valid email first." }); return; }
    setBusy("intent"); setMsg(null);
    try {
      const res = await fetch("/api/subscribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, source: "paypal-intent", event: "paypal_intent" }),
      });
      const data = await res.json();
      if (data.ok) {
        signIn(email);
        refresh();
        setIntentSaved(true);
        setMsg({ ok: true, text: "Email recorded — the PayPal link is ready below." });
      } else {
        setMsg({ ok: false, text: data.error || "Could not record that email. Check it and try again." });
      }
    } catch {
      setMsg({ ok: false, text: "Network error — try again." });
    } finally {
      setBusy(false);
    }
  };

  const claimPaypalMe = async () => {
    if (!validEmail) { setMsg({ ok: false, text: "Enter a valid email first." }); return; }
    setBusy("paypalme"); setMsg(null);
    try {
      const res = await fetch("/api/account", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email, action: "claim", method: "paypal.me",
          note: paypalNote.trim(), amountUsd: PRO_PRICE_USD,
        }),
      });
      const data = await res.json();
      if (data.ok) {
        signIn(email);
        refresh();
        setMsg({ ok: true, text: "Got it — we'll match the payment and switch your access on. You'll keep this email as your login." });
        setPaypalNote("");
      } else {
        setMsg({ ok: false, text: data.reason || "Could not record that. Email us and we'll sort it out." });
      }
    } catch {
      setMsg({ ok: false, text: "Network error — email us and we'll sort it out." });
    } finally {
      setBusy(false);
    }
  };

  const startFree = () => {
    if (!validEmail) { setMsg({ ok: false, text: "Enter a valid email first." }); return; }
    const s = signIn(email);
    refresh();
    setMsg({ ok: true, text: s.isAdmin ? "Welcome back, admin — full access enabled."
      : s.tier === "pro" ? "Subscription active — full terminal unlocked."
      : `Account created. Subscribe for $${PRO_PRICE_USD}/month to open the terminal.` });
    if (s.isAdmin || s.tier === "pro") { onDone?.(); onClose(); }
    else onDone?.();
  };

  const startPro = () => {
    if (!validEmail) { setMsg({ ok: false, text: "Enter a valid email first." }); return; }
    const s = signIn(email);
    refresh();
    if (s.isAdmin || s.tier === "pro") {
      setMsg({ ok: true, text: "This account already has full access." });
      onDone?.(); onClose();
      return;
    }
    setMsg(null);
    setStep("pay");
  };

  const verify = async () => {
    setBusy("crypto");
    setMsg(null);
    // Authoritative: the SERVER verifies the tx on-chain and issues entitlement.
    // Fail-closed — no server "ok", no access.
    const result = await serverVerifyPayment(email.trim().toLowerCase(), txHash.trim());
    if (result.ok && result.paidUntil) {
      grantPro(txHash.trim(), result.paidUntil, result.amountUsd);
      refresh();
      setMsg({ ok: true, text: `${result.reason} Full terminal unlocked — welcome.` });
      setTimeout(() => { onDone?.(); onClose(); }, 1400);
    } else {
      setMsg({ ok: false, text: result.reason });
    }
    setBusy(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={onClose}>
      <div className="w-[680px] max-w-[94vw] max-h-[90vh] overflow-y-auto bg-terminal-panel border border-terminal-border rounded-lg shadow-2xl"
        onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="px-5 py-3 border-b border-terminal-border flex items-center justify-between">
          <span className="text-sm font-bold text-terminal-green">◉ UNLOCK THE TERMINAL</span>
          <button onClick={onClose} className="text-terminal-muted hover:text-slate-200 text-sm">✕</button>
        </div>

        {step === "plans" ? (
          <div className="p-5">
            {/* Email */}
            <label className="block text-[11px] text-terminal-muted mb-1">Your email</label>
            <input
              type="email" value={email} onChange={e => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full mb-4 bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm text-slate-200 focus:border-terminal-cyan outline-none"
            />

            {/* Plans */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* FREE ACCOUNT (no terminal) */}
              <div className="border border-terminal-border rounded-lg p-4 flex flex-col">
                <div className="text-slate-200 font-bold text-sm mb-1">FREE ACCOUNT</div>
                <div className="text-2xl font-bold text-slate-100 mb-3">$0</div>
                <ul className="text-[11px] text-slate-300 space-y-1.5 flex-1">
                  <li>✓ Live scores &amp; schedules — ATP · WTA · Challenger · ITF</li>
                  <li>✓ One free match analysis a day on the home page</li>
                  <li className="text-terminal-muted">✗ The terminal (Live True P, edge, Value Board)</li>
                  <li className="text-terminal-muted">✗ Kelly staking &amp; hedge-timing signals</li>
                  <li className="text-terminal-muted">✗ Bet tracker</li>
                </ul>
                <button onClick={startFree}
                  className="mt-3 w-full py-2 rounded border border-terminal-border text-slate-200 text-xs font-bold hover:bg-terminal-bg transition">
                  CREATE FREE ACCOUNT
                </button>
              </div>

              {/* PRO */}
              <div className="border border-terminal-green/50 bg-terminal-green/5 rounded-lg p-4 flex flex-col relative">
                <div className="absolute -top-2 right-3 text-[9px] font-bold bg-terminal-green text-black px-2 py-0.5 rounded">FULL TERMINAL</div>
                <div className="text-terminal-green font-bold text-sm mb-1">PRO</div>
                <div className="text-2xl font-bold text-slate-100 mb-3">${PRO_PRICE_USD}<span className="text-xs text-terminal-muted font-normal"> / month</span></div>
                <ul className="text-[11px] text-slate-300 space-y-1.5 flex-1">
                  <li>✓ <b>Unlimited bets</b> — no trial cap</li>
                  <li>✓ <b>Live True P</b> — score-conditioned Markov engine</li>
                  <li>✓ <b>Edge vs de-vigged bookmaker odds</b>, every match</li>
                  <li>✓ <b>Value Board</b> — ranked bets with ¼-Kelly stakes</li>
                  <li>✓ <b>Hedge-timing alerts</b> (trend break / adverse move / deuce loss)</li>
                  <li>✓ Break/hold signal engine + live serve analytics</li>
                  <li>✓ <b>Bet tracker</b> — P&amp;L, ROI, closing-line value</li>
                </ul>
                <button onClick={startPro}
                  className="mt-3 w-full py-2 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90 transition">
                  SUBSCRIBE — ${PRO_PRICE_USD}/mo
                </button>
              </div>
            </div>

            {msg && (
              <div className={`mt-3 text-[11px] ${msg.ok ? "text-terminal-green" : "text-terminal-red"}`}>{msg.text}</div>
            )}
          </div>
        ) : (
          <div className="p-5">
            <button onClick={() => setStep("plans")} className="text-[10px] text-terminal-muted hover:text-slate-300 mb-3">← back to plans</button>

            <div className="text-sm font-bold text-slate-100 mb-1">Subscribe — ${PRO_PRICE_USD}/month</div>
            <div className="text-[10px] text-terminal-muted mb-4">
              Two ways to pay — scan the code or copy the address.
            </div>
            <div className="flex items-center gap-3 mb-3">
              <div className="flex-1 h-px bg-terminal-border" />
              <span className="text-[9px] text-terminal-muted">💠 PAYPAL</span>
              <div className="flex-1 h-px bg-terminal-border" />
            </div>

            {/* PayPal.me — a plain payment link, so there is no callback and
                nothing to verify against. Access is therefore NOT granted here:
                the claim is queued and confirmed by hand in /admin.

                The link is withheld until the email is captured. A PayPal.me
                transfer arrives with only a PayPal display name attached, so an
                address recorded BEFORE the money moves is the only reliable way
                to match a payment to an account. */}
            {!intentSaved ? (
              <>
                <button onClick={savePaypalIntent} disabled={!!busy || !validEmail}
                  className="flex items-center justify-center w-full min-h-[44px] rounded border border-terminal-blue/50 text-terminal-blue text-xs font-bold hover:bg-terminal-blue/10 transition disabled:opacity-40">
                  {busy === "intent" ? "CHECKING…" : `💠 PAY $${PRO_PRICE_USD} VIA PAYPAL →`}
                </button>
                <div className="mt-1.5 text-[9px] text-terminal-muted text-center">
                  {validEmail
                    ? "We record your email first, so your payment can be matched to your account"
                    : "Enter your email above first — it is how your payment gets matched to your account"}
                </div>
              </>
            ) : (
              <>
                <a href={`${PAYPAL_ME_URL}/${PRO_PRICE_USD}`} target="_blank" rel="noreferrer"
                  onClick={() => setPaypalMeOpened(true)}
                  className="flex items-center justify-center w-full min-h-[44px] rounded bg-terminal-blue/20 border border-terminal-blue/50 text-terminal-blue text-xs font-bold hover:bg-terminal-blue/30 transition">
                  💠 OPEN PAYPAL — PAY ${PRO_PRICE_USD} →
                </a>
                <div className="mt-1.5 text-[9px] text-terminal-muted text-center">
                  {email} recorded ✓
                </div>
                <div className="mt-3 flex flex-col items-center gap-2">
                  <QrCode value={`${PAYPAL_ME_URL}/${PRO_PRICE_USD}`} size={124}
                    label={`Scan to pay $${PRO_PRICE_USD}`} />
                  <div className="flex items-center gap-2 w-full">
                    <code className="flex-1 bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-[11px] text-terminal-blue break-all select-all text-center">
                      {PAYPAL_ID}
                    </code>
                    <button
                      onClick={() => { navigator.clipboard?.writeText(PAYPAL_ID); setMsg({ ok: true, text: "PayPal ID copied." }); }}
                      className="shrink-0 min-h-[36px] text-[10px] px-2 rounded border border-terminal-border text-slate-300 hover:bg-terminal-bg">
                      COPY
                    </button>
                  </div>
                </div>
              </>
            )}
            {paypalMeOpened && (
              <div className="mt-2">
                <input
                  value={paypalNote} onChange={e => setPaypalNote(e.target.value)}
                  placeholder="PayPal name or transaction ID"
                  className="w-full mb-2 bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-[11px] text-slate-200 focus:border-terminal-blue outline-none"
                />
                <button onClick={claimPaypalMe} disabled={!!busy || !paypalNote.trim()}
                  className="w-full min-h-[40px] rounded bg-terminal-blue/20 border border-terminal-blue/50 text-terminal-blue text-xs font-bold hover:bg-terminal-blue/30 transition disabled:opacity-40">
                  {busy === "paypalme" ? "SENDING…" : "I'VE PAID — REQUEST ACTIVATION"}
                </button>
                <p className="mt-1.5 text-[9px] text-terminal-muted leading-relaxed">
                  A PayPal.me transfer can&apos;t be checked automatically, so this one is switched on by hand
                  after the payment is matched — unlike the buttons above, which unlock instantly.
                </p>
              </div>
            )}

            <div className="flex items-center gap-3 my-4">
              <div className="flex-1 h-px bg-terminal-border" />
              <span className="text-[9px] text-terminal-muted">OR PAY IN CRYPTO</span>
              <div className="flex-1 h-px bg-terminal-border" />
            </div>

            <ol className="text-[11px] text-slate-300 space-y-2 mb-4 list-decimal list-inside">
              <li>Send <b>at least ${PRO_PRICE_USD} in ETH / USDC / USDT / DAI</b> (Ethereum mainnet) to:</li>
            </ol>
            <div className="flex justify-center mb-3">
              <QrCode value={PAYMENT_ADDRESS} size={124} label="Scan with your wallet" />
            </div>
            <div className="flex items-center gap-2 mb-4">
              <code className="flex-1 bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-[11px] text-terminal-cyan break-all select-all">
                {PAYMENT_ADDRESS}
              </code>
              <button
                onClick={() => { navigator.clipboard?.writeText(PAYMENT_ADDRESS); setMsg({ ok: true, text: "Address copied." }); }}
                className="shrink-0 text-[10px] px-2 py-2 rounded border border-terminal-border text-slate-300 hover:bg-terminal-bg">
                COPY
              </button>
            </div>
            <label className="block text-[11px] text-terminal-muted mb-1">2. Paste your transaction hash — access unlocks instantly once verified on-chain</label>
            <input
              value={txHash} onChange={e => setTxHash(e.target.value)}
              placeholder="0x…"
              className="w-full mb-3 bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-[11px] font-mono text-slate-200 focus:border-terminal-cyan outline-none"
            />
            <button onClick={verify} disabled={!!busy || !txHash.trim()}
              className="w-full py-2 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90 transition disabled:opacity-40">
              {busy === "crypto" ? "VERIFYING ON-CHAIN…" : "VERIFY PAYMENT & UNLOCK"}
            </button>
            {msg && (
              <div className={`mt-3 text-[11px] ${msg.ok ? "text-terminal-green" : "text-terminal-red"}`}>{msg.text}</div>
            )}
            <div className="mt-4 text-[9px] text-terminal-muted leading-relaxed">
              Verification checks the transaction on Ethereum mainnet: it must be confirmed, pay the address above,
              be worth at least <b>${PRO_PRICE_USD}</b>, and be from the last 30 days. Access runs 30 days from the
              payment, then renews on your next monthly payment. Tied to the email you entered ({email || "—"}).
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="px-5 py-2 border-t border-terminal-border text-[9px] text-terminal-muted">
          Model outputs are probabilities, not guarantees. Bet only what you can afford to lose — staking discipline (¼ Kelly, 5% cap, 2% edge floor) is part of the product.
        </div>
      </div>
    </div>
  );
}
