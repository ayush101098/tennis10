"use client";

import { useState } from "react";
import {
  PAYMENT_ADDRESS, PRO_PRICE_USD,
  signIn, verifyPaymentTx, grantPro, useTier,
} from "@/lib/auth";

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
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  if (!open) return null;

  const validEmail = /\S+@\S+\.\S+/.test(email);

  const startFree = () => {
    if (!validEmail) { setMsg({ ok: false, text: "Enter a valid email first." }); return; }
    const s = signIn(email);
    refresh();
    setMsg({ ok: true, text: s.isAdmin ? "Welcome back, admin — full access enabled." : "Free account active — pre-match probabilities unlocked." });
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
    setBusy(true);
    setMsg(null);
    const result = await verifyPaymentTx(txHash);
    if (result.ok) {
      grantPro(txHash.trim());
      refresh();
      setMsg({ ok: true, text: `${result.reason} Pro access unlocked — welcome.` });
      setTimeout(() => { onDone?.(); onClose(); }, 1200);
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
            <div className="grid grid-cols-2 gap-3">
              {/* FREE */}
              <div className="border border-terminal-border rounded-lg p-4 flex flex-col">
                <div className="text-slate-200 font-bold text-sm mb-1">FREE</div>
                <div className="text-2xl font-bold text-slate-100 mb-3">$0</div>
                <ul className="text-[11px] text-slate-300 space-y-1.5 flex-1">
                  <li>✓ Every match — ATP · WTA · Challenger · ITF</li>
                  <li>✓ Live scores &amp; schedules</li>
                  <li>✓ <b>Pre-match model probabilities</b> (NN + Elo)</li>
                  <li className="text-terminal-muted">✗ Live True P &amp; edge vs bookmaker</li>
                  <li className="text-terminal-muted">✗ Kelly staking &amp; Value Board</li>
                  <li className="text-terminal-muted">✗ Hedge-timing &amp; break/hold signals</li>
                  <li className="text-terminal-muted">✗ Bet tracker</li>
                </ul>
                <button onClick={startFree}
                  className="mt-3 w-full py-2 rounded border border-terminal-border text-slate-200 text-xs font-bold hover:bg-terminal-bg transition">
                  START FREE
                </button>
              </div>

              {/* PRO */}
              <div className="border border-terminal-green/50 bg-terminal-green/5 rounded-lg p-4 flex flex-col relative">
                <div className="absolute -top-2 right-3 text-[9px] font-bold bg-terminal-green text-black px-2 py-0.5 rounded">FULL TERMINAL</div>
                <div className="text-terminal-green font-bold text-sm mb-1">PRO</div>
                <div className="text-2xl font-bold text-slate-100 mb-3">${PRO_PRICE_USD}<span className="text-xs text-terminal-muted font-normal"> one-time</span></div>
                <ul className="text-[11px] text-slate-300 space-y-1.5 flex-1">
                  <li>✓ Everything in Free</li>
                  <li>✓ <b>Live True P</b> — score-conditioned Markov engine</li>
                  <li>✓ <b>Edge vs de-vigged bookmaker odds</b>, every match</li>
                  <li>✓ <b>Value Board</b> — ranked bets with ¼-Kelly stakes</li>
                  <li>✓ <b>Hedge-timing alerts</b> (trend break / adverse move / deuce loss)</li>
                  <li>✓ Break/hold signal engine + live serve analytics</li>
                  <li>✓ <b>Bet tracker</b> — P&amp;L, ROI, closing-line value</li>
                </ul>
                <button onClick={startPro}
                  className="mt-3 w-full py-2 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90 transition">
                  GET PRO — ${PRO_PRICE_USD}
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
            <div className="text-sm font-bold text-slate-100 mb-2">Pay ${PRO_PRICE_USD} in crypto</div>
            <ol className="text-[11px] text-slate-300 space-y-2 mb-4 list-decimal list-inside">
              <li>Send <b>${PRO_PRICE_USD} worth of ETH / USDT / USDC</b> (Ethereum mainnet) to:</li>
            </ol>
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
            <button onClick={verify} disabled={busy || !txHash.trim()}
              className="w-full py-2 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90 transition disabled:opacity-40">
              {busy ? "VERIFYING ON-CHAIN…" : "VERIFY PAYMENT & UNLOCK"}
            </button>
            {msg && (
              <div className={`mt-3 text-[11px] ${msg.ok ? "text-terminal-green" : "text-terminal-red"}`}>{msg.text}</div>
            )}
            <div className="mt-4 text-[9px] text-terminal-muted leading-relaxed">
              Verification checks the transaction on Ethereum mainnet: it must be confirmed and pay the address above
              (native ETH or an ERC-20 transfer). Access is tied to the email you entered ({email || "—"}).
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
