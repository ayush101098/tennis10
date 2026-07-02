"use client";

/**
 * Client-side auth, tiers and payment verification.
 *
 * Tiers:
 *   public — not signed in: match list + full analysis of ONE match per day
 *   free   — signed in:    pre-match model probabilities on every match
 *   pro    — $99:          full trading terminal (live True P, edge, Kelly,
 *                          hedge signals, Value Board, bet tracker)
 *
 * ADMIN_EMAIL is always pro, forever, at no charge.
 *
 * Pro access is granted after an on-chain payment to PAYMENT_ADDRESS is
 * verified: the user pastes their transaction hash and we check it against
 * public Ethereum RPCs (to-address must match; any EVM value transfer or
 * ERC-20 transfer that targets the address in `to` or the token `to` field).
 *
 * NOTE: sessions and entitlements are stored in localStorage — good for a
 * single-operator launch; move to a real backend before selling at scale.
 */

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export const ADMIN_EMAIL = "ayushmishra101098@gmail.com";
export const PAYMENT_ADDRESS = "0x905aCd442c7B3EF9BfEB0A3189f3686c1Cd0c697";
export const PRO_PRICE_USD = 99;

export type Tier = "public" | "free" | "pro";

export interface Session {
  email: string;
  tier: "free" | "pro";
  isAdmin: boolean;
  txHash?: string;
  since: number;
}

const LS_KEY = "tt_session_v1";

function normEmail(e: string): string {
  return e.trim().toLowerCase();
}

export function loadSession(): Session | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return null;
    const s: Session = JSON.parse(raw);
    // Admin is always pro regardless of what was stored
    if (s.isAdmin || normEmail(s.email) === ADMIN_EMAIL) {
      s.isAdmin = true;
      s.tier = "pro";
    }
    return s;
  } catch {
    return null;
  }
}

function saveSession(s: Session | null): void {
  if (s) localStorage.setItem(LS_KEY, JSON.stringify(s));
  else localStorage.removeItem(LS_KEY);
}

export function signIn(email: string): Session {
  const e = normEmail(email);
  const isAdmin = e === ADMIN_EMAIL;
  const prev = loadSession();
  const s: Session = {
    email: e,
    tier: isAdmin ? "pro" : prev && normEmail(prev.email) === e && prev.tier === "pro" ? "pro" : "free",
    isAdmin,
    txHash: prev?.txHash,
    since: Date.now(),
  };
  saveSession(s);
  return s;
}

export function signOut(): void {
  saveSession(null);
}

/* ── On-chain payment verification ──────────────────────────────────────── */

const RPCS = [
  "https://ethereum-rpc.publicnode.com",
  "https://1rpc.io/eth",
  "https://eth.drpc.org",
];

export interface VerifyResult {
  ok: boolean;
  reason: string;
}

/**
 * Verify a payment tx hash: the transaction must exist, be confirmed, and its
 * `to` must be the payment address (native transfer) OR an ERC-20 `transfer`
 * whose recipient (first calldata arg) is the payment address.
 */
export async function verifyPaymentTx(txHash: string): Promise<VerifyResult> {
  const hash = txHash.trim();
  if (!/^0x[0-9a-fA-F]{64}$/.test(hash)) {
    return { ok: false, reason: "That is not a valid transaction hash (0x + 64 hex chars)." };
  }
  const target = PAYMENT_ADDRESS.toLowerCase();

  for (const rpc of RPCS) {
    try {
      const res = await fetch(rpc, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "eth_getTransactionByHash", params: [hash] }),
      });
      if (!res.ok) continue;
      const json = await res.json();
      const tx = json.result;
      if (!tx) return { ok: false, reason: "Transaction not found on Ethereum mainnet. Wait for confirmation and try again." };
      if (!tx.blockNumber) return { ok: false, reason: "Transaction is still pending — try again once it confirms." };

      const to = (tx.to || "").toLowerCase();
      const input: string = tx.input || "0x";

      // Native ETH transfer straight to the payment address
      const valueHex = (tx.value || "0x0").replace(/^0x/, "");
      const hasValue = /[1-9a-f]/i.test(valueHex);
      if (to === target && hasValue) {
        return { ok: true, reason: "Payment verified (ETH transfer)." };
      }
      // ERC-20 transfer(address,uint256) → recipient is the first calldata word
      if (input.startsWith("0xa9059cbb") && input.length >= 10 + 64) {
        const recipient = "0x" + input.slice(10 + 24, 10 + 64).toLowerCase();
        if (recipient === target) {
          return { ok: true, reason: "Payment verified (token transfer)." };
        }
      }
      return { ok: false, reason: "That transaction does not pay the access address." };
    } catch {
      continue; // try next RPC
    }
  }
  return { ok: false, reason: "Could not reach Ethereum RPC to verify — check your connection and retry." };
}

/** Upgrade the current session to pro after a verified payment. */
export function grantPro(txHash: string): Session | null {
  const s = loadSession();
  if (!s) return null;
  const up: Session = { ...s, tier: "pro", txHash };
  saveSession(up);
  return up;
}

/* ── Public free-analysis quota (1 match per day without an account) ────── */

const QUOTA_KEY = "tt_public_analysis";

/** Returns the match id the public visitor is allowed to analyse today (if any). */
export function getPublicAnalysisId(): string | null {
  try {
    const raw = localStorage.getItem(QUOTA_KEY);
    if (!raw) return null;
    const { day, id } = JSON.parse(raw);
    return day === new Date().toDateString() ? id : null;
  } catch {
    return null;
  }
}

/**
 * Claim the single free analysis slot for a match. Returns true if this match
 * may be analysed (either it already holds the slot, or the slot was free).
 */
export function claimPublicAnalysis(matchId: string): boolean {
  const current = getPublicAnalysisId();
  if (current === matchId) return true;
  if (current !== null) return false;
  localStorage.setItem(QUOTA_KEY, JSON.stringify({ day: new Date().toDateString(), id: matchId }));
  return true;
}

/* ── React context ───────────────────────────────────────────────────────── */

interface TierContextValue {
  session: Session | null;
  tier: Tier;
  refresh: () => void;
}

const TierContext = createContext<TierContextValue>({ session: null, tier: "public", refresh: () => {} });

export function TierProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const refresh = () => setSession(loadSession());
  useEffect(refresh, []);
  const tier: Tier = session ? session.tier : "public";
  return <TierContext.Provider value={{ session, tier, refresh }}>{children}</TierContext.Provider>;
}

export function useTier(): TierContextValue {
  return useContext(TierContext);
}
