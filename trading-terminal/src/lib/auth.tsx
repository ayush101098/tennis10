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
 * ADMIN_EMAILS are always pro, forever, at no charge.
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

export const ADMIN_EMAILS = new Set([
  "ayushmishra101098@gmail.com",
  "mishrapriyanka9515@gmail.com",
  "sahil7goyal18@gmail.com",
  "yuvamsharma98@gmail.com",
  // First paying customer (tx 0x4548736331c3…, $100.31 ETH, 2026-07-23).
  // Added here to unblock access while the Netlify verify function's blob write
  // is failing ("Could not record entitlement"). NOTE: admin is free-forever AND
  // opens /admin (leads, revenue, payments) — downgrade to the TIME_GRANTS entry
  // below, which already grants this email access to 2026-09-02, once verify is
  // fixed. See also the accounts DB record for the real payment.
  "mateimo012@gmail.com",
]);
export const PAYMENT_ADDRESS = "0x905aCd442c7B3EF9BfEB0A3189f3686c1Cd0c697";
export const PRO_PRICE_USD = 100;            // monthly subscription
// PayPal.me is a plain payment link — it produces no callback and nothing the
// site can verify, so payments through it are confirmed by hand (see the claim
// flow in PricingModal). The API-based PayPal button above it unlocks instantly.
export const PAYPAL_ME_URL = "https://paypal.me/jessefuture10";
export const PAYPAL_ID = "paypal.me/jessefuture10";

/**
 * UPI — the practical route for Indian customers, where cards and PayPal both
 * carry friction. Like PayPal.me it produces no callback, so it is confirmed by
 * hand; unlike PayPal.me, the QR can carry the amount, so the payer does not
 * have to type it.
 *
 * USD_INR is a display rate for prefilling the QR, not a settlement rate. It is
 * deliberately generous rather than exact — a QR that asks for slightly more
 * than the price is a bad trade against one that asks for too little and leaves
 * the customer underpaid and unactivated. Update it when it drifts.
 */
export const UPI_ID = "tennisalpha.ybl";
export const USD_INR = 88;
export const upiUri = (usd: number): string =>
  `upi://pay?pa=${encodeURIComponent(UPI_ID)}&pn=${encodeURIComponent("Tennis Alpha")}`
  + `&am=${Math.round(usd * USD_INR)}&cu=INR&tn=${encodeURIComponent("Tennis Alpha subscription")}`;
export const SUBSCRIPTION_DAYS = 30;          // access window per payment
export const MIN_PAYMENT_USD = 100;           // payments below this are rejected
export const TRIAL_DAYS = 1;                  // free trial granted on first sign-up
/**
 * How the trial is described everywhere. A one-day trial reads as "24 hours",
 * not "1 day" — and deriving it means the copy cannot drift from the length the
 * server actually grants (netlify/functions/account.js is the authority).
 */
export const TRIAL_LABEL = TRIAL_DAYS === 1 ? "24-hour" : `${TRIAL_DAYS}-day`;
export const TRIAL_LENGTH = TRIAL_DAYS === 1 ? "24 hours" : `${TRIAL_DAYS} days`;
/**
 * Free trials are OFF (operator, 2026-08-14): the terminal is for subscribers
 * and operator-issued grants only.
 *
 * This flag governs COPY ONLY. The authority is TRIALS_ENABLED in
 * netlify/functions/account.js, which is what decides whether a grant is
 * actually written — a client flag alone would be an invitation to flip it in
 * devtools. Both must be turned back on together to re-open trials.
 */
export const TRIALS_ENABLED = false;
export const FREE_BET_LIMIT = 0;              // 0 = no free trial; every user must hold an active subscription

// Stablecoins we can price 1:1 for the payment-amount guardrail (mainnet).
const STABLES: Record<string, { sym: string; dec: number }> = {
  "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": { sym: "USDC", dec: 6 },
  "0xdac17f958d2ee523a2206206994597c13d831ec7": { sym: "USDT", dec: 6 },
  "0x6b175474e89094c44da98b954eedeac495271d0f": { sym: "DAI", dec: 18 },
};

/**
 * Time-boxed pro grants (manual comps for off-platform payments): email ->
 * expiry epoch-ms. While Date.now() < expiry the email is treated as pro on any
 * browser it signs in from; after expiry it lapses back to free automatically
 * (see loadSession/signIn). Unlike ADMIN_EMAILS this is NOT free-forever.
 */
export const TIME_GRANTS: Record<string, number> = {
  // paid off-platform 2026-07-21 13:53Z; access REVOKED 2026-07-22 by operator.
  // Kept (expiry in the past) rather than deleted so loadSession actively
  // downgrades any already-"pro" session for this email back to free.
  "x7kobe@gmail.com": 1,
  // paid off-platform 2026-08-03; full access for one month → 2026-09-02.
  // Mirrored in the account DB (grants[]); this entry makes it effective
  // immediately on any browser without waiting on a server round-trip.
  "mateimo012@gmail.com": 1788339135450,
  // Comped by the operator 2026-08-07 for a trial period → 2026-09-06 23:59Z.
  // Lapses back to free on its own; shorten the expiry (or set it to 1) to end
  // it sooner — a past expiry actively downgrades an existing pro session.
  "blokhin.ia.9801@gmail.com": 1788739199000,
  // PAID customer, 2026-08-07 — one month of full access → 2026-09-07 23:59Z.
  // This entry makes it effective on any browser immediately; it should also be
  // recorded in the account DB (payments[] via /admin) so revenue is accounted
  // for rather than looking like a comp.
  "nsanity937@gmail.com": 1788825599000,
  // Comped by the operator 2026-08-13 — two days of full access → 2026-08-15
  // 23:59Z. NOTE: the request gave the local part only ("utkarsh.srivastava98");
  // gmail is assumed, matching every other grant here. A grant is keyed by the
  // exact address, so if the domain is different this silently does nothing —
  // change the key rather than adding a second entry.
  "utkarsh.srivastava98@gmail.com": 1786838399000,
};

/** Grant expiry for an email if one is currently ACTIVE, else null. */
export function activeGrantExpiry(email: string): number | null {
  const exp = TIME_GRANTS[normEmail(email)];
  return exp !== undefined && Date.now() < exp ? exp : null;
}

export type Tier = "public" | "free" | "pro";

export interface Session {
  email: string;
  tier: "free" | "pro";
  isAdmin: boolean;
  txHash?: string;
  paidUntil?: number;   // epoch-ms the current subscription lapses (undefined = never paid)
  since: number;
}

const LS_KEY = "tt_session_v1";

function normEmail(e: string): string {
  return e.trim().toLowerCase();
}

/** True if this session currently has PAID access: admin, an active comp grant,
 *  or a subscription payment whose 30-day window hasn't lapsed. */
export function subActive(s: Session | null): boolean {
  if (!s) return false;
  const e = normEmail(s.email);
  if (s.isAdmin || ADMIN_EMAILS.has(e)) return true;
  if (activeGrantExpiry(e)) return true;
  return typeof s.paidUntil === "number" && s.paidUntil > Date.now();
}

export function loadSession(): Session | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return null;
    const s: Session = JSON.parse(raw);
    // The subscription is the single source of truth for tier — an expired
    // paidUntil (or a lapsed comp) drops the user straight back to free.
    const before = s.tier;
    s.tier = subActive(s) ? "pro" : "free";
    if (s.isAdmin || ADMIN_EMAILS.has(normEmail(s.email))) s.isAdmin = true;
    // Write the corrected tier back, so the stored session can't disagree with
    // what the app renders (a granted user previously stayed "free" on disk
    // while showing PRO everywhere — confusing to debug, and a trap for any
    // code that reads localStorage directly instead of going through subActive).
    if (s.tier !== before) saveSession(s);
    return s;
  } catch {
    return null;
  }
}

function saveSession(s: Session | null): void {
  // Once the seat is gone, nothing may write a session back except a fresh
  // sign-in (which clears the flag first).
  if (s && seatLost) return;
  if (s) localStorage.setItem(LS_KEY, JSON.stringify(s));
  else localStorage.removeItem(LS_KEY);
}

export function signIn(email: string): Session {
  seatLost = false;   // signing in claims the seat for this device
  const e = normEmail(email);
  const isAdmin = ADMIN_EMAILS.has(e);
  const prev = loadSession();
  // Carry over an existing subscription only for the same email — paidUntil is
  // the source of truth, and subActive() enforces its expiry.
  const samePrev = prev && normEmail(prev.email) === e ? prev : null;
  const s: Session = {
    email: e,
    tier: "free",  // set correctly just below via subActive
    isAdmin,
    txHash: samePrev?.txHash,
    paidUntil: samePrev?.paidUntil,
    since: Date.now(),
  };
  s.tier = subActive(s) ? "pro" : "free";
  saveSession(s);
  // Remembered separately so a redirect-based payment return (PayPal) can still
  // identify the payer even if the session was cleared while they were away.
  try {
    localStorage.setItem("tt_last_email", e);
    // A fresh sign-in resolves any previous eviction notice.
    localStorage.removeItem("tt_evicted");
  } catch { /* non-critical */ }
  // Record the login in the account database (fire-and-forget: sign-in must
  // never block on it). Without this, logins existed only in this browser's
  // localStorage and there was no way to see who was actually using the app.
  recordLogin(e);
  return s;
}

/* ── one email, one device ─────────────────────────────────────────────────
 *
 * A subscription is per person, not per address book. The seat is bound to a
 * device id generated here and held server-side; signing in somewhere else
 * takes the seat and the previous device is signed out on its next check.
 *
 * Last-device-wins rather than refusing the new one: someone who changes phone
 * or clears their browser must not be locked out of what they paid for. The
 * deterrent against sharing is that the two devices evict each other, not a
 * wall the paying customer hits first.
 *
 * This is enforcement, not security — the id lives in localStorage and can be
 * copied. It stops casual password-sharing, which is what it is for.
 */
const DEVICE_KEY = "tt_device_v1";

/**
 * Set the moment this device is found to have lost the seat.
 *
 * syncEntitlement() runs concurrently on load and ends by writing the session
 * back; without this flag it resurrected the session milliseconds after the
 * device check signed it out, and the eviction silently did nothing. Any
 * explicit sign-in clears it.
 */
let seatLost = false;

export function deviceId(): string {
  if (typeof window === "undefined") return "";
  try {
    let id = localStorage.getItem(DEVICE_KEY);
    if (!id) {
      id = (crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`);
      localStorage.setItem(DEVICE_KEY, id);
    }
    return id;
  } catch {
    return "";
  }
}

/**
 * Tell the account DB this email just signed in. Never throws.
 *
 * The response carries any trial the server just granted, so a new account
 * becomes pro immediately rather than on the next page load — a signup that
 * promises free access and then shows a locked terminal is the worst possible
 * first impression.
 */
export function recordLogin(email: string, source?: string): void {
  // Also capture it as a LEAD. Sign-ups previously landed only in the accounts
  // store, so anyone who signed up — including every Google sign-in and every
  // trial — was missing from the leads list and from the Google Sheet mirror,
  // which reads /api/subscribe. Two stores, two purposes: accounts answers
  // "who has access", leads answers "who gave us their address".
  try {
    void import("./subscribe")
      .then(m => m.captureLead(normEmail(email), source || "signup"))
      .catch(() => {});
  } catch { /* never block sign-in */ }

  try {
    void fetch("/api/account", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: normEmail(email), source, deviceId: deviceId() }),
    })
      .then(r => r.json().catch(() => null))
      .then((data) => {
        // A protected account signing in from a device that is not its bound
        // one is refused outright. Without this the client would keep granting
        // admin locally from ADMIN_EMAILS — which is in the public bundle —
        // and the server check would be decoration.
        if (data?.deviceRejected) {
          seatLost = true;
          signOut();
          try { localStorage.setItem("tt_device_locked", "1"); } catch { /* cosmetic */ }
          window.dispatchEvent(new Event("tt-session-changed"));
          return;
        }
        if (!data?.paidUntil) return;
        const cur = loadSession();
        if (!cur || normEmail(cur.email) !== normEmail(email)) return;
        const up: Session = { ...cur, paidUntil: data.paidUntil };
        up.tier = subActive(up) ? "pro" : "free";
        saveSession(up);
        if (data.trialGranted) {
          try { localStorage.setItem("tt_trial_started", String(Date.now())); } catch { /* cosmetic */ }
        }
        window.dispatchEvent(new Event("tt-session-changed"));
      })
      .catch(() => {});
  } catch { /* never block sign-in */ }
}

/**
 * Does this device still hold the seat for this email?
 *
 * Returns true on any network or server failure — an outage must not sign
 * paying customers out. Only an explicit `deviceOk: false` evicts.
 */
export async function deviceStillValid(email: string): Promise<boolean> {
  try {
    const res = await fetch(
      `/api/account?email=${encodeURIComponent(normEmail(email))}&deviceId=${encodeURIComponent(deviceId())}`,
      { cache: "no-store" },
    );
    if (!res.ok) return true;
    const data = await res.json();
    // `locked` means a protected account on the wrong device: never treat that
    // as valid, even transiently.
    if (data.locked) return false;
    return data.deviceOk !== false;
  } catch {
    return true;
  }
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
  paidUntil?: number;   // subscription expiry (block time + 30d) on success
  amountUsd?: number;   // verified USD value of the payment
}

async function rpc<T = unknown>(url: string, method: string, params: unknown[]): Promise<T | null> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
  });
  if (!res.ok) return null;
  return (await res.json()).result ?? null;
}

async function ethUsd(): Promise<number | null> {
  try {
    const j = await fetch("https://api.coinbase.com/v2/prices/ETH-USD/spot").then((r) => r.json());
    const p = parseFloat(j.data.amount);
    return Number.isFinite(p) ? p : null;
  } catch {
    return null;
  }
}

/**
 * Verify a subscription payment. REAL guardrail — the tx must:
 *   1. exist and be confirmed,
 *   2. pay PAYMENT_ADDRESS (native ETH, or a USDC/USDT/DAI transfer),
 *   3. be worth at least $100 (MIN_PAYMENT_USD), and
 *   4. be no older than the 30-day window (so an old tx can't be re-used forever).
 * On success returns paidUntil = block time + 30 days.
 */
export async function verifyPaymentTx(txHash: string): Promise<VerifyResult> {
  const hash = txHash.trim();
  if (!/^0x[0-9a-fA-F]{64}$/.test(hash)) {
    return { ok: false, reason: "That is not a valid transaction hash (0x + 64 hex chars)." };
  }
  const target = PAYMENT_ADDRESS.toLowerCase();

  for (const url of RPCS) {
    try {
      const tx = await rpc<{ to: string; input: string; value: string; blockNumber: string }>(
        url, "eth_getTransactionByHash", [hash]);
      if (!tx) return { ok: false, reason: "Transaction not found on Ethereum mainnet. Wait for confirmation and try again." };
      if (!tx.blockNumber) return { ok: false, reason: "Transaction is still pending — try again once it confirms." };

      const to = (tx.to || "").toLowerCase();
      const input: string = tx.input || "0x";

      // Determine the USD value paid to the access address.
      let usd: number | null = null;
      let kind = "";
      const valueWei = BigInt(tx.value || "0x0");
      if (to === target && valueWei > BigInt(0)) {
        const price = await ethUsd();
        if (price == null) return { ok: false, reason: "Couldn't fetch the ETH price to verify the amount — pay in USDC/USDT for an instant check, or retry." };
        usd = (Number(valueWei) / 1e18) * price;
        kind = "ETH";
      } else if (input.startsWith("0xa9059cbb") && input.length >= 10 + 128) {
        const recipient = "0x" + input.slice(10 + 24, 10 + 64).toLowerCase();
        if (recipient !== target) {
          return { ok: false, reason: "That transaction does not pay the access address." };
        }
        const stable = STABLES[to];
        if (!stable) {
          return { ok: false, reason: "Unsupported token. Pay in ETH, USDC, USDT or DAI." };
        }
        const amount = BigInt("0x" + input.slice(10 + 64, 10 + 128));
        usd = Number(amount) / 10 ** stable.dec;
        kind = stable.sym;
      } else {
        return { ok: false, reason: "That transaction does not pay the access address." };
      }

      if (usd == null || usd + 0.5 < MIN_PAYMENT_USD) {
        return { ok: false, reason: `Payment is ~$${(usd || 0).toFixed(2)} ${kind} — the subscription is $${MIN_PAYMENT_USD}/month. Send at least $${MIN_PAYMENT_USD}.` };
      }

      // Block timestamp -> 30-day access window. Reject stale (re-used) txs.
      const block = await rpc<{ timestamp: string }>(url, "eth_getBlockByNumber", [tx.blockNumber, false]);
      const blockMs = block ? Number(BigInt(block.timestamp)) * 1000 : Date.now();
      const paidUntil = blockMs + SUBSCRIPTION_DAYS * 86400000;
      if (paidUntil <= Date.now()) {
        return { ok: false, reason: "That payment is more than 30 days old — the subscription has lapsed. Send a new monthly payment." };
      }

      return { ok: true, reason: `Payment verified — $${usd.toFixed(2)} ${kind}. Access until ${new Date(paidUntil).toLocaleDateString()}.`, paidUntil, amountUsd: usd };
    } catch {
      continue; // try next RPC
    }
  }
  return { ok: false, reason: "Could not reach Ethereum RPC to verify — check your connection and retry." };
}

/** Upgrade the current session to a paid subscription until `paidUntil`. */
export function grantPro(txHash: string, paidUntil: number, amountUsd?: number): Session | null {
  const s = loadSession();
  if (!s) return null;
  const up: Session = { ...s, tier: "pro", txHash, paidUntil };
  saveSession(up);
  // Link the payment to the email server-side so "who paid" is answerable
  // (the on-chain tx has no email). Fire-and-forget — never blocks the grant.
  import("./subscribe").then((m) => m.recordPayment(up.email, txHash, amountUsd != null ? String(Math.round(amountUsd)) : undefined)).catch(() => {});
  import("@/components/Analytics").then((m) => m.trackEvent("Payment")).catch(() => {});
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

/* ── Free-bet trial quota (3 bets before a subscription is required) ─────── */

const FREE_BETS_PREFIX = "tt_free_bets_";

function freeBetsKey(email: string): string {
  return FREE_BETS_PREFIX + normEmail(email || "anon");
}

/** How many of the FREE_BET_LIMIT trial bets this email has left. */
export function freeBetsRemaining(email: string): number {
  try {
    const used = parseInt(localStorage.getItem(freeBetsKey(email)) || "0", 10) || 0;
    return Math.max(0, FREE_BET_LIMIT - used);
  } catch {
    return 0;
  }
}

/** Spend one trial bet. Returns the number remaining afterwards. */
export function consumeFreeBet(email: string): number {
  try {
    const key = freeBetsKey(email);
    const used = parseInt(localStorage.getItem(key) || "0", 10) || 0;
    localStorage.setItem(key, String(used + 1));
    return Math.max(0, FREE_BET_LIMIT - (used + 1));
  } catch {
    return 0;
  }
}

/**
 * The single access gate for the terminal: a paid/active subscription (or admin
 * / comp) OR trial bets still remaining. No session at all -> no access.
 */
export function canAccessTerminal(s: Session | null): boolean {
  if (!s) return false;
  return subActive(s) || freeBetsRemaining(s.email) > 0;
}

/**
 * Reconcile the local session against the SERVER's authoritative entitlement.
 * The server (which verified the payment on-chain) is the source of truth, so a
 * hand-edited localStorage paidUntil is overwritten here. If the server can't be
 * reached we keep the current session rather than locking a real subscriber out
 * on a transient network blip. Admins/comps bypass (handled by subActive).
 */
export async function syncEntitlement(): Promise<Session | null> {
  const s = loadSession();
  if (!s) return null;
  if (s.isAdmin || ADMIN_EMAILS.has(normEmail(s.email)) || activeGrantExpiry(s.email)) return s;
  const { serverEntitlement } = await import("./entitlement");
  const ent = await serverEntitlement(s.email);
  if (!ent) return s;                       // server unreachable -> don't lock out
  const up: Session = { ...s, paidUntil: ent.paidUntil || undefined };
  up.tier = subActive(up) ? "pro" : "free";
  saveSession(up);
  return up;
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
  useEffect(() => {
    const s0 = loadSession();
    setSession(s0);
    // Server is authoritative — reconcile on load so a spoofed localStorage
    // entitlement is corrected before the terminal renders as unlocked.
    syncEntitlement().then((s) => s && setSession(s)).catch(() => {});

    // One email, one device. Checked on load and every 2 minutes, so a seat
    // taken elsewhere ends this session rather than running both in parallel.
    // Only an explicit rejection signs anyone out — see deviceStillValid.
    let stop = false;
    const check = async () => {
      const cur = loadSession();
      if (!cur?.email || stop) return;
      if (await deviceStillValid(cur.email)) return;
      seatLost = true;
      signOut();
      if (!stop) {
        setSession(null);
        try {
          localStorage.setItem("tt_evicted", "1");
        } catch { /* the banner is a nicety, the sign-out is the point */ }
      }
    };
    check();
    const iv = setInterval(check, 120_000);

    // recordLogin fires this once the server has answered — that is when a
    // freshly granted trial becomes visible.
    const onChanged = () => setSession(loadSession());
    window.addEventListener("tt-session-changed", onChanged);

    return () => {
      stop = true;
      clearInterval(iv);
      window.removeEventListener("tt-session-changed", onChanged);
    };
  }, []);
  const tier: Tier = session ? session.tier : "public";
  return <TierContext.Provider value={{ session, tier, refresh }}>{children}</TierContext.Provider>;
}

export function useTier(): TierContextValue {
  return useContext(TierContext);
}
