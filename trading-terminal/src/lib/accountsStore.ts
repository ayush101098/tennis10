/**
 * THE account database — one record per email, the single place that answers
 * "who signed in, who paid, and until when".
 *
 * Before this, that answer was split across three stores with different shapes:
 *   entitlements (byEmail/claims)  — paid access, written by the verifier
 *   leads (leads/payments)         — email capture + a duplicate payment log
 *   nothing at all                 — logins were never recorded anywhere;
 *                                    sign-in was purely client-side localStorage
 * so there was no way to see who was actually using the product.
 *
 * Storage mirrors the rest of the app: a JSON file in local dev, Netlify Blobs
 * in production (same shape either way, so the admin view is identical).
 */

export interface AccountGrant {
  until: number;        // epoch-ms the grant expires
  reason: string;
  by: string;           // who issued it (operator id / "system")
  ts: number;
}

export interface AccountPayment {
  txHash: string;
  amountUsd: number;
  from?: string;
  ts: number;
}

export interface Account {
  email: string;
  firstSeen: number;
  lastLogin: number;
  loginCount: number;
  source?: string;          // where they first arrived from
  paidUntil: number;        // 0 = never paid; max of payments + grants
  payments: AccountPayment[];
  grants: AccountGrant[];
}

export type AccountsDB = Record<string, Account>;

export const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
export const normEmail = (e: string) => String(e || "").trim().toLowerCase();

/** A fresh, empty record. */
export function blankAccount(email: string, now = Date.now()): Account {
  return {
    email,
    firstSeen: now,
    lastLogin: now,
    loginCount: 0,
    paidUntil: 0,
    payments: [],
    grants: [],
  };
}

/** paidUntil recomputed from the underlying facts, so it can never drift. */
export function recomputePaidUntil(a: Account): number {
  const fromPayments = a.payments.reduce((m, p) => Math.max(m, p.ts + 30 * 86400000), 0);
  const fromGrants = a.grants.reduce((m, g) => Math.max(m, g.until), 0);
  return Math.max(fromPayments, fromGrants);
}

export function isActive(a: Account, now = Date.now()): boolean {
  return a.paidUntil > now;
}

/** Record a login, creating the account on first sight. */
export function applyLogin(db: AccountsDB, email: string, source?: string, now = Date.now()): Account {
  const key = normEmail(email);
  const acct = db[key] || blankAccount(key, now);
  acct.lastLogin = now;
  acct.loginCount += 1;
  if (source && !acct.source) acct.source = source;
  acct.paidUntil = recomputePaidUntil(acct);
  db[key] = acct;
  return acct;
}

/** Record a verified payment (idempotent on txHash). */
export function applyPayment(db: AccountsDB, email: string, p: AccountPayment): Account {
  const key = normEmail(email);
  const acct = db[key] || blankAccount(key, p.ts);
  if (!acct.payments.some(x => x.txHash.toLowerCase() === p.txHash.toLowerCase())) {
    acct.payments.push(p);
  }
  acct.paidUntil = recomputePaidUntil(acct);
  db[key] = acct;
  return acct;
}

/** Issue a manual grant (comp / off-platform payment). */
export function applyGrant(db: AccountsDB, email: string, g: AccountGrant): Account {
  const key = normEmail(email);
  const acct = db[key] || blankAccount(key, g.ts);
  acct.grants.push(g);
  acct.paidUntil = recomputePaidUntil(acct);
  db[key] = acct;
  return acct;
}

/** Admin-facing summary rows, most recently active first. */
export function summarize(db: AccountsDB, now = Date.now()) {
  const rows = Object.values(db).map(a => ({
    email: a.email,
    firstSeen: a.firstSeen,
    lastLogin: a.lastLogin,
    loginCount: a.loginCount,
    source: a.source || "",
    active: a.paidUntil > now,
    paidUntil: a.paidUntil,
    daysLeft: a.paidUntil > now ? Math.ceil((a.paidUntil - now) / 86400000) : 0,
    totalPaidUsd: a.payments.reduce((s, p) => s + (p.amountUsd || 0), 0),
    payments: a.payments.length,
    grants: a.grants.length,
  }));
  rows.sort((x, y) => y.lastLogin - x.lastLogin);
  const counts = {
    accounts: rows.length,
    active: rows.filter(r => r.active).length,
    paying: rows.filter(r => r.payments > 0).length,
    comped: rows.filter(r => r.grants > 0 && r.payments === 0).length,
    revenueUsd: rows.reduce((s, r) => s + r.totalPaidUsd, 0),
  };
  return { rows, counts };
}
