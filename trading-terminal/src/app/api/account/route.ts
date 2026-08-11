import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import {
  EMAIL_RE, normEmail, applyLogin, applyPayment, applyGrant, summarize,
  recomputePaidUntil, blankAccount, type AccountsDB,
} from "@/lib/accountsStore";

/**
 * The account database endpoint — logins, payments and grants in one place.
 *
 * POST { email, source? }                    -> record a login (called on sign-in)
 * POST { email, action:"grant", days, reason, adminToken } -> issue a manual grant
 * GET  ?email=            -> { active, paidUntil } for one account (public)
 * GET  (x-admin-token)    -> full roster + counts (admin only)
 *
 * Local dev persists to accounts.local.json; production uses the Netlify
 * function twin backed by Blobs. On first read this backfills from the older
 * entitlements/leads files so no existing customer is lost.
 */

export const dynamic = "force-dynamic";

const FILE = path.join(process.cwd(), "accounts.local.json");
const ENTITLEMENTS = path.join(process.cwd(), "entitlements.local.json");
const LEADS = path.join(process.cwd(), "leads.local.json");

async function readJson<T>(p: string, fallback: T): Promise<T> {
  try { return JSON.parse(await fs.readFile(p, "utf8")); } catch { return fallback; }
}

/** Merge the legacy stores in, so the unified DB starts complete. */
async function backfill(db: AccountsDB): Promise<AccountsDB> {
  const ent = await readJson<{ byEmail?: Record<string, { paidUntil: number; txHash: string; from: string; amountUsd: number; verifiedAt: number }> }>(ENTITLEMENTS, {});
  for (const [email, e] of Object.entries(ent.byEmail || {})) {
    const key = normEmail(email);
    if (!db[key]) db[key] = blankAccount(key, e.verifiedAt || Date.now());
    if (e.txHash && !db[key].payments.some(p => p.txHash === e.txHash)) {
      // Payment time must be the ON-CHAIN block time, not verifiedAt: expiry is
      // derived as ts + 30d, and the entitlement's paidUntil is already
      // blockMs + 30d, so recover blockMs from it. Using verifiedAt here would
      // hand a late verifier extra access (pay, wait 29 days, verify → ~59).
      const blockMs = e.paidUntil ? e.paidUntil - 30 * 86400000 : (e.verifiedAt || Date.now());
      db[key].payments.push({
        txHash: e.txHash, amountUsd: e.amountUsd || 0, from: e.from, ts: blockMs,
      });
    }
    // Preserve an entitlement that predates this store. Idempotent: the grant
    // is only added once, so repeated backfills can't extend anyone's access.
    const MIGRATED = "migrated from entitlements store";
    if ((e.paidUntil || 0) > recomputePaidUntil(db[key]) &&
        !db[key].grants.some(g => g.reason === MIGRATED && g.until === e.paidUntil)) {
      db[key].grants.push({
        until: e.paidUntil, reason: MIGRATED,
        by: "system", ts: e.verifiedAt || Date.now(),
      });
    }
    db[key].paidUntil = recomputePaidUntil(db[key]);
  }

  const leads = await readJson<{ leads?: { email: string; ts: number; lastSeen?: number; source?: string }[] }>(LEADS, {});
  for (const l of leads.leads || []) {
    const key = normEmail(l.email);
    if (!EMAIL_RE.test(key)) continue;
    if (!db[key]) db[key] = blankAccount(key, l.ts || Date.now());
    db[key].firstSeen = Math.min(db[key].firstSeen, l.ts || Date.now());
    if (l.source && !db[key].source) db[key].source = l.source;
    db[key].paidUntil = recomputePaidUntil(db[key]);
  }
  return db;
}

/** Backfill runs on every load, not just the first: legacy stores keep receiving
 *  writes from the older endpoints, and gating on an empty file silently
 *  stranded every pre-existing customer. It is idempotent by construction. */
async function load(): Promise<AccountsDB> {
  return backfill(await readJson<AccountsDB>(FILE, {}));
}
async function save(db: AccountsDB) {
  await fs.writeFile(FILE, JSON.stringify(db, null, 2));
}

function adminOk(req: NextRequest): boolean {
  const token = process.env.LEADS_ADMIN_TOKEN;
  return !!token && req.headers.get("x-admin-token") === token;
}

export async function GET(req: NextRequest) {
  const db = await load();
  const email = normEmail(req.nextUrl.searchParams.get("email") || "");

  if (email) {
    if (!EMAIL_RE.test(email)) {
      return NextResponse.json({ error: "email required" }, { status: 400 });
    }
    const a = db[email];
    const paidUntil = a?.paidUntil || 0;
    return NextResponse.json(
      { active: paidUntil > Date.now(), paidUntil },
      { headers: { "Cache-Control": "no-store" } },
    );
  }

  if (!adminOk(req)) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  await save(db);   // persist any backfill
  return NextResponse.json(summarize(db), { headers: { "Cache-Control": "no-store" } });
}

export async function POST(req: NextRequest) {
  let body: Record<string, unknown> = {};
  try { body = await req.json(); } catch { /* ignore */ }
  const email = normEmail(String(body.email || ""));
  if (!EMAIL_RE.test(email)) {
    return NextResponse.json({ ok: false, reason: "A valid email is required." }, { status: 400 });
  }
  const db = await load();

  // ── manual grant (admin only) ──
  if (body.action === "grant") {
    const token = process.env.LEADS_ADMIN_TOKEN;
    if (!token || String(body.adminToken || "") !== token) {
      return NextResponse.json({ ok: false, reason: "unauthorized" }, { status: 401 });
    }
    const days = Math.max(1, Math.min(3650, Number(body.days) || 30));
    const acct = applyGrant(db, email, {
      until: Date.now() + days * 86400000,
      reason: String(body.reason || "manual grant"),
      by: String(body.by || "operator"),
      ts: Date.now(),
    });
    await save(db);
    return NextResponse.json({ ok: true, email, paidUntil: acct.paidUntil, days });
  }

  // ── payment (written by the verifier after on-chain confirmation) ──
  if (body.txHash) {
    const acct = applyPayment(db, email, {
      txHash: String(body.txHash),
      amountUsd: Number(body.amountUsd) || 0,
      from: body.from ? String(body.from) : undefined,
      ts: Number(body.ts) || Date.now(),
    });
    await save(db);
    return NextResponse.json({ ok: true, email, paidUntil: acct.paidUntil });
  }

  // ── login ──
  const acct = applyLogin(db, email, body.source ? String(body.source) : undefined);
  await save(db);
  return NextResponse.json(
    { ok: true, email, active: acct.paidUntil > Date.now(), paidUntil: acct.paidUntil },
    { headers: { "Cache-Control": "no-store" } },
  );
}
