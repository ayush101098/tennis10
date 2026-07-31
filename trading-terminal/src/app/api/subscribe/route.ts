import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

/**
 * Local-dev email capture + payment ledger. The static export strips this route
 * (see netlify.toml `rm -rf src/app/api`); in production the Netlify Function at
 * /.netlify/functions/subscribe serves the same contract, backed by Netlify
 * Blobs. Here we persist to a gitignored JSON file so captures are visible while
 * running `next dev`.
 *
 * POST { email, source? }                 -> capture a lead
 * POST { email, txHash, amount?, from? }  -> record a payment linked to the email
 * GET  (header x-admin-token: LEADS_ADMIN_TOKEN) -> { leads, payments, counts }
 */

export const dynamic = "force-dynamic";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const FILE = path.join(process.cwd(), "leads.local.json");
const ANALYTICS_FILE = path.join(process.cwd(), "analytics.local.json");

type Ev = { ts: number; path: string; ref: string; vid: string };
function aggregateTraffic(events: Ev[]) {
  const byPath: Record<string, number> = {}, byRef: Record<string, number> = {}, byDay: Record<string, number> = {};
  const vids = new Set<string>();
  for (const e of events) {
    vids.add(e.vid || "?");
    byPath[e.path] = (byPath[e.path] || 0) + 1;
    let ref = "direct";
    try { ref = e.ref ? new URL(e.ref, "http://x").hostname || "direct" : "direct"; } catch { /* keep */ }
    byRef[ref] = (byRef[ref] || 0) + 1;
    const day = new Date(e.ts).toISOString().slice(0, 10);
    byDay[day] = (byDay[day] || 0) + 1;
  }
  const top = (o: Record<string, number>) => Object.entries(o).map(([k, v]) => ({ k, v })).sort((a, b) => b.v - a.v).slice(0, 10);
  const days = [];
  for (let i = 13; i >= 0; i--) {
    const d = new Date(Date.now() - i * 86400000).toISOString().slice(0, 10);
    days.push({ day: d, count: byDay[d] || 0 });
  }
  return { views: events.length, uniques: vids.size, byPath: top(byPath), byRef: top(byRef), byDay: days, recent: events.slice(-25).reverse() };
}
async function loadAnalytics(): Promise<Ev[]> {
  try { return JSON.parse(await fs.readFile(ANALYTICS_FILE, "utf8")); } catch { return []; }
}

type Lead = { email: string; ts: number; lastSeen: number; source: string; paid: boolean };
type Payment = { email: string; txHash: string; amount: string | null; from: string | null; ts: number };
type DB = { leads: Lead[]; payments: Payment[] };

async function load(): Promise<DB> {
  try {
    return JSON.parse(await fs.readFile(FILE, "utf8"));
  } catch {
    return { leads: [], payments: [] };
  }
}
async function save(db: DB) {
  await fs.writeFile(FILE, JSON.stringify(db, null, 2));
}

export async function GET(req: NextRequest) {
  const token = process.env.LEADS_ADMIN_TOKEN;
  if (!token || req.headers.get("x-admin-token") !== token) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const db = await load();
  const events = await loadAnalytics();
  return NextResponse.json({
    counts: { leads: db.leads.length, payments: db.payments.length, views: events.length },
    ...db,
    traffic: aggregateTraffic(events),
  }, { headers: { "Cache-Control": "no-store" } });
}

export async function POST(req: NextRequest) {
  let body: Record<string, unknown> = {};
  try { body = await req.json(); } catch { /* ignore */ }
  const email = String(body.email || "").trim().toLowerCase();
  if (!EMAIL_RE.test(email)) {
    return NextResponse.json({ error: "Enter a valid email address." }, { status: 400 });
  }
  const now = Date.now();
  const db = await load();

  if (body.txHash) {
    const txHash = String(body.txHash).trim();
    if (!db.payments.some((p) => p.txHash === txHash)) {
      db.payments.push({
        email, txHash,
        amount: body.amount != null ? String(body.amount) : null,
        from: body.from ? String(body.from) : null,
        ts: now,
      });
    }
  }

  const existing = db.leads.find((l) => l.email === email);
  if (existing) {
    existing.lastSeen = now;
    if (body.txHash) existing.paid = true;
  } else {
    db.leads.push({
      email, ts: now, lastSeen: now,
      source: body.source ? String(body.source).slice(0, 40) : "cta",
      paid: !!body.txHash,
    });
  }
  await save(db);
  return NextResponse.json({ ok: true }, { headers: { "Cache-Control": "no-store" } });
}
