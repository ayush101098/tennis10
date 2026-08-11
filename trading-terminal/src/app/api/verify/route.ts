import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

/**
 * Local-dev mirror of netlify/functions/verify.js (the authoritative payment
 * verifier). Stripped from the static export; production uses the Netlify
 * function via the netlify.toml redirect.
 *
 * NOTE: server-side fetch may be sandboxed in local dev, so real on-chain
 * verification can be unavailable here. Set VERIFY_ALLOW_MOCK=true in .env.local
 * to accept a mock tx hash (0xdead…) for testing the entitlement/reuse flow.
 * This flag must NEVER be set in production — the Netlify function has no mock.
 */

export const dynamic = "force-dynamic";

const PAYMENT_ADDRESS = "0x905aCd442c7B3EF9BfEB0A3189f3686c1Cd0c697";
const MIN_PAYMENT_USD = 100;
const SUBSCRIPTION_DAYS = 30;
const RPCS = ["https://ethereum-rpc.publicnode.com", "https://1rpc.io/eth", "https://eth.drpc.org"];
const STABLES: Record<string, { sym: string; dec: number }> = {
  "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": { sym: "USDC", dec: 6 },
  "0xdac17f958d2ee523a2206206994597c13d831ec7": { sym: "USDT", dec: 6 },
  "0x6b175474e89094c44da98b954eedeac495271d0f": { sym: "DAI", dec: 18 },
};
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const FILE = path.join(process.cwd(), "entitlements.local.json");

type Ent = { paidUntil: number; txHash: string; from: string; amountUsd: number; verifiedAt: number };
type DB = { byEmail: Record<string, Ent>; claims: Record<string, string> };

async function load(): Promise<DB> {
  try { return JSON.parse(await fs.readFile(FILE, "utf8")); } catch { return { byEmail: {}, claims: {} }; }
}
async function save(db: DB) { await fs.writeFile(FILE, JSON.stringify(db, null, 2)); }

async function rpc(url: string, method: string, params: unknown[]) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }) });
  if (!res.ok) throw new Error(`rpc ${res.status}`);
  return (await res.json()).result;
}
async function ethUsd(): Promise<number | null> {
  try { const j = await (await fetch("https://api.coinbase.com/v2/prices/ETH-USD/spot")).json();
    const p = parseFloat(j.data.amount); return Number.isFinite(p) ? p : null; } catch { return null; }
}

type Verdict = { ok: boolean; reason: string; paidUntil?: number; amountUsd?: number; from?: string };

async function verifyOnChain(txHash: string): Promise<Verdict> {
  const target = PAYMENT_ADDRESS.toLowerCase();
  let lastErr: unknown;
  for (const url of RPCS) {
    try {
      const tx = await rpc(url, "eth_getTransactionByHash", [txHash]);
      if (!tx) return { ok: false, reason: "Transaction not found on Ethereum mainnet." };
      if (!tx.blockNumber) return { ok: false, reason: "Transaction is still pending — retry once confirmed." };
      const to = (tx.to || "").toLowerCase();
      const input: string = tx.input || "0x";
      let usd: number | null = null, kind = "";
      const from = (tx.from || "").toLowerCase();
      const valueWei = BigInt(tx.value || "0x0");
      if (to === target && valueWei > BigInt(0)) {
        const price = await ethUsd();
        if (price == null) return { ok: false, reason: "Couldn't price ETH — pay in USDC/USDT or retry." };
        usd = (Number(valueWei) / 1e18) * price; kind = "ETH";
      } else if (input.startsWith("0xa9059cbb") && input.length >= 10 + 128) {
        const recipient = "0x" + input.slice(10 + 24, 10 + 64).toLowerCase();
        if (recipient !== target) return { ok: false, reason: "That transaction does not pay the access address." };
        const stable = STABLES[to];
        if (!stable) return { ok: false, reason: "Unsupported token — pay in ETH, USDC, USDT or DAI." };
        usd = Number(BigInt("0x" + input.slice(10 + 64, 10 + 128))) / 10 ** stable.dec; kind = stable.sym;
      } else {
        return { ok: false, reason: "That transaction does not pay the access address." };
      }
      if (usd + 0.5 < MIN_PAYMENT_USD) return { ok: false, reason: `Payment is ~$${usd.toFixed(2)} ${kind} — subscription is $${MIN_PAYMENT_USD}/month.` };
      const block = await rpc(url, "eth_getBlockByNumber", [tx.blockNumber, false]);
      const blockMs = block ? Number(BigInt(block.timestamp)) * 1000 : Date.now();
      const paidUntil = blockMs + SUBSCRIPTION_DAYS * 86400000;
      if (paidUntil <= Date.now()) return { ok: false, reason: "That payment is more than 30 days old." };
      return { ok: true, reason: `Verified $${usd.toFixed(2)} ${kind}.`, paidUntil, amountUsd: usd, from };
    } catch (e) { lastErr = e; }
  }
  throw lastErr || new Error("all RPCs failed");
}

export async function GET(req: NextRequest) {
  const email = String(req.nextUrl.searchParams.get("email") || "").trim().toLowerCase();
  if (!EMAIL_RE.test(email)) return NextResponse.json({ error: "email required" }, { status: 400 });
  const db = await load();
  const paidUntil = db.byEmail[email]?.paidUntil || 0;
  return NextResponse.json({ active: paidUntil > Date.now(), paidUntil }, { headers: { "Cache-Control": "no-store" } });
}

export async function POST(req: NextRequest) {
  let body: Record<string, unknown> = {};
  try { body = await req.json(); } catch { /* ignore */ }
  const email = String(body.email || "").trim().toLowerCase();
  const txHash = String(body.txHash || "").trim();
  if (!EMAIL_RE.test(email)) return NextResponse.json({ ok: false, reason: "A valid email is required." }, { status: 400 });
  if (!/^0x[0-9a-fA-F]{64}$/.test(txHash)) return NextResponse.json({ ok: false, reason: "That is not a valid transaction hash." }, { status: 400 });

  let v: Verdict;
  // DEV-ONLY mock (never enabled in production). A 0xdead… hash grants a $100/30d entitlement.
  if (process.env.VERIFY_ALLOW_MOCK === "true" && /^0xdead/.test(txHash)) {
    v = { ok: true, reason: "Verified (dev mock).", paidUntil: Date.now() + SUBSCRIPTION_DAYS * 86400000, amountUsd: 100, from: "0xmock" };
  } else {
    try { v = await verifyOnChain(txHash); }
    catch { return NextResponse.json({ ok: false, reason: "Verification service can't reach the chain right now — try again." }, { status: 503 }); }
  }
  if (!v.ok) return NextResponse.json(v);

  const db = await load();
  const owner = db.claims[txHash];
  if (owner && owner !== email) {
    return NextResponse.json({ ok: false, reason: "This transaction is already linked to another account." }, { status: 409 });
  }
  db.claims[txHash] = email;
  const prev = db.byEmail[email]?.paidUntil || 0;
  db.byEmail[email] = { paidUntil: Math.max(prev, v.paidUntil!), txHash, from: v.from!, amountUsd: Math.round(v.amountUsd!), verifiedAt: Date.now() };
  await save(db);

  // Mirror into the unified account database so the roster shows this payer.
  // `ts` MUST be the on-chain block time, not now: the account DB derives
  // payment expiry as ts + 30d, so stamping it with the verification time would
  // let someone pay, sit on the receipt for 29 days, then verify and collect
  // ~59 days of access. v.paidUntil is blockMs + 30d, so subtract that back out.
  const blockMs = v.paidUntil! - SUBSCRIPTION_DAYS * 86400000;
  // Best-effort: a verified payment must never fail because the mirror did.
  try {
    await fetch(new URL("/api/account", req.nextUrl.origin), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email, txHash, amountUsd: Math.round(v.amountUsd!), from: v.from, ts: blockMs,
      }),
    });
  } catch { /* entitlement above is already persisted */ }
  return NextResponse.json({ ok: true, reason: v.reason, paidUntil: v.paidUntil, amountUsd: v.amountUsd }, { headers: { "Cache-Control": "no-store" } });
}
