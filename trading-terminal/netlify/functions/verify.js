/**
 * Netlify serverless function — AUTHORITATIVE payment verification + entitlement.
 *
 * This is the security boundary. The browser can lie about localStorage, so the
 * server (not the client) decides who has paid:
 *
 *   POST /api/verify  { email, txHash }
 *     - verifies the tx ON-CHAIN, server-side: it must pay PAYMENT_ADDRESS, be
 *       worth >= $100, and be <= 30 days old (block timestamp).
 *     - binds the tx to the FIRST email that claims it — a public tx hash can't
 *       be re-used to unlock a second account.
 *     - persists entitlement in Netlify Blobs and returns the authoritative
 *       paidUntil. The client stores nothing it can forge into access.
 *
 *   GET /api/verify?email=<e>
 *     - returns the server's current { active, paidUntil } for that email. The
 *       client calls this on load and trusts THIS over localStorage.
 *
 * Fail-closed: if the chain can't be reached, verification is refused (never
 * granted). Ceiling: a static client-rendered app can't stop a user who edits
 * the shipped JS — closing that needs the premium data served from here too.
 */

const PAYMENT_ADDRESS = "0x905aCd442c7B3EF9BfEB0A3189f3686c1Cd0c697";
const MIN_PAYMENT_USD = 100;
const SUBSCRIPTION_DAYS = 30;
const RPCS = [
  "https://ethereum-rpc.publicnode.com",
  "https://1rpc.io/eth",
  "https://eth.drpc.org",
];
const STABLES = {
  "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": { sym: "USDC", dec: 6 },
  "0xdac17f958d2ee523a2206206994597c13d831ec7": { sym: "USDT", dec: 6 },
  "0x6b175474e89094c44da98b954eedeac495271d0f": { sym: "DAI", dec: 18 },
};

const STORE = "entitlements";
const BYEMAIL = "byEmail";   // { email: { paidUntil, txHash, from, amountUsd, verifiedAt } }
const CLAIMS = "claims";     // { txHash: email }
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function store() {
  const { getStore } = require("@netlify/blobs");
  return getStore(STORE);
}
const reply = (statusCode, obj) => ({
  statusCode,
  headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  body: JSON.stringify(obj),
});

async function rpc(url, method, params) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
  });
  if (!res.ok) throw new Error(`rpc ${res.status}`);
  return (await res.json()).result;
}

async function ethUsd() {
  try {
    const j = await (await fetch("https://api.coinbase.com/v2/prices/ETH-USD/spot")).json();
    const p = parseFloat(j.data.amount);
    return Number.isFinite(p) ? p : null;
  } catch { return null; }
}

/** On-chain verification. Returns { ok, reason, paidUntil?, amountUsd?, from? }.
 *  Throws only on network failure (caller fails closed). */
async function verifyOnChain(txHash) {
  const target = PAYMENT_ADDRESS.toLowerCase();
  let lastErr;
  for (const url of RPCS) {
    try {
      const tx = await rpc(url, "eth_getTransactionByHash", [txHash]);
      if (!tx) return { ok: false, reason: "Transaction not found on Ethereum mainnet." };
      if (!tx.blockNumber) return { ok: false, reason: "Transaction is still pending — retry once confirmed." };

      const to = (tx.to || "").toLowerCase();
      const input = tx.input || "0x";
      let usd = null, kind = "", from = (tx.from || "").toLowerCase();

      const valueWei = BigInt(tx.value || "0x0");
      if (to === target && valueWei > BigInt(0)) {
        const price = await ethUsd();
        if (price == null) return { ok: false, reason: "Couldn't price ETH to verify the amount — pay in USDC/USDT or retry." };
        usd = (Number(valueWei) / 1e18) * price; kind = "ETH";
      } else if (input.startsWith("0xa9059cbb") && input.length >= 10 + 128) {
        const recipient = "0x" + input.slice(10 + 24, 10 + 64).toLowerCase();
        if (recipient !== target) return { ok: false, reason: "That transaction does not pay the access address." };
        const stable = STABLES[to];
        if (!stable) return { ok: false, reason: "Unsupported token — pay in ETH, USDC, USDT or DAI." };
        usd = Number(BigInt("0x" + input.slice(10 + 64, 10 + 128))) / 10 ** stable.dec;
        kind = stable.sym;
      } else {
        return { ok: false, reason: "That transaction does not pay the access address." };
      }

      if (usd + 0.5 < MIN_PAYMENT_USD) {
        return { ok: false, reason: `Payment is ~$${usd.toFixed(2)} ${kind} — the subscription is $${MIN_PAYMENT_USD}/month.` };
      }
      const block = await rpc(url, "eth_getBlockByNumber", [tx.blockNumber, false]);
      const blockMs = block ? Number(BigInt(block.timestamp)) * 1000 : Date.now();
      const paidUntil = blockMs + SUBSCRIPTION_DAYS * 86400000;
      if (paidUntil <= Date.now()) {
        return { ok: false, reason: "That payment is more than 30 days old — send a new monthly payment." };
      }
      return { ok: true, reason: `Verified $${usd.toFixed(2)} ${kind}.`, paidUntil, amountUsd: usd, from };
    } catch (e) {
      lastErr = e; // try next RPC
    }
  }
  throw lastErr || new Error("all RPCs failed");
}

exports.handler = async (event) => {
  // ── GET: authoritative entitlement lookup ──
  if (event.httpMethod === "GET") {
    const email = String((event.queryStringParameters || {}).email || "").trim().toLowerCase();
    if (!EMAIL_RE.test(email)) return reply(400, { error: "email required" });
    let ent = null;
    try { ent = (await store().get(BYEMAIL, { type: "json" }))?.[email] || null; } catch { /* none */ }
    const paidUntil = ent?.paidUntil || 0;
    return reply(200, { active: paidUntil > Date.now(), paidUntil });
  }

  if (event.httpMethod !== "POST") return reply(405, { error: "method not allowed" });

  let body = {};
  try { body = JSON.parse(event.body || "{}"); } catch { /* ignore */ }
  const email = String(body.email || "").trim().toLowerCase();
  const txHash = String(body.txHash || "").trim();
  if (!EMAIL_RE.test(email)) return reply(400, { ok: false, reason: "A valid email is required." });
  if (!/^0x[0-9a-fA-F]{64}$/.test(txHash)) return reply(400, { ok: false, reason: "That is not a valid transaction hash." });

  // Verify on-chain (fail closed on network error).
  let v;
  try {
    v = await verifyOnChain(txHash);
  } catch {
    return reply(503, { ok: false, reason: "Verification service can't reach the chain right now — try again shortly." });
  }
  if (!v.ok) return reply(200, v);

  // Persist entitlement + prevent tx re-use across accounts.
  //
  // The chain has ALREADY confirmed this payment by the time we get here, so a
  // storage failure is our bookkeeping problem, not the customer's. It used to
  // return 500 "Could not record entitlement", which locked a legitimate payer
  // out of the product they had just paid for. Now it grants access and reports
  // the real error so the operator can fix storage.
  //
  // Trade-off while storage is down: the tx-reuse check can't run, so the same
  // hash could in principle be claimed by two emails. Losing that guard is far
  // cheaper than refusing every paying customer, and reuse is auditable
  // after the fact from the chain + the account DB.
  let storeWarning = null;
  try {
    const s = store();
    const claims = (await s.get(CLAIMS, { type: "json" })) || {};
    const byEmail = (await s.get(BYEMAIL, { type: "json" })) || {};
    const owner = claims[txHash];
    if (owner && owner !== email) {
      return reply(409, { ok: false, reason: "This transaction is already linked to another account." });
    }
    claims[txHash] = email;
    const prev = byEmail[email]?.paidUntil || 0;
    byEmail[email] = {
      paidUntil: Math.max(prev, v.paidUntil),   // a new payment extends access
      txHash, from: v.from, amountUsd: Math.round(v.amountUsd), verifiedAt: Date.now(),
    };
    await s.setJSON(CLAIMS, claims);
    await s.setJSON(BYEMAIL, byEmail);
  } catch (e) {
    storeWarning = String((e && e.message) || e).slice(0, 200);
    console.error("[verify] entitlement store write FAILED:", storeWarning);
  }

  // Mirror into the leads/payments ledger so the admin dashboard sees it.
  try {
    const { getStore } = require("@netlify/blobs");
    const leads = getStore("leads");
    const payments = (await leads.get("payments", { type: "json" })) || [];
    if (!payments.some((p) => p.txHash === txHash)) {
      payments.push({ email, txHash, amount: String(Math.round(v.amountUsd)), from: v.from, ts: Date.now(), verified: true });
      await leads.setJSON("payments", payments);
    }
  } catch { /* non-critical */ }

  // Mirror into the unified account database (the roster the admin reads).
  // ts MUST be the on-chain block time, not now: the account DB derives payment
  // expiry as ts + 30d, so stamping it with the verification time would let
  // someone pay, sit on the receipt for 29 days, then verify and collect ~59
  // days of access. v.paidUntil is blockMs + 30d, so subtract that back out.
  try {
    const { getStore } = require("@netlify/blobs");
    const accounts = getStore("accounts");
    const db = (await accounts.get("byEmail", { type: "json" })) || {};
    const blockMs = v.paidUntil - SUBSCRIPTION_DAYS * 86400000;
    const now = Date.now();
    if (!db[email]) {
      db[email] = { email, firstSeen: now, lastLogin: now, loginCount: 0, paidUntil: 0, payments: [], grants: [] };
    }
    if (!db[email].payments.some((p) => p.txHash === txHash)) {
      db[email].payments.push({
        txHash, amountUsd: Math.round(v.amountUsd), from: v.from, ts: blockMs,
      });
    }
    const fromPayments = db[email].payments.reduce((m, p) => Math.max(m, p.ts + 30 * 86400000), 0);
    const fromGrants = db[email].grants.reduce((m, g) => Math.max(m, g.until), 0);
    db[email].paidUntil = Math.max(fromPayments, fromGrants);
    await accounts.setJSON("byEmail", db);
  } catch { /* non-critical — the entitlement above is already persisted */ }

  return reply(200, {
    ok: true, reason: v.reason, paidUntil: v.paidUntil, amountUsd: v.amountUsd,
    // present only when server-side persistence failed; access is still granted
    ...(storeWarning ? { storeWarning } : {}),
  });
};
