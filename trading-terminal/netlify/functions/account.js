/**
 * THE account database (production twin of src/app/api/account/route.ts).
 *
 * One record per email — logins, payments and grants — so "who signed in, who
 * paid, and until when" has a single answer. Backed by Netlify Blobs, with a
 * one-time backfill from the older `entitlements` and `leads` stores so no
 * existing customer is lost.
 *
 *   POST { email, source? }                                 -> record a login
 *   POST { email, txHash, amountUsd?, from? }                -> record a payment
 *   POST { email, action:"grant", days, reason, adminToken } -> manual grant
 *   GET  ?email=            -> { active, paidUntil }  (public, used by the client)
 *   GET  (x-admin-token)    -> { rows, counts }       (admin roster)
 */

const { store: sharedStore } = require("./_blobs");
const { daysForAmount } = require("./_plans");

const STORE = "accounts";
const KEY = "byEmail";
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const DAY = 86400000;

const norm = (e) => String(e || "").trim().toLowerCase();
const reply = (statusCode, obj) => ({
  statusCode,
  headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  body: JSON.stringify(obj),
});

async function store(name) {
  try {
        return sharedStore(name);
  } catch {
    return null;
  }
}

function blank(email, now) {
  return { email, firstSeen: now, lastLogin: now, loginCount: 0, paidUntil: 0, payments: [], grants: [], claims: [] };
}

function recompute(a) {
  // Window follows the amount paid — a $7 day pass must not buy 30 days, and a
  // $999 lifetime must not expire in a month. Legacy rows with no amount keep
  // the old 30-day assumption so nobody who already paid loses access.
  const fromPayments = a.payments.reduce((m, p) => {
    const days = p.amountUsd ? daysForAmount(p.amountUsd) : 30;
    return Math.max(m, p.ts + days * DAY);
  }, 0);
  const fromGrants = a.grants.reduce((m, g) => Math.max(m, g.until), 0);
  return Math.max(fromPayments, fromGrants);
}

/** Pull the legacy stores in the first time we build the roster. */
async function backfill(db) {
  const ent = await store("entitlements");
  if (ent) {
    try {
      const byEmail = (await ent.get("byEmail", { type: "json" })) || {};
      for (const [email, e] of Object.entries(byEmail)) {
        const k = norm(email);
        if (!db[k]) db[k] = blank(k, e.verifiedAt || Date.now());
        if (e.txHash && !db[k].payments.some((p) => p.txHash === e.txHash)) {
          // Payment time must be the ON-CHAIN block time, not verifiedAt: expiry
          // is ts + 30d and the entitlement's paidUntil is already blockMs + 30d,
          // so recover blockMs from it. verifiedAt would hand a late verifier
          // extra access (pay, wait 29 days, verify → ~59 days).
          const blockMs = e.paidUntil ? e.paidUntil - 30 * DAY : (e.verifiedAt || Date.now());
          db[k].payments.push({
            txHash: e.txHash, amountUsd: e.amountUsd || 0, from: e.from, ts: blockMs,
          });
        }
        // idempotent: only add the migration grant once, so repeated
        // backfills can never extend anyone's access window
        const MIGRATED = "migrated from entitlements store";
        if ((e.paidUntil || 0) > recompute(db[k]) &&
            !db[k].grants.some((g) => g.reason === MIGRATED && g.until === e.paidUntil)) {
          db[k].grants.push({
            until: e.paidUntil, reason: MIGRATED,
            by: "system", ts: e.verifiedAt || Date.now(),
          });
        }
        db[k].paidUntil = recompute(db[k]);
      }
    } catch { /* best effort */ }
  }
  const leads = await store("leads");
  if (leads) {
    try {
      const list = (await leads.get("leads", { type: "json" })) || [];
      for (const l of list) {
        const k = norm(l.email);
        if (!EMAIL_RE.test(k)) continue;
        if (!db[k]) db[k] = blank(k, l.ts || Date.now());
        db[k].firstSeen = Math.min(db[k].firstSeen, l.ts || Date.now());
        if (l.source && !db[k].source) db[k].source = l.source;
        db[k].paidUntil = recompute(db[k]);
      }
    } catch { /* best effort */ }
  }
  return db;
}

// Backfill on every load, not just the first: the legacy stores keep receiving
// writes from the older endpoints, and gating on an empty DB silently stranded
// every pre-existing customer. Idempotent by construction.
async function load(s) {
  let db = {};
  if (s) {
    try { db = (await s.get(KEY, { type: "json" })) || {}; } catch { db = {}; }
  }
  return backfill(db);
}

async function save(s, db) {
  if (s) { try { await s.setJSON(KEY, db); } catch { /* best effort */ } }
}

function summarize(db, now) {
  const rows = Object.values(db).map((a) => ({
    email: a.email,
    firstSeen: a.firstSeen,
    lastLogin: a.lastLogin,
    loginCount: a.loginCount,
    source: a.source || "",
    active: a.paidUntil > now,
    paidUntil: a.paidUntil,
    daysLeft: a.paidUntil > now ? Math.ceil((a.paidUntil - now) / DAY) : 0,
    totalPaidUsd: a.payments.reduce((s, p) => s + (p.amountUsd || 0), 0),
    payments: a.payments.length,
    grants: a.grants.length,
    // repeated device changes on a paid account = a shared password
    deviceChanges: a.deviceChanges || 0,
    // unverified claims awaiting manual confirmation
    pending: (a.claims || []).filter((c) => c.status === "pending"),
  }));
  rows.sort((x, y) => y.lastLogin - x.lastLogin);
  return {
    rows,
    counts: {
      accounts: rows.length,
      active: rows.filter((r) => r.active).length,
      paying: rows.filter((r) => r.payments > 0).length,
      comped: rows.filter((r) => r.grants > 0 && r.payments === 0).length,
      revenueUsd: rows.reduce((s, r) => s + r.totalPaidUsd, 0),
      pendingClaims: rows.reduce((s, r) => s + r.pending.length, 0),
    },
  };
}

exports.handler = async (event) => {
  const s = await store(STORE);
  const now = Date.now();

  if (event.httpMethod === "GET") {
    const email = norm((event.queryStringParameters || {}).email || "");
    const db = await load(s);
    if (email) {
      if (!EMAIL_RE.test(email)) return reply(400, { error: "email required" });
      const acct = db[email] || {};
      const paidUntil = acct.paidUntil || 0;
      const deviceId = String((event.queryStringParameters || {}).deviceId || "").slice(0, 64);
      // deviceOk is true when no device is bound yet, so an account that
      // predates this feature is never locked out of its own seat.
      const deviceOk = !deviceId || !acct.deviceId || acct.deviceId === deviceId;
      return reply(200, { active: paidUntil > now, paidUntil, deviceOk });
    }
    const token = process.env.LEADS_ADMIN_TOKEN;
    const hdr = event.headers["x-admin-token"] || event.headers["X-Admin-Token"];
    if (!token || hdr !== token) return reply(401, { error: "unauthorized" });
    await save(s, db);
    return reply(200, summarize(db, now));
  }

  if (event.httpMethod !== "POST") return reply(405, { error: "method not allowed" });

  let body = {};
  try { body = JSON.parse(event.body || "{}"); } catch { /* ignore */ }
  const email = norm(body.email);
  if (!EMAIL_RE.test(email)) return reply(400, { ok: false, reason: "A valid email is required." });

  const db = await load(s);

  if (body.action === "grant") {
    const token = process.env.LEADS_ADMIN_TOKEN;
    if (!token || String(body.adminToken || "") !== token) {
      return reply(401, { ok: false, reason: "unauthorized" });
    }
    // 36500 so a lifetime comp is expressible, not silently clipped to 10 years
    const days = Math.max(1, Math.min(36500, Number(body.days) || 30));
    if (!db[email]) db[email] = blank(email, now);
    db[email].grants.push({
      until: now + days * DAY,
      reason: String(body.reason || "manual grant"),
      by: String(body.by || "operator"),
      ts: now,
    });
    // Granting is the act of confirming a claim, so clear the queue for this
    // email — otherwise /admin keeps showing work that is already done.
    for (const c of db[email].claims || []) {
      if (c.status === "pending") { c.status = "confirmed"; c.confirmedAt = now; }
    }
    db[email].paidUntil = recompute(db[email]);
    await save(s, db);
    return reply(200, { ok: true, email, paidUntil: db[email].paidUntil, days });
  }

  // ── unverified payment claim ──
  // PayPal.me (and any other off-platform transfer) gives the site no callback
  // and nothing to verify against, so a claim NEVER grants access — it queues
  // the customer for manual confirmation in /admin. Anything that self-granted
  // here would be a free-access button with extra steps.
  if (body.action === "claim") {
    if (!db[email]) db[email] = blank(email, now);
    if (!db[email].claims) db[email].claims = [];
    db[email].claims.push({
      method: String(body.method || "paypal.me").slice(0, 24),
      note: String(body.note || "").slice(0, 120),
      amountUsd: Number(body.amountUsd) || 0,
      ts: now,
      status: "pending",
    });
    await save(s, db);
    return reply(200, { ok: true, email, pending: true });
  }

  if (body.txHash) {
    if (!db[email]) db[email] = blank(email, now);
    if (!db[email].payments.some((p) => p.txHash === String(body.txHash))) {
      db[email].payments.push({
        txHash: String(body.txHash),
        amountUsd: Number(body.amountUsd) || 0,
        from: body.from ? String(body.from) : undefined,
        ts: Number(body.ts) || now,
      });
    }
    db[email].paidUntil = recompute(db[email]);
    await save(s, db);
    return reply(200, { ok: true, email, paidUntil: db[email].paidUntil });
  }

  // ── login, and the single-device binding ──
  //
  // One email, one device. Enforced LAST-DEVICE-WINS rather than by refusing
  // the new one: a customer who changes phone, clears storage or opens a
  // private window must never be locked out of something they paid for, and a
  // hard refusal turns every such case into a support ticket. Signing in
  // elsewhere silently evicts the previous device instead — the shared-password
  // case becomes two people fighting over one session, which is the deterrent.
  if (!db[email]) db[email] = blank(email, now);
  const deviceId = String(body.deviceId || "").slice(0, 64);
  let evicted = false;
  if (deviceId) {
    if (db[email].deviceId && db[email].deviceId !== deviceId) {
      evicted = true;
      db[email].deviceChangedAt = now;
      db[email].deviceChanges = (db[email].deviceChanges || 0) + 1;
    }
    db[email].deviceId = deviceId;
    db[email].deviceSeenAt = now;
  }
  db[email].lastLogin = now;
  db[email].loginCount += 1;
  if (body.source && !db[email].source) db[email].source = String(body.source);
  db[email].paidUntil = recompute(db[email]);
  await save(s, db);
  return reply(200, {
    ok: true, email,
    active: db[email].paidUntil > now,
    paidUntil: db[email].paidUntil,
    // true when this sign-in took the seat from another device
    evicted,
  });
};
