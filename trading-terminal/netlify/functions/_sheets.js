/**
 * Google Sheets append via a service account.
 *
 * The alternative transport (an Apps Script Web App) needs no key but has to be
 * deployed by hand from the sheet's UI. This path is pure server-to-server: a
 * robot account signs a JWT, swaps it for an access token, and appends a row.
 * The cost is a private key in the environment and the sheet shared with the
 * robot's address.
 *
 * Env:
 *   GOOGLE_SERVICE_ACCOUNT_EMAIL   ...@...iam.gserviceaccount.com
 *   GOOGLE_PRIVATE_KEY             the PEM, newlines may be escaped as \n
 *   GOOGLE_SHEET_ID                defaults to the waitlist sheet
 *   GOOGLE_SHEET_TAB               defaults to the first tab
 *
 * No googleapis dependency: it is ~50MB for one append, and everything needed
 * is an RS256 signature plus two HTTPS calls.
 */

const crypto = require("crypto");

const TOKEN_URL = "https://oauth2.googleapis.com/token";
const SCOPE = "https://www.googleapis.com/auth/spreadsheets";
const DEFAULT_SHEET_ID = "1CDJls5iS71bsWzb3rEMgCQrhRWjoQh64mzFDwJi9d-4";

const b64url = (buf) =>
  Buffer.from(buf).toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

/** Netlify stores multi-line values with literal \n; PEM parsing needs real ones. */
function normalizeKey(raw) {
  const key = String(raw || "").trim().replace(/\\n/g, "\n");
  // Some UIs also wrap the whole value in quotes.
  return key.replace(/^"(.*)"$/s, "$1");
}

function config() {
  const email = process.env.GOOGLE_SERVICE_ACCOUNT_EMAIL;
  const key = normalizeKey(process.env.GOOGLE_PRIVATE_KEY);
  if (!email || !key) return null;
  return {
    email,
    key,
    sheetId: process.env.GOOGLE_SHEET_ID || DEFAULT_SHEET_ID,
    tab: process.env.GOOGLE_SHEET_TAB || "",
  };
}

/** Build and sign the JWT assertion Google swaps for an access token. */
function assertion(cfg, now = Math.floor(Date.now() / 1000)) {
  const header = b64url(JSON.stringify({ alg: "RS256", typ: "JWT" }));
  const claims = b64url(JSON.stringify({
    iss: cfg.email,
    scope: SCOPE,
    aud: TOKEN_URL,
    iat: now,
    exp: now + 3600,
  }));
  const signature = b64url(
    crypto.createSign("RSA-SHA256").update(`${header}.${claims}`).sign(cfg.key),
  );
  return `${header}.${claims}.${signature}`;
}

// Access tokens last an hour; a warm container should not re-mint one per
// signup. Cached with a 60s safety margin.
let cached = { token: null, expires: 0 };

async function accessToken(cfg) {
  if (cached.token && Date.now() < cached.expires) return cached.token;
  const res = await fetch(TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
      assertion: assertion(cfg),
    }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.access_token) {
    throw new Error(`token ${res.status} ${data.error_description || data.error || ""}`.trim());
  }
  cached = { token: data.access_token, expires: Date.now() + (data.expires_in - 60) * 1000 };
  return cached.token;
}

const api = (cfg, path) =>
  `https://sheets.googleapis.com/v4/spreadsheets/${encodeURIComponent(cfg.sheetId)}${path}`;

const range = (cfg, a1) => (cfg.tab ? `${cfg.tab}!${a1}` : a1);

/**
 * Append one waitlist row, unless the address is already present.
 *
 * Returns a short status string for the caller to log — never throws, because
 * the sheet is a mirror and must not be able to fail a signup.
 */
async function appendWaitlist({ email, joinedAt }) {
  const cfg = config();
  if (!cfg) return "not configured";
  try {
    const token = await accessToken(cfg);
    const auth = { Authorization: `Bearer ${token}` };

    // Existing addresses — column A. One row per person, so a retry or a
    // re-run of the backfill cannot double up.
    const readRes = await fetch(
      `${api(cfg, `/values/${encodeURIComponent(range(cfg, "A:A"))}`)}?majorDimension=COLUMNS`,
      { headers: auth },
    );
    const read = await readRes.json().catch(() => ({}));
    if (!readRes.ok) {
      throw new Error(`read ${readRes.status} ${(read.error && read.error.message) || ""}`.trim());
    }
    const column = (read.values && read.values[0]) || [];
    const seen = column.map((v) => String(v).toLowerCase().trim());
    if (seen.includes(String(email).toLowerCase().trim())) return "duplicate";

    // Header row on a fresh sheet, so the columns are self-describing.
    const rows = [];
    if (column.length === 0) rows.push(["email", "joinedAt"]);
    rows.push([email, joinedAt]);

    const appendRes = await fetch(
      `${api(cfg, `/values/${encodeURIComponent(range(cfg, "A:B"))}:append`)}`
      + "?valueInputOption=RAW&insertDataOption=INSERT_ROWS",
      { method: "POST", headers: { ...auth, "Content-Type": "application/json" },
        body: JSON.stringify({ values: rows }) },
    );
    if (!appendRes.ok) {
      const err = await appendRes.json().catch(() => ({}));
      throw new Error(`append ${appendRes.status} ${(err.error && err.error.message) || ""}`.trim());
    }
    return "ok";
  } catch (e) {
    return String((e && e.message) || e).slice(0, 120);
  }
}

/**
 * Append many rows in one pass — one column read and one append call, rather
 * than a read+append per address. A backfill of several hundred leads has to
 * fit inside a single function invocation.
 */
async function appendWaitlistBatch(rows) {
  const cfg = config();
  if (!cfg) return { ok: false, reason: "not configured" };
  try {
    const token = await accessToken(cfg);
    const auth = { Authorization: `Bearer ${token}` };

    const readRes = await fetch(
      `${api(cfg, `/values/${encodeURIComponent(range(cfg, "A:A"))}`)}?majorDimension=COLUMNS`,
      { headers: auth },
    );
    const read = await readRes.json().catch(() => ({}));
    if (!readRes.ok) {
      throw new Error(`read ${readRes.status} ${(read.error && read.error.message) || ""}`.trim());
    }
    const column = (read.values && read.values[0]) || [];
    const seen = new Set(column.map((v) => String(v).toLowerCase().trim()));

    const values = [];
    if (column.length === 0) values.push(["email", "joinedAt"]);
    let skipped = 0;
    for (const r of rows) {
      const key = String(r.email || "").toLowerCase().trim();
      if (!key || seen.has(key)) { skipped++; continue; }
      seen.add(key);
      values.push([key, r.joinedAt]);
    }
    const added = values.length - (column.length === 0 ? 1 : 0);
    if (added <= 0) return { ok: true, added: 0, skipped };

    const appendRes = await fetch(
      `${api(cfg, `/values/${encodeURIComponent(range(cfg, "A:B"))}:append`)}`
      + "?valueInputOption=RAW&insertDataOption=INSERT_ROWS",
      { method: "POST", headers: { ...auth, "Content-Type": "application/json" },
        body: JSON.stringify({ values }) },
    );
    if (!appendRes.ok) {
      const err = await appendRes.json().catch(() => ({}));
      throw new Error(`append ${appendRes.status} ${(err.error && err.error.message) || ""}`.trim());
    }
    return { ok: true, added, skipped };
  } catch (e) {
    return { ok: false, reason: String((e && e.message) || e).slice(0, 160) };
  }
}

module.exports = { appendWaitlist, appendWaitlistBatch, assertion, normalizeKey, config };
