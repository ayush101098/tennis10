/**
 * The one place application state is read and written.
 *
 * Every handler used to call _blobs.js directly, which meant the data lived in
 * Netlify Blobs — a Netlify service, on the plan whose exhaustion took the site
 * down. This is the same interface (`store(name).get(key, {type:"json"})` and
 * `.setJSON(key, value)`) backed by whichever provider is configured, so the
 * fifteen handlers did not have to change at all:
 *
 *   BLOB_READ_WRITE_TOKEN set   -> Vercel Blob        (the destination)
 *   otherwise                   -> Netlify Blobs      (the origin, still works)
 *
 * Keeping the Netlify path alive is deliberate. It is the rollback: unset one
 * environment variable and the previous storage is serving again, with no
 * deploy and no code change. It is also what the migration script reads from.
 *
 * WHY VERCEL BLOB AND NOT KV/REDIS
 *   The access pattern is a handful of whole JSON documents, read and rewritten
 *   in one piece — a leads list, an accounts map. That is a file, not a cache,
 *   and Blob needs no marketplace provisioning step.
 *
 * CONCURRENCY, HONESTLY
 *   Read-modify-write on a whole document races if two writes land in the same
 *   instant; the last writer wins and the other lead is lost. That race existed
 *   identically on Netlify Blobs — this is not a regression — and at the
 *   current write volume (a signup every few hours) it is theoretical. It stops
 *   being theoretical somewhere around a signup a second, which is the point to
 *   move accounts to Postgres.
 */

const VERCEL_TOKEN = () => process.env.BLOB_READ_WRITE_TOKEN;

/* ── Vercel Blob ─────────────────────────────────────────────────────────── */

let vercelBlob = null;
function sdk() {
  if (!vercelBlob) vercelBlob = require("@vercel/blob");
  return vercelBlob;
}

/** Blob pathnames cannot contain the characters a store/key pair might. */
const pathFor = (storeName, key) =>
  `${String(storeName).replace(/[^A-Za-z0-9._-]/g, "_")}/${String(key).replace(/[^A-Za-z0-9._-]/g, "_")}.json`;

function vercelStore(storeName) {
  const token = VERCEL_TOKEN();
  return {
    async get(key, opts) {
      const { get } = sdk();
      const pathname = pathFor(storeName, key);
      try {
        // The store is PRIVATE — these documents are customer emails and
        // account records, and a public store would serve them to anyone who
        // guessed the pathname. get() authenticates with the token; useCache
        // is off because this is mutable state, not an asset.
        const res = await get(pathname, { access: "private", token, useCache: false });
        if (!res) return null;
        const text = await new Response(res.stream).text();
        return opts?.type === "json" ? JSON.parse(text) : text;
      } catch (e) {
        // A miss is a normal answer, not a failure. The SDK signals it with
        // BlobNotFoundError and the message "The requested blob does not
        // exist" — match both, since either could change independently.
        const name = String(e?.name || "");
        const msg = String(e?.message || "");
        if (name.includes("NotFound") || /not.?found|does not exist/i.test(msg)) return null;
        throw e;
      }
    },
    async setJSON(key, value) {
      const { put } = sdk();
      await put(pathFor(storeName, key), JSON.stringify(value), {
        token,
        access: "private",         // customer data — never publicly addressable
        contentType: "application/json",
        addRandomSuffix: false,    // a stable pathname is the whole point
        allowOverwrite: true,
        cacheControlMaxAge: 0,     // state must never be served stale
      });
    },
    async set(key, value) {
      const { put } = sdk();
      await put(pathFor(storeName, key), String(value), {
        token, access: "private", addRandomSuffix: false, allowOverwrite: true, cacheControlMaxAge: 0,
      });
    },
  };
}

/* ── dispatch ────────────────────────────────────────────────────────────── */

let lastError = null;

/**
 * Is ANY blob backend actually configured?
 *
 * Without this the store is constructed happily and then fails on the first
 * network call, once per request. Running locally that meant a doomed round
 * trip on every cache read — 219 of them in one short session, ~300ms each,
 * for a store that was never going to answer. An unconfigured backend should
 * be detected once, not rediscovered per request.
 *
 * Callers already treat a null store as "no cache", so returning null is the
 * existing contract rather than a new one.
 */
function configured() {
  if (VERCEL_TOKEN()) return true;
  const siteId = process.env.NETLIFY_SITE_ID || process.env.SITE_ID;
  const token = process.env.NETLIFY_API_TOKEN || process.env.NETLIFY_AUTH_TOKEN;
  return !!(siteId && token);
}

/**
 * Circuit breaker for a credential the backend REJECTS.
 *
 * "Configured" and "working" are different things. A token that is present but
 * expired, revoked or scoped to another site answers 401/403 on every call —
 * and retrying it per request buys nothing but latency and log noise. Seen
 * locally: 219 rejected blob reads in one short session, ~300ms each, on a
 * `BLOB_READ_WRITE_TOKEN` the backend refuses.
 *
 * After a few consecutive auth failures the store reports itself unusable for
 * a cooldown, so callers fall back to their no-cache path immediately. It
 * re-arms afterwards rather than giving up for the life of the process,
 * because a rotated token should heal without a redeploy.
 */
const AUTH_FAIL_LIMIT = 3;
const BREAKER_COOLDOWN_MS = 5 * 60 * 1000;
let authFailures = 0;
let breakerUntil = 0;

function isAuthError(e) {
  const m = String((e && e.message) || e);
  return m.includes("401") || m.includes("403")
    || /unauthor|forbidden/i.test(m);
}

function noteFailure(e) {
  if (!isAuthError(e)) return;
  authFailures += 1;
  if (authFailures >= AUTH_FAIL_LIMIT) {
    breakerUntil = Date.now() + BREAKER_COOLDOWN_MS;
    lastError = "blob credential rejected by the backend — caching disabled for "
      + `${BREAKER_COOLDOWN_MS / 60000} min. Check BLOB_READ_WRITE_TOKEN / `
      + "NETLIFY_API_TOKEN.";
  }
}

/** Wrap a store so auth failures trip the breaker instead of repeating. */
function guarded(inner) {
  if (!inner) return inner;
  const wrap = (fn) => async (...args) => {
    try {
      const out = await fn.apply(inner, args);
      authFailures = 0;                 // a success clears the count
      return out;
    } catch (e) {
      noteFailure(e);
      throw e;
    }
  };
  return new Proxy(inner, {
    get(target, prop) {
      const v = target[prop];
      return typeof v === "function" ? wrap(v) : v;
    },
  });
}

function store(name) {
  if (!configured()) {
    lastError = "no blob backend configured (set NETLIFY_API_TOKEN + SITE_ID, "
      + "or BLOB_READ_WRITE_TOKEN) — running without a cache";
    return null;
  }
  if (Date.now() < breakerUntil) return null;
  if (VERCEL_TOKEN()) {
    try {
      return guarded(vercelStore(name));
    } catch (e) {
      lastError = `vercel blob: ${String(e?.message || e).slice(0, 160)}`;
    }
  }
  // Fallback / rollback path.
  try {
    return guarded(require("./_blobs").store(name));
  } catch (e) {
    lastError = `netlify blobs: ${String(e?.message || e).slice(0, 160)}`;
    return null;
  }
}

const provider = () =>
  !configured() ? "none" : (VERCEL_TOKEN() ? "vercel-blob" : "netlify-blobs");
const storeStatus = () => ({ provider: provider(), lastError });

// blobStatus is the name the diagnostics in sofa-proxy already call. Kept so
// the switch of provider did not silently break the one endpoint whose job is
// to explain why storage is not working.
const blobStatus = () => ({
  provider: provider(),
  hasVercelToken: !!VERCEL_TOKEN(),
  lastError,
});

module.exports = { store, provider, storeStatus, blobStatus };
