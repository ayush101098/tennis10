/**
 * Shared Netlify Blobs accessor.
 *
 * getStore(name) relies on site context that Netlify normally injects into the
 * function runtime. On this site that injection does NOT happen — every call
 * throws:
 *
 *   "The environment has not been configured to use Netlify Blobs. To use it
 *    manually, supply the following properties when creating a store:
 *    siteID, token"
 *
 * Every function swallowed that and degraded silently, which is why the tennis
 * cache stayed empty, the TT feed showed nothing, and payment verification
 * returned "Could not record entitlement" to a customer who had actually paid.
 *
 * So: try the automatic context first, then fall back to explicit credentials.
 * SITE_ID is provided by Netlify automatically; the token must be supplied as
 * NETLIFY_API_TOKEN (a personal access token) in the site's env vars.
 */

let lastError = null;

function store(name) {
  let getStore;
  try {
    ({ getStore } = require("@netlify/blobs"));
  } catch (e) {
    lastError = "require('@netlify/blobs') failed: " + String((e && e.message) || e).slice(0, 140);
    return null;
  }

  // 1. automatic context (works when Netlify injects it)
  try {
    return getStore(name);
  } catch (e) {
    lastError = String((e && e.message) || e).slice(0, 200);
  }

  // 2. explicit credentials
  const siteID = process.env.NETLIFY_SITE_ID || process.env.SITE_ID;
  const token = process.env.NETLIFY_API_TOKEN || process.env.NETLIFY_AUTH_TOKEN;
  if (siteID && token) {
    try {
      return getStore({ name, siteID, token });
    } catch (e) {
      lastError += " | explicit creds failed: " + String((e && e.message) || e).slice(0, 150);
    }
  } else {
    lastError += ` | no fallback creds (siteID=${!!siteID}, token=${!!token}) —`
      + " set NETLIFY_API_TOKEN in the site env";
  }
  return null;
}

const lastBlobError = () => lastError;

const blobStatus = () => ({
  hasSiteId: !!(process.env.NETLIFY_SITE_ID || process.env.SITE_ID),
  hasApiToken: !!(process.env.NETLIFY_API_TOKEN || process.env.NETLIFY_AUTH_TOKEN),
  lastError,
});

module.exports = { store, lastBlobError, blobStatus };
