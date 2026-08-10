/**
 * Verify a Google Sign-In credential and return the email it proves.
 *
 * The browser gets an ID token (a JWT) from Google Identity Services and posts
 * it here. This endpoint asks Google whether the token is genuine — the token
 * MUST NOT be trusted client-side, because anyone can put any email in a JSON
 * body and the whole point is that this one is proven.
 *
 * Two checks that matter, and are the usual omissions:
 *   aud  — the token must have been issued for OUR client id. A token minted
 *          for some other site is perfectly valid and says nothing about a
 *          user's intent to sign in here.
 *   email_verified — a Google account can carry an unverified address.
 *
 * Uses Google's tokeninfo endpoint rather than local JWKS verification: one
 * HTTPS call, no crypto dependency, and Google is authoritative either way.
 * The cost is a round trip per sign-in, which is nothing at this volume.
 *
 * Env: GOOGLE_CLIENT_ID (also exposed to the browser as
 *      NEXT_PUBLIC_GOOGLE_CLIENT_ID — a client id is public by design).
 */

const TOKENINFO = "https://oauth2.googleapis.com/tokeninfo?id_token=";

const reply = (statusCode, obj) => ({
  statusCode,
  headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  body: JSON.stringify(obj),
});

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") return reply(405, { ok: false, reason: "method not allowed" });

  const clientId = process.env.GOOGLE_CLIENT_ID || process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
  if (!clientId) {
    return reply(503, { ok: false, reason: "Google sign-in isn't configured yet." });
  }

  let body = {};
  try { body = JSON.parse(event.body || "{}"); } catch {
    return reply(400, { ok: false, reason: "invalid JSON" });
  }
  const credential = String(body.credential || "");
  if (!credential || credential.length > 4096) {
    return reply(400, { ok: false, reason: "no credential" });
  }

  try {
    const res = await fetch(TOKENINFO + encodeURIComponent(credential));
    const info = await res.json().catch(() => ({}));
    if (!res.ok) {
      return reply(401, { ok: false, reason: "Google rejected that sign-in. Try again." });
    }
    if (info.aud !== clientId) {
      // Issued for a different site — valid, and meaningless here.
      return reply(401, { ok: false, reason: "That sign-in was not issued for this site." });
    }
    if (info.email_verified !== "true" && info.email_verified !== true) {
      return reply(401, { ok: false, reason: "That Google account has no verified email address." });
    }
    const email = String(info.email || "").trim().toLowerCase();
    if (!email) return reply(401, { ok: false, reason: "No email on that Google account." });

    return reply(200, {
      ok: true,
      email,
      name: info.name || "",
    });
  } catch (e) {
    return reply(502, { ok: false, reason: String((e && e.message) || e).slice(0, 140) });
  }
};
