/**
 * Netlify serverless function — SofaScore API proxy.
 *
 * When the frontend is deployed as a static export on Netlify, Next.js API
 * routes don't exist.  This function handles /api/sofa/* requests instead.
 *
 * SofaScore blocks ALL programmatic HTTP requests via TLS fingerprinting
 * (Varnish CDN returns 403 for Node.js fetch, curl, etc.).
 *
 * This function forwards requests to a deployed instance of sofa_proxy.py
 * (which uses tls_client with Chrome TLS fingerprint impersonation).
 *
 * Set the SOFA_PROXY_URL env var on Netlify to the deployed proxy URL:
 *   e.g. https://sofa-proxy.onrender.com
 *        https://sofa-proxy-xxxx.fly.dev
 */

exports.handler = async (event) => {
  const SOFA_PROXY_URL = process.env.SOFA_PROXY_URL;

  if (!SOFA_PROXY_URL) {
    return {
      statusCode: 503,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        error: "SOFA_PROXY_URL not configured. Deploy sofa_proxy.py and set the env var.",
      }),
    };
  }

  // Extract the SofaScore API path from the request
  const fnPrefix = "/.netlify/functions/sofa-proxy/";
  let sofaPath = event.path;
  if (sofaPath.startsWith(fnPrefix)) {
    sofaPath = sofaPath.slice(fnPrefix.length);
  } else if (sofaPath.startsWith("/api/sofa/")) {
    sofaPath = sofaPath.slice("/api/sofa/".length);
  }

  // Forward to the deployed sofa_proxy.py (which handles TLS fingerprinting)
  const url = `${SOFA_PROXY_URL.replace(/\/$/, "")}/${sofaPath}`;

  try {
    const res = await fetch(url, {
      headers: { Accept: "application/json" },
    });

    const body = await res.text();

    return {
      statusCode: res.status,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "public, s-maxage=3, stale-while-revalidate=5",
        "Access-Control-Allow-Origin": "*",
      },
      body,
    };
  } catch (err) {
    console.error("[sofa-proxy]", err);
    return {
      statusCode: 502,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        error: "Failed to reach sofa_proxy — is it deployed?",
        proxyUrl: SOFA_PROXY_URL,
      }),
    };
  }
};
