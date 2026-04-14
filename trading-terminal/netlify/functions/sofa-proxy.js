/**
 * Netlify serverless function — SofaScore API proxy.
 *
 * When the frontend is deployed as a static export on Netlify, Next.js API
 * routes don't exist.  This function handles /api/sofa/* requests instead.
 */

exports.handler = async (event) => {
  const SOFA_BASE = "https://www.sofascore.com/api/v1";

  // event.path is like /.netlify/functions/sofa-proxy/sport/tennis/...
  // The redirect in netlify.toml rewrites /api/sofa/* → /.netlify/functions/sofa-proxy/*
  // Extract the trailing path after the function name
  const fnPrefix = "/.netlify/functions/sofa-proxy/";
  let sofaPath = event.path;
  if (sofaPath.startsWith(fnPrefix)) {
    sofaPath = sofaPath.slice(fnPrefix.length);
  } else if (sofaPath.startsWith("/api/sofa/")) {
    sofaPath = sofaPath.slice("/api/sofa/".length);
  }

  const url = `${SOFA_BASE}/${sofaPath}`;

  try {
    const res = await fetch(url, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        Accept: "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        Referer: "https://www.sofascore.com/",
        Origin: "https://www.sofascore.com",
      },
    });

    const body = await res.text();

    return {
      statusCode: res.status,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "public, s-maxage=30, stale-while-revalidate=60",
        "Access-Control-Allow-Origin": "*",
      },
      body,
    };
  } catch (err) {
    console.error("[sofa-proxy]", err);
    return {
      statusCode: 502,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Failed to fetch from SofaScore" }),
    };
  }
};
