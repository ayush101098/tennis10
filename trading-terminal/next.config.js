/** @type {import('next').NextConfig} */

// Netlify: static export + Netlify Functions (its build strips src/app/api).
// Vercel:  a normal Next app — the API routes ARE the backend, served by the
//          same handlers as the Netlify functions via src/lib/netlifyAdapter.
// Docker/GCP: standalone.
const isNetlify = !!process.env.NETLIFY;
const isVercel = !!process.env.VERCEL;

const nextConfig = {
  reactStrictMode: false, // avoid double-mount of WS
  output: isNetlify ? "export" : isVercel ? undefined : "standalone",
  // Static export cannot optimise images; on Vercel it can, but every image
  // here is a data: URI or an SVG, so there is nothing to gain and the
  // unoptimised path is identical on both hosts.
  images: { unoptimized: true },
};
module.exports = nextConfig;
