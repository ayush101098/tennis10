/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false, // avoid double-mount of WS
  // Netlify sets NETLIFY=true automatically; use static export there,
  // standalone for Docker/GCP deployment
  output: process.env.NETLIFY ? 'export' : 'standalone',
  images: { unoptimized: true }, // required for static export
};
module.exports = nextConfig;
