/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false, // avoid double-mount of WS
  output: 'standalone',   // required for Docker deployment
};
module.exports = nextConfig;
