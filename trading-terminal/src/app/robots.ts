import type { MetadataRoute } from "next";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://tennisalpha.in";

/**
 * robots.txt, emitted as a static file by the export.
 *
 * /admin is disallowed because it is an operator console — it is token-gated,
 * but there is no reason for it to sit in a search index either.
 */
export default function robots(): MetadataRoute.Robots {
  return {
    rules: [{ userAgent: "*", allow: "/", disallow: ["/admin"] }],
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
