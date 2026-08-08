import { PLANS } from "@/lib/plans";

/**
 * Structured data.
 *
 * Emitted server-side as a plain <script type="application/ld+json"> so it is
 * present in the exported HTML — a client-injected blob would be invisible to
 * crawlers that do not execute JS, which defeats the point.
 *
 * Only facts the site actually backs are described. No aggregateRating: Google
 * requires review markup to correspond to reviews visible on the page, and
 * inventing one is both a manual-action risk and a lie to customers.
 */

const SITE = process.env.NEXT_PUBLIC_SITE_URL || "https://tennisalpha.in";

function Ld({ data }: { data: unknown }) {
  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(data) }}
    />
  );
}

/** The product itself — homepage and terminal. */
export function SoftwareApplicationLd() {
  return (
    <Ld data={{
      "@context": "https://schema.org",
      "@type": "SoftwareApplication",
      name: "Tennis Alpha",
      applicationCategory: "FinanceApplication",
      operatingSystem: "Web",
      url: SITE,
      description:
        "Live win-probability model for professional tennis — neural network prior re-priced by a score-conditioned Markov engine on every point, compared against de-vigged bookmaker odds, with ¼-Kelly staking and hedge timing.",
      offers: PLANS.map(p => ({
        "@type": "Offer",
        name: p.label,
        price: String(p.usd),
        priceCurrency: "USD",
        url: `${SITE}/terminal`,
      })),
      publisher: { "@type": "Organization", name: "Nexxore Labs", url: SITE },
    }} />
  );
}

/** Breadcrumbs for any non-home page. */
export function BreadcrumbLd({ trail }: { trail: { name: string; path: string }[] }) {
  return (
    <Ld data={{
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: [{ name: "Home", path: "/" }, ...trail].map((c, i) => ({
        "@type": "ListItem",
        position: i + 1,
        name: c.name,
        item: `${SITE}${c.path === "/" ? "" : c.path}`,
      })),
    }} />
  );
}

/** Methodology questions, answered on the manual page. */
export function FaqLd({ qa }: { qa: { q: string; a: string }[] }) {
  return (
    <Ld data={{
      "@context": "https://schema.org",
      "@type": "FAQPage",
      mainEntity: qa.map(({ q, a }) => ({
        "@type": "Question",
        name: q,
        acceptedAnswer: { "@type": "Answer", text: a },
      })),
    }} />
  );
}
