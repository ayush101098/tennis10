/**
 * Provider-agnostic web analytics tag, injected once in the app shell.
 *
 * Configure with ONE of these (set in Netlify env / .env.local), then redeploy:
 *   NEXT_PUBLIC_PLAUSIBLE_DOMAIN = yourdomain.com        (Plausible — privacy-first, recommended)
 *   NEXT_PUBLIC_GA_ID            = G-XXXXXXXXXX           (Google Analytics 4)
 *
 * GA4 defaults to this site's own property. NEXT_PUBLIC_* is inlined at build
 * time, so an env-only setup would need the var present in the Netlify build —
 * the default keeps the tag working without that. A measurement ID is public
 * by construction (it ships in the HTML of every page it tags), not a secret.
 * Custom events (signups, payments) can be sent from anywhere via trackEvent().
 */

const PLAUSIBLE = process.env.NEXT_PUBLIC_PLAUSIBLE_DOMAIN;
const GA_ID = process.env.NEXT_PUBLIC_GA_ID || "G-WTTMQK5ZZJ";

export default function Analytics() {
  if (PLAUSIBLE) {
    return (
      <>
        <script defer data-domain={PLAUSIBLE} src="https://plausible.io/js/script.tagged-events.js" />
        <script
          dangerouslySetInnerHTML={{
            __html:
              "window.plausible=window.plausible||function(){(window.plausible.q=window.plausible.q||[]).push(arguments)}",
          }}
        />
      </>
    );
  }
  if (GA_ID) {
    return (
      <>
        <script async src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`} />
        <script
          dangerouslySetInnerHTML={{
            __html: `window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments)}gtag('js',new Date());gtag('config','${GA_ID}');`,
          }}
        />
      </>
    );
  }
  return null;
}

/** Fire a named conversion event to whichever provider is configured. */
export function trackEvent(name: string, props?: Record<string, string | number>): void {
  if (typeof window === "undefined") return;
  const w = window as unknown as {
    plausible?: (n: string, o?: { props?: Record<string, unknown> }) => void;
    gtag?: (...a: unknown[]) => void;
  };
  try {
    if (w.plausible) w.plausible(name, props ? { props } : undefined);
    if (w.gtag) w.gtag("event", name, props || {});
  } catch {
    /* analytics must never break the app */
  }
}
