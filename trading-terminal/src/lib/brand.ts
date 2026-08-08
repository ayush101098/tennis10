/** Brand-level constants shared across the site. */

export const X_URL = "https://x.com/future_jesse";

/**
 * Telegram invite. Socials renders the button only when this is non-empty, so
 * clearing it removes Telegram everywhere rather than leaving a dead link.
 */
export const TELEGRAM_URL = "https://t.me/+n2KRYAFVyBhhZDA9";

/** Legal entity shown in the footer copyright line. */
export const LEGAL_NAME = "Nexxore Labs";

/**
 * Search-console ownership tokens.
 *
 * Both consoles accept a meta tag in the homepage <head>, which is the least
 * fragile option here: a DNS TXT record proves the whole domain but has to be
 * re-added if nameservers ever move, and an uploaded HTML file is easy to lose
 * in a rebuild. Paste the token between the quotes and redeploy — an empty
 * string renders no tag at all, so a half-configured site never emits a
 * meaningless one.
 *
 * Google:  Search Console -> Add property -> URL prefix -> HTML tag
 *          (copy only the content="..." value, not the whole tag)
 * Bing:    Webmaster Tools -> Add site -> Meta tag  — or skip it entirely and
 *          use "Import from Google Search Console", which needs no token.
 */
export const GOOGLE_SITE_VERIFICATION = "";
export const BING_SITE_VERIFICATION = "";
