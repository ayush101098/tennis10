"use client";

import { useEffect, useState } from "react";

/**
 * The Tennis Alpha wordmark.
 *
 * Two modes, chosen automatically:
 *
 *   1. If /brand/wordmark.svg (or .png) has been dropped into public/, it is
 *      used — that is the real artwork and always wins.
 *   2. Otherwise the name is set in Kaushan Script, the closest widely
 *      available match to the brush lettering, in the brand green.
 *
 * Written this way so the site carries the brand now and picks up the real
 * asset the moment it lands, with no code change. Drop the file at
 * trading-terminal/public/brand/wordmark.svg and it takes over.
 */

const BRAND_GREEN = "#4CA85E";

export default function Wordmark({ size = 18, mark = true, text = true }: {
  /** cap height of the lettering in px */
  size?: number;
  /** show the bird mark alongside, when its file exists */
  mark?: boolean;
  /**
   * Show the name as well as the mark. Set false for a mark-only lockup.
   *
   * The name does not simply disappear when this is off — it moves to the
   * wrapper's aria-label, which is already present. A bare icon with no
   * accessible name is an unlabelled link for anyone using a screen reader,
   * and this one is the site's home link.
   */
  text?: boolean;
}) {
  const [hasArt, setHasArt] = useState(false);
  const [hasMark, setHasMark] = useState(false);

  // A HEAD probe rather than an <img onError>, so a missing file never paints
  // a broken-image glyph in the header.
  useEffect(() => {
    let alive = true;
    const probe = (url: string, set: (v: boolean) => void) =>
      fetch(url, { method: "HEAD" })
        .then(r => { if (alive && r.ok && !String(r.headers.get("content-type") || "").includes("html")) set(true); })
        .catch(() => {});
    probe("/brand/wordmark.svg", setHasArt);
    if (mark) probe("/brand/mark.svg", setHasMark);
    return () => { alive = false; };
  }, [mark]);

  return (
    <span className="inline-flex items-center gap-2 shrink-0" role="img" aria-label="Tennis Alpha">
      {mark && hasMark && (
        // eslint-disable-next-line @next/next/no-img-element -- static brand asset
        <img src="/brand/mark.svg" alt="" height={size * 1.6} style={{ height: size * 1.6, width: "auto" }} />
      )}
      {/* Mark-only lockup, before the HEAD probe has resolved (or if the asset
          is missing): without this the header renders empty on first paint,
          because the name is hidden and the mark has not been confirmed yet.
          A monogram is a better first frame than a hole where the logo goes. */}
      {!text && !hasMark && (
        <span
          className="inline-flex items-center justify-center rounded font-bold"
          style={{
            width: size * 1.6, height: size * 1.6, fontSize: size * 0.78,
            color: BRAND_GREEN, border: `1.5px solid ${BRAND_GREEN}66`,
            letterSpacing: "-0.02em",
          }}
        >
          TA
        </span>
      )}
      {!text ? null : hasArt ? (
        // eslint-disable-next-line @next/next/no-img-element -- static brand asset
        <img src="/brand/wordmark.svg" alt="Tennis Alpha"
          style={{ height: size * 1.5, width: "auto", display: "block" }} />
      ) : (
        <span className="brand-wordmark" style={{ fontSize: size * 0.92, color: BRAND_GREEN }}>
          Tennis Alpha
        </span>
      )}
    </span>
  );
}
