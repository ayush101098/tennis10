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

export default function Wordmark({ size = 18, mark = true }: {
  /** cap height of the lettering in px */
  size?: number;
  /** show the bird mark alongside, when its file exists */
  mark?: boolean;
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
    <span className="inline-flex items-center gap-2 shrink-0" aria-label="Tennis Alpha">
      {mark && hasMark && (
        // eslint-disable-next-line @next/next/no-img-element -- static brand asset
        <img src="/brand/mark.svg" alt="" height={size * 1.4} style={{ height: size * 1.4, width: "auto" }} />
      )}
      {hasArt ? (
        // eslint-disable-next-line @next/next/no-img-element -- static brand asset
        <img src="/brand/wordmark.svg" alt="Tennis Alpha"
          style={{ height: size * 1.5, width: "auto", display: "block" }} />
      ) : (
        <span className="brand-wordmark" style={{ fontSize: size * 1.5, color: BRAND_GREEN }}>
          tennis alpha
        </span>
      )}
    </span>
  );
}
