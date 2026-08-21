"use client";

import { useEffect, useState } from "react";

/**
 * Dynamic tennis backdrop for the hero.
 *
 * TWO MODES, chosen automatically — the same pattern Wordmark uses, so the site
 * looks finished now and upgrades the moment real artwork lands:
 *
 *   1. If /brand/hero.jpg (or .webp) exists in public/, it is used as a
 *      photographic backdrop under a dark scrim.
 *   2. Otherwise a court is drawn in SVG.
 *
 * Why vector is the default rather than a stock photograph: an SVG court is
 * resolution-independent by construction — it is exactly as crisp on a 5K
 * display as on a phone, at about 2KB — where a "high resolution" JPEG large
 * enough for a 2560px hero costs hundreds of KB on every first paint and still
 * softens when the viewport goes wider. It also carries no licensing exposure,
 * which matters on a page that takes payment.
 *
 * MOTION
 * The ball traces a rally and the court drifts, both slowly. This is behind
 * body copy, so it has to stay legible: everything runs at low contrast and the
 * whole thing is disabled under `prefers-reduced-motion`, which is a real
 * accessibility need here — parallax and looping motion are common migraine and
 * vestibular triggers.
 */
export default function CourtBackdrop({ live = 0 }: { live?: number }) {
  const [photo, setPhoto] = useState<string | null>(null);
  const [reduced, setReduced] = useState(false);

  // HEAD probe rather than <img onError>, so a missing file never paints a
  // broken-image glyph behind the headline.
  useEffect(() => {
    let alive = true;
    (async () => {
      for (const url of ["/brand/hero.webp", "/brand/hero.jpg"]) {
        try {
          const r = await fetch(url, { method: "HEAD" });
          const ct = String(r.headers.get("content-type") || "");
          if (alive && r.ok && ct.startsWith("image/")) { setPhoto(url); return; }
        } catch { /* absent — fall through to the vector court */ }
      }
    })();
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => setReduced(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  const animate = !reduced;

  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden select-none">
      {photo ? (
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${photo})`, opacity: 0.22 }}
        />
      ) : (
        <svg
          // Anchored to the bottom and shown WHOLE. Centring it and scaling to
          // 170% cropped the court to a few diagonals crossing the middle of the
          // hero, which read as stray lines over the form rather than as a
          // court. Sitting it under the text lets the perspective do its job:
          // the page gains a horizon instead of a texture.
          className={`absolute bottom-0 left-0 h-[62%] w-full ${animate ? "court-drift" : ""}`}
          // Geometry is drawn for a WIDE, SHALLOW band because that is the
          // shape of a hero. An 800x520 viewBox forced `slice` to zoom until
          // only the outer tramlines survived, which read as stray diagonals.
          // A 3:1 viewBox with a strongly foreshortened court fills the band and
          // still reads as a court.
          viewBox="0 0 1200 400"
          fill="none"
          preserveAspectRatio="xMidYMax slice"
        >
          <defs>
            <linearGradient id="cbFade" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#4CA85E" stopOpacity="0.05" />
              <stop offset="40%" stopColor="#4CA85E" stopOpacity="0.42" />
              <stop offset="100%" stopColor="#4CA85E" stopOpacity="0.72" />
            </linearGradient>
            <radialGradient id="cbGlow" cx="50%" cy="95%" r="62%">
              <stop offset="0%" stopColor="#4CA85E" stopOpacity="0.16" />
              <stop offset="100%" stopColor="#4CA85E" stopOpacity="0" />
            </radialGradient>
            <filter id="cbSoft" x="-60%" y="-60%" width="220%" height="220%">
              <feGaussianBlur stdDeviation="6" />
            </filter>
          </defs>

          <ellipse cx="600" cy="380" rx="640" ry="220" fill="url(#cbGlow)" />

          {/* One-point perspective, vanishing above centre. Doubles court
              outside, singles inside, service boxes between net and baselines. */}
          <g stroke="url(#cbFade)" strokeWidth="2" strokeLinecap="round">
            {/* baselines */}
            <path d="M480 60 H720" />
            <path d="M120 360 H1080" />
            {/* doubles sidelines */}
            <path d="M480 60 L120 360" />
            <path d="M720 60 L1080 360" />
            {/* singles sidelines */}
            <path d="M505 60 L210 360" />
            <path d="M695 60 L990 360" />
            {/* service lines */}
            <path d="M426 140 H774" />
            <path d="M288 280 H912" />
            {/* centre service line */}
            <path d="M600 140 V280" />
          </g>

          {/* Net: posts and a sagging cord. This is the element that makes the
              lines read as tennis rather than as a perspective grid. */}
          <g stroke="#4CA85E" strokeOpacity="0.6" strokeWidth="2">
            <path d="M300 200 V150" />
            <path d="M900 200 V150" />
            <path d="M300 156 Q600 196 900 156" />
          </g>
          <path d="M300 156 Q600 196 900 156 L900 200 Q600 240 300 200 Z"
                fill="#4CA85E" fillOpacity="0.06" />

          {/* Rally: a plausible cross-court exchange, not decoration drifting. */}
          <path id="cbRally"
                d="M330 330 Q600 130 880 300 Q600 380 340 250 Q600 110 860 330"
                stroke="#4CA85E" strokeOpacity={animate ? 0.18 : 0.1}
                strokeWidth="1.2" strokeDasharray="4 9" fill="none" />
          <g filter="url(#cbSoft)">
            <circle r="7" fill="#D6F35B" fillOpacity="0.45">
              {animate && (
                <animateMotion dur="15s" repeatCount="indefinite" rotate="auto">
                  <mpath href="#cbRally" />
                </animateMotion>
              )}
            </circle>
          </g>
          <circle r="3" fill="#EAF7A8" fillOpacity="0.9">
            {animate && (
              <animateMotion dur="15s" repeatCount="indefinite" rotate="auto">
                <mpath href="#cbRally" />
              </animateMotion>
            )}
          </circle>
        </svg>
      )}

      {/* Scrim. The hero sets body copy over this, and contrast is not
          negotiable — the backdrop loses wherever the two compete. */}
      <div className="absolute inset-0 bg-gradient-to-b from-terminal-bg via-terminal-bg/55 to-terminal-bg/85" />

      {/* The one genuinely live element: a pulse only while matches are in
          play, so the page is visibly reacting to the feed rather than looping
          the same animation whatever is happening. */}
      {live > 0 && (
        <div className="absolute inset-x-0 top-0 flex justify-center">
          <div className={`mt-1 h-px w-40 bg-terminal-green/40 ${animate ? "animate-pulse" : ""}`} />
        </div>
      )}

      <style jsx>{`
        .court-drift {
          animation: cbDrift 46s ease-in-out infinite alternate;
        }
        @keyframes cbDrift {
          from { transform: scale(1) rotate(0deg); }
          to   { transform: scale(1.04) rotate(0.4deg); }
        }
        @media (prefers-reduced-motion: reduce) {
          .court-drift { animation: none; }
        }
      `}</style>
    </div>
  );
}
