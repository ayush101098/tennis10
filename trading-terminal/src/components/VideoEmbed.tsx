"use client";

import { useState } from "react";

/**
 * Click-to-play YouTube facade.
 *
 * The previous markup mounted the YouTube iframe on page load behind
 * `loading="lazy"`. Two problems with that, one cosmetic and one not:
 *
 *   · Until the iframe actually loads, the slot paints as a bare black
 *     rectangle — no poster, no title, no affordance. On a landing page that is
 *     a dead hole where the product demo should be.
 *   · The embed pulls ~1MB of YouTube player JS and sets third-party cookies for
 *     every visitor, including the large majority who never press play.
 *
 * So the poster frame is a plain <img> and the iframe is only created on click.
 * The page shows a real image immediately, costs nothing until someone wants the
 * video, and contacts YouTube only when they do.
 *
 * `maxresdefault` does not exist for every upload, so onError steps down to
 * `hqdefault`, which always does — otherwise a missing max-res thumbnail would
 * reintroduce exactly the empty box this replaces.
 */
export default function VideoEmbed({ id, title }: { id: string; title: string }) {
  const [playing, setPlaying] = useState(false);
  const [thumb, setThumb] = useState(`https://i.ytimg.com/vi/${id}/maxresdefault.jpg`);

  return (
    <div
      className="relative w-full rounded-lg overflow-hidden border border-terminal-border bg-terminal-panel"
      style={{ aspectRatio: "16 / 9" }}
    >
      {playing ? (
        <iframe
          className="absolute inset-0 w-full h-full"
          src={`https://www.youtube-nocookie.com/embed/${id}?rel=0&autoplay=1`}
          title={title}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          referrerPolicy="strict-origin-when-cross-origin"
          allowFullScreen
        />
      ) : (
        <button
          type="button"
          onClick={() => setPlaying(true)}
          className="group absolute inset-0 w-full h-full cursor-pointer"
          aria-label={`Play video: ${title}`}
        >
          {/* eslint-disable-next-line @next/next/no-img-element -- third-party
              thumbnail on a fixed CDN path; next/image would proxy it for no gain
              and images are unoptimised in this build anyway. */}
          <img
            src={thumb}
            alt=""
            className="absolute inset-0 w-full h-full object-cover"
            loading="lazy"
            onError={() => setThumb(`https://i.ytimg.com/vi/${id}/hqdefault.jpg`)}
          />
          <span className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/20 to-black/30 transition-opacity group-hover:opacity-80" />

          <span className="absolute inset-0 flex items-center justify-center">
            <span className="flex items-center justify-center w-16 h-16 rounded-full bg-terminal-green text-black shadow-lg transition-transform group-hover:scale-110">
              {/* Play triangle, drawn rather than typed: the ▶ glyph renders at a
                  different size and baseline on every platform. */}
              <svg width="22" height="24" viewBox="0 0 22 24" fill="currentColor" aria-hidden>
                <path d="M21 12 0 24V0z" />
              </svg>
            </span>
          </span>

          <span className="absolute left-0 right-0 bottom-0 p-3 sm:p-4 text-left">
            <span className="block text-[12px] sm:text-sm font-bold text-slate-100 leading-snug">
              {title}
            </span>
            <span className="mt-1 block text-[10px] text-slate-300/80">
              Click to play · loads only when you press play
            </span>
          </span>
        </button>
      )}
    </div>
  );
}
