"use client";

import { X_URL, TELEGRAM_URL } from "@/lib/brand";

/** X (Twitter) mark — inline so a 12px glyph costs no request. */
function XIcon({ size = 13 }: { size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="currentColor" aria-hidden="true">
      <path d="M18.9 2H22l-7.4 8.5L23 22h-6.8l-5.3-7-6.1 7H1.7l7.9-9.1L1 2h7l4.8 6.4L18.9 2Zm-1.2 18h1.9L7.4 3.9H5.4L17.7 20Z" />
    </svg>
  );
}

function TelegramIcon({ size = 13 }: { size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="currentColor" aria-hidden="true">
      <path d="M21.9 4.3 18.9 19c-.2 1-.8 1.3-1.7.8l-4.6-3.4-2.2 2.1c-.2.3-.5.5-1 .5l.3-4.7 8.5-7.7c.4-.3-.1-.5-.6-.2L6.9 13 2.4 11.6c-1-.3-1-1 .2-1.4l18-6.9c.8-.3 1.5.2 1.3 1Z" />
    </svg>
  );
}

/**
 * Social links.
 *
 * Telegram is rendered ONLY when TELEGRAM_URL is set — shipping a placeholder
 * href would put a dead link in front of every visitor. Fill it in
 * src/lib/brand.ts and it appears everywhere at once.
 */
export default function Socials({ variant = "nav" }: { variant?: "nav" | "footer" }) {
  const nav = variant === "nav";
  const cls = nav
    ? "inline-flex items-center justify-center w-9 h-9 rounded border border-terminal-border text-terminal-muted hover:text-slate-100 hover:bg-terminal-panel transition"
    : "inline-flex items-center gap-1.5 min-h-[36px] px-2 text-terminal-muted hover:text-slate-300 transition";

  return (
    <div className={`flex items-center ${nav ? "gap-1.5" : "gap-3"}`}>
      <a href={X_URL} target="_blank" rel="noreferrer" aria-label="Tennis Alpha on X" className={cls}>
        <XIcon />{!nav && <span>@future_jesse</span>}
      </a>
      {TELEGRAM_URL && (
        <a href={TELEGRAM_URL} target="_blank" rel="noreferrer" aria-label="Tennis Alpha on Telegram" className={cls}>
          <TelegramIcon />{!nav && <span>Telegram</span>}
        </a>
      )}
    </div>
  );
}
