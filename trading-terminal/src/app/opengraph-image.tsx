import { ImageResponse } from "next/og";

/**
 * Social share card, rendered once at build time.
 *
 * The site is a static export, so this is generated during `next build` rather
 * than per-request — which is why it carries no live data. It exists because a
 * link with no image is a link nobody clicks, and this product is shared
 * through X and Telegram, where the card IS the pitch.
 */
export const alt = "Tennis Alpha — live win probability, edge and Kelly stakes for professional tennis";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OgImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%", height: "100%", display: "flex", flexDirection: "column",
          justifyContent: "center", padding: "72px",
          background: "#0a0e17", color: "#e2e8f0", fontFamily: "monospace",
        }}
      >
        {/* The mark is drawn, not typed: next/og ships no font covering ◉ and
            it rendered as a tofu box. */}
        <div style={{ display: "flex", alignItems: "center", gap: 16, color: "#4CA85E", fontSize: 34 }}>
          <div style={{
            width: 26, height: 26, borderRadius: 13,
            border: "6px solid #4CA85E", display: "flex",
          }} />
          <span>tennis alpha</span>
        </div>
        <div style={{ fontSize: 62, fontWeight: 700, color: "#f1f5f9", lineHeight: 1.15, marginTop: 26 }}>
          True probabilities for every
        </div>
        <div style={{ fontSize: 62, fontWeight: 700, color: "#22c55e", lineHeight: 1.15 }}>
          professional tennis match.
        </div>
        <div style={{ fontSize: 26, color: "#94a3b8", marginTop: 30, lineHeight: 1.4 }}>
          Live True P · edge vs de-vigged odds · ¼-Kelly stakes · hedge timing
        </div>
        <div style={{ display: "flex", gap: 18, marginTop: 34, fontSize: 22, color: "#475569" }}>
          <span>ATP</span><span>·</span><span>WTA</span><span>·</span>
          <span>Challenger</span><span>·</span><span>W125</span><span>·</span><span>ITF</span>
        </div>
      </div>
    ),
    size,
  );
}
