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
        {/* The TA monogram, drawn with primitives: next/og has no SVG import
            and no font covering the dial glyphs, so the mark is composed from
            boxes here rather than referenced. */}
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          <div style={{ display: "flex", position: "relative", width: 54, height: 54 }}>
            <div style={{ position: "absolute", left: 2, top: 12, width: 30, height: 6, background: "#F1F5F9", display: "flex" }} />
            <div style={{ position: "absolute", left: 14, top: 12, width: 6, height: 32, background: "#F1F5F9", display: "flex" }} />
            <div style={{ position: "absolute", left: 26, top: 44, width: 10, height: 6, background: "#22c55e", display: "flex", transform: "rotate(-72deg)" }} />
            <div style={{ position: "absolute", left: 30, top: 12, width: 6, height: 32, background: "#22c55e", display: "flex", transform: "rotate(-18deg)" }} />
            <div style={{ position: "absolute", left: 42, top: 12, width: 6, height: 32, background: "#22c55e", display: "flex", transform: "rotate(18deg)" }} />
          </div>
          <span style={{ color: "#4CA85E", fontSize: 30, letterSpacing: 4 }}>TENNIS ALPHA</span>
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
