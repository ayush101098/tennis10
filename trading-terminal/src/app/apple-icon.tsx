import { ImageResponse } from "next/og";

/**
 * iOS home-screen icon.
 *
 * A PNG, not the SVG the browser tab uses: iOS does not render SVG touch icons,
 * so apple-icon.svg was simply 404ing. Built from primitives at build time —
 * next/og cannot import the mark file.
 */
export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div style={{
        width: "100%", height: "100%", display: "flex",
        alignItems: "center", justifyContent: "center", background: "#0a0e17",
      }}>
        <div style={{ display: "flex", position: "relative", width: 108, height: 108 }}>
          {/* T */}
          <div style={{ position: "absolute", left: 6, top: 30, width: 60, height: 12, background: "#F1F5F9", display: "flex" }} />
          <div style={{ position: "absolute", left: 30, top: 30, width: 12, height: 62, background: "#F1F5F9", display: "flex" }} />
          {/* A — two legs meeting at an apex */}
          <div style={{ position: "absolute", left: 58, top: 30, width: 12, height: 62, background: "#22c55e", display: "flex", transform: "rotate(-16deg)" }} />
          <div style={{ position: "absolute", left: 82, top: 30, width: 12, height: 62, background: "#22c55e", display: "flex", transform: "rotate(16deg)" }} />
        </div>
      </div>
    ),
    size,
  );
}
