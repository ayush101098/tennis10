/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // Small phones (iPhone SE / mini) sit below this; labels collapse to
      // their icon under it so the terminal nav never wraps.
      screens: { xs: "400px" },
      colors: {
        terminal: {
          bg: "#0a0e17",
          panel: "#111827",
          border: "#1e293b",
          muted: "#475569",
          green: "#22c55e",
          red: "#ef4444",
          yellow: "#eab308",
          blue: "#3b82f6",
          cyan: "#06b6d4",
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', "monospace"],
      },
    },
  },
  plugins: [],
};
