/** @type {import('tailwindcss').Config} */

/**
 * Design tokens. The single source of values for the UI — see /DESIGN.md.
 *
 * The `terminal.*` names are kept as ALIASES onto the same values rather than
 * deleted: they appear in roughly 8,000 lines of existing components, and
 * renaming them all in one pass would produce an unreviewable diff with real
 * regression risk in a product that trades money. New code uses the semantic
 * names; the old names resolve to identical colours, so both can coexist while
 * screens migrate one at a time.
 */

const palette = {
  bg: "#0a0e17",
  surface: "#111827",
  elevated: "#161f30",
  border: "#1e293b",
  borderStrong: "#334155",

  // Text. `muted` was #475569 (2.55:1 on bg) — a WCAG AA failure used for the
  // majority of small labels in the app. It is now #94a3b8 (7.53:1).
  text: "#e2e8f0",
  textStrong: "#f1f5f9",
  textMuted: "#94a3b8",
  textFaint: "#64748b",

  primary: "#22c55e",
  accent: "#06b6d4",
  warning: "#eab308",
  danger: "#ef4444",
  info: "#3b82f6",
};

module.exports = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // Small phones (iPhone SE / mini) sit below this; labels collapse to
      // their icon under it so the terminal nav never wraps.
      screens: { xs: "400px" },

      colors: {
        bg: palette.bg,
        surface: palette.surface,
        elevated: palette.elevated,
        border: palette.border,
        "border-strong": palette.borderStrong,

        content: palette.text,
        "content-strong": palette.textStrong,
        "content-muted": palette.textMuted,
        "content-faint": palette.textFaint,

        primary: palette.primary,
        accent: palette.accent,
        warning: palette.warning,
        danger: palette.danger,
        info: palette.info,

        // ── Legacy aliases (same values) ──
        terminal: {
          bg: palette.bg,
          panel: palette.surface,
          elevated: palette.elevated,
          border: palette.border,
          muted: palette.textMuted,
          green: palette.primary,
          red: palette.danger,
          yellow: palette.warning,
          blue: palette.info,
          cyan: palette.accent,
        },
        // `slate-*` is used ~278 times for text. Pinning the handful of steps
        // actually used onto the token values keeps those screens consistent
        // with the system while they migrate, instead of running a second
        // uncontrolled palette alongside it.
        slate: {
          100: palette.textStrong,
          200: palette.text,
          300: palette.text,
          400: palette.textMuted,
          // Deliberately NOT textFaint: legacy `text-slate-500` carries real
          // secondary content, and textFaint (4.06:1) fails AA for small text.
          // Mapping it to muted makes every un-migrated use compliant by
          // construction rather than one screen at a time.
          500: palette.textMuted,
          600: palette.borderStrong,
          700: palette.border,
        },
      },

      // Seven steps, replacing 14 arbitrary px sizes. 7px/8px are gone: they
      // were used 114 times and are not legible on any display.
      fontSize: {
        micro: ["10px", { lineHeight: "14px", letterSpacing: "0.06em" }],
        xs: ["11px", { lineHeight: "16px" }],
        sm: ["12px", { lineHeight: "18px" }],
        base: ["13px", { lineHeight: "20px" }],
        md: ["15px", { lineHeight: "24px" }],
        lg: ["17px", { lineHeight: "26px" }],
        xl: ["22px", { lineHeight: "30px" }],
        // Display steps — marketing headings and the single largest figure on
        // a screen. Listed so they are part of the system rather than falling
        // through to Tailwind's defaults unnoticed.
        "2xl": ["28px", { lineHeight: "34px" }],
        "3xl": ["34px", { lineHeight: "40px" }],
      },

      borderRadius: {
        sm: "3px",   // controls
        DEFAULT: "3px",
        md: "6px",   // containers
        lg: "6px",   // alias — prevents a fourth radius creeping back in
      },

      boxShadow: {
        // The only shadow in the system. Things that float, nothing else.
        overlay: "0 16px 40px -12px rgba(0,0,0,0.7), 0 0 0 1px rgba(30,41,59,0.9)",
      },

      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', "monospace"],
        sans: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
        serif: ['"Source Serif 4"', "Georgia", "serif"],
      },

      transitionDuration: {
        fast: "120ms",
        base: "200ms",
      },
      transitionTimingFunction: {
        standard: "cubic-bezier(0.2, 0, 0, 1)",
      },
    },
  },
  plugins: [],
};
