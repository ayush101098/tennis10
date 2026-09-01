# Tennis Alpha — Design System

The single source of truth for the application's visual language. If a value is
not in this file, it should not appear in a component.

**The product is an instrument, not a landing page.** It prices live tennis
matches against live markets. Density, alignment and legibility of numbers
outrank decoration everywhere. The visual language is that of a trading
terminal: flat surfaces separated by hairlines, monospace tabular figures, and
colour used only to carry meaning (direction, state, risk) — never to dress a
page up.

---

## 0. Audit findings this system exists to fix

Measured on the codebase before this system was introduced:

| Finding | Evidence | Resolution |
|---|---|---|
| Muted text fails contrast | `terminal-muted` `#475569` = **2.55:1** on background — WCAG AA needs 4.5:1 — used for the majority of small labels | `text-muted` is now `#94a3b8` (**7.53:1**) |
| No type scale | **536** arbitrary `text-[Npx]` uses across **14** distinct sizes (7, 8, 9, 10, 11, 11.5, 12, 12.5, 13, 14, 15, 18, 22, 26px) | 7 named steps; 7px and 8px text removed entirely |
| Two parallel colour systems | **278** `text-slate-*` uses alongside `terminal-*` tokens, plus **20** raw hex values in TSX | One semantic token set |
| No keyboard focus | **92** buttons, **28** inputs, **zero** `focus-visible` rings | Global focus ring on every interactive element |
| Emoji as icons | **164** emoji used as UI iconography | SVG icon set (`ui/Icon`) |
| Undocumented radii | `rounded` (180) / `rounded-lg` (27) / `rounded-full` (32) / `rounded-t` (1) mixed arbitrarily | 3 tokens: `sm`, `md`, `full` |

What was **not** wrong, and was deliberately left alone: shadows (only 4 uses),
gradients (2 files), animation (15 uses). The app was not over-decorated. Its
problems were consistency and accessibility, not excess.

---

## 1. Colour

Semantic tokens only. **Never** write a hex value or a `slate-*` / `gray-*`
class in a component.

### Surfaces

| Token | Value | Use |
|---|---|---|
| `bg` | `#0a0e17` | Page background. The only true black-ish surface. |
| `surface` | `#111827` | Panels, table headers, cards that need separation from the page. |
| `elevated` | `#161f30` | Modals, popovers, dropdowns — anything floating above the page. |
| `border` | `#1e293b` | Default hairline. Separates everything. |
| `border-strong` | `#334155` | Emphasised separation: active tab, selected row, focused control. |

### Text

| Token | Value | Contrast on `bg` | Use |
|---|---|---|---|
| `text-strong` | `#f1f5f9` | 16.9:1 | Headings, key figures, the number the row exists for. |
| `text` | `#e2e8f0` | 15.7:1 | Default body and table content. |
| `text-muted` | `#94a3b8` | 7.5:1 | Labels, captions, secondary metadata. |
| `text-faint` | `#64748b` | 4.1:1 | **Non-essential only** — disabled text, decorative rules. Fails AA for body text; never use it to carry information. |

### Meaning

Colour encodes state. It is never chosen for looks.

| Token | Value | Contrast | Means |
|---|---|---|---|
| `primary` | `#22c55e` | 8.5:1 | The model's position, positive edge, confirmation, primary action. |
| `accent` | `#06b6d4` | 8.0:1 | Market/venue data, links, secondary emphasis. |
| `warning` | `#eab308` | 10.1:1 | Needs attention before acting — suspect data, stale feed, live volatility. |
| `danger` | `#ef4444` | 5.1:1 | Loss, negative edge, destructive action, hard errors. |
| `info` | `#3b82f6` | 5.3:1 | Neutral system information. |

Every meaning colour passes AA at 4.5:1 on both `bg` and `surface`.

**Rules**
- Green means *the model likes it*, never merely "good". Red means *money down
  or data wrong*, never merely "stop".
- Colour is never the only signal. Every coloured state also carries a label,
  glyph or position — required for the ~8% of men with colour-vision
  deficiency, which for a red/green P&L product is the whole audience.
- Tinted backgrounds use the meaning colour at 8–12% alpha over `surface`, with
  the border at 30–40%. No other tint strengths.

---

## 2. Typography

Three families, each with one job. This is the product's existing voice,
formalised rather than replaced.

| Family | Role |
|---|---|
| **JetBrains Mono** | All data: prices, probabilities, percentages, scores, table figures, column labels. Tabular figures keep columns aligned — the reason the terminal is monospace at all. |
| **IBM Plex Sans** | All interface text: buttons, labels, navigation, body copy, prose. |
| **Source Serif 4** | Marketing headings only (`.marketing h1/h2/h3`). It signals research rather than tipster. It never appears in the terminal or in any control. |

### Scale

Seven steps. Nothing between them.

| Token | Size / line-height | Use |
|---|---|---|
| `text-micro` | 10px / 14px | Uppercase column headers and micro-labels **only**. Never sentences. Always `+0.06em` tracking. |
| `text-xs` | 11px / 16px | Dense table cells, secondary metadata in a row. The floor for anything a user must read. |
| `text-sm` | 12px / 18px | Default terminal body — rows, panel content, controls. |
| `text-base` | 13px / 20px | Emphasised row content, dense prose. |
| `text-md` | 15px / 24px | Marketing body copy, modal text. |
| `text-lg` | 17px / 26px | Section subheadings, key figures in a stat block. |
| `text-xl` | 22px / 30px | Panel titles, modal titles, the largest number on screen. |
| `text-2xl` | 28px / 34px | Display — section headings on marketing pages. |
| `text-3xl` | 34px / 40px | Display — the largest text on any screen. |

Marketing headings scale fluidly and sit outside this table:
`h1: clamp(30px, 4.2vw, 46px)`, `h2: clamp(22px, 3vw, 32px)`, `h3: 18px`.

**Rules**
- **7px and 8px text is banned.** It was used 114 times and is not legible.
  11px is the floor for content; 10px is permitted only for uppercase labels,
  which read larger than their nominal size.
- Numbers are **always** `font-mono` with `tabular-nums`. A column of figures
  that does not align is a bug.
- Weight carries hierarchy before size does: `600` for emphasis, `700` for the
  single most important element in a group. Never `800`/`900`.
- Maximum three type sizes in any one component. If you need a fourth, the
  component is doing too much.

---

## 3. Spacing

The 4px grid. Use Tailwind's default scale, which already matches:

`4 · 8 · 12 · 16 · 24 · 32 · 48 · 64` → `1 · 2 · 3 · 4 · 6 · 8 · 12 · 16`

**Rules**
- No arbitrary spacing (`p-[13px]`). If the grid does not fit, the layout is
  wrong, not the grid.
- Standard paddings: table cell `px-3 py-2`; panel body `p-4`; panel header
  `px-4 py-2.5`; section gap `py-12` (mobile) / `py-16` (desktop).
- Related things sit closer than unrelated things. A label 4px from its value
  and 16px from the next pair needs no divider.

---

## 4. Radius

| Token | Value | Use |
|---|---|---|
| `rounded-sm` | 3px | Controls: buttons, inputs, badges, tabs. |
| `rounded-md` | 6px | Containers: panels, modals, cards. |
| `rounded-full` | — | Dots, pills, avatars only. |

Nothing else. No `rounded-xl`, no `rounded-2xl`. Large radii read as consumer
software and fight the instrument.

---

## 5. Elevation

Borders do the work of separation; shadows are reserved for things that
genuinely float.

| Token | Use |
|---|---|
| *(none)* | Default. Panels and cards separate with `border border-border`. |
| `shadow-overlay` | Modals, dropdowns, popovers, toasts — anything over content. |

**Never** put a shadow on a card in the page flow. That is the single most
reliable tell of a generated interface.

---

## 6. Buttons

One height system, four variants. All: `rounded-sm`, `font-medium`,
`text-sm`, `120ms` transition, focus ring per §12.

| Size | Height | Padding | Use |
|---|---|---|---|
| `sm` | 28px | `px-2.5` | Inside dense rows and panel headers. |
| `md` | 36px | `px-3.5` | Default. |
| `lg` | 44px | `px-5` | Primary page actions, mobile CTAs. |

| Variant | Rest | Hover | Active | Use |
|---|---|---|---|---|
| `primary` | `bg-primary` / black text | `opacity-90` | `opacity-80` | The one action that matters on the screen. |
| `secondary` | `bg-surface` / `border-border` / `text` | `border-border-strong` | `bg-elevated` | Everything ordinary. |
| `ghost` | transparent / `text-muted` | `text` + `bg-surface` | `bg-elevated` | Tertiary, toolbars, dismissals. |
| `danger` | `bg-danger` / white text | `opacity-90` | `opacity-80` | Destructive and irreversible only. |

**States**
- `disabled`: `opacity-50`, `cursor-not-allowed`, `pointer-events-none`.
  Disabled buttons must be accompanied by an explanation of what would enable
  them.
- `loading`: spinner replaces the label's leading icon, **width is preserved**
  so the layout does not jump; button is `aria-busy` and non-interactive.
- Touch targets are ≥44px on coarse pointers regardless of visual size.

**Rules**
- One `primary` per view. Two primaries mean neither is.
- Buttons that do the same thing look the same everywhere.
- Label with a verb (`Start free trial`), never a noun (`Free trial`).

---

## 7. Inputs

| Property | Value |
|---|---|
| Height | 36px (`sm`: 28px inline) |
| Padding | `px-3` |
| Background | `bg` (recessed against `surface`) |
| Border | `border-border` |
| Radius | `rounded-sm` |
| Text | `text-sm`; **16px on coarse pointers** — iOS zooms and never restores below that |
| Focus | `border-accent` + focus ring; never `outline: none` alone |
| Error | `border-danger` + message below, wired via `aria-describedby` |
| Disabled | `opacity-50`, `cursor-not-allowed` |

Every input has a `<label>` or an `aria-label`. Placeholder text is not a
label — it disappears exactly when the user needs it.

Numeric inputs use `font-mono` and `inputMode="decimal"`.

---

## 8. Cards

A card is a container that groups content **and has a boundary that means
something**. Most information does not need one.

**Use a card when:** the group is independently actionable or dismissible; it
sits over the page (modal, popover); or it is one repeated item in a
collection of peers.

**Do not use a card when:** it holds a single statistic (use a stat cell in a
bordered row); it wraps a whole page section (use spacing and a heading);
it contains one paragraph; or it is nested inside another card.

Nested cards are banned. Wrapping every element in a bordered rounded box is
the most recognisable generated-UI pattern and it destroys hierarchy — if
everything is separated, nothing is grouped.

---

## 9. Tables

Tables are the primary interface of this product.

| Property | Value |
|---|---|
| Row height | 36px default, 32px dense |
| Header | `text-micro`, uppercase, `text-muted`, `bg-surface`, `border-b border-border`, sticky |
| Cell padding | `px-3 py-2` |
| Row separator | `border-b border-border`; last row none |
| Hover | `bg-surface/60` |
| Selected | `bg-accent/10` + 2px `border-l border-accent` |
| Alignment | Text left; **all numbers right, `font-mono`, `tabular-nums`** |

**Numeric formatting**
- Probabilities/percentages: 1 decimal (`64.2%`); whole numbers in dense cells.
- Odds: 2 decimals (`2.10`).
- Money: no decimals under $1,000; thousands separated.
- Signed values carry the sign (`+4.1%`), and the colour matches the sign.
- Missing data is an em-dash `—` in `text-faint`, never `0`, `N/A` or blank.

**Empty state**: never a blank panel. Say what would appear here, and why it
is not there yet (§11).

Below `640px` a table either scrolls horizontally in its own container — never
the page — or becomes a stacked list. It never squeezes columns to illegibility.

---

## 10. Layout

| Property | Value |
|---|---|
| Content max-width | `1180px` (boards, dense views) |
| Prose max-width | `720px` — anything read as sentences |
| Page padding | `16px` mobile, `24px` ≥640px |
| Section rhythm | `py-12` mobile, `py-16` desktop |
| Header | 56px, sticky, `bg/95` + `backdrop-blur` |

**Breakpoints** (Tailwind defaults plus `xs`): `xs 400 · sm 640 · md 768 ·
lg 1024 · xl 1280`.

Mobile is not a smaller desktop: the board becomes a list, the analysis pane
moves below the list rather than beside it, and nested scroll containers are
removed so the page scrolls once.

---

## 11. States

Every data surface defines four states. A component that only handles the
success case is unfinished.

- **Loading** — skeletons matching final layout, or a labelled indicator.
  Never a bare spinner in an empty panel; never a layout that jumps on arrival.
- **Empty** — a heading, one sentence of cause, and the action that changes it.
  "No US Open matches on today's card" beats "No data".
- **Error** — what failed, whose fault it is, and whether it self-recovers.
  Distinguish *outage* from *genuinely nothing*: they look identical to a user
  and must not. Never show a raw exception.
- **Stale** — data present but not current. Say so inline; do not silently
  render an old number as if it were live. In a trading product a stale price
  shown as live is the most expensive possible bug.

---

## 12. Focus & accessibility

Non-negotiable, and the largest gap this system closes.

- Every interactive element shows `:focus-visible` — a 2px `accent` ring at
  2px offset. Applied globally in `globals.css`; do not remove it locally.
- Focus order follows visual order. No positive `tabindex`.
- Anything clickable is a `<button>` or `<a>`. A clickable `<div>` needs
  `role`, `tabIndex={0}` and Enter/Space handlers — prefer the real element.
- Icon-only controls require `aria-label`.
- Contrast: 4.5:1 for text, 3:1 for UI boundaries and large text.
- Live regions (`aria-live="polite"`) for values that change without
  interaction, so a screen reader is not silent while prices move.
- Respect `prefers-reduced-motion`: all non-essential motion is removed.

---

## 13. Motion

Motion communicates change. It never decorates.

| Token | Value | Use |
|---|---|---|
| `duration-fast` | 120ms | Hover, focus, colour changes. |
| `duration-base` | 200ms | Expand/collapse, panel transitions. |
| Easing | `cubic-bezier(0.2, 0, 0, 1)` | Everything. |

**Allowed**: opacity and colour transitions; height for disclosure; a 2s pulse
on a genuine live indicator; a shimmer on skeletons.

**Banned**: bounce, float, parallax, glow, scale-on-hover over 1.02, entrance
animations on page load, anything looping that is not communicating live state.

All of it is disabled under `prefers-reduced-motion: reduce`.

---

## 14. Iconography

SVG only, from `ui/Icon`. 16px default on a 16px grid, `1.5px` stroke,
`currentColor`.

**Emoji are not icons.** They render differently on every platform, cannot be
recoloured, carry no accessible name, and are the clearest signal of a
generated interface. The 164 emoji in the original code are being replaced.

The one exception is genuine content — a flag in a player's name, say — which
is data, not iconography.

---

## 15. Conflicts with prior decisions

Documented per the brief, with the resolution chosen:

1. **9px/10px terminal text vs. legibility.** The terminal legitimately trades
   size for density, but 7–8px is illegible on any display. *Resolution:* 11px
   floor for content, 10px for uppercase labels only. Rows grew ~1px; no
   column was lost.
2. **Three font families.** More than a strict system would allow.
   *Resolution:* kept, because each has a distinct and enforced job (data /
   interface / marketing headings). The serif is confined to marketing headings
   and never enters the product surface.
3. **Emoji carried meaning in dense rows** where a text label would not fit.
   *Resolution:* replaced with SVG icons at the same footprint, which also gain
   accessible names.

4. **Marketing prose styles vs. product chrome.** Product panels live inside
   `.marketing` on the landing page, and `.marketing p` (specificity 0,1,1)
   silently beat every `text-*` utility (0,1,0) — panel titles rendered as 32px
   serif and panel copy at 15px regardless of the classes the component set.
   *Resolution:* the marketing defaults are wrapped in `:where(.marketing)`,
   dropping them to element specificity (0,0,1) so any utility wins while
   unclassed prose is still styled. Prefer this over per-component escape
   hatches: it fixes every element, not the ones someone remembered to mark.
5. **`terminal-*` class names are used in ~8,000 lines.** *Resolution:* the
   token names are kept as aliases so the migration is incremental rather than
   a single unreviewable rewrite. New code uses semantic names; `terminal-*`
   maps onto the same values. See `AGENTS.md` for migration status.
