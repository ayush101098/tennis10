# UI QA Checklist

Run this against any screen before shipping a change to it. It is deliberately
a *manual* checklist: the automated suite (`npm run test:visual`) catches
regressions in things that already work, and catches roughly a third of
accessibility problems. It cannot tell you whether a screen makes sense.

**Automated first, then this.**

```bash
npm run test          # unit — pricing, staking, parlay math, name resolution
npm run test:visual   # Playwright — screenshots, a11y, interaction contracts
```

---

## Screen inventory

| Screen | Route | Migrated to the design system | Notes |
|---|---|---|---|
| Landing | `/` | Partial — nav, hero, value board, parlay builder | The rest of the page uses legacy `terminal-*` aliases, which resolve to the same token values |
| Terminal | `/terminal` | Not yet — legacy classes | Gated; opens the pricing modal for signed-out visitors |
| Calculator | `/calculator` | Tokens only | Contrast and link fixes applied |
| Manual | `/manual` | Tokens only | Markdown-rendered; scroll regions are keyboard-reachable |
| Resources | `/resources` | Tokens only | |
| Table tennis | `/tt` | Not yet — legacy classes | |
| Admin | `/admin` | Tokens only | Internal |

Legacy screens are **not broken** — `terminal-*` and `slate-*` map onto the
same token values (see `tailwind.config.js`), so they inherit the corrected
contrast and radii automatically. What they do not yet have is the shared
component behaviour. Migrate a screen when you are changing it anyway; do not
open a rewrite for its own sake.

---

## Visual

- [ ] **Alignment** — one grid. Nothing is off by a pixel or two from the thing above it.
- [ ] **Spacing** — every value from the 4px scale. No `p-[13px]`.
- [ ] **Hierarchy** — squint at it. The most important thing is still the most prominent.
- [ ] **Typography** — at most three sizes in a component; all from the scale; no 7–8px text.
- [ ] **Numbers** — `font-mono`, `tabular-nums`, right-aligned; columns line up exactly.
- [ ] **Colour** — semantic tokens only. No hex, no `slate-*` in new code.
- [ ] **Colour carries meaning** — green = model likes it, red = loss or bad data. Never decorative. A state colour used for prominence is a bug.
- [ ] **Radius** — `sm` on controls, `md` on containers. Nothing else.
- [ ] **Elevation** — no shadow on anything in the page flow.
- [ ] **No nested cards** — a bordered box inside a bordered box means the grouping is wrong.
- [ ] **Density** — matches neighbouring screens. A row here looks like a row there.

## Interaction

- [ ] **Hover** — every interactive element responds; nothing non-interactive does.
- [ ] **Focus** — Tab through the whole screen. The ring is visible on every stop, never clipped by `overflow: hidden`.
- [ ] **Focus order** — follows visual order.
- [ ] **Active** — pressing gives immediate feedback.
- [ ] **Disabled** — looks disabled, is not focusable, and the reason is stated somewhere.
- [ ] **Loading** — skeleton matches the final layout; nothing jumps on arrival; `aria-busy` set.
- [ ] **Error** — says what failed, whose fault, whether it recovers. No raw exceptions.
- [ ] **Success** — confirmed visibly; destructive actions confirmed before, not after.
- [ ] **Toggle state** — carried by `aria-pressed`/`aria-selected`, and the accessible name does **not** change when it flips.

## States

Check all four for every data surface. Force them — do not assume.

- [ ] **Loading** — throttle the network.
- [ ] **Empty** — stub the source to return nothing. Does it explain itself?
- [ ] **Error** — make the source fail. Is an outage distinguishable from "genuinely nothing"?
- [ ] **Stale** — is out-of-date data labelled as out of date? *In a trading product a stale price shown as live is the most expensive bug possible.*

## Responsive

Test at 390 (phone), 768 (tablet), 1280 (laptop), 1920 (desktop).

- [ ] **No horizontal page scroll** at any width. Wide content scrolls inside its own container.
- [ ] **Navigation adapts** — collapses without losing anything.
- [ ] **Tables** — scroll in their own container or become a stacked list. Never squeezed to illegibility.
- [ ] **Cards don't become towers** on mobile.
- [ ] **Type scales** — nothing under 11px; inputs at 16px on touch (iOS zooms below that and never zooms back).
- [ ] **Touch targets ≥ 44px** on coarse pointers.
- [ ] **Hierarchy survives** — the most important thing is still first and still obvious.
- [ ] **Nothing hover-only** — touch devices have no hover, so anything only in a tooltip does not exist on a phone.

## Accessibility

- [ ] `npm run test:visual` passes (axe, wcag2a/2aa/21a/21aa, critical + serious).
- [ ] **Keyboard-only pass** — complete the screen's primary task without a mouse.
- [ ] **Escape** closes dialogs; focus returns to the trigger.
- [ ] **Focus trapped** in modals; Tab cannot reach the page behind.
- [ ] **Contrast** — 4.5:1 text, 3:1 boundaries. Never fake de-emphasis with `opacity` on text: it silently drops contrast below the floor.
- [ ] **Every control has a name** — icon-only buttons need `aria-label`.
- [ ] **Every input has a label** — a placeholder is not a label.
- [ ] **Semantic HTML** — `<button>` for actions, `<a>` for navigation, real headings, no clickable `<div>`.
- [ ] **Heading outline** — one `h1`, no skipped levels.
- [ ] **Live values** announce (`aria-live="polite"`) — a screen reader should not be silent while prices move.
- [ ] **Reduced motion** respected.
- [ ] **Not colour alone** — every coloured state also has a label, sign or position.

## Product quality

The questions the automated suite cannot ask.

- [ ] **Is the primary action obvious** within two seconds? Is there exactly one?
- [ ] **Is anything here unnecessary?** Delete it and see if the screen is worse.
- [ ] **Is the density right** for the task — dense enough to compare, loose enough to read?
- [ ] **Does the screen answer the question the user arrived with?**
- [ ] **Does it look like the same product** as the screen before it?
- [ ] **Does any number mislead?** A figure shown confidently that the model is not confident about is a product defect, not a display detail.
- [ ] **Would a stranger think this was generated?** If yes, name the reason and fix it (§ below).

---

## The vibe-code audit

Run this last, as a reviewer who has never seen the code. Findings from the
last pass, and what was done:

| Tell | Found | Fixed |
|---|---|---|
| Emoji as icons | 164 across the app | SVG set in `ui/Icon`; migrated screens carry none |
| Everything in a rounded card | Panels nested inside panels on the landing page | `Panel` is the one container; nesting banned |
| Typography with no system | 536 arbitrary sizes, 14 steps | 9-step scale; 7–8px removed |
| Colour with no meaning | Board used the `warning` colour for prominence, not for warning | Neutral; prominence comes from position |
| Contradictory signals | A "STRONG" badge on a row inside "DO NOT BET" | Badge suppressed on quarantined rows |
| Misleading colour | Suspect row's edge rendered in positive accent | Rendered in `danger` |
| Marketing type leaking into product chrome | Panel titles rendered as 32px serif; panel body copy at 15px, ignoring `text-xs` | Marketing defaults wrapped in `:where()` so any utility class wins |
| Competing CTAs | Three button-shaped actions in the hero | One primary; the rest are links |
| Shouty labels | `LAUNCH TERMINAL →`, `SEE TODAY'S MATCHES ↓` | Sentence case, icon arrows |
| Fake de-emphasis | `opacity-70`/`opacity-60` on text | Removed — the group heading already says why |

Recheck each of these whenever a new screen is added.
