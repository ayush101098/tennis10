# AGENTS.md

Working agreement for anyone — human or agent — changing this repository.

Read `/DESIGN.md` before touching the UI and `/docs/ui-qa.md` before shipping
it. This file covers how the codebase is organised and the rules that are easy
to break without noticing.

---

## What this product is

A tennis trading terminal. It prices live matches with a model (neural network
⊕ Elo pre-match, a score-conditioned Markov engine once live) and compares that
price against a live market to find edge.

**People risk money on the numbers this app displays.** That single fact drives
most of the rules below: a number shown confidently that the model is not
confident about is a product defect, not a rounding detail.

---

## Layout

```
trading-terminal/          Next.js app (the product)
  src/app/                 routes — /, /terminal, /calculator, /manual, /resources, /tt, /admin
  src/components/ui/       the design system primitives — start here
  src/components/          feature components
  src/lib/                 pricing, schedule, market and model logic
  src/lib/__tests__/       vitest unit tests
  netlify/functions/       the production backend
  tests/visual/            Playwright visual + a11y QA
execution/                 Python trading pipeline (Polymarket)
DESIGN.md                  the visual language — the source of truth
docs/ui-qa.md              per-screen QA checklist
```

---

## Commands

```bash
npm run dev                # local dev on :3000
npm run test               # vitest — pricing, staking, parlay maths, name resolution
npm run test:visual        # Playwright — screenshots, axe, interaction contracts
npm run test:visual:update # re-baseline screenshots (review the diff first)
npx tsc --noEmit           # typecheck
npm run build              # production build
```

Both suites must pass before shipping. `tsc --noEmit` is not optional — the
static export will fail the build otherwise.

---

## UI rules

The full system is in `/DESIGN.md`. The ones most often broken:

1. **Tokens only.** No hex, no `slate-*`/`gray-*` in new code. Semantic names:
   `bg`, `surface`, `elevated`, `border`, `content`, `content-muted`,
   `primary`, `accent`, `warning`, `danger`, `info`.
2. **Use the primitives.** `ui/Button`, `ui/Field`, `ui/Panel`, `ui/Table`,
   `ui/Modal`, `ui/Tabs`, `ui/Icon`. If two things look similar, they should be
   the same component.
3. **No emoji as icons.** `ui/Icon` only. Emoji are unstyleable, unnameable and
   the clearest tell of generated UI.
4. **Never `outline: none`.** The global `:focus-visible` ring in `globals.css`
   is the only thing standing between this app and an unusable keyboard
   experience.
5. **Never fake de-emphasis with `opacity` on text.** It drops contrast below
   the accessible floor. Use `content-muted`.
6. **Colour must mean something.** Using `warning` to make a panel look
   important is a bug.
7. **Every data surface handles four states**: loading, empty, error, stale.
   An outage must never look like a quiet day.
8. **Numbers are `font-mono tabular-nums`, right-aligned**, formatted through
   the helpers in `ui/Table` so the same quantity is never formatted two ways.
9. **A toggle's accessible name does not change when it toggles** — that is
   what `aria-pressed` is for.

### Migration status

`terminal-*` and `slate-*` are **aliases onto the token values**, not a second
palette. They exist because the class names appear in ~8,000 lines, and a
single sweeping rename in a money-handling product is an unreviewable diff.

- New code: semantic names.
- Touching an old screen: migrate that screen.
- Do not open a rewrite of screens you are not otherwise changing.

Migrated so far: nav, hero, US Open value board, parlay builder, and every
`ui/` primitive. Everything else uses aliases and is correct but not yet
sharing component behaviour.

---

## Domain rules that are not style preferences

These encode money decisions. Do not "simplify" them away.

- **No prior, no opinion.** A match whose players cannot be ranked gets no
  value assessment. A coin flip against a market that knows the players is
  ignorance, not edge.
- **The 2% edge floor and ¼-Kelly / 5% cap are the product.** Never surface a
  bet below the floor as actionable.
- **Edges over 20% are quarantined, never recommended.** A gap that size means
  the data or the model is wrong. Show it, label it, do not bet it.
- **Live matches use live prices.** Comparing a score-conditioned probability
  against a pre-match price fabricates edge.
- **De-vig before comparing.** Raw implied probabilities include the spread;
  skipping this reads the spread as edge.
- **Parlays multiply model error along with payout.** The reality check beside
  the headline edge is not decoration — do not remove it to make the number
  look better.

### Known model limitation

The model currently sits ~13 percentage points from the market per leg on the
US Open card, and about 14pp above it on favourites — measured, not estimated
(see `TENNIS_DATA_SOURCES_AUDIT.md`). That is model error, not edge. Until it
is recalibrated against closing lines, the boards are best read as a
disagreement monitor. **Do not add UI that presents these edges as more
certain than they are.**

---

## Data sources

- **SofaScore** — schedule, live scores, point-by-point, statistics. Its
  **odds** endpoints currently 403, and the daily bulk odds feed returns a
  stale, disjoint event-id space, so odds cannot be joined from it.
- **Polymarket** — the working price of record, and the venue the pipeline
  trades. `lib/pmValue.ts` prices the board from it.
- **ESPN** — 403s server requests and contributes no individual matches. Dead
  weight; do not build on it.
- `public/rankings.json` is regenerated by `generate_rankings.py` and feeds the
  Elo prior. It goes stale silently — check its date when priors look wrong.

The SofaScore feed names players `"Borges N."`, not `"Nuno Borges"`. Name
resolution lives in `scheduleService.lookupEntry` and `polymarket.surname`, both
unit-tested. Breaking either silently unprices the entire board.

---

## Testing

- **Unit tests for anything that produces a number a user might bet on.**
- **Assert behaviour, not appearance** — `aria-pressed`, not a colour class.
- **Do not assert a property you have not verified.** A test that passes for
  the wrong reason is worse than no test: an earlier parlay test asserted that
  longer tickets tolerate less model error, which is false, and passed only on
  floating-point noise.
- **Visual baselines must be reproducible.** Stub every network source and
  freeze the clock (`tests/visual/fixtures.ts`). A flaky visual test gets muted,
  and a muted test is worse than none.
- When a fixture makes a feature render empty, fix the fixture — a test that
  skips itself is not coverage.

---

## Conventions

- Comments explain **why**, not what. The non-obvious constraint, the failure
  that motivated the code, the thing the next person will otherwise undo.
- Match the surrounding style rather than introducing a new one.
- Commit only when asked. Branch first if on `main`.
