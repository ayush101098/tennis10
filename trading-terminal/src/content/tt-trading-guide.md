# TT Terminal — Trading Guide

> How to actually use the 🏓 table-tennis terminal: what each number is, where the
> real edges are in these leagues, and the discipline rules. Written in the same
> spirit as the tennis manual's hard lesson — the model is a tool for finding
> mispriced lines, not an oracle. Every claim below is checkable against
> `tabletennis/site/metrics.json` (also shown live in the terminal header).

---

## 1. What you're looking at

The terminal covers the high-frequency factory leagues — **Setka Cup (UKR), Liga Pro
(RUS), Czech Liga Pro, TT Cup (POL), WTT** — the same markets the plan targets:
matches every 25–30 minutes, best-of-5 races to 11, all day long.

Two engines feed it:

| Layer | Where it shows | What it is |
|---|---|---|
| **Pre-match** | scheduled rows, edge panel | Walk-forward Elo ⊕ GBDT over 59k matches: form, game-win rate, streaks, H2H, rest hours, 24h load, experience |
| **Live True P** | 🔴 LIVE rows, sparkline, edge panel | Exact race-to-11/win-by-2 recursion conditioned on the current game + point score, **anchored to the pre-match model by inversion**, plus a character residual capped at ±15pp |

The character residual is the interesting part: per-player traits learned from how
they historically deviate from the analytic expectation — **clutch, deuce composure,
comeback, front-running, fatigue**. When it's active you'll see the 🧠 annotation,
e.g. `character −3% — Duch A. comeback trait (+0.11)`.

**Header health checks, every session:** the model tag + held-out accuracy must be
showing; if you see `⚠ live poller stale` or `⚠ pre-match file >12h old`, the
numbers on screen are dead — fix the pipeline (§6) before trading anything.

## 2. How good is the model, honestly

Held-out walk-forward validation, ~24k test rows (never random-split, features from
prior matches only — leakage-safe by construction):

| Model | Accuracy | Log loss |
|---|---|---|
| Coin flip | 50.0% | 0.6931 |
| Elo only | 55.4% | 0.6907 |
| **GBDT (shipped)** | **57.8%** | **0.6740** |

Calibration by bucket (predicted vs actual win rate):

| Predicted | n | Actually won |
|---|---|---|
| 50–60% | 18,184 | 55.3% |
| 60–70% | 4,502 | 63.7% |
| 70–80% | 845 | 70.7% |
| **80–100%** | **375** | **80.3% (pred 87.1%)** |

Read the last row twice: **the model is overconfident on its biggest favourites** —
exactly where the tennis engine burned us. Practical rule: haircut anything the
terminal shows above ~80%, and never take a heavy favourite at short odds just
because True P looks huge.

The live layer is honest too: the residual shipped only because it beat the pure
analytic baseline out-of-sample (log-loss 0.5522 → 0.5501 over 159k historical game
states). It's a small, real improvement — not magic.

## 3. Reading the Match Centre

Each row: time (or ● LIVE), players (favourite in white), league, and on live rows
the score as `games · points` (e.g. `1–2 g · 7–9` = down a game 1–2, trailing 7–9 in
the current game).

- **The big % is True P for player 1** — live it updates every ~8 seconds.
- **The sparkline** is the recent True P path. Choppy = swingy match; a staircase =
  one player steadily taking over.
- **▲/▼ ±pp** is live True P minus pre-match. This is the single most useful glance
  number: it tells you how much the market's opening price is now stale.

With no match selected, the right pane is the **Edge Board**:

- **💎 LIVE MOVERS** — biggest divergence from pre-match. These are where in-play
  prices are most likely to lag reality.
- **🧠 CHARACTER RESIDUALS IN PLAY** — matches where a player trait is actively
  shifting the number. A comeback trait firing while the scoreboard looks lost is
  precisely the spot recreational money overreacts to.
- **🎯 STRONGEST PRE-MATCH LEANS** — high-confidence scheduled picks with fair
  odds, for shopping against your book's opening lines.

## 4. The trade: where TT edges actually come from

There is no Polymarket for these leagues — you're pricing against a bookmaker, and
the journal is **paper until your book fills are real**. The realistic edge sources,
in order of quality:

1. **Stale in-play lines.** Factory-league matches are fast and books reprice
   lazily between games. The moment a game ends, compare the book's new line to
   live True P. A 30pp mover with the book still near pre-match is the trade.
2. **Deuce/clutch spots.** At 9–9+ the analytic recursion is exact while books
   shade to the favourite by reflex. If the residual also likes the underdog's
   deuce composure, the dog side is systematically a better price.
3. **Pre-match line shopping on high-confidence leans.** Only in the 60–75% band —
   that's where calibration is clean (see §2). Skip the 80%+ leans entirely.

**Mechanics, in the edge panel:**

1. Select the match, pick your side (the buttons show True P each way).
2. Type the **odds your book is actually showing** — never leave it on model-fair;
   fair odds by definition have zero edge and exist only as a reference.
3. Read the edge line. **Floor: +3%** for TT (higher than the tennis terminal's 2%
   — these books run 6–8% margins and the model is coarser here). Below the floor,
   don't. The discipline IS the system.
4. Size at **¼ Kelly, hard-capped at 2% of bankroll per bet**. The panel shows both
   Kelly numbers; TT variance is brutal — a best-of-5 to 11 is a coin that flips
   fast, and you'll have 40+ opportunities a day. Volume, not size, is where the
   EV compounds.
5. ⚡ LOG PAPER BET. Every bet, no exceptions — including the ones you'd rather
   forget. An unlogged journal is a lie you tell yourself.

## 5. The Bet Tracker is the product

The tennis terminal's most important discovery came from its journal: the engine's
claimed edge was anti-predictive (bigger shown edge → bigger losses). The TT journal
exists so we find that out **here** before real money does.

- Settle every bet honestly (WIN / LOSS / VOID) the moment the match ends.
- Watch **P@bet vs your actual hit rate**. If bets logged at ~65% are winning ~55%,
  the model is overconfident in your selection zone — widen the haircut, or stop.
- ROI over 50+ settled bets is signal. ROI over 10 is noise; do not resize on it.
- One league at a time when starting out. Setka and Liga Pro have different
  personalities (pace, walkover rates, motivation patterns) — learn one before
  spreading.

**Hard risk rules (non-negotiable):** max 2% of bankroll per bet · stop for the day
at −5 units · no bet without a logged edge ≥ +3% · nothing above True P 80% · never
chase a loss into the next 25-minute match — there is always another one, which is
exactly why tilt is more dangerous in TT than anywhere else.

## 6. Keeping the data alive (ops)

The terminal reads local pipeline artifacts via `/api/tt`. Three things must be
running:

```bash
python sofa_proxy.py                      # :3001 — TLS-impersonating data proxy
python -m tabletennis.pipeline            # daily: ingest → retrain → predictions.json
python -m tabletennis.live                # 8s in-play poller → live_predictions.json
```

The header tells you when any of these has died (`feed unreachable` / stale
warnings). Retrain roughly daily — Elo drifts fast in leagues where players log
hundreds of matches a month, and every state the live poller records grows the
training set for the momentum/serve traits that need point sequences.

---

*Model outputs are calibrated probabilities, not guarantees. These leagues exist to
generate betting volume; the books pricing them are not stupid. Bet only what you
can afford to lose.*
