# Manual Trading — Short Guide

> **Read this before the full [execution manual](./trading-manual.md).** The two disagree.
> This one is written from the settled journal (887 non-hedge bets); the other is
> written from theory. Where they conflict, this one has the receipts.

---

## 0. The finding that shapes everything below

Over 887 settled non-hedge bets, priced against real Polymarket lines:

| | Brier score (lower = better) |
|---|---|
| **The book's price** | **0.199** — sharp, calibrated in every bucket |
| **Our `true_p`** | **0.280** — worse than a 0.25 coinflip |

Our model says 72.7% on average. Those picks win **51%** of the time. Paper ROI: **−8.2%**.
Real fills would be worse — every one of those rows is a `simulated buy`, with no
slippage or fees taken out.

**And our edge number is anti-predictive:**

| Model's claimed edge | n | ROI |
|---|---|---|
| 0–5% | 200 | **+2.0%** |
| 5–10% | 182 | −3.7% |
| 10–20% | 156 | −11.0% |
| 20–40% | 198 | −6.4% |
| **40%+** | 151 | **−26.4%** |

Read that again. The *bigger* the edge the terminal shows you, the *more* you lose.
That is exactly what you'd expect: a big "edge" means our model maximally disagrees
with a market that is measurably sharper than we are. The disagreement is our error,
not our insight. Shrinking `true_p` toward 50% doesn't rescue it — every variant
still ranks bets by the same broken signal (−11% to −13% ROI at any shrink factor).

**So: the Edge Panel's edge %, the 🔥 STRONG signals, and the ¼ Kelly sizing in the
full manual are not a filter. Sizing up on them is sizing up on noise.** Until the
engine is recalibrated and re-validated out-of-sample, treat every number derived
from `true_p` as decoration.

---

## 1. How to filter bets

The honest filter right now is **almost everything fails it.** That is the correct
output of a sharp market, not a bug.

**Discard immediately:**
- Any bet whose only justification is a model edge. That's the anti-predictive signal above.
- Anything in the 40%+ edge bucket. That's the model breaking, and it's your worst ROI.
- Liquid ATP/WTA match-winner markets. Brier 0.199 means the book has already priced
  everything you know.

**The only thing worth looking at** is where you have information the book lacks —
not a different opinion, *information*. In practice that's narrow:
- **Stale lines on thin markets.** ITF/Challenger books update slowly. If you're
  watching a live feed and the price hasn't moved to a break that already happened,
  that's real, and it has nothing to do with our model.
- **Things the book can't see yet:** a visible injury, a medical timeout, conditions.
  You saw it, the price hasn't moved.

If you can't name the specific fact you know that the market doesn't, you don't have
a bet. "The model says 72%" is not such a fact.

---

## 2. Break and serious hold — what's actually sound

This is the one piece of the stack the calibration problem doesn't touch, and it's
worth being precise about why.

[`hold_prob_from_score`](../../execution/momentum.py) is **arithmetic, not prediction.**
Given a point-win-on-serve `p` and the live point score, it solves the per-point
Markov chain exactly — deuce collapses to the geometric tail `p²/(p²+q²)`. Feed it
0–40 and it tells you the server's hold chance. That's a calculation, and it's right.

```
break_prob_from_score(p, server_pts, returner_pts) = 1 − hold_prob_from_score(...)
```

The bot's thresholds ([sx_breakbot.py:50](../../execution/sx_breakbot.py#L50)):
- **Break play** — `break_prob ≥ 0.70`: the returner is about to break.
- **Serious hold** — `hold_prob ≥ 0.80`: the server is safe.

**Reading it at the table:**

| Situation | Why it matters |
|---|---|
| 0–40, 15–40 on serve | Break prob is high *by arithmetic* — no forecast involved |
| 40–0, 40–15 | Serious hold. The game is close to decided |
| Deuce | The geometric tail dominates. `p` barely separates players here — least edge, most variance |

**The catch, and it's the whole catch.** The break/hold number is only sound as a
statement about *this service game*. The bot then converts it into a **match-winner**
bet through `live_true_p` — and that's the miscalibrated model again. One likely break
is a weak match-level signal; `sx_breakbot`'s own docstring says so ("treat this as
plumbing/rehearsal, not proven +EV"). It's right.

So: **trust the break math for what it measures — a single game. Don't let it launder
a match-winner position.** And on a liquid book, a break point at 0–40 is already in
the price within a second. The arithmetic being correct doesn't make it valuable.

---

## 3. When to hedge

Hedging is mechanically sound. The formula in the full manual is correct:

```
Hedge stake = (your stake × your odds) / current opposite odds
```

But be clear about what a hedge can and cannot do:

**A hedge cannot rescue a bad entry.** It locks in whatever your position is worth
*now*. If you entered on a noise signal at a fair price, hedging converts a coinflip
into a **certain small loss** — you pay the spread on the way in and again on the way
out. The full manual's "guaranteed profit either way" only holds if the entry had
genuine value. On this journal's evidence, the entries mostly didn't.

**Hedge when:**
- The price moved your way for a reason **you can name** that isn't the model. You're
  taking money the market gave you, and that's real regardless of calibration.
- You're carrying more risk on one match than you meant to. Hedging as *risk reduction*
  is always legitimate. Hedging as *alpha* is not.

**Don't hedge when:**
- The only input is `true_p` dropping by `SX_HEDGE_DROP` (0.15). That's the broken
  model triggering a real cost. The 228 hedge legs in the journal are all negative-edge
  by our own model — that's expected for insurance, but insurance against a signal
  that's noise is just paying twice.

**The rule:** size so you never *need* to hedge. A hedge you were forced into is a
sizing error you already made.

---

## 4. If you trade this manually today

1. **Don't size off the edge number.** It's anti-predictive; that's measured, not theoretical.
2. **Assume the book is right** unless you can name what you know that it doesn't.
3. **Use the break/hold math for the game in front of you**, not to justify a match bet.
4. **Keep stakes small enough that hedging is a choice.**
5. **Paper everything, and run `python -m execution.settle` + `python -m execution.calibration`.**
   The moment a bucket shows large `n` *and* positive ROI, that's your first real signal.
   Nothing in the journal qualifies yet.

---

## 5. The actual fix

The engine needs recalibration before any of this is tradeable — an isotonic or Platt
fit of `true_p` against settled outcomes, validated **out-of-sample**, with the
market price as the baseline to beat. Right now we don't beat it; we lose to it by
0.08 Brier on the same events.

Note the bar: a recalibrated model that merely *matches* the book still has no edge.
Calibration is table stakes, not alpha. It makes the number honest — it doesn't make
it profitable.

---

*Sourced from 887 settled non-hedge bets in `tennis_betting.db` (all paper fills at
real Polymarket prices). Reproduce with `python -m execution.calibration`.*
