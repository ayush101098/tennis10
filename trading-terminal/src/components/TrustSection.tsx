/**
 * Why anyone should believe the numbers.
 *
 * This replaces the methodology block that used to sit here — the one that
 * explained the neural network, the Markov re-pricing and the de-vigging. That
 * block answered "how does it work", which almost nobody asked. This one
 * answers "has it been right", which is the only question a stranger actually
 * has.
 *
 * EVERY FIGURE BELOW IS MEASURED, and the measurement is named so it can be
 * argued with. Nothing here is a projection, a backtest of a strategy that was
 * chosen after seeing the data, or a round number that felt persuasive. Where
 * the result is unflattering it is stated anyway — a page that only reports
 * wins is the page every tipster has, and it is why nobody believes them.
 *
 * TO KEEP IT HONEST AS IT GROWS
 *   The corpus grows daily, so these numbers go stale. They are constants here
 *   rather than a live query on purpose: a marketing claim should change when a
 *   person decides to re-measure and re-publish, not silently drift overnight.
 *   Re-run scripts/calibrate-game-tree.ts and edit this file.
 */

/** Measured 2026-09-18 — see scripts/calibrate-game-tree.ts. */
const RECORD = {
  measuredOn: "18 September 2026",
  games: "2,741",
  matches: "587",
  testMatches: "91",
  calibrationError: "1.6",
  worstBucket: "6.5",
};

export default function TrustSection() {
  return (
    <section id="track-record" className="marketing px-4 sm:px-6 py-14 max-w-[900px] mx-auto">
      <h2 className="text-center text-lg font-bold text-slate-100 mb-1">
        Has it actually been right?
      </h2>
      <p className="text-center text-[12px] text-terminal-muted mb-8 max-w-[560px] mx-auto">
        The only question worth answering. Here is how we check, what came back,
        and where we are not good enough to be worth listening to.
      </p>

      <div className="grid gap-3 sm:grid-cols-3 mb-8">
        <Stat
          figure={RECORD.games}
          label="games graded"
          sub={`across ${RECORD.matches} matches`}
        />
        <Stat
          figure={`${RECORD.calibrationError} pts`}
          label="average error"
          sub="when we say 40%, it happens about 40% of the time"
        />
        <Stat
          figure={RECORD.testMatches}
          label="matches held back"
          sub="graded on matches the model never saw"
        />
      </div>

      {/* The part that makes the rest believable. */}
      <div className="border border-terminal-border rounded-lg p-5 bg-terminal-panel/30 space-y-4">
        <div>
          <div className="text-[11px] font-bold text-terminal-green mb-1">HOW WE CHECK</div>
          <p className="text-[12px] text-slate-300 leading-relaxed">
            We keep every live match we have ever shown. To grade ourselves we
            replay them point by point, ask the model what it would have said at
            each moment, and compare that against what actually happened next.
            Crucially, the matches used to check are not the matches used to
            build — otherwise a model can score itself on its own homework.
          </p>
        </div>

        <div>
          <div className="text-[11px] font-bold text-terminal-yellow mb-1">WHERE WE ARE NOT GOOD ENOUGH</div>
          <p className="text-[12px] text-slate-300 leading-relaxed">
            In lopsided positions — a player two points from an easy hold — our
            numbers were overconfident by as much as 40 points before we
            corrected them, and even now we do not trust them at the extremes.
            So in those situations the terminal says <b className="text-slate-200">“too
            one-sided to call reliably”</b> and shows nothing. That is deliberate.
            A confident number we have not earned is worse than no number.
          </p>
        </div>

        <div>
          <div className="text-[11px] font-bold text-terminal-muted mb-1">WHAT THIS IS NOT</div>
          <p className="text-[12px] text-slate-300 leading-relaxed">
            Not a tipping service and not a profit claim. We publish
            probabilities and the evidence behind them. Whether that is worth
            money to you depends on what you do with it, and on prices we do not
            control.
          </p>
        </div>

        <p className="text-[10px] text-terminal-muted border-t border-terminal-border pt-3">
          Measured {RECORD.measuredOn}. Worst single bucket {RECORD.worstBucket} points off.
          These figures are re-measured by hand rather than generated live, so
          what you see is a number a person checked.
        </p>
      </div>

      {/* Deliberately empty until there is something real to put in it. An
          empty shelf is recoverable; a shelf of invented testimonials is not. */}
      <PublicCalls />
    </section>
  );
}

function Stat({ figure, label, sub }: { figure: string; label: string; sub: string }) {
  return (
    <div className="border border-terminal-border rounded-lg p-4 text-center bg-terminal-panel/20">
      <div className="text-2xl font-bold text-slate-100 font-mono tabular-nums">{figure}</div>
      <div className="text-[11px] font-bold text-slate-300 mt-0.5">{label}</div>
      <div className="text-[10px] text-terminal-muted mt-1 leading-snug">{sub}</div>
    </div>
  );
}

/**
 * Published calls, newest first.
 *
 * Fill CALLS as predictions are posted and settled. The shape is deliberately
 * awkward to fake: each entry needs the date it was published, what was said
 * BEFORE the match, and what happened — so a losing call cannot quietly be left
 * out without leaving a gap in the dates.
 */
const CALLS: { date: string; call: string; result: "won" | "lost"; link?: string }[] = [];

function PublicCalls() {
  if (!CALLS.length) {
    return (
      <div className="mt-6 text-center text-[11px] text-terminal-muted">
        Daily published calls start here. Each one goes up before the match and is
        settled the next day, win or lose.
      </div>
    );
  }
  const won = CALLS.filter(c => c.result === "won").length;
  return (
    <div className="mt-8">
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-[11px] font-bold text-slate-200">Published calls</span>
        <span className="text-[10px] text-terminal-muted">{won} of {CALLS.length} correct</span>
      </div>
      <div className="border border-terminal-border rounded-lg divide-y divide-terminal-border overflow-hidden">
        {CALLS.map(c => (
          <div key={c.date + c.call} className="flex items-center gap-3 px-3 py-2 text-[11px]">
            <span className="text-terminal-muted font-mono shrink-0">{c.date}</span>
            <span className="text-slate-300 flex-1 min-w-0 truncate">{c.call}</span>
            <span className={`font-bold shrink-0 ${c.result === "won" ? "text-terminal-green" : "text-terminal-red"}`}>
              {c.result === "won" ? "✓" : "✗"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
