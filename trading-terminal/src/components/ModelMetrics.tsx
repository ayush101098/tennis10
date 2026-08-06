/**
 * Held-out model performance, published as measured.
 *
 * Every number here comes from a real evaluation artifact in the repo — no
 * marketing figures, and deliberately no ROI. Sources:
 *
 *   tennis  model_evaluation_results.csv, written by train_and_evaluate.py
 *           (temporal split: train 2020-21, validate 2022, test 2023-24)
 *   rank baseline
 *           computed over the same 2023-24 window in tennis_data.db:
 *           5,970 matches where both ATP/WTA ranks were known, scoring
 *           "the higher-ranked player wins"
 *   table tennis
 *           tabletennis/site/metrics.json — walk-forward, 95,622 train rows,
 *           23,906 held-out test rows, with Elo-only and coin-flip baselines
 *
 * The ROI column in model_evaluation_results.csv is NOT surfaced: it reports
 * +94.9% for the Markov model, which is not a credible out-of-sample return,
 * and live calibration work measured the deployed engine as overconfident with
 * negative ROI. Publishing it would be a false claim.
 *
 * Form: a dot plot, not bars. The interesting differences are 1-2 points wide,
 * which zero-based bars cannot show and truncated bars would misrepresent.
 * The baseline is a reference rule, not a competing series, so the whole panel
 * uses one mark colour and states every delta in text as well as position.
 */

interface Model {
  name: string;
  accuracy: number;
  logLoss?: number;
  brier?: number;
  note?: string;
}

interface Board {
  sport: string;
  split: string;
  testN: string;
  domain: [number, number];
  baseline: { name: string; accuracy: number; note: string };
  floor?: { name: string; accuracy: number };
  models: Model[];
  asOf: string;
}

const TENNIS: Board = {
  sport: "🎾 Tennis — match winner",
  split: "Trained 2020–21 · validated 2022 · tested on 2023–24, never seen in training",
  testN: "2023–24 hold-out",
  domain: [55, 68],
  baseline: {
    name: "Higher-ranked player wins",
    accuracy: 63.58,
    note: "5,970 matches with both ranks known",
  },
  floor: { name: "Coin flip", accuracy: 50 },
  models: [
    { name: "Logistic regression", accuracy: 63.90, logLoss: 0.6336, brier: 0.2220 },
    { name: "Neural network", accuracy: 63.57, logLoss: 0.6338, brier: 0.2220 },
    { name: "Markov engine", accuracy: 62.50, logLoss: 0.6478, brier: 0.2279 },
    { name: "Meta-ensemble (LR+NN)", accuracy: 61.65, logLoss: 0.6558, brier: 0.2318 },
  ],
  asOf: "December 2025",
};

const TT: Board = {
  sport: "🏓 Table tennis — match winner",
  split: "Walk-forward: 95,622 training rows, 23,906 held-out rows scored strictly forward in time",
  testN: "23,906 hold-out matches",
  domain: [48, 60],
  baseline: {
    name: "Elo only",
    accuracy: 55.38,
    note: "log loss 0.6907",
  },
  floor: { name: "Coin flip", accuracy: 50 },
  models: [
    { name: "Gradient boosting (live)", accuracy: 57.79, logLoss: 0.6740 },
    { name: "Logistic regression", accuracy: 56.80, logLoss: 0.6772 },
  ],
  asOf: "August 2026",
};

const pct = (n: number) => `${n.toFixed(2)}%`;
const delta = (n: number, base: number) => `${n - base >= 0 ? "+" : "−"}${Math.abs(n - base).toFixed(2)} pts`;

function Row({ m, board }: { m: Model; board: Board }) {
  const [lo, hi] = board.domain;
  const x = ((m.accuracy - lo) / (hi - lo)) * 100;
  const beats = m.accuracy >= board.baseline.accuracy;
  return (
    <div className="grid grid-cols-[minmax(0,1.5fr)_minmax(60px,1fr)_auto] items-center gap-2 sm:gap-3 py-1.5">
      <span className="text-[10px] sm:text-[11px] text-slate-300 leading-tight">{m.name}</span>
      <div className="relative h-4" aria-hidden="true">
        {/* track */}
        <div className="absolute inset-x-0 top-1/2 h-px bg-terminal-border" />
        {/* baseline reference rule */}
        <div className="absolute top-0 bottom-0 w-px bg-terminal-muted"
          style={{ left: `${((board.baseline.accuracy - lo) / (hi - lo)) * 100}%` }} />
        {/* the model */}
        <div className="absolute top-1/2 w-2.5 h-2.5 rounded-full bg-terminal-green ring-2 ring-terminal-bg"
          style={{ left: `${x}%`, transform: "translate(-50%,-50%)" }} />
      </div>
      <span className="text-[10px] sm:text-[11px] tabular-nums text-right whitespace-nowrap">
        <span className="text-slate-100 font-bold">{pct(m.accuracy)}</span>{" "}
        {/* the delta is stated in text, never signalled by colour alone */}
        <span className={beats ? "text-terminal-green" : "text-terminal-red"}>
          {beats ? "▲" : "▼"} {delta(m.accuracy, board.baseline.accuracy)}
        </span>
      </span>
    </div>
  );
}

function BoardPanel({ board }: { board: Board }) {
  const [lo, hi] = board.domain;
  const basePct = ((board.baseline.accuracy - lo) / (hi - lo)) * 100;
  return (
    <div className="border border-terminal-border rounded-lg bg-terminal-panel/30 p-4">
      <div className="mb-1 text-[11px] font-bold text-slate-100">{board.sport}</div>
      <div className="text-[9px] text-terminal-muted leading-relaxed mb-3">{board.split}</div>

      {board.models.map(m => <Row key={m.name} m={m} board={board} />)}

      {/* baseline, labelled in place — it is the thing every model is judged against */}
      <div className="grid grid-cols-[minmax(0,1.5fr)_minmax(60px,1fr)_auto] items-center gap-2 sm:gap-3 pt-1.5 mt-1 border-t border-terminal-border/60">
        <span className="text-[10px] sm:text-[11px] text-terminal-muted leading-tight">{board.baseline.name}</span>
        <div className="relative h-4" aria-hidden="true">
          <div className="absolute inset-x-0 top-1/2 h-px bg-terminal-border" />
          <div className="absolute top-0 bottom-0 w-px bg-terminal-muted" style={{ left: `${basePct}%` }} />
          <div className="absolute top-1/2 w-2.5 h-2.5 rounded-full border border-terminal-muted bg-terminal-bg"
            style={{ left: `${basePct}%`, transform: "translate(-50%,-50%)" }} />
        </div>
        <span className="text-[10px] sm:text-[11px] tabular-nums text-right text-terminal-muted whitespace-nowrap">
          {pct(board.baseline.accuracy)} <span className="text-[9px]">baseline</span>
        </span>
      </div>

      {/* Axis range. The two panels are scaled to their own sport, so dot
          positions are only comparable within a panel — say so, in place. */}
      <div className="grid grid-cols-[minmax(0,1.5fr)_minmax(60px,1fr)_auto] gap-2 sm:gap-3 mt-1">
        <span />
        <span className="flex justify-between text-[8px] text-terminal-muted tabular-nums">
          <span>{lo}%</span><span>{hi}%</span>
        </span>
        <span />
      </div>

      {/* the numbers again as a table — position is never the only channel */}
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-[9px] tabular-nums">
          <thead>
            <tr className="text-terminal-muted text-left">
              <th className="font-normal py-1 pr-2">Model</th>
              <th className="font-normal py-1 px-2 text-right">Accuracy</th>
              <th className="font-normal py-1 px-2 text-right">Log loss</th>
              <th className="font-normal py-1 pl-2 text-right">Brier</th>
            </tr>
          </thead>
          <tbody className="text-slate-400">
            {board.models.map(m => (
              <tr key={m.name} className="border-t border-terminal-border/40">
                <td className="py-1 pr-2">{m.name}</td>
                <td className="py-1 px-2 text-right text-slate-200">{pct(m.accuracy)}</td>
                <td className="py-1 px-2 text-right">{m.logLoss?.toFixed(4) ?? "—"}</td>
                <td className="py-1 pl-2 text-right">{m.brier?.toFixed(4) ?? "—"}</td>
              </tr>
            ))}
            <tr className="border-t border-terminal-border/40 text-terminal-muted">
              <td className="py-1 pr-2">{board.baseline.name} <span className="text-[8px]">({board.baseline.note})</span></td>
              <td className="py-1 px-2 text-right">{pct(board.baseline.accuracy)}</td>
              <td className="py-1 px-2 text-right">—</td>
              <td className="py-1 pl-2 text-right">—</td>
            </tr>
            {board.floor && (
              <tr className="border-t border-terminal-border/40 text-terminal-muted">
                <td className="py-1 pr-2">{board.floor.name}</td>
                <td className="py-1 px-2 text-right">{pct(board.floor.accuracy)}</td>
                <td className="py-1 px-2 text-right">0.6931</td>
                <td className="py-1 pl-2 text-right">0.2500</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="mt-2 text-[9px] text-terminal-muted">
        {board.testN} · evaluated {board.asOf}
      </div>
    </div>
  );
}

export default function ModelMetrics() {
  return (
    <section className="px-4 sm:px-6 pb-14 max-w-[1000px] mx-auto">
      <h2 className="text-center text-lg font-bold text-slate-100 mb-1">Measured, not claimed</h2>
      <p className="text-center text-[11px] text-terminal-muted mb-6 max-w-[620px] mx-auto leading-relaxed">
        Held-out performance against the baseline that actually matters — beating a coin flip is
        not an achievement, beating the obvious heuristic is. These are the numbers our own
        evaluation scripts produce, including the ones that are unflattering.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <BoardPanel board={TENNIS} />
        <BoardPanel board={TT} />
      </div>

      <p className="mt-4 text-[9px] text-terminal-muted leading-relaxed max-w-[760px] mx-auto text-center">
        <b className="text-slate-400">How to read this.</b> On tennis, our best model beats
        &ldquo;back the higher-ranked player&rdquo; by 0.3 points of accuracy — a real but small
        edge, and two of the four models sit below that line. The table-tennis model clears its
        Elo baseline by 2.4 points on a much larger hold-out. We publish accuracy, log loss and
        Brier score because they are verifiable; we do not publish a backtested ROI, because
        out-of-sample returns depend on prices we did not trade at.
      </p>
    </section>
  );
}
