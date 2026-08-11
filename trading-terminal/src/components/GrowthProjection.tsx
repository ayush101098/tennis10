"use client";

import { useMemo, useState } from "react";
import { EDGE_FLOOR, MAX_BANKROLL_FRACTION } from "@/lib/scheduleService";

/**
 * "If I bet this way, how many events until my bankroll reaches X?"
 *
 * The honest answer is a distribution, not a number, so this shows the
 * EXPECTED path and says so. The maths is Kelly's log-growth rate:
 *
 *   g = p·ln(1 + f·b) + (1−p)·ln(1 − f)
 *
 * where f is the fraction staked, b = odds − 1, and p the true win
 * probability. Bankroll compounds as B·e^(g·n), so the events needed to reach
 * a target is ln(target/start) / g.
 *
 * Two things this deliberately does NOT do:
 *   - promise the number. Variance at ¼ Kelly is large; a 20-bet losing run
 *     inside a winning strategy is ordinary, and a projection that hides that
 *     is how people over-stake.
 *   - accept an edge under the 2% floor. Below it the product says no bet, so
 *     projecting growth from one would contradict the discipline it sells.
 */

const KELLY_FRACTION = 0.25;   // ¼ Kelly, as enforced everywhere else

const money = (n: number) =>
  n >= 1000 ? `$${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : `$${Math.round(n)}`;

export default function GrowthProjection() {
  const [bankroll, setBankroll] = useState(1000);
  const [target, setTarget] = useState(5000);
  const [edgePct, setEdgePct] = useState(5);
  const [odds, setOdds] = useState(2.0);

  const r = useMemo(() => {
    const b = odds - 1;
    const edge = edgePct / 100;
    // Edge is defined against the market's implied probability, so the true
    // win probability is implied + edge.
    const implied = 1 / odds;
    const p = Math.min(0.98, Math.max(0.02, implied + edge));

    const fullKelly = b > 0 ? (p * b - (1 - p)) / b : 0;
    const staked = Math.max(0, Math.min(fullKelly * KELLY_FRACTION, MAX_BANKROLL_FRACTION));

    const belowFloor = edge < EDGE_FLOOR;
    const noGrowth = staked <= 0 || fullKelly <= 0;

    // Expected log-growth per event.
    const g = noGrowth ? 0
      : p * Math.log(1 + staked * b) + (1 - p) * Math.log(1 - staked);

    const events = g > 0 && target > bankroll ? Math.log(target / bankroll) / g : Infinity;

    // Milestones along the way — the shape of compounding is the point.
    const milestones: { at: number; events: number }[] = [];
    if (g > 0 && target > bankroll) {
      for (const mult of [1.5, 2, 3, 5, 10]) {
        const amount = bankroll * mult;
        if (amount > target * 1.01) break;
        milestones.push({ at: amount, events: Math.ceil(Math.log(mult) / g) });
      }
    }

    return {
      p, fullKelly, staked, stakeDollars: bankroll * staked,
      g, events, milestones, belowFloor, noGrowth,
      perEventPct: (Math.exp(g) - 1) * 100,
    };
  }, [bankroll, target, edgePct, odds]);

  return (
    <div className="border border-terminal-border rounded-lg bg-terminal-panel/30 p-4">
      <h2 className="text-sm font-bold text-slate-100 mb-1">
        How long to reach your target
      </h2>
      <p className="text-[10px] text-terminal-muted mb-3 leading-relaxed">
        At ¼ Kelly with a {(MAX_BANKROLL_FRACTION * 100).toFixed(0)}% cap — the same rules the
        terminal enforces.
      </p>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <NumField label="Bankroll" prefix="$" value={bankroll} onChange={setBankroll} step={100} min={1} />
        <NumField label="Target" prefix="$" value={target} onChange={setTarget} step={500} min={1} />
        <NumField label="Edge per bet" suffix="%" value={edgePct} onChange={setEdgePct} step={0.5} min={0} />
        <NumField label="Typical odds" value={odds} onChange={setOdds} step={0.05} min={1.01} />
      </div>

      {r.belowFloor ? (
        <div className="rounded border border-terminal-border bg-terminal-bg/60 p-3">
          <div className="text-[11px] font-bold text-terminal-muted mb-1">⊘ Below the {(EDGE_FLOOR * 100).toFixed(0)}% edge floor</div>
          <p className="text-[10px] text-terminal-muted leading-relaxed">
            At {edgePct}% edge there is no bet to size. An edge smaller than the model&apos;s own
            error is not an edge — the terminal declines these, so projecting growth from one
            would be projecting from nothing.
          </p>
        </div>
      ) : r.noGrowth ? (
        <div className="text-[11px] text-terminal-red">
          These numbers have no positive expectation — at {odds.toFixed(2)} you need better than{" "}
          {((1 / odds) * 100).toFixed(1)}% to win long-run.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
            <Stat label="Stake per event" value={money(r.stakeDollars)}
              note={`${(r.staked * 100).toFixed(1)}% of bankroll`} tone="green" />
            <Stat label="True win prob" value={`${(r.p * 100).toFixed(1)}%`} note="implied + your edge" />
            <Stat label="Growth per event" value={`${r.perEventPct >= 0 ? "+" : ""}${r.perEventPct.toFixed(2)}%`}
              note="expected, compounding" />
            <Stat label={`Events to ${money(target)}`}
              value={Number.isFinite(r.events) ? String(Math.ceil(r.events)) : "—"}
              note={Number.isFinite(r.events) ? "at this edge" : "target below bankroll"} tone="yellow" />
          </div>

          {r.milestones.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-[10px] tabular-nums">
                <thead>
                  <tr className="text-terminal-muted text-left">
                    <th className="font-normal py-1 pr-3">Bankroll reaches</th>
                    <th className="font-normal py-1 pr-3 text-right">Events</th>
                    <th className="font-normal py-1 text-right">Stake by then</th>
                  </tr>
                </thead>
                <tbody className="text-slate-300">
                  {r.milestones.map(m => (
                    <tr key={m.at} className="border-t border-terminal-border/40">
                      <td className="py-1 pr-3 text-slate-100 font-bold">{money(m.at)}</td>
                      <td className="py-1 pr-3 text-right">{m.events}</td>
                      <td className="py-1 text-right text-terminal-muted">{money(m.at * r.staked)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <p className="mt-3 text-[9px] text-terminal-muted leading-relaxed">
            <b className="text-slate-400">This is an expectation, not a schedule.</b> Kelly growth is
            the average of a wide distribution — at {(r.staked * 100).toFixed(1)}% per event a run of
            ten losses costs about {((1 - Math.pow(1 - r.staked, 10)) * 100).toFixed(0)}% of bankroll and is
            entirely normal inside a winning strategy. The stake shown is a percentage, so it falls
            with your bankroll as well as rising with it; that is what stops a bad run ending you.
            It also assumes every bet really carries {edgePct}% — if the model is wrong about that,
            the growth is negative and the same maths runs in reverse.
          </p>
        </>
      )}
    </div>
  );
}

function NumField({ label, value, onChange, step, min, prefix, suffix }: {
  label: string; value: number; onChange: (n: number) => void;
  step: number; min: number; prefix?: string; suffix?: string;
}) {
  return (
    <label className="block">
      <span className="block text-[9px] text-terminal-muted mb-1">{label}</span>
      <span className="flex items-center gap-1 bg-terminal-bg border border-terminal-border rounded px-2 focus-within:border-terminal-cyan">
        {prefix && <span className="text-[11px] text-terminal-muted">{prefix}</span>}
        <input type="number" value={value} step={step} min={min}
          onChange={e => onChange(Math.max(min, parseFloat(e.target.value) || min))}
          className="w-full bg-transparent py-2 text-[12px] text-slate-100 outline-none" />
        {suffix && <span className="text-[11px] text-terminal-muted">{suffix}</span>}
      </span>
    </label>
  );
}

function Stat({ label, value, note, tone }: {
  label: string; value: string; note?: string; tone?: "green" | "yellow";
}) {
  const color = tone === "green" ? "text-terminal-green" : tone === "yellow" ? "text-terminal-yellow" : "text-slate-100";
  return (
    <div className="rounded border border-terminal-border bg-terminal-bg/50 p-2">
      <div className="text-[9px] text-terminal-muted">{label}</div>
      <div className={`text-lg font-bold ${color}`}>{value}</div>
      {note && <div className="text-[9px] text-terminal-muted">{note}</div>}
    </div>
  );
}
