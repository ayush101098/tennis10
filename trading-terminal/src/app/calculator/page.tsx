"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import Wordmark from "@/components/Wordmark";
import Socials from "@/components/Socials";
import GrowthProjection from "@/components/GrowthProjection";
import { BreadcrumbLd } from "@/components/JsonLd";
import { fetchScheduleClient } from "@/lib/scheduleService";
import type { ScheduleData, ScheduledMatch } from "@/lib/scheduleService";
import { useTier } from "@/lib/auth";
import PricingModal from "@/components/PricingModal";
import { usePortfolio } from "@/hooks/usePortfolio";
import {
  portfolioTiers,
  stakePlan,
  devig,
  MIN_EDGE,
  STRONG_EDGE,
  SUSPECT_EDGE,
  MAX_ENTRY_FRACTION,
  type StakeClass,
} from "@/lib/portfolio";

/**
 * KELLY CALCULATOR — punch in your portfolio once and immediately see the
 * recommended stake for every match on today's board, plus a standalone
 * what-if calculator. All sizing follows the Trading Execution Manual:
 * ¼ Kelly, 2% edge floor, 6u (6%) per-event cap, three-tier bankroll.
 */
export default function CalculatorPage() {
  const { tier } = useTier();
  const isPro = tier === "pro";
  const [portfolio, setPortfolio] = usePortfolio();
  const [pricingOpen, setPricingOpen] = useState(false);

  const t = portfolioTiers(portfolio);
  const money = (n: number) =>
    "$" + Math.round(n).toLocaleString("en-US");

  return (
    <div className="marketing min-h-screen bg-terminal-bg text-slate-200">
      <BreadcrumbLd trail={[{ name: "Kelly Calculator", path: "/calculator" }]} />
      {/* ── Nav ── */}
      <nav className="sticky top-0 z-40 flex items-center justify-between gap-2 px-3 sm:px-6 py-3 border-b border-terminal-border bg-terminal-bg/95 backdrop-blur">
        <Link href="/" className="hover:opacity-80">
          <Wordmark size={16} />
        </Link>
        <div className="flex items-center gap-2 sm:gap-3 text-[11px] shrink-0">
          <Socials />
          <Link href="/manual" className="hidden sm:inline text-terminal-muted hover:text-slate-200">Manual</Link>
          <Link href="/terminal" className="inline-flex items-center min-h-[40px] font-bold px-3 rounded bg-terminal-green text-black hover:opacity-90">
            LAUNCH TERMINAL →
          </Link>
        </div>
      </nav>

      <main className="px-4 sm:px-6 py-8 max-w-[1100px] mx-auto">
        <header className="mb-6">
          <div className="text-[10px] uppercase tracking-wider text-terminal-cyan font-bold mb-1">Kelly Calculator</div>
          <h1 className="text-2xl sm:text-3xl font-bold text-slate-100">Your stake on every match — instantly.</h1>
          <p className="mt-2 text-[13px] text-slate-400 max-w-[680px] leading-relaxed">
            Set your portfolio once. We size every bet at ¼ Kelly against the model&apos;s edge,
            capped at 6% per event, with the 2% edge floor and three-tier bankroll from the{" "}
            <Link href="/manual" className="text-accent underline underline-offset-2 decoration-1 decoration-current/40 hover:decoration-current">execution manual</Link>.
          </p>
        </header>

        {/* ── Portfolio input + tier breakdown ── */}
        <section className="border border-terminal-border rounded-lg bg-terminal-panel/30 p-4 sm:p-5 mb-6">
          <div className="flex flex-wrap items-end gap-4">
            <label className="block">
              <span className="block text-[10px] uppercase tracking-wider text-terminal-muted font-bold mb-1">Your portfolio</span>
              <div className="flex items-center gap-1 text-2xl font-bold">
                <span className="text-terminal-muted">$</span>
                <input
                  type="number"
                  min={1}
                  value={portfolio}
                  onChange={e => setPortfolio(parseInt(e.target.value) || 0)}
                  className="w-[160px] bg-terminal-bg border border-terminal-border rounded px-3 py-1.5 text-slate-100 focus:border-terminal-cyan outline-none"
                />
              </div>
            </label>
            <div className="flex flex-wrap gap-1.5">
              {[500, 1000, 5000, 10000].map(v => (
                <button key={v} onClick={() => setPortfolio(v)}
                  className={`text-[11px] px-2.5 py-1 rounded border ${portfolio === v ? "border-terminal-cyan text-terminal-cyan" : "border-terminal-border text-terminal-muted hover:text-slate-200"}`}>
                  {money(v)}
                </button>
              ))}
            </div>
            <div className="ml-auto text-[11px] text-terminal-muted">
              1 unit = <span className="text-slate-200 font-mono">{money(t.unit)}</span> <span className="text-content-muted">(1% of bankroll)</span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mt-5">
            <TierCard label="Tier A · Core" value={money(t.tierA)} note="60% — main trading bankroll" tone="green" />
            <TierCard label="Tier B · Hedge" value={money(t.tierB)} note="30% — hedge reserve" tone="cyan" />
            <TierCard label="Tier C · Emergency" value={money(t.tierC)} note="10% — 🚨 signals only" tone="muted" />
            <TierCard label="Max per event" value={money(t.maxEntry)} note="6u hard cap on entry" tone="yellow" />
            <TierCard label="Max exposure" value={money(t.maxExposure)} note="15u across all events" tone="yellow" />
          </div>
        </section>

        <div className="grid lg:grid-cols-[1fr_360px] gap-6">
          {/* ── Every-match stakes (Pro) ── */}
          <section>
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-bold text-slate-100">Today&apos;s board — recommended stakes</h2>
              <span className="text-[10px] text-terminal-muted">¼ Kelly · edge ≥ {MIN_EDGE * 100}%</span>
            </div>
            {isPro ? (
              <LiveStakeBoard portfolio={portfolio} />
            ) : (
              <LockedBoard onUpgrade={() => setPricingOpen(true)} />
            )}
          </section>

          {/* ── What-if calculator (public) ── */}
          <section>
            <h2 className="text-sm font-bold text-slate-100 mb-2">What-if calculator</h2>
            <WhatIf portfolio={portfolio} />
          </section>
        </div>

        {/* ── Edge → stake quick reference for this portfolio ── */}
        <section className="mt-8">
          <div className="mb-6"><GrowthProjection /></div>

          <h2 className="text-sm font-bold text-slate-100 mb-2">Edge → stake, for {money(portfolio)}</h2>
          <EdgeReference portfolio={portfolio} />
        </section>

        <footer className="mt-10 pt-6 border-t border-terminal-border text-center text-[9px] text-terminal-muted leading-relaxed">
          Stakes are guidance, not guarantees. ¼ Kelly, 6% per-event cap and the 2% edge floor exist to keep you solvent through variance.
          Bet only what you can afford to lose.
        </footer>
      </main>

      <PricingModal open={pricingOpen} onClose={() => setPricingOpen(false)} onDone={() => {}} />
    </div>
  );
}

/* ── Portfolio tier card ── */
function TierCard({ label, value, note, tone }: {
  label: string; value: string; note: string; tone: "green" | "cyan" | "yellow" | "muted";
}) {
  const c = tone === "green" ? "text-terminal-green" : tone === "cyan" ? "text-terminal-cyan"
    : tone === "yellow" ? "text-terminal-yellow" : "text-slate-300";
  return (
    <div className="border border-terminal-border rounded-lg p-3 bg-terminal-bg/40">
      <div className="text-[9px] uppercase tracking-wider text-terminal-muted font-bold">{label}</div>
      <div className={`text-lg font-bold font-mono mt-0.5 ${c}`}>{value}</div>
      <div className="text-[9px] text-terminal-muted mt-0.5">{note}</div>
    </div>
  );
}

/* ── Live every-match stake board (Pro) ── */
function LiveStakeBoard({ portfolio }: { portfolio: number }) {
  const [data, setData] = useState<ScheduleData | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    const load = () => fetchScheduleClient().then(setData).catch(() => setErr(true));
    load();
    const iv = setInterval(load, 45_000);
    return () => clearInterval(iv);
  }, []);

  const rows = useMemo(() => {
    if (!data) return [];
    return [...data.today]
      .filter(m => (m.status === "live" || m.status === "scheduled") && m.value && !m.value.suspect && m.value.edge >= MIN_EDGE)
      .sort((a, b) => b.value!.edge - a.value!.edge)
      .slice(0, 50);
  }, [data]);

  if (err) return <Empty msg="Couldn't load the schedule. Try again shortly." />;
  if (!data) return <Empty msg="Loading today's board…" pulse />;
  if (rows.length === 0) return <Empty msg={`No edges above the ${MIN_EDGE * 100}% floor right now — waiting is the trade.`} />;

  const money = (n: number) => "$" + Math.round(n).toLocaleString("en-US");

  return (
    <div className="border border-terminal-border rounded-lg overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-[8px] uppercase tracking-wider text-terminal-muted bg-terminal-panel/50">
              <th className="text-left font-bold px-3 py-1.5">Bet</th>
              <th className="text-right font-bold px-2 py-1.5">Odds</th>
              <th className="text-right font-bold px-2 py-1.5">Edge</th>
              <th className="text-right font-bold px-2 py-1.5">Stake</th>
              <th className="text-right font-bold px-2 py-1.5">Units</th>
              <th className="text-right font-bold px-3 py-1.5">Size</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(m => {
              const v = m.value!;
              const plan = stakePlan(portfolio, v.trueP, v.odds, v.edge);
              const strong = v.edge >= STRONG_EDGE;
              const live = m.status === "live";
              return (
                <tr key={m.id} className="border-t border-terminal-border hover:bg-terminal-panel/30">
                  <td className="px-3 py-1.5 min-w-[180px]">
                    <div className="flex items-center gap-1.5">
                      {live && <span className="text-[8px] text-terminal-green font-bold">● LIVE</span>}
                      <span className="text-slate-100 font-medium truncate">{v.player}</span>
                      <span className="text-terminal-muted truncate">v {v.side === 1 ? m.player2 : m.player1}</span>
                    </div>
                    <div className="text-[8px] text-terminal-muted truncate">{m.tour} · {m.surface}{v.live ? " · live Markov" : ""}</div>
                  </td>
                  <td className="text-right px-2 font-mono text-terminal-yellow">{v.odds.toFixed(2)}</td>
                  <td className={`text-right px-2 font-mono font-bold ${strong ? "text-terminal-green" : "text-terminal-yellow"}`}>+{(v.edge * 100).toFixed(1)}%</td>
                  <td className="text-right px-2 font-mono font-bold text-terminal-green">{money(plan.stake)}</td>
                  <td className="text-right px-2 font-mono text-slate-300">{plan.units.toFixed(1)}u{plan.capped ? "*" : ""}</td>
                  <td className="text-right px-3"><SizeTag c={plan.classification} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="px-3 py-2 text-[9px] text-terminal-muted border-t border-terminal-border">
        Stake = ¼ Kelly on model edge, capped at {MAX_ENTRY_FRACTION * 100}% of bankroll. <span className="font-mono">*</span> = size bound by the 6u cap.
      </div>
    </div>
  );
}

function SizeTag({ c }: { c: StakeClass }) {
  if (c === "NONE") return <span className="text-[8px] text-terminal-muted">—</span>;
  const tone: Record<Exclude<StakeClass, "NONE">, string> = {
    MICRO: "text-terminal-muted border-terminal-border",
    SMALL: "text-slate-300 border-terminal-border",
    MEDIUM: "text-terminal-cyan border-terminal-cyan/40",
    LARGE: "text-terminal-green border-terminal-green/40",
    MAX: "text-black bg-terminal-green border-terminal-green",
  };
  return <span className={`inline-block text-[8px] font-bold px-1.5 py-0.5 rounded border ${tone[c]}`}>{c}</span>;
}

function LockedBoard({ onUpgrade }: { onUpgrade: () => void }) {
  return (
    <div className="border border-terminal-border rounded-lg bg-terminal-panel/30 p-8 flex flex-col items-center justify-center gap-3 text-center min-h-[260px]">
      <div className="text-3xl">🔒</div>
      <div className="text-terminal-green font-bold text-sm">PRO FEATURE</div>
      <div className="text-[11px] text-slate-300 max-w-[360px] leading-relaxed">
        The live board sizes every match on today&apos;s schedule against the model&apos;s edge for your exact portfolio.
        The what-if calculator on the right is free — try it with any odds and probability.
      </div>
      <button onClick={onUpgrade} className="mt-1 px-4 py-2 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90">
        GO PRO — $99
      </button>
    </div>
  );
}

function Empty({ msg, pulse }: { msg: string; pulse?: boolean }) {
  return (
    <div className={`border border-terminal-border rounded-lg bg-terminal-panel/20 p-8 text-center text-[11px] text-terminal-muted min-h-[160px] flex items-center justify-center ${pulse ? "animate-pulse" : ""}`}>
      {msg}
    </div>
  );
}

/* ── Standalone what-if calculator (public) ── */
function WhatIf({ portfolio }: { portfolio: number }) {
  const [prob, setProb] = useState(60); // your win probability %
  const [odds, setOdds] = useState(2.0); // decimal odds offered
  const [oppOdds, setOppOdds] = useState<string>(""); // optional, to de-vig

  const trueP = Math.min(0.999, Math.max(0.001, prob / 100));
  const marketP = useMemo(() => {
    const opp = parseFloat(oppOdds);
    if (opp > 1) {
      const dv = devig(odds, opp);
      if (dv) return dv.a;
    }
    return odds > 1 ? 1 / odds : 0;
  }, [odds, oppOdds]);

  const edge = trueP - marketP;
  const plan = stakePlan(portfolio, trueP, odds, edge);
  const fairOdds = trueP > 0 ? 1 / trueP : 0;
  const money = (n: number) => "$" + Math.round(n).toLocaleString("en-US");

  const bettable = edge >= MIN_EDGE && edge < SUSPECT_EDGE;

  return (
    <div className="border border-terminal-border rounded-lg bg-terminal-panel/30 p-4">
      <div className="space-y-3">
        <Field label={`Your win probability — ${prob}%`}>
          <input type="range" min={1} max={99} value={prob} onChange={e => setProb(parseInt(e.target.value))}
            className="w-full accent-terminal-green" />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Odds offered">
            <input type="number" step={0.01} min={1.01} value={odds}
              onChange={e => setOdds(parseFloat(e.target.value) || 1.01)}
              className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1.5 text-slate-100 font-mono focus:border-terminal-cyan outline-none" />
          </Field>
          <Field label="Opponent odds (optional)">
            <input type="number" step={0.01} min={1.01} value={oppOdds} placeholder="de-vig"
              onChange={e => setOppOdds(e.target.value)}
              className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1.5 text-slate-100 font-mono focus:border-terminal-cyan outline-none" />
          </Field>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-y-1.5 gap-x-3 mt-4 text-[11px] font-mono">
        <Stat label="Fair odds" value={fairOdds ? fairOdds.toFixed(2) : "—"} />
        <Stat label="Market P" value={(marketP * 100).toFixed(1) + "%"} />
        <Stat label="Edge" value={(edge >= 0 ? "+" : "") + (edge * 100).toFixed(1) + "%"}
          color={edge >= STRONG_EDGE ? "green" : edge >= MIN_EDGE ? "yellow" : "red"} />
        <Stat label="Full Kelly" value={(plan.fullKelly * 100).toFixed(1) + "%"} />
      </div>

      <div className={`mt-4 rounded-lg border p-3 text-center ${
        bettable ? "border-terminal-green/40 bg-terminal-green/5"
          : edge >= SUSPECT_EDGE ? "border-terminal-red/40 bg-terminal-red/5"
          : "border-terminal-border bg-terminal-bg/40"}`}>
        {edge >= SUSPECT_EDGE ? (
          <div className="text-[11px] text-terminal-red font-bold">⚠ Edge &gt; {SUSPECT_EDGE * 100}% — check your inputs, this is almost always bad data, not a bet.</div>
        ) : bettable ? (
          <>
            <div className="text-[9px] uppercase tracking-wider text-terminal-muted font-bold">Recommended stake</div>
            <div className="text-2xl font-bold font-mono text-terminal-green mt-0.5">{money(plan.stake)}</div>
            <div className="text-[10px] text-terminal-muted mt-0.5">
              {plan.units.toFixed(2)} units · {plan.classification}{plan.capped ? " · capped at 6u" : ""}
              {" · to win "}{money(plan.stake * (odds - 1))}
            </div>
          </>
        ) : (
          <div className="text-[11px] text-terminal-muted">
            No bet — edge is below the {MIN_EDGE * 100}% floor. The discipline is the product.
          </div>
        )}
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="block text-[10px] text-terminal-muted mb-1">{label}</span>
      {children}
    </label>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: "green" | "yellow" | "red" }) {
  const c = color === "green" ? "text-terminal-green" : color === "yellow" ? "text-terminal-yellow" : color === "red" ? "text-terminal-red" : "text-slate-200";
  return (
    <div className="flex items-center justify-between border-b border-terminal-border/60 pb-1">
      <span className="text-terminal-muted">{label}</span>
      <span className={`font-bold ${c}`}>{value}</span>
    </div>
  );
}

/* ── Edge → stake reference for the current portfolio ── */
function EdgeReference({ portfolio }: { portfolio: number }) {
  // Illustrative: at even (2.00) odds, edge maps directly to a Kelly fraction.
  const rows = [0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.2];
  const money = (n: number) => "$" + Math.round(n).toLocaleString("en-US");
  return (
    <div className="border border-terminal-border rounded-lg overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-[8px] uppercase tracking-wider text-terminal-muted bg-terminal-panel/50">
              <th className="text-left font-bold px-3 py-1.5">Edge (at 2.00 odds)</th>
              <th className="text-right font-bold px-3 py-1.5">¼ Kelly stake</th>
              <th className="text-right font-bold px-3 py-1.5">Units</th>
              <th className="text-right font-bold px-3 py-1.5">Classification</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(e => {
              // At 2.00 odds, trueP = 0.5 + edge (market fair P = 0.5).
              const trueP = 0.5 + e;
              const plan = stakePlan(portfolio, trueP, 2.0, e);
              return (
                <tr key={e} className="border-t border-terminal-border">
                  <td className="px-3 py-1.5 font-mono text-slate-200">+{(e * 100).toFixed(0)}%</td>
                  <td className="px-3 py-1.5 text-right font-mono font-bold text-terminal-green">{money(plan.stake)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-slate-300">{plan.units.toFixed(1)}u{plan.capped ? "*" : ""}</td>
                  <td className="px-3 py-1.5 text-right"><SizeTag c={plan.classification} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="px-3 py-2 text-[9px] text-terminal-muted border-t border-terminal-border">
        Reference at even odds — real stakes use each match&apos;s actual odds. <span className="font-mono">*</span> = bound by the 6u per-event cap.
      </div>
    </div>
  );
}
