"use client";

import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { fetchScheduleClient } from "@/lib/scheduleService";
import type { ScheduleData } from "@/lib/scheduleService";
import { scanForOpportunities, rankOpportunities } from "@/lib/arbitrage/screener";
import { allocateStakes } from "@/lib/arbitrage/math";
import type { ArbitrageOpportunity, OpportunityClassification } from "@/lib/arbitrage/types";
import { PROVIDER_REGISTRY, type ProviderStatus } from "@/lib/arbitrage/providers/registry";
import { evaluateArbHedge, vanishedHedgeSignal, type HedgeSignal, type PositionAction } from "@/lib/arbitrage/hedgeSignal";

/**
 * Read-only arbitrage screener tab.
 *
 * SCOPE: this scans TennisAlpha's own two data sources — Polymarket's two
 * outcome tokens against each other (real, executable arbitrage when
 * mispriced) and the model's True P against Polymarket (model-based EV,
 * never labeled arbitrage — see classify.ts). It does not place trades.
 * A "verified" row is still something to check on Polymarket before betting
 * real money on it: quotes move between detection and execution, and this
 * screener has no order-book depth, only best-ask price (see
 * polymarketAdapter.ts on why liquidity reads "unknown" rather than a
 * fabricated number).
 */

const SCAN_INTERVAL_MS = 20_000; // independent of LiveGrid's poll; this tab only runs while open

const CLASS_LABEL: Record<OpportunityClassification, string> = {
  verified_arbitrage: "VERIFIED ARB",
  conditional_arbitrage: "CONDITIONAL",
  model_positive_ev: "MODEL EV",
  invalid: "NO EDGE",
};
const CLASS_COLOR: Record<OpportunityClassification, string> = {
  verified_arbitrage: "text-terminal-green bg-terminal-green/10 border-terminal-green/40",
  conditional_arbitrage: "text-terminal-yellow bg-terminal-yellow/10 border-terminal-yellow/40",
  model_positive_ev: "text-terminal-cyan bg-terminal-cyan/10 border-terminal-cyan/40",
  invalid: "text-terminal-muted bg-terminal-bg/40 border-terminal-border",
};

type ClassFilter = "all" | OpportunityClassification;

export default function ArbitrageScreener() {
  const [schedule, setSchedule] = useState<ScheduleData | null>(null);
  const [opportunities, setOpportunities] = useState<ArbitrageOpportunity[]>([]);
  const [scanning, setScanning] = useState(true);
  const [lastScan, setLastScan] = useState<number | null>(null);
  const [classFilter, setClassFilter] = useState<ClassFilter>("all");
  const [selected, setSelected] = useState<ArbitrageOpportunity | null>(null);
  const [whatIfStake, setWhatIfStake] = useState(1000);
  const [showProviders, setShowProviders] = useState(false);
  const [hedgeSignals, setHedgeSignals] = useState<Map<string, HedgeSignal>>(new Map());
  const scanningRef = useRef(false);
  // Entry ROI per opportunity id — the reference point every later hedge
  // signal is measured against. Opportunity ids are now deterministic (see
  // classify.ts's stableId), so the SAME real comparison keeps the same id
  // scan after scan and this map actually tracks something continuous.
  const entryRoiRef = useRef<Map<string, number>>(new Map());
  // Ids from the last REAL scan result (not counting one-cycle "vanished"
  // rows) — the reference set for detecting what dropped out this cycle.
  const lastRealIdsRef = useRef<Set<string>>(new Set());
  const lastOppByIdRef = useRef<Map<string, ArbitrageOpportunity>>(new Map());

  const runScan = useCallback(async () => {
    if (scanningRef.current) return;
    scanningRef.current = true;
    setScanning(true);
    try {
      const data = await fetchScheduleClient();
      setSchedule(data);
      const matches = [...data.today, ...data.tomorrow];
      const found = await scanForOpportunities(matches);
      // Only surface something worth a screener row — a raw "invalid" flood
      // (every match with S >= 1, which is most of them, correctly) would
      // bury the handful of real signals under noise.
      const worthShowing = found.filter(o => o.classification !== "invalid");

      // ── Hedge signal: entry ROI vs current ROI per opportunity id ──
      const nextHedges = new Map<string, HedgeSignal>();
      const currentIds = new Set<string>();
      for (const opp of worthShowing) {
        currentIds.add(opp.opportunityId);
        const roi = opp.netRoiPct ?? opp.theoreticalRoiPct;
        if (!entryRoiRef.current.has(opp.opportunityId)) {
          entryRoiRef.current.set(opp.opportunityId, roi);
        }
        const entryRoi = entryRoiRef.current.get(opp.opportunityId)!;
        nextHedges.set(opp.opportunityId, evaluateArbHedge(opp, entryRoi, roi));
      }

      // Opportunities that were real last scan and are gone this scan get one
      // more cycle as an explicit STOP row instead of silently vanishing —
      // "this closed" is itself a signal, not just an empty gap in the table.
      const vanishedRows: ArbitrageOpportunity[] = [];
      for (const id of lastRealIdsRef.current) {
        if (currentIds.has(id)) continue;
        const cached = lastOppByIdRef.current.get(id);
        if (!cached) continue;
        const entryRoi = entryRoiRef.current.get(id) ?? cached.theoreticalRoiPct;
        nextHedges.set(id, vanishedHedgeSignal(entryRoi));
        vanishedRows.push(cached);
        entryRoiRef.current.delete(id); // one cycle shown, then this id starts fresh if it ever returns
      }

      const display = rankOpportunities([...worthShowing, ...vanishedRows]);
      setOpportunities(display);
      setHedgeSignals(nextHedges);
      lastRealIdsRef.current = currentIds;
      lastOppByIdRef.current = new Map(worthShowing.map(o => [o.opportunityId, o]));
      setLastScan(Date.now());
    } catch (err) {
      console.error("[ArbitrageScreener] scan failed:", err);
    } finally {
      scanningRef.current = false;
      setScanning(false);
    }
  }, []);

  useEffect(() => {
    runScan();
    const iv = setInterval(() => { if (document.visibilityState === "visible") runScan(); }, SCAN_INTERVAL_MS);
    return () => clearInterval(iv);
  }, [runScan]);

  const filtered = useMemo(
    () => classFilter === "all" ? opportunities : opportunities.filter(o => o.classification === classFilter),
    [opportunities, classFilter],
  );

  const counts = useMemo(() => {
    const c: Record<OpportunityClassification, number> = {
      verified_arbitrage: 0, conditional_arbitrage: 0, model_positive_ev: 0, invalid: 0,
    };
    for (const o of opportunities) c[o.classification]++;
    return c;
  }, [opportunities]);

  const bestVerifiedRoi = useMemo(() => {
    const verified = opportunities.filter(o => o.classification === "verified_arbitrage");
    if (!verified.length) return null;
    return Math.max(...verified.map(o => o.netRoiPct ?? o.theoreticalRoiPct));
  }, [opportunities]);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Summary bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 p-3 border-b border-terminal-border shrink-0">
        <SummaryCard label="VERIFIED ARB" value={counts.verified_arbitrage} tone="text-terminal-green" />
        <SummaryCard label="CONDITIONAL" value={counts.conditional_arbitrage} tone="text-terminal-yellow" />
        <SummaryCard label="MODEL EV" value={counts.model_positive_ev} tone="text-terminal-cyan" />
        <SummaryCard
          label="BEST VERIFIED ROI"
          value={bestVerifiedRoi != null ? `${bestVerifiedRoi.toFixed(2)}%` : "—"}
          tone="text-terminal-green"
        />
      </div>

      {/* Filters + status */}
      <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-terminal-border shrink-0 flex-wrap">
        <div className="flex items-center gap-1 flex-wrap">
          {(["all", "verified_arbitrage", "conditional_arbitrage", "model_positive_ev"] as ClassFilter[]).map(f => (
            <button
              key={f}
              onClick={() => setClassFilter(f)}
              className={`px-2 py-1 rounded text-[10px] font-bold border ${
                classFilter === f
                  ? "border-terminal-yellow text-terminal-yellow bg-terminal-yellow/10"
                  : "border-terminal-border text-terminal-muted hover:text-slate-300"
              }`}
            >
              {f === "all" ? "ALL" : CLASS_LABEL[f]}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowProviders(v => !v)}
            className="px-2 py-1 rounded text-[10px] font-bold border border-terminal-border text-terminal-muted hover:text-slate-300"
          >
            PROVIDERS {showProviders ? "▲" : "▼"}
          </button>
          <div className="text-[10px] text-terminal-muted font-mono">
            {scanning ? "scanning…" : lastScan ? `updated ${Math.round((Date.now() - lastScan) / 1000)}s ago` : ""}
          </div>
        </div>
      </div>

      {showProviders && <ProviderHealthPanel />}

      {/* Table */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full text-terminal-muted text-xs p-6 text-center">
            {scanning
              ? "Scanning live matches against Polymarket…"
              : "No opportunities right now. This scans Polymarket's own two-sided price against itself and against TennisAlpha's model — thin, real signals only, not a promise something is always here."}
          </div>
        ) : (
          <table className="w-full text-[11px] font-mono">
            <thead className="sticky top-0 bg-terminal-panel border-b border-terminal-border">
              <tr className="text-terminal-muted text-left">
                <Th>MATCH</Th><Th>CLASS</Th><Th>SEL A</Th><Th>ODDS A</Th>
                <Th>SEL B</Th><Th>ODDS B</Th><Th>ROI</Th><Th>PROFIT</Th><Th>HEDGE</Th><Th>FLAGS</Th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(o => (
                <tr
                  key={o.opportunityId}
                  onClick={() => { setSelected(o); setWhatIfStake(o.totalStake); }}
                  className="border-b border-terminal-border/50 hover:bg-terminal-elevated/60 cursor-pointer"
                >
                  <Td>{o.event.player1} v {o.event.player2}<div className="text-terminal-muted">{o.event.tournament}</div></Td>
                  <Td><span className={`px-1.5 py-0.5 rounded border text-[9px] font-bold ${CLASS_COLOR[o.classification]}`}>{CLASS_LABEL[o.classification]}</span></Td>
                  <Td>{o.legs[0]?.playerName} <span className="text-terminal-muted">({o.legs[0]?.provider})</span></Td>
                  <Td className="tabular-nums">{o.legs[0]?.odds.toFixed(2)}</Td>
                  <Td>{o.legs[1]?.playerName} <span className="text-terminal-muted">({o.legs[1]?.provider})</span></Td>
                  <Td className="tabular-nums">{o.legs[1]?.odds.toFixed(2)}</Td>
                  <Td className={`tabular-nums ${o.theoreticalRoiPct > 0 ? "text-terminal-green" : "text-terminal-red"}`}>
                    {o.theoreticalRoiPct.toFixed(2)}%
                  </Td>
                  <Td className="tabular-nums">${o.theoreticalProfit.toFixed(2)}</Td>
                  <Td><HedgeChip signal={hedgeSignals.get(o.opportunityId)} /></Td>
                  <Td>{o.riskFlags.length ? <span className="text-terminal-yellow">{o.riskFlags.length}</span> : <span className="text-terminal-muted">—</span>}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {selected && (
        <OpportunityDrawer
          opp={selected}
          hedge={hedgeSignals.get(selected.opportunityId)}
          stake={whatIfStake}
          onStakeChange={setWhatIfStake}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}

const STATUS_LABEL: Record<ProviderStatus, string> = {
  live: "LIVE",
  configured_no_key: "NEEDS KEY",
  no_public_api: "NO PUBLIC API",
};
const STATUS_COLOR: Record<ProviderStatus, string> = {
  live: "text-terminal-green",
  configured_no_key: "text-terminal-yellow",
  no_public_api: "text-terminal-muted",
};

/**
 * Honest venue coverage — every provider on the requested comparison list,
 * with its real status. A venue with no public API stays listed as "no
 * public api", not silently omitted, so it's clear this isn't scanning
 * bookmakers it claims to without actually reaching them.
 */
function ProviderHealthPanel() {
  return (
    <div className="border-b border-terminal-border p-3 space-y-2 max-h-64 overflow-y-auto">
      <div className="text-[10px] text-terminal-muted">
        Only providers marked LIVE feed the table above. The rest need real account credentials this app doesn&apos;t have — see each one&apos;s hover for what integrating it would require.
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
        {PROVIDER_REGISTRY.map(p => (
          <div
            key={p.id}
            title={p.authNote}
            className="flex items-center justify-between gap-2 rounded border border-terminal-border bg-terminal-bg/40 px-2 py-1 text-[10px]"
          >
            <span className="truncate">{p.displayName} <span className="text-terminal-muted">({p.category === "sportsbook" ? "book" : "market"})</span></span>
            <span className={`font-bold whitespace-nowrap ${STATUS_COLOR[p.status]}`}>{STATUS_LABEL[p.status]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function SummaryCard({ label, value, tone }: { label: string; value: string | number; tone: string }) {
  return (
    <div className="rounded border border-terminal-border bg-terminal-bg/40 p-2">
      <div className="text-[9px] text-terminal-muted font-bold">{label}</div>
      <div className={`text-lg font-bold tabular-nums ${tone}`}>{value}</div>
    </div>
  );
}
function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-2 py-1.5 font-bold text-[10px] whitespace-nowrap">{children}</th>;
}
function Td({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-2 py-1.5 align-top ${className}`}>{children}</td>;
}

const HEDGE_COLOR: Record<PositionAction, string> = {
  HOLD: "text-terminal-green bg-terminal-green/10 border-terminal-green/40",
  HEDGE: "text-terminal-yellow bg-terminal-yellow/10 border-terminal-yellow/40",
  STOP: "text-terminal-red bg-terminal-red/10 border-terminal-red/40",
};

function HedgeChip({ signal }: { signal: HedgeSignal | undefined }) {
  if (!signal) return <span className="text-terminal-muted">—</span>;
  return (
    <span
      title={signal.reason}
      className={`px-1.5 py-0.5 rounded border text-[9px] font-bold ${HEDGE_COLOR[signal.action]}`}
    >
      {signal.action}
    </span>
  );
}

function OpportunityDrawer({
  opp, hedge, stake, onStakeChange, onClose,
}: {
  opp: ArbitrageOpportunity;
  hedge: HedgeSignal | undefined;
  stake: number;
  onStakeChange: (n: number) => void;
  onClose: () => void;
}) {
  // The what-if calculator calls the SAME allocateStakes() the detection
  // engine used — no second, drifting copy of the formula in the UI.
  const odds = opp.legs.map(l => l.odds);
  const alloc = allocateStakes(odds, stake);

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="w-full sm:max-w-lg sm:rounded-lg border border-terminal-border bg-terminal-panel p-4 space-y-3 max-h-[90vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <div>
            <div className="font-bold text-sm">{opp.event.player1} vs {opp.event.player2}</div>
            <div className="text-[10px] text-terminal-muted">{opp.event.tournament} · {opp.event.tour} · {opp.event.status}</div>
          </div>
          <button onClick={onClose} className="text-terminal-muted hover:text-slate-200 text-lg leading-none">×</button>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <div className={`inline-block px-2 py-1 rounded border text-[10px] font-bold ${CLASS_COLOR[opp.classification]}`}>
            {CLASS_LABEL[opp.classification]}
          </div>
          {hedge && (
            <div className={`inline-block px-2 py-1 rounded border text-[10px] font-bold ${HEDGE_COLOR[hedge.action]}`}>
              {hedge.action}
            </div>
          )}
        </div>

        {hedge && (
          <div className={`text-[10px] rounded p-2 border ${
            hedge.action === "STOP" ? "text-terminal-red bg-terminal-red/5 border-terminal-red/30"
              : hedge.action === "HEDGE" ? "text-terminal-yellow bg-terminal-yellow/5 border-terminal-yellow/30"
              : "text-terminal-green bg-terminal-green/5 border-terminal-green/30"
          }`}>
            <div className="font-bold mb-0.5">
              HEDGE SIGNAL: {hedge.action} — entry {hedge.entryRoiPct.toFixed(2)}% → now {hedge.currentRoiPct.toFixed(2)}%
              {" "}({hedge.driftPct >= 0 ? "+" : ""}{hedge.driftPct.toFixed(2)}pp since detection)
            </div>
            {hedge.reason}
          </div>
        )}

        {opp.classification === "model_positive_ev" && (
          <div className="text-[10px] text-terminal-cyan bg-terminal-cyan/5 border border-terminal-cyan/30 rounded p-2">
            This compares TennisAlpha&apos;s model probability to a market price — a probabilistic edge, not a
            guaranteed payoff. It is not arbitrage.
          </div>
        )}

        <div className="rounded border border-terminal-border bg-terminal-bg/40 p-2.5 space-y-2">
          <div className="text-[10px] font-bold text-terminal-muted">LEGS</div>
          {opp.legs.map((l, i) => (
            <div key={i} className="flex items-center justify-between text-[11px] font-mono">
              <span>{l.playerName} <span className="text-terminal-muted">({l.provider})</span></span>
              <span className="tabular-nums">{l.odds.toFixed(3)}</span>
            </div>
          ))}
        </div>

        <div className="space-y-1.5">
          <label className="text-[10px] font-bold text-terminal-muted">WHAT-IF TOTAL STAKE (USD)</label>
          <input
            type="number"
            value={stake}
            min={1}
            onChange={e => onStakeChange(Math.max(1, Number(e.target.value) || 0))}
            className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1.5 text-sm font-mono"
          />
        </div>

        <div className="rounded border border-terminal-border bg-terminal-bg/40 p-2.5 space-y-1.5 text-[11px] font-mono">
          {opp.legs.map((l, i) => (
            <div key={i} className="flex items-center justify-between">
              <span>{l.playerName}</span>
              <span className="tabular-nums">${alloc.stakes[i].toFixed(2)}</span>
            </div>
          ))}
          <div className="border-t border-terminal-border pt-1.5 flex items-center justify-between font-bold">
            <span>Gross payout (any outcome)</span>
            <span className="tabular-nums">${alloc.grossPayout.toFixed(2)}</span>
          </div>
          <div className={`flex items-center justify-between font-bold ${alloc.grossProfit >= 0 ? "text-terminal-green" : "text-terminal-red"}`}>
            <span>Gross profit</span>
            <span className="tabular-nums">${alloc.grossProfit.toFixed(2)} ({alloc.roiPct.toFixed(2)}%)</span>
          </div>
          {opp.netProfit == null && (
            <div className="text-terminal-yellow text-[10px] pt-1">
              Net figures unknown — {opp.riskFlags.join(", ") || "execution conditions unverified"}. Gross above is
              theoretical only.
            </div>
          )}
        </div>

        <div className="rounded border border-terminal-border bg-terminal-bg/40 p-2.5 space-y-1 text-[11px]">
          <div className="text-[10px] font-bold text-terminal-muted mb-1">EXECUTION CONDITIONS</div>
          <Row label="Liquidity" value={opp.liquidityStatus} warn={opp.liquidityStatus !== "known"} />
          <Row label="Settlement" value={opp.settlementStatus} warn={opp.settlementStatus !== "compatible"} />
          <Row label="Quote age" value={opp.quoteAgeMs != null ? `${Math.round(opp.quoteAgeMs / 1000)}s` : "unknown"} warn={opp.quoteAgeMs == null} />
        </div>

        {opp.riskFlags.length > 0 && (
          <div className="rounded border border-terminal-yellow/40 bg-terminal-yellow/5 p-2.5">
            <div className="text-[10px] font-bold text-terminal-yellow mb-1">RISK FLAGS</div>
            <ul className="text-[10px] text-terminal-yellow space-y-0.5 list-disc list-inside">
              {opp.riskFlags.map(f => <li key={f}>{f.replace(/_/g, " ")}</li>)}
            </ul>
          </div>
        )}

        <div className="text-[9px] text-terminal-muted">
          Detected {new Date(opp.detectedAt).toLocaleTimeString()} · read-only — no orders are placed by this screen.
        </div>
      </div>
    </div>
  );
}

function Row({ label, value, warn }: { label: string; value: string; warn: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-terminal-muted">{label}</span>
      <span className={warn ? "text-terminal-yellow" : "text-terminal-green"}>{value}</span>
    </div>
  );
}
