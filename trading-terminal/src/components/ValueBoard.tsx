"use client";

import { useMemo, useState } from "react";
import type { ScheduledMatch } from "@/lib/scheduleService";
import { usePolymarket } from "@/hooks/usePolymarket";
import { fixtureKey, outcomePrice, type PmFixture } from "@/lib/polymarket";
import TradeTicket, { type TicketTarget } from "@/components/TradeTicket";

/**
 * VALUE BOARD — betting intelligence for every match on the schedule.
 *
 * Each row is one match, ranked by model edge vs the de-vigged bookmaker
 * market. True P comes from the NN+Elo ensemble pre-match and switches to the
 * score-conditioned Markov engine once the match is live. Stakes are ¼ Kelly
 * capped at 5% of bankroll; nothing below the 2% edge floor is flagged as a bet.
 */

const MIN_EDGE = 0.02;       // minimum edge to flag a bet
const STRONG_EDGE = 0.05;    // strong-signal threshold
const KELLY_FRACTION = 0.25; // quarter Kelly
const KELLY_CAP = 0.05;      // never stake more than 5% of bankroll

interface Props {
  matches: ScheduledMatch[];
  onSelectMatch?: (m: ScheduledMatch) => void;
}

export default function ValueBoard({ matches, onSelectMatch }: Props) {
  const [bankroll, setBankroll] = useState(1000);
  const [liveOnly, setLiveOnly] = useState(false);
  const [ticket, setTicket] = useState<TicketTarget | null>(null);
  const pmIndex = usePolymarket();

  const { actionable, watchlist, suspects, priced, unpriced } = useMemo(() => {
    const pool = matches.filter(m =>
      (m.status === "live" || m.status === "scheduled") && (!liveOnly || m.status === "live"));
    const withValue = pool.filter(m => m.value);
    const byEdge = [...withValue].sort((a, b) => (b.value!.edge - a.value!.edge));
    return {
      actionable: byEdge.filter(m => m.value!.edge >= MIN_EDGE && !m.value!.suspect),
      watchlist: byEdge.filter(m => m.value!.edge > 0 && m.value!.edge < MIN_EDGE),
      suspects: byEdge.filter(m => m.value!.suspect),
      priced: byEdge.length,
      unpriced: pool.length - byEdge.length,
    };
  }, [matches, liveOnly]);

  const hedges = useMemo(
    () => matches.filter(m => m.status === "live" && m.liveScore?.hedgeAlert?.shouldHedge),
    [matches],
  );

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Controls */}
      <div className="flex items-center gap-3 px-3 py-1.5 border-b border-terminal-border shrink-0 text-[10px]">
        <label className="flex items-center gap-1 text-terminal-muted">
          Bankroll $
          <input
            type="number"
            value={bankroll}
            onChange={e => setBankroll(parseInt(e.target.value) || 100)}
            className="w-[70px] bg-terminal-bg border border-terminal-border rounded px-1.5 py-0.5 text-slate-200 focus:border-terminal-cyan outline-none"
          />
        </label>
        <button
          onClick={() => setLiveOnly(v => !v)}
          className={`px-2 py-0.5 rounded font-bold ${liveOnly ? "bg-terminal-green/20 text-terminal-green" : "text-terminal-muted hover:text-slate-300"}`}
        >
          🔴 LIVE ONLY
        </button>
        <span className="ml-auto text-terminal-muted">
          {priced} priced · {unpriced} awaiting odds
        </span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* ── Hedge alerts first: protecting open positions beats new bets ── */}
        {hedges.length > 0 && (
          <BoardSection title={`🛑 HEDGE ALERTS (${hedges.length})`} tone="red">
            {hedges.map(m => (
              <button key={m.id} onClick={() => onSelectMatch?.(m)}
                className="w-full text-left px-3 py-1.5 border-b border-terminal-border bg-terminal-red/5 hover:bg-terminal-red/10">
                <div className="flex items-center gap-2 text-[10px]">
                  <span className={`font-bold ${m.liveScore!.hedgeAlert!.urgency === "IMMEDIATE" ? "text-terminal-red animate-pulse" : "text-terminal-yellow"}`}>
                    {m.liveScore!.hedgeAlert!.urgency}
                  </span>
                  <span className="text-slate-200 flex-1 truncate">{m.player1} vs {m.player2}</span>
                  <span className="text-terminal-muted">{m.liveScore!.hedgeAlert!.reason}</span>
                  {m.liveScore!.hedgeAlert!.hedgeSizeMultiplier !== undefined && (
                    <span className="text-terminal-cyan font-mono">×{m.liveScore!.hedgeAlert!.hedgeSizeMultiplier}</span>
                  )}
                </div>
              </button>
            ))}
          </BoardSection>
        )}

        {/* ── Actionable value bets ── */}
        <BoardSection title={`💎 VALUE BETS — edge ≥ ${MIN_EDGE * 100}% (${actionable.length})`} tone="green">
          <HeaderRow />
          {actionable.map(m => <ValueRow key={m.id} m={m} pm={pmIndex.get(fixtureKey(m.player1, m.player2))} bankroll={bankroll} onClick={() => onSelectMatch?.(m)} onTrade={setTicket} />)}
          {actionable.length === 0 && (
            <div className="text-terminal-muted text-[10px] text-center py-4">
              No edges above the {MIN_EDGE * 100}% floor right now — the discipline IS the system. Wait.
            </div>
          )}
        </BoardSection>

        {/* ── Marginal / watchlist ── */}
        {watchlist.length > 0 && (
          <BoardSection title={`👁 WATCHLIST — positive but below floor (${watchlist.length})`} tone="muted">
            <HeaderRow />
            {watchlist.map(m => <ValueRow key={m.id} m={m} pm={pmIndex.get(fixtureKey(m.player1, m.player2))} bankroll={bankroll} onClick={() => onSelectMatch?.(m)} onTrade={setTicket} dim />)}
          </BoardSection>
        )}

        {/* ── Suspect: edge too large to be real — data problem, not free money ── */}
        {suspects.length > 0 && (
          <BoardSection title={`⚠ SUSPECT DATA — edge >20% means bad inputs, DO NOT BET (${suspects.length})`} tone="red">
            <HeaderRow />
            {suspects.map(m => <ValueRow key={m.id} m={m} pm={pmIndex.get(fixtureKey(m.player1, m.player2))} bankroll={bankroll} onClick={() => onSelectMatch?.(m)} onTrade={setTicket} dim />)}
          </BoardSection>
        )}

        {/* Method note */}
        <div className="px-3 py-2 text-[9px] text-terminal-muted leading-relaxed">
          True P: neural network (42k-match trained, Platt-calibrated) ⊕ Elo pre-match → tour-aware Markov once live.
          Edge vs de-vigged market. Stake = ¼ Kelly, capped at {KELLY_CAP * 100}% of bankroll.
          Only bet edges ≥ {MIN_EDGE * 100}%. Unranked fields (deep ITF quallies) stay unpriced until live.
        </div>
      </div>

      {ticket && <TradeTicket target={ticket} bankroll={bankroll} onClose={() => setTicket(null)} />}
    </div>
  );
}

function HeaderRow() {
  return (
    <div className="grid grid-cols-[44px_1fr_60px_44px_44px_48px_52px_60px_64px_56px] gap-1 px-3 py-1 text-[8px] font-bold text-terminal-muted uppercase tracking-wider border-b border-terminal-border">
      <span>Status</span>
      <span>Bet</span>
      <span className="text-right">True P</span>
      <span className="text-right">Odds</span>
      <span className="text-right">Mkt P</span>
      <span className="text-right">Edge</span>
      <span className="text-right">¼ Kelly</span>
      <span className="text-right">Signal</span>
      <span className="text-right">Polymkt</span>
      <span className="text-right">Trade</span>
    </div>
  );
}

function ValueRow({ m, pm, bankroll, onClick, onTrade, dim }: {
  m: ScheduledMatch; pm?: PmFixture; bankroll: number; onClick: () => void;
  onTrade: (t: TicketTarget) => void; dim?: boolean;
}) {
  const v = m.value!;
  const stake = Math.round(bankroll * Math.min(v.kelly * KELLY_FRACTION, KELLY_CAP));
  const strong = v.edge >= STRONG_EDGE;
  const live = m.status === "live";

  const openTicket = (e: React.MouseEvent) => {
    e.stopPropagation();
    onTrade({ match: m, fixture: pm, initialMarket: "match", initialPick: v.player });
  };

  return (
    <button onClick={onClick}
      className={`w-full grid grid-cols-[44px_1fr_60px_44px_44px_48px_52px_60px_64px_56px] gap-1 px-3 py-1.5 border-b border-terminal-border text-left items-center hover:bg-terminal-panel/40 ${dim ? "opacity-60" : ""}`}>
      {/* status */}
      <span className={`text-[9px] font-bold ${live ? "text-terminal-green" : "text-terminal-muted"}`}>
        {live ? "● LIVE" : m.start_time || "TBD"}
      </span>
      {/* bet description */}
      <span className="min-w-0">
        <span className="block text-[10px] text-slate-100 font-medium truncate">
          {v.player} <span className="text-terminal-muted font-normal">v {v.side === 1 ? m.player2 : m.player1}</span>
        </span>
        <span className="block text-[8px] text-terminal-muted truncate">
          {m.tour} · {m.tournament} · {m.surface}
          {v.live && <span className="text-terminal-cyan"> · live Markov</span>}
        </span>
      </span>
      <span className="text-right text-[10px] font-mono text-slate-200">{(v.trueP * 100).toFixed(1)}%</span>
      <span className="text-right text-[10px] font-mono text-terminal-yellow">{v.odds.toFixed(2)}</span>
      <span className="text-right text-[10px] font-mono text-terminal-muted">{(v.marketP * 100).toFixed(1)}%</span>
      <span className={`text-right text-[10px] font-mono font-bold ${strong ? "text-terminal-green" : v.edge >= MIN_EDGE ? "text-terminal-yellow" : "text-terminal-muted"}`}>
        +{(v.edge * 100).toFixed(1)}%
      </span>
      <span className={`text-right text-[10px] font-mono ${stake > 0 ? "text-terminal-green font-bold" : "text-terminal-muted"}`}>
        ${stake}
      </span>
      <span className={`text-right text-[9px] font-bold ${strong ? "text-terminal-green" : v.edge >= MIN_EDGE ? "text-terminal-yellow" : "text-terminal-muted"}`}>
        {v.suspect ? "⚠ DATA?" : strong ? "🔥 STRONG" : v.edge >= MIN_EDGE ? "⚡ VALUE" : "— watch"}
      </span>
      <PolymarketCell pm={pm} pick={v.player} trueP={v.trueP} onOpen={openTicket} />
      <span className="text-right">
        <span onClick={openTicket}
          className={`inline-block text-[8px] font-bold px-1.5 py-0.5 rounded border cursor-pointer ${
            strong || v.edge >= MIN_EDGE
              ? "text-black bg-terminal-green border-terminal-green hover:opacity-90"
              : "text-terminal-cyan border-terminal-cyan/40 hover:bg-terminal-cyan/10"
          }`}
          title={`Open trade ticket — buy on Polymarket or paper trade${stake > 0 ? ` (¼ Kelly ≈ $${stake})` : ""}`}>
          ⚡ TRADE
        </span>
      </span>
    </button>
  );
}

/**
 * Live Polymarket quote for the model's pick, next to the TRADE CTA.
 * Shows the PM price (¢) and edge vs True P; click opens the trade ticket
 * (match + any listed set-winner markets).
 */
function PolymarketCell({ pm, pick, trueP, onOpen }: {
  pm?: PmFixture; pick: string; trueP: number; onOpen: (e: React.MouseEvent) => void;
}) {
  const market = pm?.match;
  const price = market ? outcomePrice(market, pick) : null;
  if (!market || price === null) {
    return <span className="text-right text-[9px] text-terminal-muted">—</span>;
  }
  const pmEdge = trueP - price;
  const tone = pmEdge >= STRONG_EDGE ? "text-terminal-green border-terminal-green/40 hover:bg-terminal-green/10"
    : pmEdge >= MIN_EDGE ? "text-terminal-yellow border-terminal-yellow/40 hover:bg-terminal-yellow/10"
    : "text-terminal-muted border-terminal-border hover:bg-terminal-bg";
  const sets = pm!.sets.map(s => s.question).join("\n");
  return (
    <span className="text-right">
      <span
        onClick={onOpen}
        title={`${market.question}\nBuy ${pick} @ ${(price * 100).toFixed(1)}¢ · edge vs True P ${(pmEdge * 100).toFixed(1)}%${sets ? `\nAlso listed:\n${sets}` : ""}`}
        className={`inline-block text-[8px] font-bold font-mono px-1.5 py-0.5 rounded border cursor-pointer ${tone}`}
      >
        {Math.round(price * 100)}¢ {pmEdge >= 0 ? "+" : ""}{(pmEdge * 100).toFixed(1)}
      </span>
    </span>
  );
}

function BoardSection({ title, tone, children }: {
  title: string; tone: "green" | "red" | "muted"; children: React.ReactNode;
}) {
  const c = tone === "green" ? "text-terminal-green" : tone === "red" ? "text-terminal-red" : "text-terminal-muted";
  return (
    <div>
      <div className={`px-3 py-1 bg-terminal-panel/50 border-y border-terminal-border sticky top-0 z-10 text-[10px] font-bold tracking-wider ${c}`}>
        {title}
      </div>
      {children}
    </div>
  );
}
