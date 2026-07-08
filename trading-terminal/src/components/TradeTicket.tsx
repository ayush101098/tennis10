"use client";

import { useEffect, useMemo, useState } from "react";
import type { ScheduledMatch } from "@/lib/scheduleService";
import { trackBet } from "@/components/BetTracker";
import { useTier } from "@/lib/auth";
import {
  eventUrl, fetchQuote, outcomeIndex,
  type PmFixture, type PmMarket, type PmMarketType,
} from "@/lib/polymarket";
import {
  connectPolymarket, disconnectPolymarket, hasWallet, loadPmConnection,
  placeBuyOrder, PM_CHANGED_EVENT, type PmConnection,
} from "@/lib/pmTrading";

/**
 * TRADE TICKET — the "take this trade" CTA behind every signal.
 *
 * One modal per fixture: pick the market (match winner or any listed set
 * winner), pick the side, and either fire a real order into Polymarket's
 * CLOB (wallet connected) or log a paper trade at the same live price.
 * Both paths land in the signed-in user's bet journal.
 */

const KELLY_FRACTION = 0.25;
const KELLY_CAP = 0.05;

export interface TicketTarget {
  match: ScheduledMatch;
  fixture?: PmFixture;
  initialMarket?: PmMarketType;
  initialPick?: string; // player name
}

interface Props {
  target: TicketTarget;
  bankroll: number;
  onClose: () => void;
}

type Phase = { kind: "idle" } | { kind: "placing" }
  | { kind: "done"; venue: "polymarket" | "paper"; detail: string }
  | { kind: "error"; message: string };

export default function TradeTicket({ target, bankroll, onClose }: Props) {
  const { session } = useTier();
  const m = target.match;
  const fixture = target.fixture;

  const markets = useMemo(() => {
    const out = new Map<PmMarketType, PmMarket>();
    if (fixture?.match) out.set("match", fixture.match);
    for (const s of fixture?.sets ?? []) out.set(s.marketType, s);
    return out;
  }, [fixture]);

  const [marketType, setMarketType] = useState<PmMarketType>(
    target.initialMarket && (target.initialMarket === "match" || markets.has(target.initialMarket))
      ? target.initialMarket : "match");
  const pmMarket = markets.get(marketType);

  // Model frame: match-winner probabilities (live Markov when available)
  const p1Prob = m.liveScore?.trueProbabilities?.p1MatchProb ?? m.p1_win_prob;
  const p2Prob = m.liveScore?.trueProbabilities?.p2MatchProb ?? m.p2_win_prob;
  const defaultPick = target.initialPick ?? (p1Prob >= p2Prob ? m.player1 : m.player2);
  const [pick, setPick] = useState(defaultPick);
  const pickProb = marketType === "match"
    ? (pick === m.player1 ? p1Prob : p2Prob)
    : null; // no set-winner model — trade set markets on price only

  // ── Live price for the picked outcome ──
  const outcomeIdx = pmMarket ? outcomeIndex(pmMarket, pick) : -1;
  const snapshot = pmMarket && outcomeIdx >= 0 ? pmMarket.prices[outcomeIdx] : null;
  const [ask, setAsk] = useState<number | null>(null);
  const [bid, setBid] = useState<number | null>(null);
  useEffect(() => {
    setAsk(null); setBid(null);
    const tokenId = pmMarket && outcomeIdx >= 0 ? pmMarket.tokenIds[outcomeIdx] : null;
    if (!tokenId) return;
    let alive = true;
    const load = () => fetchQuote(tokenId)
      .then(q => { if (alive) { setAsk(q.bestAsk); setBid(q.bestBid); } })
      .catch(() => { /* keep gamma snapshot */ });
    load();
    const t = setInterval(load, 10_000);
    return () => { alive = false; clearInterval(t); };
  }, [pmMarket, outcomeIdx]);

  // Tradeable price: live ask ▸ gamma snapshot ▸ book odds (paper-only fallback)
  const bookP = m.value && m.value.player === pick ? 1 / m.value.odds : null;
  const price = ask ?? (snapshot && snapshot > 0 && snapshot < 1 ? snapshot : null) ?? bookP;
  const priceSource = ask !== null ? "live CLOB ask" : snapshot ? "PM snapshot" : bookP ? "book odds" : null;

  const edge = pickProb !== null && price !== null ? pickProb - price : null;
  const kelly = pickProb !== null && price !== null && price < 1
    ? Math.max(0, (pickProb - price) / (1 - price)) : 0;
  const suggested = Math.max(0, Math.round(bankroll * Math.min(kelly * KELLY_FRACTION, KELLY_CAP)));

  const [stakeStr, setStakeStr] = useState(String(suggested > 0 ? suggested : 10));
  useEffect(() => { setStakeStr(String(suggested > 0 ? suggested : 10)); }, [suggested, marketType, pick]);
  const stake = parseFloat(stakeStr) || 0;
  const shares = price ? Math.floor((stake / price) * 100) / 100 : 0;

  // ── Polymarket connection (per signed-in user) ──
  const [conn, setConn] = useState<PmConnection | null>(null);
  const [funder, setFunder] = useState("");
  const [showConnect, setShowConnect] = useState(false);
  useEffect(() => {
    const load = () => setConn(loadPmConnection(session?.email));
    load();
    window.addEventListener(PM_CHANGED_EVENT, load);
    return () => window.removeEventListener(PM_CHANGED_EVENT, load);
  }, [session?.email]);

  const [phase, setPhase] = useState<Phase>({ kind: "idle" });

  const logBet = (venue: "polymarket" | "paper", fillPrice: number, filledShares: number, orderId?: string) => {
    trackBet({
      match: `${m.player1} v ${m.player2}`,
      tour: m.tour,
      selection: pick,
      odds: 1 / fillPrice,
      stake: Math.round(filledShares * fillPrice * 100) / 100,
      edgeAtEntry: edge ?? undefined,
      truePAtEntry: pickProb ?? undefined,
      venue,
      market: marketType,
      price: fillPrice,
      shares: filledShares,
      orderId,
    });
  };

  const doConnect = async () => {
    setPhase({ kind: "placing" });
    try {
      await connectPolymarket(session?.email, funder.trim() || undefined);
      setShowConnect(false);
      setPhase({ kind: "idle" });
    } catch (err) {
      setPhase({ kind: "error", message: (err as Error).message });
    }
  };

  const doRealTrade = async () => {
    if (!conn || !pmMarket || outcomeIdx < 0 || !price || stake <= 0) return;
    setPhase({ kind: "placing" });
    try {
      const placed = await placeBuyOrder(conn, pmMarket, outcomeIdx, price, stake);
      logBet("polymarket", placed.priceLimit, placed.shares, placed.orderId);
      setPhase({
        kind: "done", venue: "polymarket",
        detail: `${placed.shares.toFixed(2)} shares @ ≤${Math.round(placed.priceLimit * 100)}¢ · ${placed.status}${placed.orderId ? ` · ${placed.orderId.slice(0, 10)}…` : ""}`,
      });
    } catch (err) {
      setPhase({ kind: "error", message: (err as Error).message });
    }
  };

  const doPaperTrade = () => {
    if (!price || stake <= 0) return;
    logBet("paper", price, shares);
    setPhase({
      kind: "done", venue: "paper",
      detail: `${shares.toFixed(2)} shares @ ${Math.round(price * 100)}¢ · filled at ${priceSource}`,
    });
  };

  const setTabs: PmMarketType[] = ["set1", "set2", "set3"];

  return (
    <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4" onClick={onClose}>
      <div className="w-full max-w-[440px] bg-terminal-panel border border-terminal-border rounded-lg overflow-hidden"
        onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border">
          <div className="min-w-0">
            <div className="text-[11px] font-bold text-slate-100 truncate">⚡ TAKE TRADE — {m.player1} vs {m.player2}</div>
            <div className="text-[9px] text-terminal-muted truncate">{m.tour} · {m.tournament} · {m.status === "live" ? "🔴 LIVE" : m.start_time || "scheduled"}</div>
          </div>
          <button onClick={onClose} className="text-terminal-muted hover:text-slate-200 text-sm px-1">✕</button>
        </div>

        {phase.kind === "done" ? (
          <div className="p-5 text-center space-y-2">
            <div className="text-2xl">{phase.venue === "polymarket" ? "⬢" : "📝"}</div>
            <div className={`text-sm font-bold ${phase.venue === "polymarket" ? "text-terminal-green" : "text-terminal-cyan"}`}>
              {phase.venue === "polymarket" ? "ORDER SENT TO POLYMARKET" : "PAPER TRADE LOGGED"}
            </div>
            <div className="text-[10px] text-slate-300">{pick} — {marketType === "match" ? "match winner" : marketType.replace("set", "set ") + " winner"}</div>
            <div className="text-[10px] font-mono text-terminal-muted">{phase.detail}</div>
            <div className="text-[9px] text-terminal-muted">Logged to your bet journal ({session?.email || "guest"}).</div>
            <button onClick={onClose}
              className="mt-2 px-4 py-1.5 rounded bg-terminal-green text-black text-[10px] font-bold hover:opacity-90">DONE</button>
          </div>
        ) : (
          <div className="p-3 space-y-3">
            {/* Market tabs */}
            <div className="flex gap-1">
              <Tab active={marketType === "match"} onClick={() => setMarketType("match")} label="MATCH" />
              {setTabs.map(t => (
                <Tab key={t} active={marketType === t} disabled={!markets.has(t)}
                  onClick={() => markets.has(t) && setMarketType(t)}
                  label={t.replace("set", "SET ")} title={markets.has(t) ? markets.get(t)!.question : "Not listed on Polymarket"} />
              ))}
              {pmMarket && (
                <a href={eventUrl(pmMarket)} target="_blank" rel="noopener noreferrer"
                  className="ml-auto text-[9px] text-terminal-muted hover:text-terminal-cyan self-center">polymarket ↗</a>
              )}
            </div>

            {marketType !== "match" && !pmMarket ? (
              <div className="text-[10px] text-terminal-muted text-center py-4">This set market is not listed on Polymarket for this fixture.</div>
            ) : (
              <>
                {/* Side selection */}
                <div className="grid grid-cols-2 gap-1.5">
                  {[m.player1, m.player2].map(p => {
                    const active = pick === p;
                    const prob = marketType === "match" ? (p === m.player1 ? p1Prob : p2Prob) : null;
                    return (
                      <button key={p} onClick={() => setPick(p)}
                        className={`px-2 py-1.5 rounded border text-left ${active ? "border-terminal-green bg-terminal-green/10" : "border-terminal-border hover:bg-terminal-bg"}`}>
                        <div className={`text-[10px] font-medium truncate ${active ? "text-terminal-green" : "text-slate-200"}`}>{p}</div>
                        <div className="text-[8px] text-terminal-muted font-mono">{prob !== null ? `True P ${(prob * 100).toFixed(1)}%` : "price only"}</div>
                      </button>
                    );
                  })}
                </div>

                {/* Quote strip */}
                <div className="grid grid-cols-4 gap-px bg-terminal-border rounded overflow-hidden text-center">
                  <Cell label="ASK" value={price !== null ? `${Math.round(price * 100)}¢` : "—"} tone="yellow" sub={priceSource || undefined} />
                  <Cell label="BID" value={bid !== null ? `${Math.round(bid * 100)}¢` : "—"} />
                  <Cell label="TRUE P" value={pickProb !== null ? `${(pickProb * 100).toFixed(1)}%` : "n/a"} />
                  <Cell label="EDGE" value={edge !== null ? `${edge >= 0 ? "+" : ""}${(edge * 100).toFixed(1)}%` : "—"}
                    tone={edge !== null ? (edge >= 0.05 ? "green" : edge >= 0.02 ? "yellow" : "red") : undefined} />
                </div>
                {marketType !== "match" && (
                  <div className="text-[8px] text-terminal-muted -mt-1">Set-winner markets have no model number — you are trading price, not model edge.</div>
                )}

                {/* Stake */}
                <div className="flex items-center gap-2">
                  <label className="text-[9px] text-terminal-muted shrink-0">Stake $</label>
                  <input type="number" min="1" value={stakeStr} onChange={e => setStakeStr(e.target.value)}
                    className="w-[80px] bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[11px] text-slate-200 font-mono outline-none focus:border-terminal-cyan" />
                  {suggested > 0 && (
                    <button onClick={() => setStakeStr(String(suggested))}
                      className="text-[8px] font-bold px-1.5 py-0.5 rounded border border-terminal-cyan/40 text-terminal-cyan hover:bg-terminal-cyan/10">
                      ¼ KELLY ${suggested}
                    </button>
                  )}
                  <span className="ml-auto text-[9px] text-terminal-muted font-mono">
                    {price !== null && stake > 0 ? `≈ ${shares.toFixed(2)} sh → $${shares.toFixed(2)} if it wins` : ""}
                  </span>
                </div>

                {phase.kind === "error" && (
                  <div className="text-[9px] text-terminal-red bg-terminal-red/10 border border-terminal-red/30 rounded px-2 py-1.5">{phase.message}</div>
                )}

                {/* Actions */}
                <div className="space-y-1.5">
                  {conn ? (
                    <>
                      <button onClick={doRealTrade}
                        disabled={phase.kind === "placing" || !pmMarket || outcomeIdx < 0 || !price || stake <= 0}
                        className="w-full py-2 rounded bg-terminal-green text-black text-[11px] font-bold hover:opacity-90 disabled:opacity-40">
                        {phase.kind === "placing" ? "SIGNING & SENDING…" : `⬢ BUY ${pick.split(" ").pop()?.toUpperCase()} ON POLYMARKET — $${stake || 0}`}
                      </button>
                      <div className="flex items-center justify-between text-[8px] text-terminal-muted">
                        <span>connected: <span className="font-mono">{conn.address.slice(0, 6)}…{conn.address.slice(-4)}</span>{conn.sigType === 2 ? " (proxy)" : ""}</span>
                        <button onClick={() => disconnectPolymarket(session?.email)} className="hover:text-terminal-red">disconnect</button>
                      </div>
                    </>
                  ) : showConnect ? (
                    <div className="border border-terminal-border rounded p-2 space-y-1.5">
                      <div className="text-[9px] text-slate-300">
                        Connect the wallet you use on Polymarket. If your Polymarket account was created on
                        polymarket.com with this wallet, paste your <b>profile address</b> (where your USDC sits) below.
                      </div>
                      <input placeholder="Polymarket profile address (optional, 0x…)" value={funder} onChange={e => setFunder(e.target.value)}
                        className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[9px] text-slate-200 font-mono outline-none focus:border-terminal-cyan" />
                      <div className="flex gap-1.5">
                        <button onClick={doConnect} disabled={phase.kind === "placing"}
                          className="flex-1 py-1.5 rounded bg-terminal-green text-black text-[10px] font-bold hover:opacity-90 disabled:opacity-40">
                          {phase.kind === "placing" ? "CONNECTING…" : "CONNECT WALLET"}
                        </button>
                        <button onClick={() => setShowConnect(false)}
                          className="px-3 py-1.5 rounded border border-terminal-border text-[10px] text-terminal-muted hover:text-slate-300">cancel</button>
                      </div>
                    </div>
                  ) : (
                    <button onClick={() => (hasWallet() ? setShowConnect(true) : setPhase({ kind: "error", message: "No wallet extension found. Install MetaMask to trade for real — paper trading works without it." }))}
                      disabled={!pmMarket}
                      className="w-full py-2 rounded border border-terminal-green/50 text-terminal-green text-[11px] font-bold hover:bg-terminal-green/10 disabled:opacity-40">
                      ⬢ CONNECT POLYMARKET TO TRADE LIVE
                    </button>
                  )}

                  <button onClick={doPaperTrade} disabled={!price || stake <= 0 || phase.kind === "placing"}
                    className="w-full py-2 rounded border border-terminal-cyan/50 text-terminal-cyan text-[11px] font-bold hover:bg-terminal-cyan/10 disabled:opacity-40">
                    📝 PAPER TRADE — SAME PRICE, NO MONEY
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function Tab({ active, disabled, onClick, label, title }: {
  active: boolean; disabled?: boolean; onClick: () => void; label: string; title?: string;
}) {
  return (
    <button onClick={onClick} title={title} disabled={disabled}
      className={`text-[9px] font-bold px-2 py-1 rounded ${
        active ? "bg-terminal-green/20 text-terminal-green"
          : disabled ? "text-terminal-border cursor-not-allowed"
          : "text-terminal-muted hover:text-slate-300"}`}>
      {label}
    </button>
  );
}

function Cell({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: "green" | "yellow" | "red" }) {
  const c = tone === "green" ? "text-terminal-green" : tone === "yellow" ? "text-terminal-yellow" : tone === "red" ? "text-terminal-red" : "text-slate-200";
  return (
    <div className="bg-terminal-panel px-1 py-1.5">
      <div className="text-[7px] text-terminal-muted font-bold tracking-wider">{label}</div>
      <div className={`text-[11px] font-bold font-mono ${c}`}>{value}</div>
      {sub && <div className="text-[7px] text-terminal-muted truncate">{sub}</div>}
    </div>
  );
}
