"use client";

import { useEffect, useMemo, useState } from "react";
import { loadSession } from "@/lib/auth";

/**
 * BET TRACKER — the discipline half of the system.
 * Every user gets their own journal (keyed by signed-in email): every trade
 * taken from a signal — real Polymarket orders and paper trades alike — lands
 * here with the model edge at entry. Settle results; watch P&L, ROI, and
 * average edge taken.
 */

export type BetVenue = "polymarket" | "paper" | "manual";

export interface TrackedBet {
  id: string;
  placedAt: number;
  match: string;        // "Player A v Player B"
  tour: string;
  selection: string;    // player backed
  odds: number;         // decimal odds (1/price for PM-style entries)
  stake: number;        // USD risked
  edgeAtEntry?: number; // model edge when the bet was placed
  truePAtEntry?: number;
  venue?: BetVenue;     // polymarket = real order | paper = simulated fill
  market?: "match" | "set1" | "set2" | "set3";
  price?: number;       // entry price 0..1 (PM-style)
  shares?: number;      // PM-style: payout = $1/share on win
  orderId?: string;     // CLOB order id for real trades
  status: "open" | "won" | "lost" | "void";
}

const LEGACY_KEY = "tt_bets_v1";
const KEY_PREFIX = "tt_bets_v2_";

/** Per-user storage key — each signed-in email keeps its own journal. */
function betsKey(): string {
  const email = loadSession()?.email?.trim().toLowerCase();
  return KEY_PREFIX + (email || "guest");
}

function loadBets(): TrackedBet[] {
  try {
    const key = betsKey();
    let bets: TrackedBet[] = JSON.parse(localStorage.getItem(key) || "[]");
    // One-time migration: pre-user bets join the current user's journal
    const legacy = localStorage.getItem(LEGACY_KEY);
    if (legacy) {
      try { bets = [...bets, ...JSON.parse(legacy)]; } catch { /* drop corrupt */ }
      localStorage.removeItem(LEGACY_KEY);
      localStorage.setItem(key, JSON.stringify(bets));
    }
    return bets;
  } catch {
    return [];
  }
}

function saveBets(bets: TrackedBet[]): void {
  localStorage.setItem(betsKey(), JSON.stringify(bets));
}

/** Add a bet from anywhere in the app (Trade Ticket, Value Board, manual). */
export function trackBet(bet: Omit<TrackedBet, "id" | "placedAt" | "status">): void {
  const bets = loadBets();
  bets.unshift({ ...bet, id: `bet_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`, placedAt: Date.now(), status: "open" });
  saveBets(bets);
  window.dispatchEvent(new Event("tt-bets-changed"));
}

function pnl(b: TrackedBet): number {
  if (b.status === "won") return b.shares ? b.shares - b.stake : b.stake * (b.odds - 1);
  if (b.status === "lost") return -b.stake;
  return 0;
}

function venueOf(b: TrackedBet): BetVenue {
  return b.venue || "manual";
}

export default function BetTracker() {
  const [bets, setBets] = useState<TrackedBet[]>([]);
  const [venueFilter, setVenueFilter] = useState<"all" | BetVenue>("all");
  const [form, setForm] = useState({ match: "", selection: "", odds: "2.00", stake: "25" });
  const email = loadSession()?.email || "guest";

  const reload = () => setBets(loadBets());
  useEffect(() => {
    reload();
    window.addEventListener("tt-bets-changed", reload);
    return () => window.removeEventListener("tt-bets-changed", reload);
  }, []);

  const settle = (id: string, status: TrackedBet["status"]) => {
    const next = bets.map(b => (b.id === id ? { ...b, status } : b));
    saveBets(next); setBets(next);
  };
  const remove = (id: string) => {
    const next = bets.filter(b => b.id !== id);
    saveBets(next); setBets(next);
  };

  const visible = venueFilter === "all" ? bets : bets.filter(b => venueOf(b) === venueFilter);

  const stats = useMemo(() => {
    const settled = visible.filter(b => b.status === "won" || b.status === "lost");
    const staked = settled.reduce((s, b) => s + b.stake, 0);
    const profit = settled.reduce((s, b) => s + pnl(b), 0);
    const wins = settled.filter(b => b.status === "won").length;
    const openRisk = visible.filter(b => b.status === "open").reduce((s, b) => s + b.stake, 0);
    const edges = visible.filter(b => b.edgeAtEntry !== undefined);
    const avgEdge = edges.length ? edges.reduce((s, b) => s + (b.edgeAtEntry || 0), 0) / edges.length : 0;
    return {
      n: settled.length, wins, staked, profit,
      roi: staked > 0 ? profit / staked : 0,
      openRisk, avgEdge,
    };
  }, [visible]);

  const addManual = () => {
    const odds = parseFloat(form.odds), stake = parseFloat(form.stake);
    if (!form.match.trim() || !form.selection.trim() || !(odds > 1) || !(stake > 0)) return;
    trackBet({ match: form.match.trim(), tour: "", selection: form.selection.trim(), odds, stake, venue: "manual" });
    setForm({ ...form, match: "", selection: "" });
  };

  const counts = {
    all: bets.length,
    polymarket: bets.filter(b => venueOf(b) === "polymarket").length,
    paper: bets.filter(b => venueOf(b) === "paper").length,
    manual: bets.filter(b => venueOf(b) === "manual").length,
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* P&L strip */}
      <div className="grid grid-cols-6 gap-px bg-terminal-border shrink-0">
        <Stat label="SETTLED" value={`${stats.n}`} />
        <Stat label="WIN RATE" value={stats.n ? `${Math.round((stats.wins / stats.n) * 100)}%` : "—"} />
        <Stat label="STAKED" value={`$${stats.staked.toFixed(0)}`} />
        <Stat label="P&L" value={`${stats.profit >= 0 ? "+" : ""}$${stats.profit.toFixed(2)}`} tone={stats.profit > 0 ? "green" : stats.profit < 0 ? "red" : undefined} />
        <Stat label="ROI" value={stats.n ? `${(stats.roi * 100).toFixed(1)}%` : "—"} tone={stats.roi > 0 ? "green" : stats.roi < 0 ? "red" : undefined} />
        <Stat label="OPEN RISK" value={`$${stats.openRisk.toFixed(0)}`} />
      </div>

      {/* Venue filter + whose journal this is */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-terminal-border shrink-0 text-[9px]">
        {([["all", `ALL (${counts.all})`],
           ["polymarket", `⬢ POLYMARKET (${counts.polymarket})`],
           ["paper", `📝 PAPER (${counts.paper})`],
           ["manual", `✍ MANUAL (${counts.manual})`]] as const).map(([v, label]) => (
          <button key={v} onClick={() => setVenueFilter(v)}
            className={`font-bold px-2 py-0.5 rounded ${venueFilter === v ? "bg-terminal-cyan/20 text-terminal-cyan" : "text-terminal-muted hover:text-slate-300"}`}>
            {label}
          </button>
        ))}
        <span className="ml-auto text-terminal-muted">journal · {email}</span>
      </div>

      {/* Manual entry */}
      <div className="flex items-center gap-1.5 px-3 py-2 border-b border-terminal-border shrink-0">
        <input placeholder="Match (A v B)" value={form.match} onChange={e => setForm({ ...form, match: e.target.value })}
          className="flex-1 bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[10px] text-slate-200 outline-none focus:border-terminal-cyan" />
        <input placeholder="Selection" value={form.selection} onChange={e => setForm({ ...form, selection: e.target.value })}
          className="w-[120px] bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[10px] text-slate-200 outline-none focus:border-terminal-cyan" />
        <input placeholder="Odds" type="number" step="0.01" value={form.odds} onChange={e => setForm({ ...form, odds: e.target.value })}
          className="w-[60px] bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[10px] text-slate-200 outline-none focus:border-terminal-cyan" />
        <input placeholder="Stake" type="number" value={form.stake} onChange={e => setForm({ ...form, stake: e.target.value })}
          className="w-[60px] bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-[10px] text-slate-200 outline-none focus:border-terminal-cyan" />
        <button onClick={addManual}
          className="text-[10px] font-bold px-3 py-1 rounded bg-terminal-green/20 text-terminal-green border border-terminal-green/40 hover:bg-terminal-green/30">
          + LOG BET
        </button>
      </div>

      {/* Bet list */}
      <div className="flex-1 overflow-y-auto">
        {visible.length === 0 && (
          <div className="text-terminal-muted text-[11px] text-center py-10">
            No bets logged yet. Take a trade from the Value Board (⚡ TRADE) or add one manually above.
          </div>
        )}
        {visible.map(b => (
          <div key={b.id} className={`flex items-center gap-2 px-3 py-1.5 border-b border-terminal-border text-[10px] ${b.status !== "open" ? "opacity-70" : ""}`}>
            <span className="w-[74px] shrink-0 text-terminal-muted">{new Date(b.placedAt).toLocaleDateString("en-GB", { day: "2-digit", month: "short" })} {new Date(b.placedAt).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}</span>
            <VenueChip venue={venueOf(b)} />
            <span className="flex-1 min-w-0 truncate text-slate-300">
              {b.match}{b.tour ? ` · ${b.tour}` : ""}
              {b.market && b.market !== "match" && <span className="text-terminal-cyan"> · {b.market.replace("set", "SET ")} WINNER</span>}
            </span>
            <span className="w-[110px] shrink-0 truncate text-slate-100 font-medium">{b.selection}</span>
            <span className="w-[46px] shrink-0 text-right font-mono text-terminal-yellow" title={b.price !== undefined ? `entry ${(b.price * 100).toFixed(1)}¢` : undefined}>
              {b.price !== undefined ? `${Math.round(b.price * 100)}¢` : b.odds.toFixed(2)}
            </span>
            <span className="w-[45px] shrink-0 text-right font-mono text-slate-200">${b.stake}</span>
            <span className="w-[42px] shrink-0 text-right font-mono text-terminal-muted">{b.edgeAtEntry !== undefined ? `+${(b.edgeAtEntry * 100).toFixed(1)}%` : "—"}</span>
            <span className={`w-[55px] shrink-0 text-right font-mono font-bold ${pnl(b) > 0 ? "text-terminal-green" : pnl(b) < 0 ? "text-terminal-red" : "text-terminal-muted"}`}>
              {b.status === "open" ? "OPEN" : b.status === "void" ? "VOID" : `${pnl(b) >= 0 ? "+" : ""}$${pnl(b).toFixed(0)}`}
            </span>
            <span className="w-[104px] shrink-0 flex gap-1 justify-end">
              {b.status === "open" ? (
                <>
                  <MiniBtn label="W" tone="green" onClick={() => settle(b.id, "won")} />
                  <MiniBtn label="L" tone="red" onClick={() => settle(b.id, "lost")} />
                  <MiniBtn label="V" onClick={() => settle(b.id, "void")} />
                </>
              ) : (
                <MiniBtn label="reopen" onClick={() => settle(b.id, "open")} />
              )}
              <MiniBtn label="✕" onClick={() => remove(b.id)} />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function VenueChip({ venue }: { venue: BetVenue }) {
  const style = venue === "polymarket" ? "text-terminal-green border-terminal-green/40 bg-terminal-green/10"
    : venue === "paper" ? "text-terminal-cyan border-terminal-cyan/40 bg-terminal-cyan/10"
    : "text-terminal-muted border-terminal-border";
  const label = venue === "polymarket" ? "PM" : venue === "paper" ? "PAPER" : "MAN";
  return <span className={`w-[44px] shrink-0 text-center text-[8px] font-bold px-1 py-0.5 rounded border ${style}`}>{label}</span>;
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: "green" | "red" }) {
  const c = tone === "green" ? "text-terminal-green" : tone === "red" ? "text-terminal-red" : "text-slate-100";
  return (
    <div className="bg-terminal-panel px-3 py-2">
      <div className="text-[8px] text-terminal-muted font-bold tracking-wider">{label}</div>
      <div className={`text-sm font-bold font-mono ${c}`}>{value}</div>
    </div>
  );
}

function MiniBtn({ label, tone, onClick }: { label: string; tone?: "green" | "red"; onClick: () => void }) {
  const c = tone === "green" ? "text-terminal-green border-terminal-green/40 hover:bg-terminal-green/10"
    : tone === "red" ? "text-terminal-red border-terminal-red/40 hover:bg-terminal-red/10"
    : "text-terminal-muted border-terminal-border hover:bg-terminal-bg";
  return (
    <button onClick={onClick} className={`text-[8px] font-bold px-1.5 py-0.5 rounded border ${c}`}>{label}</button>
  );
}
