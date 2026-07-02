"use client";

import { useEffect, useMemo, useState } from "react";

/**
 * BET TRACKER — the discipline half of the system.
 * Log every bet with the model edge at entry; settle results; watch P&L, ROI,
 * and average edge taken. Persisted in localStorage per browser.
 */

export interface TrackedBet {
  id: string;
  placedAt: number;
  match: string;        // "Player A v Player B"
  tour: string;
  selection: string;    // player backed
  odds: number;
  stake: number;
  edgeAtEntry?: number; // model edge when the bet was placed
  truePAtEntry?: number;
  status: "open" | "won" | "lost" | "void";
}

const LS_KEY = "tt_bets_v1";

function loadBets(): TrackedBet[] {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveBets(bets: TrackedBet[]): void {
  localStorage.setItem(LS_KEY, JSON.stringify(bets));
}

/** Add a bet from anywhere in the app (e.g. Value Board row). */
export function trackBet(bet: Omit<TrackedBet, "id" | "placedAt" | "status">): void {
  const bets = loadBets();
  bets.unshift({ ...bet, id: `bet_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`, placedAt: Date.now(), status: "open" });
  saveBets(bets);
  window.dispatchEvent(new Event("tt-bets-changed"));
}

function pnl(b: TrackedBet): number {
  if (b.status === "won") return b.stake * (b.odds - 1);
  if (b.status === "lost") return -b.stake;
  return 0;
}

export default function BetTracker() {
  const [bets, setBets] = useState<TrackedBet[]>([]);
  const [form, setForm] = useState({ match: "", selection: "", odds: "2.00", stake: "25" });

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

  const stats = useMemo(() => {
    const settled = bets.filter(b => b.status === "won" || b.status === "lost");
    const staked = settled.reduce((s, b) => s + b.stake, 0);
    const profit = settled.reduce((s, b) => s + pnl(b), 0);
    const wins = settled.filter(b => b.status === "won").length;
    const openRisk = bets.filter(b => b.status === "open").reduce((s, b) => s + b.stake, 0);
    const edges = bets.filter(b => b.edgeAtEntry !== undefined);
    const avgEdge = edges.length ? edges.reduce((s, b) => s + (b.edgeAtEntry || 0), 0) / edges.length : 0;
    return {
      n: settled.length, wins, staked, profit,
      roi: staked > 0 ? profit / staked : 0,
      openRisk, avgEdge,
    };
  }, [bets]);

  const addManual = () => {
    const odds = parseFloat(form.odds), stake = parseFloat(form.stake);
    if (!form.match.trim() || !form.selection.trim() || !(odds > 1) || !(stake > 0)) return;
    trackBet({ match: form.match.trim(), tour: "", selection: form.selection.trim(), odds, stake });
    setForm({ ...form, match: "", selection: "" });
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
        {bets.length === 0 && (
          <div className="text-terminal-muted text-[11px] text-center py-10">
            No bets logged yet. Track bets from the Value Board (💰 TRACK) or add one manually above.
          </div>
        )}
        {bets.map(b => (
          <div key={b.id} className={`flex items-center gap-2 px-3 py-1.5 border-b border-terminal-border text-[10px] ${b.status !== "open" ? "opacity-70" : ""}`}>
            <span className="w-[74px] shrink-0 text-terminal-muted">{new Date(b.placedAt).toLocaleDateString("en-GB", { day: "2-digit", month: "short" })} {new Date(b.placedAt).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}</span>
            <span className="flex-1 min-w-0 truncate text-slate-300">{b.match}{b.tour ? ` · ${b.tour}` : ""}</span>
            <span className="w-[110px] shrink-0 truncate text-slate-100 font-medium">{b.selection}</span>
            <span className="w-[40px] shrink-0 text-right font-mono text-terminal-yellow">{b.odds.toFixed(2)}</span>
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
