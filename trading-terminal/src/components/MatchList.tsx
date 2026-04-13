"use client";

import { useState } from "react";
import type { MatchListItem, MatchSetupPayload } from "@/lib/types";
import { setupMatch, fetchMatches } from "@/lib/api";

interface Props {
  matches: MatchListItem[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onMatchCreated: (id: string) => void;
}

export default function MatchList({ matches, activeId, onSelect, onMatchCreated }: Props) {
  const [showSetup, setShowSetup] = useState(false);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border">
        <span className="text-[10px] font-bold text-terminal-muted uppercase tracking-wider">
          MATCHES
        </span>
        <button
          onClick={() => setShowSetup(!showSetup)}
          className="text-[10px] bg-terminal-green/20 text-terminal-green border border-terminal-green/40 rounded px-2 py-0.5 hover:bg-terminal-green/30"
        >
          + NEW
        </button>
      </div>

      {showSetup && <SetupForm onCreated={(id) => { onMatchCreated(id); setShowSetup(false); }} />}

      <div className="flex-1 overflow-y-auto">
        {matches.map((m) => (
          <button
            key={m.match_id}
            onClick={() => onSelect(m.match_id)}
            className={`w-full text-left px-3 py-2 border-b border-terminal-border text-xs hover:bg-terminal-panel/80 transition ${
              activeId === m.match_id ? "bg-terminal-blue/10 border-l-2 border-l-terminal-blue" : ""
            }`}
          >
            <div className="flex items-center gap-1.5">
              {m.is_live && <span className="w-1.5 h-1.5 rounded-full bg-terminal-green live-dot" />}
              <span className="truncate">{m.label}</span>
            </div>
            <div className="text-terminal-muted text-[10px] mt-0.5">{m.score_summary}</div>
          </button>
        ))}
        {matches.length === 0 && (
          <div className="text-terminal-muted text-[10px] text-center py-4">No matches</div>
        )}
      </div>
    </div>
  );
}

/* ── Setup Form ── */

function SetupForm({ onCreated }: { onCreated: (id: string) => void }) {
  const [form, setForm] = useState<MatchSetupPayload>({
    player1_name: "Djokovic",
    player2_name: "Alcaraz",
    surface: "Hard",
    best_of: 3,
    tournament: "ATP Masters",
    tournament_level: "M",
    initial_server: 1,
    p1_serve_pct: 65,
    p2_serve_pct: 63,
    p1_return_pct: 38,
    p2_return_pct: 36,
    p1_rank: 2,
    p2_rank: 1,
    p1_ranking_points: 9500,
    p2_ranking_points: 11000,
    p1_first_serve_pct: 65,
    p2_first_serve_pct: 63,
    p1_first_serve_win_pct: 74,
    p2_first_serve_win_pct: 72,
    p1_second_serve_win_pct: 55,
    p2_second_serve_win_pct: 52,
    p1_bp_save_pct: 65,
    p2_bp_save_pct: 62,
    p1_win_rate: 82,
    p2_win_rate: 78,
    bankroll: 10000,
  });

  const submit = async () => {
    const res = await setupMatch(form);
    if (res.match_id) onCreated(res.match_id);
  };

  return (
    <div className="p-3 border-b border-terminal-border bg-terminal-bg space-y-2 max-h-[60vh] overflow-y-auto">
      <div className="grid grid-cols-2 gap-2">
        <Input label="Player 1" value={form.player1_name} onChange={(v) => setForm({ ...form, player1_name: v })} />
        <Input label="Player 2" value={form.player2_name} onChange={(v) => setForm({ ...form, player2_name: v })} />
        <Select label="Surface" value={form.surface} options={["Hard", "Clay", "Grass"]} onChange={(v) => setForm({ ...form, surface: v })} />
        <Select label="Best Of" value={String(form.best_of)} options={["3", "5"]} onChange={(v) => setForm({ ...form, best_of: Number(v) })} />
        <Select label="Level" value={form.tournament_level || ""} options={["", "G", "M", "500", "250"]} onChange={(v) => setForm({ ...form, tournament_level: v })} />
        <Input label="Bankroll" value={String(form.bankroll)} onChange={(v) => setForm({ ...form, bankroll: Number(v) })} />
      </div>
      <div className="text-[9px] text-terminal-cyan uppercase font-bold mt-1">Serve / Return</div>
      <div className="grid grid-cols-2 gap-2">
        <Input label="P1 Serve %" value={String(form.p1_serve_pct)} onChange={(v) => setForm({ ...form, p1_serve_pct: Number(v) })} />
        <Input label="P2 Serve %" value={String(form.p2_serve_pct)} onChange={(v) => setForm({ ...form, p2_serve_pct: Number(v) })} />
        <Input label="P1 Return %" value={String(form.p1_return_pct)} onChange={(v) => setForm({ ...form, p1_return_pct: Number(v) })} />
        <Input label="P2 Return %" value={String(form.p2_return_pct)} onChange={(v) => setForm({ ...form, p2_return_pct: Number(v) })} />
      </div>
      <div className="text-[9px] text-terminal-cyan uppercase font-bold mt-1">Career Stats (for ML)</div>
      <div className="grid grid-cols-2 gap-2">
        <Input label="P1 Rank" value={String(form.p1_rank)} onChange={(v) => setForm({ ...form, p1_rank: Number(v) })} />
        <Input label="P2 Rank" value={String(form.p2_rank)} onChange={(v) => setForm({ ...form, p2_rank: Number(v) })} />
        <Input label="P1 Pts" value={String(form.p1_ranking_points ?? 1000)} onChange={(v) => setForm({ ...form, p1_ranking_points: Number(v) })} />
        <Input label="P2 Pts" value={String(form.p2_ranking_points ?? 1000)} onChange={(v) => setForm({ ...form, p2_ranking_points: Number(v) })} />
        <Input label="P1 1st %" value={String(form.p1_first_serve_pct ?? 62)} onChange={(v) => setForm({ ...form, p1_first_serve_pct: Number(v) })} />
        <Input label="P2 1st %" value={String(form.p2_first_serve_pct ?? 62)} onChange={(v) => setForm({ ...form, p2_first_serve_pct: Number(v) })} />
        <Input label="P1 1stW%" value={String(form.p1_first_serve_win_pct ?? 70)} onChange={(v) => setForm({ ...form, p1_first_serve_win_pct: Number(v) })} />
        <Input label="P2 1stW%" value={String(form.p2_first_serve_win_pct ?? 70)} onChange={(v) => setForm({ ...form, p2_first_serve_win_pct: Number(v) })} />
        <Input label="P1 2ndW%" value={String(form.p1_second_serve_win_pct ?? 50)} onChange={(v) => setForm({ ...form, p1_second_serve_win_pct: Number(v) })} />
        <Input label="P2 2ndW%" value={String(form.p2_second_serve_win_pct ?? 50)} onChange={(v) => setForm({ ...form, p2_second_serve_win_pct: Number(v) })} />
        <Input label="P1 BPSv%" value={String(form.p1_bp_save_pct ?? 60)} onChange={(v) => setForm({ ...form, p1_bp_save_pct: Number(v) })} />
        <Input label="P2 BPSv%" value={String(form.p2_bp_save_pct ?? 60)} onChange={(v) => setForm({ ...form, p2_bp_save_pct: Number(v) })} />
        <Input label="P1 Win%" value={String(form.p1_win_rate ?? 50)} onChange={(v) => setForm({ ...form, p1_win_rate: Number(v) })} />
        <Input label="P2 Win%" value={String(form.p2_win_rate ?? 50)} onChange={(v) => setForm({ ...form, p2_win_rate: Number(v) })} />
      </div>
      <button onClick={submit} className="w-full bg-terminal-green/20 text-terminal-green border border-terminal-green/40 rounded py-1 text-xs font-bold hover:bg-terminal-green/30">
        CREATE MATCH
      </button>
    </div>
  );
}

function Input({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <div className="text-[9px] text-terminal-muted uppercase">{label}</div>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-0.5 text-xs text-slate-200 focus:border-terminal-blue outline-none"
      />
    </div>
  );
}

function Select({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <div>
      <div className="text-[9px] text-terminal-muted uppercase">{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-0.5 text-xs text-slate-200 focus:border-terminal-blue outline-none"
      >
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}
