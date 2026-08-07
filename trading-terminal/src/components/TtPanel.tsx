"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

/**
 * 🏓 Table-tennis intelligence — the TT half of the unified terminal.
 *
 * Data comes from the tabletennis/ pipeline via /api/tt:
 *   pre-match  → walk-forward Elo + GBDT (predictions.json)
 *   live True P → exact race-to-11 analytic recursion anchored to the model,
 *                 plus the tanh-bounded character residual (live_predictions.json)
 * Paper-bet journal in localStorage `tt_ttbets_v1_<email>` (guest when signed out).
 */

/* ── data shapes (tabletennis/site JSON) ── */

interface TtPrediction {
  event_id: number;
  category: string;
  tournament: string;
  start_ts: number;
  status: string;
  p1: string;
  p2: string;
  p1_win: number;
  p2_win: number;
  elo_p1: number;
  elo_p2: number;
  matches_known: number;
  confidence: "high" | "medium" | "low" | string;
}

interface TtLiveMatch {
  event_id: number;
  category: string;
  tournament: string;
  p1: string;
  p2: string;
  games: [number, number];
  points: [number, number];
  best_of: number;
  pre_match_p1: number;
  analytic_p1: number;
  residual: number;
  p1_win: number;
  annotation: string;
  history: [number, number][]; // [ts, p1_win]
}

interface TtFeed {
  predictions: { generated_ts: number; model: string; predictions: TtPrediction[] } | null;
  live: { generated_ts: number; matches: TtLiveMatch[] } | null;
  // metrics.json reports every candidate model plus which one won; the flat
  // accuracy/log_loss fields never existed, so the held-out chip below was
  // always blank. Read the best model's numbers instead.
  metrics: {
    best_model?: string;
    n_test_rows?: number;
    models?: Record<string, { accuracy?: number; log_loss?: number }>;
  } | null;
}

/** Held-out accuracy of the winning model, or null if metrics are unusable. */
function heldOutAccuracy(metrics: TtFeed["metrics"]): number | null {
  const models = metrics?.models;
  if (!models) return null;
  const best = metrics?.best_model && models[metrics.best_model];
  const acc = (best || Object.values(models)[0])?.accuracy;
  return typeof acc === "number" ? acc : null;
}

/* ── paper bet journal ── */

interface TtBet {
  id: string;
  ts: number;
  eventId: number;
  category: string;
  match: string;       // "P1 v P2"
  pick: string;
  side: 1 | 2;
  stake: number;
  odds: number;        // decimal odds taken
  pAtBet: number;      // model True P for the pick when placed
  live: boolean;
  status: "open" | "won" | "lost" | "void";
}

const betsKey = (email: string) => `tt_ttbets_v1_${email.toLowerCase()}`;
const loadBets = (email: string): TtBet[] => {
  try { return JSON.parse(localStorage.getItem(betsKey(email)) || "[]"); } catch { return []; }
};
const saveBets = (email: string, bets: TtBet[]) =>
  localStorage.setItem(betsKey(email), JSON.stringify(bets));

const pnl = (b: TtBet) =>
  b.status === "won" ? b.stake * (b.odds - 1) : b.status === "lost" ? -b.stake : 0;

/* ── data hook: poll /api/tt every 8 s (matches the live poller cadence) ── */

/**
 * Static snapshot fallback.
 *
 * /api/tt serves whatever the local pipeline pushed into Netlify Blobs — but
 * when the blob store is unavailable on the deployed site (no injected context
 * and no NETLIFY_API_TOKEN) it returns all-nulls and the TT tab goes dark.
 * So the pipeline also drops predictions.json/metrics.json into
 * trading-terminal/public/tt/, which the static export ships with the deploy;
 * we read those when the live endpoint has nothing. Pre-match board only —
 * in-play True P still needs the push path, since it changes every 8s.
 */
async function loadSnapshot(): Promise<TtFeed | null> {
  const get = async (name: string) => {
    try {
      const r = await fetch(`/tt/${name}`, { cache: "no-store" });
      return r.ok ? await r.json() : null;
    } catch { return null; }
  };
  const [predictions, metrics] = await Promise.all([get("predictions.json"), get("metrics.json")]);
  return predictions ? { predictions, live: null, metrics } : null;
}

function useTtFeed() {
  const [feed, setFeed] = useState<TtFeed | null>(null);
  const [snapshot, setSnapshot] = useState(false);
  const [error, setError] = useState(false);
  useEffect(() => {
    let alive = true;
    const load = async () => {
      let data: TtFeed | null = null;
      try {
        // Adopt the head-inline prefetch on the first poll (see Prefetch.tsx),
        // which is already in flight before this component exists.
        const store = (window as unknown as { __ttPrefetch?: Record<string, Promise<unknown>> }).__ttPrefetch;
        const warmed = store?.["/api/tt"];
        if (warmed) {
          delete store!["/api/tt"];
          data = (await warmed) as TtFeed | null;
        }
        if (!data) {
          const res = await fetch("/api/tt", { cache: "no-store" });
          data = await res.json();
        }
      } catch {
        data = null;
      }
      const empty = !data || (!data.predictions && !data.live);
      if (empty) {
        const snap = await loadSnapshot();
        if (!alive) return;
        if (snap) { setFeed(snap); setSnapshot(true); setError(false); return; }
      }
      if (!alive) return;
      if (data) { setFeed(data); setSnapshot(false); setError(false); }
      else { setError(true); }
    };
    load();
    const t = setInterval(load, 8000);
    return () => { alive = false; clearInterval(t); };
  }, []);
  return { feed, error, snapshot };
}

/* ── match centre ── */

type Row = {
  pred: TtPrediction;
  live?: TtLiveMatch;
  status: "live" | "scheduled" | "done";
};

export function TtMatchCentre({ email }: { email: string }) {
  const { feed, error, snapshot } = useTtFeed();
  const [cat, setCat] = useState<string>("ALL");
  const [statusFilter, setStatusFilter] = useState<"all" | "live" | "sched">("all");
  const [selected, setSelected] = useState<number | null>(null);

  const rows = useMemo<Row[]>(() => {
    if (!feed?.predictions) return [];
    const liveById = new Map((feed.live?.matches ?? []).map(m => [m.event_id, m]));
    const seen = new Set<number>();
    const out: Row[] = feed.predictions.predictions
      .filter(p => p.status !== "canceled")
      .map(p => {
        seen.add(p.event_id);
        const live = liveById.get(p.event_id);
        return {
          pred: p,
          live,
          status: live ? "live" : p.status === "finished" ? "done" : "scheduled",
        } as Row;
      });
    // live matches the pre-match file never saw (started after last predict run)
    liveById.forEach((m, id) => {
      if (seen.has(id)) return;
      out.push({
        live: m,
        status: "live",
        pred: {
          event_id: id, category: m.category, tournament: m.tournament,
          start_ts: 0, status: "inprogress", p1: m.p1, p2: m.p2,
          p1_win: m.pre_match_p1, p2_win: 1 - m.pre_match_p1,
          elo_p1: 0, elo_p2: 0, matches_known: 0, confidence: "live",
        },
      });
    });
    return out.sort((a, b) =>
      (a.status === "live" ? 0 : 1) - (b.status === "live" ? 0 : 1) ||
      a.pred.start_ts - b.pred.start_ts);
  }, [feed]);

  const cats = useMemo(() => {
    const c = new Map<string, number>();
    rows.forEach(r => c.set(r.pred.category, (c.get(r.pred.category) || 0) + 1));
    return [...c.entries()].sort((a, b) => b[1] - a[1]);
  }, [rows]);

  const visible = rows.filter(r =>
    (cat === "ALL" || r.pred.category === cat) &&
    (statusFilter === "all" || (statusFilter === "live" ? r.status === "live" : r.status === "scheduled")));

  const liveCount = rows.filter(r => r.status === "live").length;
  const selectedRow = rows.find(r => r.pred.event_id === selected) ?? null;

  const heldOut = heldOutAccuracy(feed?.metrics ?? null);

  // a deploy snapshot only refreshes on redeploy, so say how old it is —
  // a day-old file means yesterday's slate, which must not read as today's
  const snapshotAgeH = snapshot && feed?.predictions
    ? Math.floor((Date.now() / 1000 - feed.predictions.generated_ts) / 3600)
    : null;

  const liveStale = feed?.live && Date.now() / 1000 - feed.live.generated_ts > 90;
  const preStale = feed?.predictions && Date.now() / 1000 - feed.predictions.generated_ts > 12 * 3600;
  // Production serves whatever the local pipeline last pushed (see
  // tabletennis/push.py). Reaching the endpoint but getting nothing back means
  // nothing has been pushed yet — say so, instead of an endless "Loading…".
  const noData = !!feed && !feed.predictions && !feed.live;

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* filter bar */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-terminal-border shrink-0 text-[10px] overflow-x-auto">
        {["ALL", ...cats.map(([c]) => c)].map(c => (
          <button key={c} onClick={() => setCat(c)}
            className={`px-2 py-0.5 rounded font-bold whitespace-nowrap ${cat === c ? "bg-terminal-cyan/20 text-terminal-cyan" : "text-terminal-muted hover:text-slate-300"}`}>
            {c}{c !== "ALL" && ` (${cats.find(([n]) => n === c)?.[1]})`}
          </button>
        ))}
        <span className="text-terminal-border">│</span>
        <button onClick={() => setStatusFilter("all")}
          className={`px-2 py-0.5 rounded font-bold ${statusFilter === "all" ? "bg-terminal-cyan/20 text-terminal-cyan" : "text-terminal-muted"}`}>
          ALL ({rows.length})
        </button>
        <button onClick={() => setStatusFilter("live")}
          className={`px-2 py-0.5 rounded font-bold ${statusFilter === "live" ? "bg-terminal-green/20 text-terminal-green" : "text-terminal-muted"}`}>
          🔴 LIVE ({liveCount})
        </button>
        <button onClick={() => setStatusFilter("sched")}
          className={`px-2 py-0.5 rounded font-bold ${statusFilter === "sched" ? "bg-terminal-yellow/20 text-terminal-yellow" : "text-terminal-muted"}`}>
          📋 SCHED
        </button>
        <span className="ml-auto text-terminal-muted whitespace-nowrap">
          {error ? <span className="text-terminal-red">feed unreachable</span>
            : noData ? <span className="text-terminal-yellow">⚠ no data pushed yet — run python -m tabletennis.push</span>
            : snapshot ? (
              <span className="text-terminal-yellow"
                title="Live push feed is offline, so this is the pre-match snapshot shipped with the deploy. In-play True P needs the push path.">
                ⚠ snapshot{snapshotAgeH != null && ` ${snapshotAgeH}h old`} — pre-match only, no live feed
              </span>
            )
            : liveStale ? <span className="text-terminal-yellow">⚠ live poller stale — run python -m tabletennis.live</span>
            : preStale ? <span className="text-terminal-yellow">⚠ pre-match file &gt;12h old</span>
            : `${rows.length} fixtures · model ${feed?.predictions?.model ?? "…"}`}
          {heldOut != null && (
            <span title={`Held-out walk-forward validation (${feed?.metrics?.best_model ?? "best model"}, `
              + `n=${feed?.metrics?.n_test_rows ?? "?"}) — see tabletennis/site/metrics.json`}>
              {" "}· held-out acc {(heldOut * 100).toFixed(1)}%
            </span>
          )}
        </span>
      </div>

      {!feed ? (
        <div className="flex-1 flex items-center justify-center text-[11px] text-terminal-muted">
          Loading table-tennis intelligence…
        </div>
      ) : noData ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-2 text-center px-6">
          <div className="text-2xl">🏓</div>
          <div className="text-[12px] font-bold text-slate-200">No table-tennis data available</div>
          <div className="text-[10px] text-terminal-muted max-w-[420px] leading-relaxed">
            The feed is reachable but empty. TT predictions are generated locally (the model
            needs the SofaScore proxy) and pushed to the site, so the pipeline has to be
            running and pushing:
            <span className="block mt-1 font-mono text-terminal-cyan">
              python -m tabletennis.live<br />
              python -m tabletennis.push
            </span>
            <span className="block mt-1">
              No deploy snapshot either — <span className="font-mono text-terminal-cyan">
              python -m tabletennis.push --snapshot-only</span> then redeploy to ship the
              pre-match board without the live feed.
            </span>
          </div>
        </div>
      ) : (
        <div className="flex-1 min-h-0 flex">
          {/* left: match list */}
          <div className="w-1/2 border-r border-terminal-border overflow-y-auto">
            {visible.map(r => (
              <MatchRow key={r.pred.event_id} row={r}
                selected={selected === r.pred.event_id}
                onClick={() => setSelected(r.pred.event_id)} />
            ))}
            {visible.length === 0 && (
              <div className="text-terminal-muted text-[10px] text-center py-6">no fixtures match the filter</div>
            )}
          </div>
          {/* right: edge panel */}
          <div className="w-1/2 overflow-y-auto">
            {selectedRow ? (
              <EdgePanel row={selectedRow} email={email} />
            ) : (
              <EdgeBoard rows={rows} onSelect={setSelected} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function fmtTime(ts: number) {
  if (!ts) return "LIVE";
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function MatchRow({ row, selected, onClick }: { row: Row; selected: boolean; onClick: () => void }) {
  const { pred, live, status } = row;
  const p1 = live ? live.p1_win : pred.p1_win;
  const fav = p1 >= 0.5 ? 1 : 2;
  const delta = live ? live.p1_win - live.pre_match_p1 : 0;
  return (
    <button onClick={onClick}
      className={`w-full text-left px-3 py-1.5 border-b border-terminal-border hover:bg-terminal-panel/40 ${selected ? "bg-terminal-panel/60" : ""}`}>
      <div className="flex items-center gap-2">
        <span className={`text-[9px] font-bold w-[42px] shrink-0 ${status === "live" ? "text-terminal-green" : "text-terminal-muted"}`}>
          {status === "live" ? <span className="live-dot">● LIVE</span> : fmtTime(pred.start_ts)}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-[10px] truncate">
            <span className={fav === 1 ? "text-slate-100 font-medium" : "text-slate-400"}>{pred.p1}</span>
            <span className="text-terminal-muted"> v </span>
            <span className={fav === 2 ? "text-slate-100 font-medium" : "text-slate-400"}>{pred.p2}</span>
          </span>
          <span className="block text-[8px] text-terminal-muted truncate">
            {pred.category}
            {live && <span className="text-terminal-cyan font-mono"> · {live.games[0]}–{live.games[1]} g · {live.points[0]}–{live.points[1]}</span>}
          </span>
        </span>
        {live && <Sparkline history={live.history} />}
        <span className="text-right shrink-0 w-[52px]">
          <span className={`block text-[10px] font-mono font-bold ${status === "live" ? "text-terminal-green" : "text-slate-200"}`}>
            {(p1 * 100).toFixed(0)}%
          </span>
          {live && Math.abs(delta) >= 0.02 && (
            <span className={`block text-[8px] font-mono ${delta > 0 ? "text-terminal-green" : "text-terminal-red"}`}>
              {delta > 0 ? "▲" : "▼"} {delta > 0 ? "+" : ""}{(delta * 100).toFixed(0)}pp
            </span>
          )}
        </span>
      </div>
    </button>
  );
}

/** live True P history as a tiny inline SVG */
function Sparkline({ history }: { history: [number, number][] }) {
  if (history.length < 2) return null;
  const w = 48, h = 16;
  const ps = history.slice(-40).map(([, p]) => p);
  const min = Math.min(...ps), max = Math.max(...ps);
  const span = Math.max(max - min, 0.04);
  const pts = ps.map((p, i) =>
    `${(i / (ps.length - 1)) * w},${h - ((p - min) / span) * (h - 2) - 1}`).join(" ");
  const up = ps[ps.length - 1] >= ps[0];
  return (
    <svg width={w} height={h} className="shrink-0" aria-hidden>
      <polyline points={pts} fill="none" strokeWidth="1.2"
        stroke={up ? "#22c55e" : "#ef4444"} opacity="0.9" />
    </svg>
  );
}

/* ── edge board: what the intelligence flags right now ── */

function EdgeBoard({ rows, onSelect }: { rows: Row[]; onSelect: (id: number) => void }) {
  const movers = rows
    .filter(r => r.live)
    .map(r => ({ r, div: r.live!.p1_win - r.live!.pre_match_p1 }))
    .sort((a, b) => Math.abs(b.div) - Math.abs(a.div))
    .slice(0, 12);
  const characters = rows.filter(r => r.live?.annotation);
  const confident = rows
    .filter(r => r.status === "scheduled" && r.pred.confidence === "high")
    .map(r => ({ r, lean: Math.abs(r.pred.p1_win - 0.5) }))
    .sort((a, b) => b.lean - a.lean)
    .slice(0, 10);

  return (
    <div>
      <SectionHead tone="green">💎 LIVE MOVERS — True P vs pre-match ({movers.length})</SectionHead>
      {movers.map(({ r, div }) => (
        <button key={r.pred.event_id} onClick={() => onSelect(r.pred.event_id)}
          className="w-full text-left px-3 py-1.5 border-b border-terminal-border hover:bg-terminal-panel/40 flex items-center gap-2 text-[10px]">
          <span className={`font-mono font-bold w-[58px] ${div > 0 ? "text-terminal-green" : "text-terminal-red"}`}>
            {div > 0 ? "▲" : "▼"} {div > 0 ? "+" : ""}{(div * 100).toFixed(1)}pp
          </span>
          <span className="flex-1 truncate text-slate-200">{r.pred.p1} v {r.pred.p2}</span>
          <span className="text-terminal-muted font-mono">
            {(r.live!.p1_win * 100).toFixed(0)}% · {r.live!.games[0]}–{r.live!.games[1]}
          </span>
        </button>
      ))}
      {movers.length === 0 && <Empty>no live matches right now</Empty>}

      {characters.length > 0 && (
        <>
          <SectionHead tone="cyan">🧠 CHARACTER RESIDUALS IN PLAY ({characters.length})</SectionHead>
          {characters.map(r => (
            <button key={r.pred.event_id} onClick={() => onSelect(r.pred.event_id)}
              className="w-full text-left px-3 py-1.5 border-b border-terminal-border hover:bg-terminal-panel/40 text-[10px]">
              <span className="text-slate-200">{r.pred.p1} v {r.pred.p2}</span>
              <span className="block text-[9px] text-terminal-cyan">{r.live!.annotation}</span>
            </button>
          ))}
        </>
      )}

      <SectionHead tone="yellow">🎯 STRONGEST PRE-MATCH LEANS — high confidence ({confident.length})</SectionHead>
      {confident.map(({ r }) => {
        const fav = r.pred.p1_win >= 0.5 ? r.pred.p1 : r.pred.p2;
        const p = Math.max(r.pred.p1_win, r.pred.p2_win);
        return (
          <button key={r.pred.event_id} onClick={() => onSelect(r.pred.event_id)}
            className="w-full text-left px-3 py-1.5 border-b border-terminal-border hover:bg-terminal-panel/40 flex items-center gap-2 text-[10px]">
            <span className="text-terminal-muted w-[42px]">{fmtTime(r.pred.start_ts)}</span>
            <span className="flex-1 truncate">
              <span className="text-slate-100 font-medium">{fav}</span>
              <span className="text-terminal-muted"> v {fav === r.pred.p1 ? r.pred.p2 : r.pred.p1}</span>
            </span>
            <span className="text-terminal-green font-mono font-bold">{(p * 100).toFixed(1)}%</span>
            <span className="text-terminal-muted font-mono">fair {(1 / p).toFixed(2)}</span>
          </button>
        );
      })}
      {confident.length === 0 && <Empty>no high-confidence scheduled fixtures</Empty>}

      <div className="px-3 py-2 text-[9px] text-terminal-muted leading-relaxed">
        True P: walk-forward Elo ⊕ GBDT pre-match → exact race-to-11/win-by-2 recursion once live,
        anchored to the model by inversion, plus a ±15pp-capped character residual (clutch, deuce
        composure, comeback, front-running, fatigue). Select a match to open the edge panel and bet ticket.
      </div>
    </div>
  );
}

/* ── edge panel: one match, full intelligence + bet ticket ── */

function EdgePanel({ row, email }: { row: Row; email: string }) {
  const { pred, live } = row;
  const p1 = live ? live.p1_win : pred.p1_win;
  const [side, setSide] = useState<1 | 2>(p1 >= 0.5 ? 1 : 2);
  const pick = side === 1 ? pred.p1 : pred.p2;
  const pPick = side === 1 ? p1 : 1 - p1;
  const fair = pPick > 0 ? 1 / pPick : 0;
  const [odds, setOdds] = useState<string>("");
  const [stake, setStake] = useState<string>("25");
  const [placed, setPlaced] = useState(false);

  useEffect(() => { setSide(p1 >= 0.5 ? 1 : 2); setPlaced(false); setOdds(""); }, [pred.event_id]); // eslint-disable-line react-hooks/exhaustive-deps

  const oddsN = parseFloat(odds) || fair;
  const stakeN = Math.max(0, parseFloat(stake) || 0);
  const edge = pPick - 1 / oddsN;
  const kelly = oddsN > 1 ? Math.max(0, (pPick * oddsN - 1) / (oddsN - 1)) : 0;

  const place = () => {
    if (!stakeN) return;
    const bets = loadBets(email);
    bets.unshift({
      id: `${pred.event_id}-${Date.now()}`,
      ts: Date.now(),
      eventId: pred.event_id,
      category: pred.category,
      match: `${pred.p1} v ${pred.p2}`,
      pick, side, stake: stakeN, odds: oddsN, pAtBet: pPick,
      live: !!live, status: "open",
    });
    saveBets(email, bets);
    setPlaced(true);
  };

  return (
    <div className="p-4 space-y-3">
      <div className="text-center">
        <div className="text-terminal-yellow font-bold text-[11px]">⚡ EDGE ANALYSIS</div>
        <div className="text-slate-100 text-sm font-bold mt-1">{pred.p1} vs {pred.p2}</div>
        <div className="text-[9px] text-terminal-muted">
          {pred.category} · {pred.tournament} · best of {live?.best_of ?? 5}
          {live && <span className="text-terminal-green font-bold"> · ● LIVE {live.games[0]}–{live.games[1]} g, {live.points[0]}–{live.points[1]}</span>}
        </div>
      </div>

      {/* model probability bars */}
      <Panel title={live ? "LIVE TRUE P" : "MODEL PROBABILITY"}>
        <ProbBar name={pred.p1} p={p1} tone="green" />
        <ProbBar name={pred.p2} p={1 - p1} tone="cyan" />
        <div className="text-[8px] text-terminal-muted text-center pt-1">
          {live
            ? `pre-match ${(live.pre_match_p1 * 100).toFixed(1)}% → analytic ${(live.analytic_p1 * 100).toFixed(1)}% + residual ${live.residual >= 0 ? "+" : ""}${(live.residual * 100).toFixed(1)}pp`
            : `Elo ${pred.elo_p1.toFixed(0)} vs ${pred.elo_p2.toFixed(0)} · ${pred.matches_known} matches known · confidence ${pred.confidence}`}
        </div>
        {live?.annotation && (
          <div className="text-[9px] text-terminal-cyan text-center">🧠 {live.annotation}</div>
        )}
        {live && live.history.length > 1 && (
          <div className="flex justify-center pt-1"><BigSparkline history={live.history} /></div>
        )}
      </Panel>

      {/* bet ticket */}
      <Panel title="PAPER BET TICKET">
        <div className="flex gap-2">
          {([1, 2] as const).map(s => (
            <button key={s} onClick={() => setSide(s)}
              className={`flex-1 px-2 py-1.5 rounded border text-[10px] font-bold truncate ${
                side === s ? "bg-terminal-green/15 border-terminal-green text-terminal-green"
                  : "border-terminal-border text-terminal-muted hover:text-slate-300"}`}>
              {s === 1 ? pred.p1 : pred.p2} · {((s === 1 ? p1 : 1 - p1) * 100).toFixed(0)}%
            </button>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <label className="text-terminal-muted">
            Odds taken (fair {fair.toFixed(2)})
            <input value={odds} onChange={e => setOdds(e.target.value)} placeholder={fair.toFixed(2)}
              className="w-full mt-0.5 bg-terminal-bg border border-terminal-border rounded px-1.5 py-1 text-slate-200 focus:border-terminal-cyan outline-none font-mono" />
          </label>
          <label className="text-terminal-muted">
            Stake $
            <input value={stake} onChange={e => setStake(e.target.value)}
              className="w-full mt-0.5 bg-terminal-bg border border-terminal-border rounded px-1.5 py-1 text-slate-200 focus:border-terminal-cyan outline-none font-mono" />
          </label>
        </div>
        <div className="flex items-center justify-between text-[9px] font-mono">
          <span className={edge >= 0.02 ? "text-terminal-green font-bold" : edge > 0 ? "text-terminal-yellow" : "text-terminal-red"}>
            edge {edge >= 0 ? "+" : ""}{(edge * 100).toFixed(1)}%
          </span>
          <span className="text-terminal-muted">full Kelly {(kelly * 100).toFixed(0)}% · ¼ Kelly {(kelly * 25).toFixed(0)}%</span>
          <span className="text-terminal-muted">to win ${(stakeN * (oddsN - 1)).toFixed(0)}</span>
        </div>
        {placed ? (
          <div className="text-center text-[10px] text-terminal-green font-bold py-1.5">
            ✓ LOGGED TO BET TRACKER
          </div>
        ) : (
          <button onClick={place} disabled={!stakeN}
            className="w-full py-2 rounded bg-terminal-green text-black text-[11px] font-bold hover:opacity-90 disabled:opacity-40">
            ⚡ LOG PAPER BET — {pick} @ {oddsN.toFixed(2)}
          </button>
        )}
        <div className="text-[8px] text-terminal-muted text-center">
          Paper journal only (localStorage) — TT leagues have no Polymarket listing. Enter the odds your
          book shows; blank uses model-fair. Settle from the TT BETS view.
        </div>
      </Panel>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border border-terminal-border rounded">
      <div className="px-2.5 py-1 text-[8px] font-bold tracking-wider text-terminal-muted uppercase border-b border-terminal-border bg-terminal-panel/50">
        {title}
      </div>
      <div className="p-2.5 space-y-1.5">{children}</div>
    </div>
  );
}

function ProbBar({ name, p, tone }: { name: string; p: number; tone: "green" | "cyan" }) {
  const c = tone === "green" ? "bg-terminal-green" : "bg-terminal-cyan";
  const t = tone === "green" ? "text-terminal-green" : "text-terminal-cyan";
  return (
    <div>
      <div className="flex justify-between text-[10px]">
        <span className={`font-bold ${t}`}>{name}</span>
        <span className="font-mono text-slate-200">{(p * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 bg-terminal-bg rounded overflow-hidden">
        <div className={`h-full ${c} rounded`} style={{ width: `${p * 100}%` }} />
      </div>
    </div>
  );
}

function BigSparkline({ history }: { history: [number, number][] }) {
  const w = 260, h = 44;
  const ps = history.map(([, p]) => p);
  const pts = ps.map((p, i) =>
    `${(i / Math.max(ps.length - 1, 1)) * w},${h - p * (h - 4) - 2}`).join(" ");
  return (
    <svg width={w} height={h} className="border border-terminal-border rounded bg-terminal-bg" aria-label="live win probability history">
      <line x1="0" y1={h - 0.5 * (h - 4) - 2} x2={w} y2={h - 0.5 * (h - 4) - 2} stroke="#334155" strokeDasharray="3 3" strokeWidth="0.5" />
      <polyline points={pts} fill="none" stroke="#22c55e" strokeWidth="1.5" />
    </svg>
  );
}

function SectionHead({ tone, children }: { tone: "green" | "cyan" | "yellow"; children: React.ReactNode }) {
  const c = tone === "green" ? "text-terminal-green" : tone === "cyan" ? "text-terminal-cyan" : "text-terminal-yellow";
  return (
    <div className={`px-3 py-1 bg-terminal-panel/50 border-y border-terminal-border sticky top-0 z-10 text-[10px] font-bold tracking-wider ${c}`}>
      {children}
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="text-terminal-muted text-[10px] text-center py-4">{children}</div>;
}

/* ── TT bet tracker ── */

export function TtBetTracker({ email }: { email: string }) {
  const [bets, setBets] = useState<TtBet[]>([]);
  useEffect(() => { setBets(loadBets(email)); }, [email]);

  const settle = useCallback((id: string, status: TtBet["status"]) => {
    setBets(prev => {
      const next = prev.map(b => (b.id === id ? { ...b, status } : b));
      saveBets(email, next);
      return next;
    });
  }, [email]);

  const settled = bets.filter(b => b.status === "won" || b.status === "lost");
  const staked = settled.reduce((s, b) => s + b.stake, 0);
  const profit = settled.reduce((s, b) => s + pnl(b), 0);
  const open = bets.filter(b => b.status === "open");

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="flex items-center gap-4 px-3 py-1.5 border-b border-terminal-border shrink-0 text-[10px] font-mono">
        <span className="text-terminal-muted">OPEN <b className="text-slate-200">{open.length}</b></span>
        <span className="text-terminal-muted">RECORD <b className="text-slate-200">
          {settled.filter(b => b.status === "won").length}–{settled.filter(b => b.status === "lost").length}
        </b></span>
        <span className="text-terminal-muted">STAKED <b className="text-slate-200">${staked.toFixed(0)}</b></span>
        <span className={`font-bold ${profit >= 0 ? "text-terminal-green" : "text-terminal-red"}`}>
          P/L {profit >= 0 ? "+" : ""}${profit.toFixed(2)}
        </span>
        {staked > 0 && (
          <span className={`font-bold ${profit >= 0 ? "text-terminal-green" : "text-terminal-red"}`}>
            ROI {((profit / staked) * 100).toFixed(1)}%
          </span>
        )}
        <span className="ml-auto text-terminal-muted">🏓 journal · {email}</span>
      </div>
      <div className="flex-1 overflow-y-auto">
        <div className="grid grid-cols-[90px_1fr_90px_50px_50px_54px_60px_120px] gap-1 px-3 py-1 text-[8px] font-bold text-terminal-muted uppercase tracking-wider border-b border-terminal-border">
          <span>Placed</span><span>Bet</span><span>League</span>
          <span className="text-right">Odds</span><span className="text-right">P@bet</span>
          <span className="text-right">Stake</span><span className="text-right">P/L</span>
          <span className="text-right">Status</span>
        </div>
        {bets.map(b => (
          <div key={b.id}
            className="grid grid-cols-[90px_1fr_90px_50px_50px_54px_60px_120px] gap-1 px-3 py-1.5 border-b border-terminal-border items-center text-[10px]">
            <span className="text-terminal-muted text-[9px]">
              {new Date(b.ts).toLocaleString([], { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })}
            </span>
            <span className="min-w-0">
              <span className="block truncate text-slate-100 font-medium">{b.pick}</span>
              <span className="block truncate text-[8px] text-terminal-muted">
                {b.match}{b.live && <span className="text-terminal-cyan"> · in-play</span>}
              </span>
            </span>
            <span className="text-[9px] text-terminal-muted truncate">{b.category}</span>
            <span className="text-right font-mono text-terminal-yellow">{b.odds.toFixed(2)}</span>
            <span className="text-right font-mono text-terminal-muted">{(b.pAtBet * 100).toFixed(0)}%</span>
            <span className="text-right font-mono text-slate-200">${b.stake.toFixed(0)}</span>
            <span className={`text-right font-mono font-bold ${
              b.status === "won" ? "text-terminal-green" : b.status === "lost" ? "text-terminal-red" : "text-terminal-muted"}`}>
              {b.status === "open" || b.status === "void" ? "—" : `${pnl(b) >= 0 ? "+" : ""}$${pnl(b).toFixed(2)}`}
            </span>
            <span className="text-right">
              {b.status === "open" ? (
                <span className="inline-flex gap-1">
                  <button onClick={() => settle(b.id, "won")}
                    className="text-[8px] font-bold px-1.5 py-0.5 rounded border border-terminal-green/40 text-terminal-green hover:bg-terminal-green/10">WIN</button>
                  <button onClick={() => settle(b.id, "lost")}
                    className="text-[8px] font-bold px-1.5 py-0.5 rounded border border-terminal-red/40 text-terminal-red hover:bg-terminal-red/10">LOSS</button>
                  <button onClick={() => settle(b.id, "void")}
                    className="text-[8px] font-bold px-1.5 py-0.5 rounded border border-terminal-border text-terminal-muted hover:bg-terminal-bg">VOID</button>
                </span>
              ) : (
                <span className={`text-[9px] font-bold ${
                  b.status === "won" ? "text-terminal-green" : b.status === "lost" ? "text-terminal-red" : "text-terminal-muted"}`}>
                  {b.status.toUpperCase()}
                </span>
              )}
            </span>
          </div>
        ))}
        {bets.length === 0 && (
          <div className="text-terminal-muted text-[10px] text-center py-8">
            No TT bets yet — log one from any match&apos;s edge panel in the 🏓 TABLE TENNIS view.
          </div>
        )}
      </div>
    </div>
  );
}
