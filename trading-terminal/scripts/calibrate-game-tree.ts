/**
 * Calibration harness for the joint game tree — PRD §35–§36, model metrics only.
 *
 * Replays reconstructed games from the point corpus through lib/gameTree — the
 * SAME engine the terminal runs, not a reimplementation — and answers two
 * questions in order:
 *
 *   1. Which estimate of p produces the most honest game probabilities?
 *      (a global base rate, the match's own hold rate, or live serve stats)
 *   2. Does an isotonic correction fix the tail overconfidence?
 *
 * WHAT THIS DELIBERATELY DOES NOT DO
 *   No market, no edge, no P&L. Nothing in the data feed prices a game, so a
 *   trading backtest would have to invent the counterparty. This answers what
 *   comes first and needs no market: are the probabilities honest?
 *
 * THE SPLIT IS BY MATCH, NOT BY GAME
 *   Games inside one match share a server, a surface and a day. Splitting at the
 *   game level leaks the test set into training through those shared traits and
 *   reports a calibration that will not survive contact with tomorrow's play.
 *
 * Run:  npx vite-node scripts/calibrate-game-tree.ts <games.json>
 */

import { readFileSync } from "node:fs";
import { gameTree } from "../src/lib/gameTree";

interface Game {
  fs: string;
  held: boolean;
  deuce: boolean;
  obs: [number, number][];      // (serverPts, returnerPts) snapshots
  n_obs: number;
  p_stat: number | null;        // point-in-time p from serve stats, pre-game
  n_serve: number;
}

type Pair = { p: number; outcome: boolean };

const all: Game[] = JSON.parse(readFileSync(process.argv[2], "utf8"));
/** Only games where every candidate estimator can be compared on equal terms. */
const games = all.filter(g => g.p_stat !== null);

/** p implied by a target hold rate, by inverting the tree at 0-0. */
function invertHoldRate(target: number): number {
  let lo = 0.2, hi = 0.95;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (gameTree(mid, 0, 0).pServer < target) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

/* ── the three estimates of p ─────────────────────────────────────────────── */

const GLOBAL_P = invertHoldRate(games.filter(g => g.held).length / games.length);

/** Per-match p from the match's own hold rate, excluding the game being scored. */
const byMatch = new Map<string, Game[]>();
for (const g of games) {
  const a = byMatch.get(g.fs) ?? []; a.push(g); byMatch.set(g.fs, a);
}
const looP = new Map<Game, number>();
for (const [, gs] of byMatch) {
  if (gs.length < 6) continue;
  const holds = gs.filter(x => x.held).length;
  for (const g of gs) {
    const r = (holds - (g.held ? 1 : 0)) / (gs.length - 1);
    looP.set(g, invertHoldRate(Math.min(Math.max(r, 0.05), 0.95)));
  }
}

/* ── train / test split, by match ─────────────────────────────────────────── */

const matchIds = [...new Set(games.map(g => g.fs))].sort();
const testMatches = new Set(matchIds.filter((_, i) => i % 2 === 1));
const isTest = (g: Game) => testMatches.has(g.fs);

/** Every in-game state worth a prediction, paired with what actually happened. */
function pairsFor(pick: (g: Game) => number | undefined, only: (g: Game) => boolean): Pair[] {
  const out: Pair[] = [];
  for (const g of games) {
    if (!only(g)) continue;
    const p = pick(g);
    if (p === undefined) continue;
    for (const [a, b] of g.obs) {
      if (a >= 4 && a - b >= 2) continue;   // already decided
      if (b >= 4 && b - a >= 2) continue;
      out.push({ p: gameTree(p, a, b).pServer, outcome: g.held });
    }
  }
  return out;
}

/* ── isotonic regression (pool-adjacent-violators) ────────────────────────── */

/**
 * Fits a monotone map from predicted to observed probability. Monotone because
 * the ordering of the model's predictions is the part worth keeping — the tree
 * knows 40-0 beats 0-40 — while the spacing is what it gets wrong. Isotonic
 * repairs the spacing and cannot reverse the ordering.
 */
function fitIsotonic(train: Pair[]): (p: number) => number {
  const pts = [...train].sort((a, b) => a.p - b.p);
  const v: number[] = [], w: number[] = [], x: number[] = [];
  for (const { p, outcome } of pts) {
    v.push(outcome ? 1 : 0); w.push(1); x.push(p);
    while (v.length > 1 && v[v.length - 2] > v[v.length - 1]) {
      // All three arrays must shrink together. Popping x once while popping v
      // and w twice desynchronised them, and the lookup then read past the end
      // of v and returned undefined — which surfaced as NaN predictions.
      const y2 = v.pop()!, w2 = w.pop()!, x2 = x.pop()!;
      const y1 = v.pop()!, w1 = w.pop()!; x.pop();
      v.push((y1 * w1 + y2 * w2) / (w1 + w2)); w.push(w1 + w2); x.push(x2);
    }
  }
  return (p: number) => {
    let lo = 0, hi = x.length - 1;
    if (p <= x[0]) return v[0];
    if (p >= x[hi]) return v[hi];
    while (lo < hi) { const m = (lo + hi) >> 1; if (x[m] < p) lo = m + 1; else hi = m; }
    return v[lo];
  };
}

/* ── scoring ──────────────────────────────────────────────────────────────── */

function score(label: string, pairs: Pair[], showTable = true) {
  if (!pairs.length) { console.log(`${label}: no observations\n`); return; }
  const n = pairs.length;
  const base = pairs.filter(x => x.outcome).length / n;
  let brier = 0, ll = 0;
  for (const { p, outcome } of pairs) {
    const y = outcome ? 1 : 0;
    brier += (p - y) ** 2;
    const c = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
    ll += -(y * Math.log(c) + (1 - y) * Math.log(1 - c));
  }
  brier /= n; ll /= n;
  const ref = base * (1 - base);            // always predict the base rate
  const skill = 1 - brier / ref;

  // Mean |predicted - actual| across buckets: the number that says how far a
  // quoted probability is from the truth, which is what an edge is measured in.
  const bins = Array.from({ length: 10 }, () => ({ n: 0, sp: 0, so: 0 }));
  for (const { p, outcome } of pairs) {
    const b = bins[Math.min(9, Math.floor(p * 10))];
    b.n++; b.sp += p; b.so += outcome ? 1 : 0;
  }
  let gapW = 0, gapN = 0, worst = 0;
  for (const b of bins) {
    if (b.n < 20) continue;
    const gap = Math.abs(b.so / b.n - b.sp / b.n);
    gapW += gap * b.n; gapN += b.n; worst = Math.max(worst, gap);
  }

  console.log(`── ${label} ──`);
  console.log(`  n ${n}   base ${(base * 100).toFixed(1)}%   Brier ${brier.toFixed(4)} (ref ${ref.toFixed(4)})`
    + `   skill ${skill >= 0 ? "+" : ""}${(skill * 100).toFixed(1)}%   logloss ${ll.toFixed(4)}`);
  console.log(`  calibration error: mean ${(100 * gapW / gapN).toFixed(1)}pts   worst bucket ${(worst * 100).toFixed(1)}pts`);
  if (showTable) {
    console.log(`     bucket    n   predicted   actual    gap`);
    for (let i = 0; i < 10; i++) {
      const b = bins[i];
      if (b.n < 20) continue;
      const pr = b.sp / b.n, ac = b.so / b.n, gp = ac - pr;
      console.log(`     ${(i * 10).toString().padStart(2)}-${i * 10 + 10}%  ${String(b.n).padStart(5)}   `
        + `${(pr * 100).toFixed(1).padStart(6)}%  ${(ac * 100).toFixed(1).padStart(6)}%  ${gp >= 0 ? "+" : ""}${(gp * 100).toFixed(1)}`);
    }
  }
  console.log();
}

/* ── run ──────────────────────────────────────────────────────────────────── */

console.log(`games ${games.length}  matches ${matchIds.length}  `
  + `(train ${matchIds.length - testMatches.size} / test ${testMatches.size} matches)`);
console.log(`global p ${GLOBAL_P.toFixed(4)}   corpus hold ${(100 * games.filter(g => g.held).length / games.length).toFixed(1)}%\n`);

console.log("═══ HELD-OUT TEST SET — same games, three estimates of p ═══\n");
const testGlobal = pairsFor(() => GLOBAL_P, isTest);
const testLoo = pairsFor(g => looP.get(g), isTest);
const testStat = pairsFor(g => g.p_stat ?? undefined, isTest);
score("p = global base rate", testGlobal, false);
score("p = match hold rate (leave-one-game-out)", testLoo, false);
score("p = live serve stats, point-in-time", testStat);

/**
 * SHRINKAGE toward the tour base rate.
 *
 * The raw serve-stats result below is the clue: a p measured off ~12 serve
 * points is extremely noisy, and the tree amplifies noise in p rather than
 * averaging it out — hold probability is a steep sigmoid in p, so a server who
 * has happened to win 9 of 12 reads as p=0.75 and prices as a near-certain
 * hold. Pulling the estimate toward a prior in proportion to how little
 * evidence stands behind it is the standard repair.
 *
 *   p_shrunk = (n·p_observed + k·p_prior) / (n + k)
 *
 * k is the weight of the prior in serve-points. k = 40 says "trust the match's
 * own serving only once you have seen about 40 serve points", which is roughly
 * three service games.
 */
const PRIOR = GLOBAL_P;
const shrink = (k: number) => (g: Game): number | undefined =>
  g.p_stat === null ? undefined : (g.n_serve * g.p_stat + k * PRIOR) / (g.n_serve + k);

console.log("═══ SHRINKAGE — serve stats pulled toward the base rate ═══\n");
for (const k of [10, 20, 40, 80]) {
  score(`p = serve stats shrunk toward base rate (k=${k} serve pts)`, pairsFor(shrink(k), isTest), false);
}

console.log("═══ ISOTONIC CORRECTION — fitted on train matches, scored on test ═══\n");
const variants: [string, (g: Game) => number | undefined][] = [
  ["global p", () => GLOBAL_P],
  ["serve stats raw", g => g.p_stat ?? undefined],
  ["serve stats shrunk k=40", shrink(40)],
];
for (const [name, pick] of variants) {
  const iso = fitIsotonic(pairsFor(pick, g => !isTest(g)));
  const scored = pairsFor(pick, isTest).map(({ p, outcome }) => ({ p: iso(p), outcome }));
  score(`${name} + isotonic`, scored, name === "serve stats shrunk k=40");
}
