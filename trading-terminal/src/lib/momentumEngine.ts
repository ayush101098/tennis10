/**
 * Live momentum engine (client-side TypeScript port of execution/momentum.py).
 *
 * The scoreboard lags: by the time it reads "5-4, break", the momentum that
 * produced it has been building for several games. This reads the SofaScore
 * point-by-point game log and derives a forward-looking momentum signal that the
 * main engine display shows alongside True P — so the number leans ahead of the
 * score instead of only reacting to it.
 *
 *   momentumP1   recency-weighted game control in [-1, 1]. Breaking serve counts
 *                far more than holding (the surprising, informative event).
 *   serveRegP1/2 serve regression: recent hold rate vs the rate implied by the
 *                player's point-win-on-serve, minus break-point pressure. Negative
 *                = serving below baseline *before* it shows up as a break.
 *
 * Kept in lock-step with the Python engine (execution/momentum.py) so the local
 * agent and the browser terminal read momentum the same way.
 */

const DECAY = 0.79; // recency: weight_i = DECAY**i, i=0 = most recent game (~3-game half-life)
const HOLD_INFO = 0.35; // a hold is expected -> low information weight
const BREAK_INFO = 1.0; // a break is surprising -> full information weight

export interface GameResult {
  server: 1 | 2 | null;
  winner: 1 | 2 | null;
  /** running (p1,p2) point-score strings, oriented to our P1 */
  points: [string | null, string | null][];
}

export interface MomentumState {
  momentumP1: number;
  serveRegP1: number;
  serveRegP2: number;
  completedGames: number;
  recentHoldsP1: number;
  recentBreaksP1: number;
  recentHoldsP2: number;
  recentBreaksP2: number;
  hasSignal: boolean;
}

/** Probability of holding a service game given point-win-on-serve p (Barnett-Clarke). */
export function gameWinProb(p: number): number {
  p = Math.min(Math.max(p, 0.01), 0.99);
  const q = 1 - p;
  const pre = p ** 4 + 4 * p ** 4 * q + 10 * p ** 4 * q * q;
  const deuce = 20 * p ** 3 * q ** 3;
  const winDeuce = (p * p) / (p * p + q * q);
  return pre + deuce * winDeuce;
}

const PT: Record<string, number> = { "0": 0, "15": 1, "30": 2, "40": 3, AD: 4, A: 4 };

function bpFaced(points: GameResult["points"], server: 1 | 2): number {
  if (!points?.length) return 0;
  let bp = 0;
  for (const [p1pt, p2pt] of points) {
    const a = PT[String(p1pt).toUpperCase()];
    const b = PT[String(p2pt).toUpperCase()];
    if (a === undefined || b === undefined) continue;
    const returnerPt = server === 1 ? b : a;
    const serverPt = server === 1 ? a : b;
    if (returnerPt === 4 || (returnerPt === 3 && serverPt < 3)) bp += 1;
  }
  return bp;
}

/**
 * Parse a SofaScore point-by-point payload into an oriented game log
 * (oldest -> newest). `sofaHomeIsP1` maps home/away (1/2) to our P1/P2.
 */
export function parseGameLog(pbp: unknown, sofaHomeIsP1: boolean): GameResult[] {
  const sets = (pbp as { pointByPoint?: unknown[] })?.pointByPoint;
  if (!Array.isArray(sets)) return [];
  const orient = (v: unknown): 1 | 2 | null =>
    v === 1 || v === 2 ? ((sofaHomeIsP1 ? v : 3 - v) as 1 | 2) : null;

  const rows: (GameResult & { set: number; game: number })[] = [];
  for (const st of sets) {
    const setNo = (st as { set?: number })?.set ?? 0;
    const games = (st as { games?: unknown[] })?.games ?? [];
    for (const g of games) {
      const sc = (g as { score?: { serving?: unknown; scoring?: unknown } })?.score ?? {};
      const scoring = sc.scoring;
      const rawPts = ((g as { points?: unknown[] })?.points ?? []) as {
        homePoint?: string;
        awayPoint?: string;
      }[];
      const points: GameResult["points"] = rawPts.map((p) =>
        sofaHomeIsP1
          ? [p.homePoint ?? null, p.awayPoint ?? null]
          : [p.awayPoint ?? null, p.homePoint ?? null],
      );
      rows.push({
        set: setNo,
        game: (g as { game?: number })?.game ?? 0,
        server: orient(sc.serving),
        winner: scoring === 1 || scoring === 2 ? orient(scoring) : null,
        points,
      });
    }
  }
  rows.sort((a, b) => a.set - b.set || a.game - b.game);
  return rows.map(({ server, winner, points }) => ({ server, winner, points }));
}

/**
 * Compute momentum from an oriented game log. sp1/sp2 are point-win-on-serve
 * priors (career or live blend) used to set the hold-rate expectation.
 */
export function computeMomentum(
  games: GameResult[],
  sp1: number,
  sp2: number,
  window = 8,
): MomentumState {
  const completed = (games || []).filter(
    (g) => (g.winner === 1 || g.winner === 2) && (g.server === 1 || g.server === 2),
  );
  const empty: MomentumState = {
    momentumP1: 0,
    serveRegP1: 0,
    serveRegP2: 0,
    completedGames: 0,
    recentHoldsP1: 0,
    recentBreaksP1: 0,
    recentHoldsP2: 0,
    recentBreaksP2: 0,
    hasSignal: false,
  };
  if (!completed.length) return empty;

  const recent = completed.slice(-window);
  const rev = [...recent].reverse(); // i=0 most recent

  // ── composite momentum ──
  let num = 0;
  let wsum = 0;
  rev.forEach((g, i) => {
    const w = DECAY ** i;
    const val = g.winner === 1 ? 1 : -1;
    const info = g.winner !== g.server ? BREAK_INFO : HOLD_INFO;
    num += w * val * info;
    wsum += w;
  });
  const momentumP1 = wsum ? Math.tanh(num / wsum) : 0;

  // ── per-player serve regression ──
  const serveReg = (player: 1 | 2, expHold: number): number => {
    let hnum = 0;
    let hden = 0;
    let bpPressure = 0;
    let nSg = 0;
    rev.forEach((g, i) => {
      if (g.server !== player) return;
      nSg += 1;
      const w = DECAY ** i;
      hnum += w * (g.winner === player ? 1 : 0);
      hden += w;
      bpPressure += w * Math.min(bpFaced(g.points, player), 3) * 0.06;
    });
    if (hden === 0) return 0;
    let reg = hnum / hden - expHold - bpPressure / hden;
    reg *= nSg / (nSg + 2); // shrink small-sample residuals toward 0
    return Math.max(-1, Math.min(1, reg));
  };

  let h1 = 0;
  let b1 = 0;
  let h2 = 0;
  let b2 = 0;
  for (const g of recent) {
    const held = g.winner === g.server;
    if (g.winner === 1) {
      held ? (h1 += 1) : (b1 += 1);
    } else {
      held ? (h2 += 1) : (b2 += 1);
    }
  }

  return {
    momentumP1: +momentumP1.toFixed(4),
    serveRegP1: +serveReg(1, gameWinProb(sp1)).toFixed(4),
    serveRegP2: +serveReg(2, gameWinProb(sp2)).toFixed(4),
    completedGames: completed.length,
    recentHoldsP1: h1,
    recentBreaksP1: b1,
    recentHoldsP2: h2,
    recentBreaksP2: b2,
    hasSignal: true,
  };
}
