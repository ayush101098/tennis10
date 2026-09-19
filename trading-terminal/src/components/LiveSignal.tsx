"use client";

import { useEffect, useState } from "react";
import type { ScheduledMatch } from "@/lib/scheduleService";
import { displayName } from "@/lib/scheduleService";
import { observePressure, pressureSignal, type PressureSignal } from "@/lib/gamePressure";
import { POINT_INDEX } from "@/lib/gameTree";
import GameStateTree from "@/components/GameStateTree";

/**
 * The top of the match panel: what is happening, and how sure we are.
 *
 * This exists because the panel below it had grown to eleven sections — serve
 * efficiency, return pressure, hold/break, momentum, stats — each a full block
 * with its own heading. Every one of them is an input to the same question, and
 * none of them answered it. Someone opening a live match had to read the whole
 * page and do the combining themselves.
 *
 * So the order is inverted: the answer first, the evidence under it, the
 * workings further down for anyone who wants them.
 *
 * NO JARGON ON THIS SURFACE. "Markov", "Kelly", "true probability" and
 * "de-vigged" are all accurate and all meaningless to someone deciding whether
 * to watch this match. The words here are the ones a tennis viewer already has:
 * serving, break point, held, under pressure.
 */

/** Recover the server's point-win rate from the serve breakdown. */
function pointWinRate(m: ScheduledMatch): number | null {
  const b = m.liveScore?.breakHoldSignals?.serveBreakdown;
  if (!b) return null;
  const f = b.firstServeIn / 100;
  if (!(f > 0 && f < 1)) return null;
  const p = f * (b.firstServeWon / 100) + (1 - f) * (b.secondServeWon / 100);
  return p > 0 && p < 1 ? p : null;
}

export default function LiveSignal({ match: m }: { match: ScheduledMatch }) {
  const [sig, setSig] = useState<PressureSignal | null>(null);
  const [showTree, setShowTree] = useState(false);
  const live = m.status === "live";
  const ls = m.liveScore;

  // Fold every poll into the match's pressure memory. The engine is idempotent
  // on a repeated snapshot, so polling faster than points are played is safe.
  useEffect(() => {
    if (!live || !ls?.server || !ls.pointScore) { setSig(null); return; }
    const srvIdx = POINT_INDEX[String(ls.server === 1 ? ls.pointScore.p1 : ls.pointScore.p2).toUpperCase()];
    const retIdx = POINT_INDEX[String(ls.server === 1 ? ls.pointScore.p2 : ls.pointScore.p1).toUpperCase()];
    if (srvIdx === undefined || retIdx === undefined) { setSig(null); return; }

    const state = observePressure({
      matchId: m.id,
      server: ls.server,
      srvPts: srvIdx,
      retPts: retIdx,
      gamesP1: ls.currentSetGames?.p1 ?? 0,
      gamesP2: ls.currentSetGames?.p2 ?? 0,
      setIndex: (ls.completedSets?.length ?? 0) + 1,
    });
    const p = pointWinRate(m);
    setSig(pressureSignal(state, p ?? 0.6, srvIdx, retIdx));
  }, [m.id, live, ls, m]);

  // ── the headline probability ──
  // Always populated for a live match with a score now — even with the server
  // unknown, computeLiveMatchProbNoServer still runs (see scheduleService.ts's
  // attachBreakHoldSignals). "not available" below is what shows only in the
  // narrow window before the first live score has arrived at all.
  const liveP = ls?.trueProbabilities?.p1MatchProb;
  const noServer = ls?.trueProbabilities?.method === "markov-no-server";
  const shownP = liveP ?? m.p1_win_prob;
  const isLiveNumber = liveP != null;
  const favIsP1 = shownP >= 0.5;
  const favName = favIsP1 ? m.player1 : m.player2;
  const favPct = Math.round((favIsP1 ? shownP : 1 - shownP) * 100);
  // The pre-match figure for the SAME player, so the two are comparable. Taking
  // each side's own favourite would compare different players and make the
  // arrow meaningless.
  const preFavPct = Math.round((favIsP1 ? m.p1_win_prob : 1 - m.p1_win_prob) * 100);

  const serverName = ls?.server === 1 ? m.player1 : ls?.server === 2 ? m.player2 : null;

  return (
    <div className="rounded-lg border border-terminal-border bg-terminal-panel/40 p-3 space-y-3">
      {/* ── who is winning: BOTH numbers, and the move between them ──
          Showing one number and a badge meant you could never see what the
          match had done to the forecast. The pre-match number is the prior and
          the live number is the same player re-priced on the score; the gap
          between them is the single most informative thing on this panel, and
          it only exists if both are on screen at once. */}
      <div>
        <div className="text-[10px] font-bold tracking-wider text-terminal-muted mb-1.5">
          CHANCE OF WINNING — {favName.toUpperCase()}
        </div>
        <div className="flex items-stretch gap-3">
          <Figure
            label="BEFORE PLAY"
            pct={preFavPct}
            tone="muted"
            title="What the model said before a ball was struck"
          />
          <div className="flex items-center text-terminal-muted text-lg">→</div>
          {isLiveNumber ? (
            <Figure
              label="● LIVE NOW"
              pct={favPct}
              tone="live"
              delta={favPct - preFavPct}
              title="Re-priced on the current score, updated as points are played"
            />
          ) : (
            <div className="flex-1 rounded border border-dashed border-terminal-border px-2.5 py-1.5">
              <div className="text-[9px] font-bold text-terminal-muted">LIVE NOW</div>
              <div className="text-sm text-terminal-muted mt-0.5">
                {live ? "not available" : "match not started"}
              </div>
            </div>
          )}
        </div>
        {!isLiveNumber && live && (
          <div className="text-[10px] text-terminal-yellow mt-1.5">
            No live score has come through for this match yet. Showing the
            pre-match figure only.
          </div>
        )}
        {isLiveNumber && noServer && (
          <div className="text-[10px] text-terminal-muted mt-1.5">
            The feed isn&apos;t saying who is serving, so this uses the set and
            game score only — a touch less precise than when it can, but it is
            a real, live number.
          </div>
        )}
      </div>

      {/* ── GAME / SET / MATCH — who is favoured at each level ──
          One blended number hides the fact that these are three different
          questions: who wins the next few points, who wins this set, who wins
          the match. A player can be trailing in the game they are about to
          lose and still be the one favoured to win the match — that is a real
          and useful thing to see, and it disappears the moment the three are
          folded into one figure. */}
      {isLiveNumber && ls?.trueProbabilities && (
        <div className="grid grid-cols-3 gap-1.5">
          <LevelPick label="GAME" p1Prob={ls.trueProbabilities.p1GameProb} m={m}
            title={noServer ? "Averaged over both players possibly serving — the feed doesn't say who actually is" : "Who is favoured to win the game in progress"} />
          <LevelPick label="SET" p1Prob={ls.trueProbabilities.p1SetProb} m={m}
            title="Who is favoured to win the set in progress" />
          <LevelPick label="MATCH" p1Prob={ls.trueProbabilities.p1MatchProb} m={m}
            title="Who is favoured to win the match from here" />
        </div>
      )}

      {/* ── the signal ── */}
      {live && sig && sig.tier !== "NONE" && (
        <div className={`rounded border p-2.5 ${
          sig.tier === "S5_EXTREME" || sig.tier === "S6_REGIME"
            ? "border-terminal-red/50 bg-terminal-red/10"
            : sig.tier === "S3_STRONG" || sig.tier === "S4_VERY_STRONG"
            ? "border-terminal-yellow/50 bg-terminal-yellow/10"
            : "border-terminal-border bg-terminal-bg/40"
        }`}>
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <div className="text-[12px] font-bold text-slate-100">{sig.headline}</div>
              {sig.evidence.length > 0 && (
                <div className="text-[10px] text-terminal-muted mt-0.5">{sig.evidence.join(" · ")}</div>
              )}
            </div>
            <PressureMeter score={sig.score} />
          </div>

          {serverName && (
            <div className="text-[10px] text-terminal-muted mt-1.5">
              {serverName} serving
              {sig.trustworthy ? (
                <> · <span className="text-slate-300">{Math.round(sig.breakProb * 100)}% chance of being broken</span></>
              ) : (
                /* Outside the band where the model has been shown to be
                   accurate, so the number is withheld rather than dressed up.
                   Saying nothing is cheaper than being believed and wrong. */
                <> · <span className="text-terminal-muted">too one-sided to call reliably</span></>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── compact indicators, replacing four full sections ── */}
      {live && <Indicators match={m} sig={sig} />}

      {/* ── holds, breaks, and how long this is expected to go ── */}
      {live && <MatchArc match={m} />}

      {/* ── the game-level decision tree, on demand ──
          Collapsed by default: it is the densest thing on this panel, and the
          rest of the panel already answers "who's winning" — this is for
          someone who wants to see the whole shape of the game currently being
          played, not something everyone needs open every time. */}
      {live && (
        <div>
          <button onClick={() => setShowTree(v => !v)}
            className="w-full text-left text-[9px] font-bold text-terminal-muted hover:text-slate-300 flex items-center gap-1">
            <span>{showTree ? "▾" : "▸"}</span> GAME MAP — every path this game can take
          </button>
          {showTree && <div className="mt-1.5"><GameStateTree match={m} /></div>}
        </div>
      )}
    </div>
  );
}

/**
 * Holds/breaks so far, and the predicted length of the match.
 *
 * These sit together because they answer the same underlying question from
 * two directions: one looks BACK at what has already happened (who has held,
 * who has broken), the other looks FORWARD (how many more games this is
 * likely to take). Holds/breaks needs a real server reading to mean anything
 * — see the doc on ScheduledMatch.liveScore.holdsAndBreaks for why it cannot
 * be inferred from the score alone — so it has an honest fallback (games won,
 * which the score DOES tell you) rather than going blank.
 */
function MatchArc({ match: m }: { match: ScheduledMatch }) {
  const hb = m.liveScore?.holdsAndBreaks;
  const gp = m.liveScore?.gamesPrediction;
  const wl = m.liveScore?.liveWorkload;
  if (!hb && !gp) return null;

  const p1Short = displayName(m.player1);
  const p2Short = displayName(m.player2);
  const ls = m.liveScore;
  const gamesWonP1 = (ls?.completedSets ?? []).reduce((n, s) => n + s.p1, 0) + (ls?.currentSetGames?.p1 ?? 0);
  const gamesWonP2 = (ls?.completedSets ?? []).reduce((n, s) => n + s.p2, 0) + (ls?.currentSetGames?.p2 ?? 0);

  return (
    <div className="rounded border border-terminal-border bg-terminal-bg/40 p-2.5 space-y-2">
      <div>
        <div className="text-[8px] font-bold text-terminal-muted tracking-wider mb-1">HOLDS &amp; BREAKS</div>
        {hb ? (
          <div className="grid grid-cols-2 gap-x-4 text-[11px]">
            <div className="flex justify-between">
              <span className="text-slate-300">{p1Short}</span>
              <span className="font-mono">
                <span className="text-terminal-green">{hb.p1Holds} held</span>
                {" · "}
                <span className={hb.p1Breaks > 0 ? "text-terminal-red" : "text-terminal-muted"}>{hb.p1Breaks} broke</span>
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-300">{p2Short}</span>
              <span className="font-mono">
                <span className="text-terminal-green">{hb.p2Holds} held</span>
                {" · "}
                <span className={hb.p2Breaks > 0 ? "text-terminal-red" : "text-terminal-muted"}>{hb.p2Breaks} broke</span>
              </span>
            </div>
          </div>
        ) : (
          <div className="text-[10px] text-terminal-muted">
            {/* The honest fallback: games won is real; hold/break attribution is
                not derivable without a server reading, and is not guessed.
                Computed from completedSets + currentSetGames — the same two
                fields every other engine in this file (gamePressure,
                liveTelemetry, predictTotalGames) already treats as the live
                truth — rather than m.score.p1_sets, which only accumulates
                once a SET finishes and reads 0-0 for the entire first set of
                every match, the most common case there is. */}
            The feed isn&apos;t saying who serves, so holds and breaks can&apos;t be
            attributed — games won: {p1Short} {gamesWonP1}, {p2Short} {gamesWonP2} this match.
          </div>
        )}
      </div>

      {gp && (
        <div className="pt-2 border-t border-terminal-border/60">
          <div className="text-[8px] font-bold text-terminal-muted tracking-wider mb-1">GAMES PREDICTED</div>
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-bold font-mono text-slate-100">{gp.gamesSoFar}</span>
            <span className="text-[10px] text-terminal-muted">played</span>
            <span className="text-terminal-muted">→</span>
            <span className="text-sm font-bold font-mono text-terminal-green">≈{gp.expectedTotal}</span>
            <span className="text-[10px] text-terminal-muted">total, expected</span>
          </div>
          <div className="text-[9px] text-terminal-muted mt-0.5">
            About {gp.expectedRemaining} more from here, based on how this match has been serving
            {wl && !wl.incomplete ? ` (${wl.points} points observed)` : ""}.
          </div>
        </div>
      )}
    </div>
  );
}

/** One of the two probability figures, with the move since the start. */
function Figure({ label, pct, tone, delta, title }: {
  label: string; pct: number; tone: "muted" | "live"; delta?: number; title: string;
}) {
  const moved = delta !== undefined && Math.abs(delta) >= 1;
  return (
    <div title={title}
      className={`flex-1 rounded border px-2.5 py-1.5 ${
        tone === "live" ? "border-terminal-green/40 bg-terminal-green/[0.07]" : "border-terminal-border"
      }`}>
      <div className={`text-[9px] font-bold ${tone === "live" ? "text-terminal-green" : "text-terminal-muted"}`}>
        {label}
      </div>
      <div className="flex items-baseline gap-1.5 mt-0.5">
        <span className={`text-xl font-bold font-mono tabular-nums ${
          tone === "live" ? "text-slate-100" : "text-slate-400"
        }`}>{pct}%</span>
        {moved && (
          <span className={`text-[10px] font-bold font-mono ${
            delta! > 0 ? "text-terminal-green" : "text-terminal-red"
          }`}>
            {delta! > 0 ? "▲" : "▼"}{Math.abs(delta!)}
          </span>
        )}
      </div>
    </div>
  );
}

/** One GAME / SET / MATCH pick: who is favoured, and by how much. */
function LevelPick({ label, p1Prob, m, title }: {
  label: string; p1Prob: number; m: ScheduledMatch; title: string;
}) {
  const fav1 = p1Prob >= 0.5;
  const pct = Math.round((fav1 ? p1Prob : 1 - p1Prob) * 100);
  const name = displayName(fav1 ? m.player1 : m.player2);
  const tooClose = pct < 55;
  return (
    <div title={title} className="rounded border border-terminal-border bg-terminal-bg/40 px-2 py-1.5 text-center">
      <div className="text-[8px] font-bold text-terminal-muted tracking-wider">{label}</div>
      <div className={`text-sm font-bold font-mono mt-0.5 ${tooClose ? "text-slate-300" : "text-terminal-green"}`}>
        {pct}%
      </div>
      <div className="text-[9px] text-terminal-muted truncate">{tooClose ? "too close to call" : name}</div>
    </div>
  );
}

/** A 0–100 pressure bar. Small on purpose: it is context, not the answer. */
function PressureMeter({ score }: { score: number }) {
  const tone = score >= 70 ? "bg-terminal-red" : score >= 40 ? "bg-terminal-yellow" : "bg-terminal-green";
  return (
    <div className="shrink-0 text-right" title="How much pressure has built up in this match recently">
      <div className="text-[9px] text-terminal-muted">PRESSURE</div>
      <div className="text-sm font-bold font-mono tabular-nums text-slate-100">{score}</div>
      <div className="w-14 h-1 rounded-full bg-terminal-border overflow-hidden mt-0.5">
        <div className={`h-full ${tone}`} style={{ width: `${score}%` }} />
      </div>
    </div>
  );
}

/**
 * Momentum, break danger and serve form as three chips.
 *
 * These were three separate sections with headings, bars and sub-tables. At
 * that size they compete with the signal for attention while adding nothing a
 * glance can use, so each is reduced to a word and a colour.
 */
function Indicators({ match: m, sig }: { match: ScheduledMatch; sig: PressureSignal | null }) {
  const bh = m.liveScore?.breakHoldSignals;
  // Real, score-derived, and always available for a live match — see the doc
  // on ScheduledMatch.liveScore.liveMomentum. `momentum` (the sofaId-based
  // one) is left unread here: it needs a SofaScore point-by-point call that is
  // currently blocked, so on the matches this session was verified against it
  // was always undefined while liveMomentum was populated.
  const lm = m.liveScore?.liveMomentum;
  const wl = m.liveScore?.liveWorkload;
  const chips: { label: string; value: string; tone: "good" | "warn" | "bad" | "flat"; title: string }[] = [];

  if (bh) {
    const bp = Math.round(bh.breakProb * 100);
    chips.push({
      label: "BREAK CHANCE", value: `${bp}%`,
      tone: bp >= 55 ? "bad" : bp >= 35 ? "warn" : "good",
      title: "How likely the player serving is to lose this game",
    });
    chips.push({
      label: "SERVE FORM", value: bh.serverSERTier.toLowerCase(),
      tone: bh.serverSERTier === "ELITE" || bh.serverSERTier === "STRONG" ? "good"
        : bh.serverSERTier === "AVERAGE" ? "flat" : "bad",
      title: "How well the current server has been serving in this match",
    });
  }
  if (lm) {
    // p1 is signed and centred on 0: positive means P1 is outperforming their
    // own match baseline right now. Shown as a percentage, not just a name, so
    // the chip carries the same magnitude information the panel above does.
    const leader = lm.p1 > 0.02 ? displayName(m.player1) : lm.p1 < -0.02 ? displayName(m.player2) : "even";
    const mag = Math.round(Math.abs(lm.p1) * 100);
    chips.push({
      label: "MOMENTUM", value: leader === "even" ? "even" : `${leader} +${mag}%`,
      tone: leader === "even" ? "flat" : mag >= 15 ? "bad" : "warn",
      title: `Recent share of points won vs this match's own baseline, over ${lm.observedPoints} points actually observed`,
    });
  } else if (m.status === "live") {
    // Says WHY it is blank rather than omitting the chip silently — momentum
    // needs a real run of points before it means anything, and "nothing here
    // yet" is a different fact from "this cannot be measured".
    chips.push({
      label: "MOMENTUM", value: "watching",
      tone: "flat",
      title: "Building up a real point sequence before reporting a trend — needs about 12 points",
    });
  }
  if (wl) {
    chips.push({
      label: "POINTS SEEN", value: `${wl.points}${wl.incomplete ? "+" : ""}`,
      tone: "flat",
      title: `${wl.points} points observed this match, ${wl.deuceGames} at deuce, ${wl.tiebreaks} tiebreak${wl.tiebreaks === 1 ? "" : "s"}`
        + (wl.incomplete ? " — the feed skipped some points, so this is a floor, not an exact count" : ""),
    });
  }
  if (sig && sig.regime !== "NORMAL") {
    chips.push({
      label: "SET", value: sig.regime === "DOMINANCE" ? "one-sided" : "unsettled",
      tone: sig.regime === "DOMINANCE" ? "bad" : "warn",
      title: "Breaks of serve in this set so far",
    });
  }
  if (!chips.length) return null;

  const toneCls = {
    good: "text-terminal-green border-terminal-green/40",
    warn: "text-terminal-yellow border-terminal-yellow/40",
    bad: "text-terminal-red border-terminal-red/40",
    flat: "text-slate-300 border-terminal-border",
  };
  return (
    <div className="flex flex-wrap gap-1.5">
      {chips.map(c => (
        <span key={c.label} title={c.title}
          className={`inline-flex items-baseline gap-1 rounded border px-1.5 py-0.5 text-[9px] ${toneCls[c.tone]}`}>
          <span className="text-terminal-muted">{c.label}</span>
          <span className="font-bold font-mono">{c.value}</span>
        </span>
      ))}
    </div>
  );
}
