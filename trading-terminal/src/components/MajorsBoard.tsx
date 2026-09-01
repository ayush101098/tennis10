"use client";

import { useMemo } from "react";
import type { ScheduledMatch } from "@/lib/scheduleService";
import { EDGE_FLOOR, MAX_BANKROLL_FRACTION, quarterKellyStake } from "@/lib/scheduleService";
import { usePolymarket } from "@/hooks/usePolymarket";
import { fixtureKey, eventUrl, type PmFixture } from "@/lib/polymarket";
import { polymarketValue } from "@/lib/pmValue";
import { tourRank } from "@/lib/scheduleService";
import { Panel, Badge, EmptyState, ErrorState, LoadingState } from "@/components/ui/Panel";
import { pct, signedPct, odds as fmtOdds, money } from "@/components/ui/Table";
import Icon from "@/components/ui/Icon";

/**
 * MAJORS VALUE BOARD — the homepage's lead act.
 *
 * Every TOUR-LEVEL match on the card (ATP and WTA main draw, which is where
 * the Grand Slams and Masters live), ranked by the model's edge over the
 * de-vigged market. The board below this one is a schedule; this is the answer
 * to the only question a visitor actually has — what is worth betting.
 *
 * Was US Open only. A board that empties the day a slam ends is a board people
 * stop opening, and the model prices the whole tour identically — there was
 * never a reason to show one fortnight of it.
 *
 * Challenger and ITF are excluded on purpose rather than for volume: those
 * draws are full of players the rankings file does not cover, so the model has
 * no prior, the gate correctly stays silent, and the rows would be unpriced
 * padding. They remain on the full schedule below.
 *
 * It renders the same `m.value` the terminal's ValueBoard trades off, so the
 * public number and the members' number cannot drift apart. What stays gated
 * is the stake: edges are the pitch, position sizing is the product.
 *
 * Three groups, because "no bet" is a finding and hiding it would misrepresent
 * the model:
 *   BETS      edge ≥ 2% floor, not suspect
 *   WATCH     positive but under the floor — shown so an efficient market does
 *             not look like a broken board
 *   SUSPECT   edge > 20% — a data fault, never a bet
 */

const STRONG_EDGE = 0.05;
const SHOWCASE_BANKROLL = 1000;

/** Tour-level: ATP or WTA main draw. tourRank 0 is exactly that tier. */
export const isMajor = (m: { tour: string }) => tourRank(m.tour) === 0;

/** Grand slams lead the board — the matches most people came to see. */
export const isSlam = (m: { tournament: string }) =>
  /australian open|roland garros|french open|wimbledon|us open/i.test(m.tournament);

/** One priced match: the model's best side, and where the price came from. */
interface Row {
  m: ScheduledMatch;
  value: NonNullable<ScheduledMatch["value"]>;
  fixture?: PmFixture;
  source: "book" | "polymarket";
}

interface Props {
  matches: ScheduledMatch[];
  isPro: boolean;
  onSelectMatch?: (m: ScheduledMatch) => void;
  onUpgrade?: () => void;
  /** True when every upstream feed failed — an outage must not read as "no value". */
  sourcesDown?: boolean;
  loading?: boolean;
  parlayIds?: Set<string>;
  onToggleParlay?: (m: ScheduledMatch, value: NonNullable<ScheduledMatch["value"]>) => void;
}

export default function MajorsBoard({
  matches, isPro, onSelectMatch, onUpgrade, sourcesDown, loading,
  parlayIds, onToggleParlay,
}: Props) {
  const pmIndex = usePolymarket();

  const { bets, watch, suspect, priced, unpriced, liveCount, total } = useMemo(() => {
    const pool = matches.filter(m =>
      isMajor(m) && (m.status === "live" || m.status === "scheduled"));

    // Bookmaker odds first where they exist, Polymarket otherwise. SofaScore's
    // odds endpoints currently 403 for every US Open match, so in practice this
    // board is priced on Polymarket — the venue this project trades.
    const rows: Row[] = [];
    for (const m of pool) {
      const fixture = pmIndex.get(fixtureKey(m.player1, m.player2));
      const value = m.value ?? polymarketValue(m, fixture);
      if (value) rows.push({ m, value, fixture, source: m.value ? "book" : "polymarket" });
    }
    // Slam first, then edge. A slam row sitting below a 250 because its edge
    // was 0.3 points smaller reads as a bug to anyone who came for the slam.
    rows.sort((a, b) =>
      (isSlam(b.m) ? 1 : 0) - (isSlam(a.m) ? 1 : 0) || b.value.edge - a.value.edge);
    return {
      bets: rows.filter(r => r.value.edge >= EDGE_FLOOR && !r.value.suspect),
      watch: rows.filter(r => r.value.edge > 0 && r.value.edge < EDGE_FLOOR),
      suspect: rows.filter(r => r.value.suspect),
      priced: rows.length,
      unpriced: pool.length - rows.length,
      liveCount: pool.filter(m => m.status === "live").length,
      total: pool.length,
    };
  }, [matches, pmIndex]);

  const rowProps = { isPro, onSelect: onSelectMatch, onUpgrade, parlayIds, onToggleParlay };

  return (
    <div className="px-4 sm:px-6 pb-12 sm:pb-16 max-w-[1180px] mx-auto">
      <Panel
        id="majors"
        title="Majors — value board"
        icon="trophy"
        tone="neutral"
        live={liveCount > 0}
        className="scroll-mt-20"
        meta={
          <span className="font-mono tabular-nums">
            {total} matches · {priced} priced{unpriced > 0 ? ` · ${unpriced} awaiting a market` : ""}
          </span>
        }>
        {sourcesDown ? (
          <ErrorState
            title="Live data temporarily unavailable"
            body="The match feed isn’t responding, so no US Open price can be trusted right now. This is on our side — it reconnects on its own." />
        ) : loading ? (
          <LoadingState label="Pricing the US Open card" rows={4} />
        ) : total === 0 ? (
          <EmptyState
            title="No tour-level matches on today’s card"
            body="ATP and WTA main draws are between events. The board below still has every Challenger and ITF match running today." />
        ) : (
          <>
            <Group title={`Value bets — edge ≥ ${Math.round(EDGE_FLOOR * 100)}%`} count={bets.length} tone="primary">
              {bets.map(r => <ValueRow key={r.m.id} row={r} {...rowProps} />)}
              {bets.length === 0 && (
                <p className="px-4 py-5 text-center text-sm text-content-muted max-w-[60ch] mx-auto">
                  {priced === 0
                    ? "Nothing is priced yet — market prices arrive closer to first serve."
                    : `Every tour price is inside the ${Math.round(EDGE_FLOOR * 100)}% floor right now. No edge, no bet — that is the system working, not a quiet board.`}
                </p>
              )}
            </Group>

            {watch.length > 0 && (
              <Group title="Watchlist — positive, below the floor" count={watch.length} tone="neutral">
                {watch.map(r => <ValueRow key={r.m.id} row={r} {...rowProps} />)}
              </Group>
            )}

            {suspect.length > 0 && (
              <Group title="Suspect data — do not bet" count={suspect.length} tone="danger">
                {suspect.map(r => <ValueRow key={r.m.id} row={r} {...rowProps} />)}
              </Group>
            )}
          </>
        )}

        <footer className="px-4 py-3 text-xs text-content-muted leading-relaxed border-t border-border">
          True P: Platt-calibrated neural network ⊕ Elo pre-match, re-priced by a tour-aware Markov engine on
          every game once the match is live. Edge = True P − de-vigged market price, live prices for live
          matches. Stake = ¼ Kelly capped at {Math.round(MAX_BANKROLL_FRACTION * 100)}% of bankroll; nothing
          under the {Math.round(EDGE_FLOOR * 100)}% floor is a bet. Edges over 20% are quarantined as bad data,
          not free money.
        </footer>
      </Panel>
    </div>
  );
}

function Group({ title, count, tone, children }: {
  title: string; count: number; tone: "primary" | "neutral" | "danger"; children: React.ReactNode;
}) {
  const toneCls = tone === "primary" ? "text-primary"
    : tone === "danger" ? "text-danger" : "text-content-muted";
  return (
    <section className="border-b border-border last:border-b-0">
      <h3 className={`px-4 py-2 text-micro uppercase font-semibold ${toneCls} bg-bg/40`}>
        {title} <span className="font-mono tabular-nums">({count})</span>
      </h3>
      {children}
    </section>
  );
}

function ValueRow({ row, isPro, onSelect, onUpgrade, parlayIds, onToggleParlay }: {
  row: Row; isPro: boolean;
  onSelect?: (m: ScheduledMatch) => void; onUpgrade?: () => void;
  parlayIds?: Set<string>;
  onToggleParlay?: (m: ScheduledMatch, value: NonNullable<ScheduledMatch["value"]>) => void;
}) {
  const { m, value: v, fixture, source } = row;
  const live = m.status === "live";
  // Never badge a quarantined row as STRONG: that group is headed "do not
  // bet", and a confidence badge inside it says the opposite of the heading.
  const strong = v.edge >= STRONG_EDGE && !v.suspect;
  const stake = quarterKellyStake(SHOWCASE_BANKROLL, v.kelly);
  const opponent = v.side === 1 ? m.player2 : m.player1;
  const inParlay = !!parlayIds?.has(m.id);

  return (
    <div className="flex items-center gap-3 px-4 border-b border-border last:border-b-0 hover:bg-surface/40">
      {/* The row itself is the button that opens the analysis. Nested
          interactive elements sit beside it, never inside it — a button inside
          a button is invalid HTML and breaks keyboard navigation. */}
      {/* The row is the button that opens the analysis. Nested interactive
          elements sit BESIDE it, never inside — a button inside a button is
          invalid HTML and breaks keyboard navigation.

          Layout is two-line on phones and one line from `sm` up. Squeezing
          seven columns into 390px collapsed the player's name to "Sa…" and
          overlapped the badge with the odds; a name the reader cannot read is
          not a denser row, it is a broken one. */}
      <button
        onClick={() => onSelect?.(m)}
        className="flex flex-1 min-w-0 flex-col sm:flex-row sm:items-center gap-1 sm:gap-3 py-2.5 text-left">
        <span className="flex items-center gap-2 min-w-0 sm:contents">
          <span className="w-14 shrink-0 text-left">
            {live
              ? <span className="text-micro uppercase font-semibold text-primary">Live</span>
              : <span className="text-micro font-mono tabular-nums text-content-muted">{m.start_time || "TBD"}</span>}
            {m.round && <span className="block text-micro text-content-muted truncate">{m.round}</span>}
          </span>

          <span className="flex-1 min-w-0">
            <span className="flex items-center gap-1.5 min-w-0">
              <span className="text-base font-semibold text-content-strong truncate">{v.player}</span>
              {strong && <Badge tone="primary">Strong</Badge>}
            </span>
            <span className="block text-xs text-content-muted truncate">
              vs {opponent}
              {live && m.score ? ` · ${m.score.p1_sets.join("-")} / ${m.score.p2_sets.join("-")}` : ""}
            </span>
          </span>
        </span>

        {/* Figures. On a phone they sit on their own line, left-aligned and
            evenly spaced, so each keeps its label and none is truncated. */}
        <span className="flex items-center gap-4 sm:gap-3 pl-16 sm:pl-0">
          <span className="w-12 sm:w-14 shrink-0 text-left sm:text-right">
            <span className="block font-mono tabular-nums text-sm text-content">{pct(v.trueP, 0)}</span>
            <span className="block text-micro uppercase text-content-muted">true p</span>
          </span>
          <span className="w-12 shrink-0 text-left sm:text-right">
            <span className="block font-mono tabular-nums text-sm text-content">{fmtOdds(v.odds)}</span>
            <span className="block text-micro uppercase text-content-muted">odds</span>
          </span>
          <span className="w-14 shrink-0 text-left sm:text-right">
            <span className={`block font-mono tabular-nums text-base font-semibold ${
              v.suspect ? "text-danger"
                : strong ? "text-primary"
                : v.edge >= EDGE_FLOOR ? "text-accent"
                : "text-content-muted"}`}>
              {signedPct(v.edge)}
            </span>
            <span className="block text-micro uppercase text-content-muted">edge</span>
          </span>
        </span>
      </button>

      {/* Add to the ticket. A suspect row stays addable on purpose — the
          builder shows what it does to the price, which teaches more than
          hiding it would. */}
      {onToggleParlay && (
        <button
          onClick={() => onToggleParlay(m, v)}
          aria-pressed={inParlay}
          // The name stays CONSTANT and `aria-pressed` carries the state.
          // Naming it "Parlay"/"On ticket" instead meant the control was
          // unnamed on mobile (the label is hidden under 640px) and renamed
          // itself on every toggle, which is exactly what a pressed-state
          // button must not do.
          aria-label={`Add ${v.player} to the parlay ticket`}
          data-testid="parlay-toggle"
          className={`shrink-0 inline-flex items-center gap-1 h-7 px-2 rounded-sm border text-micro uppercase font-semibold
            transition-colors duration-fast ease-standard [@media(pointer:coarse)]:min-h-[44px]
            ${inParlay
              ? "border-accent/40 bg-accent/10 text-accent"
              : "border-border text-content-muted hover:text-accent hover:border-accent/40"}`}>
          <Icon name={inParlay ? "check" : "plus"} size={12} />
          <span className="hidden sm:inline">{inParlay ? "On ticket" : "Parlay"}</span>
        </button>
      )}

      {/* Stake — gated. */}
      <div className="w-16 shrink-0 text-right">
        {isPro ? (
          <>
            <span className="block font-mono tabular-nums text-sm text-warning">{money(stake)}</span>
            <span className="block text-micro uppercase text-content-muted">per $1k</span>
          </>
        ) : (
          <button onClick={onUpgrade}
            className="inline-flex items-center gap-1 text-micro uppercase font-semibold text-content-muted hover:text-primary">
            <Icon name="lock" size={11} />
            Stake
          </button>
        )}
      </div>

      {/* Where the price came from — an edge is meaningless without the market
          it is an edge against. */}
      <div className="hidden md:block w-24 shrink-0 text-right">
        {source === "polymarket" && fixture?.match ? (
          <a href={eventUrl(fixture.match)} target="_blank" rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-micro uppercase text-accent/80 hover:text-accent">
            Polymarket <Icon name="external" size={11} />
          </a>
        ) : (
          <span className="text-micro uppercase text-content-muted">Bookmaker</span>
        )}
      </div>
    </div>
  );
}
