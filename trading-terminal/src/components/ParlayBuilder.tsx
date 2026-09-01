"use client";

import { useMemo, useState } from "react";
import {
  priceParlay, stressParlay, stressTolerance, correlationWarnings,
  LEG_WARNING_THRESHOLD, MAX_STRESS_POINTS, MEASURED_GAP_PP, type ParlayLeg,
} from "@/lib/parlay";
import { EDGE_FLOOR, MAX_BANKROLL_FRACTION, quarterKellyStake } from "@/lib/scheduleService";
import { Panel, Stat, EmptyState, Badge } from "@/components/ui/Panel";
import { pct, signedPct, odds as fmtOdds, money } from "@/components/ui/Table";
import Button, { IconButton } from "@/components/ui/Button";
import Icon from "@/components/ui/Icon";

/**
 * PARLAY BUILDER.
 *
 * Combines legs picked off the value board into one ticket and prices it
 * against the market the legs came from.
 *
 * The design point worth defending: a parlay's headline edge is the most
 * flattering number this product can display, and the least trustworthy. A
 * model that runs hot on each leg keeps a smaller and smaller SHARE of its
 * stated edge as legs are added. So the reality check is not a footnote here —
 * it sits beside the headline, and the tolerance line says plainly how many
 * points of per-leg error the ticket can absorb before it is -EV.
 *
 * Note tolerance itself does not vary with leg count: if every leg is g points
 * hot, removing g points from each reproduces the market product exactly, for
 * any number of legs. Length risk lives in the retained-edge share, the
 * collapsing win probability, and correlation — not in that number.
 */

const SHOWCASE_BANKROLL = 1000;

interface Props {
  legs: ParlayLeg[];
  isPro: boolean;
  onRemove: (matchId: string) => void;
  onClear: () => void;
  onUpgrade?: () => void;
}

export default function ParlayBuilder({ legs, isPro, onRemove, onClear, onUpgrade }: Props) {
  // Points of per-leg model overconfidence to assume. Defaults to the gap
  // actually measured on this card, so the flattering number is never shown
  // without the measured correction beside it.
  const [stress, setStress] = useState(MEASURED_GAP_PP);

  const base = useMemo(() => priceParlay(legs), [legs]);
  const stressed = useMemo(() => stressParlay(legs, stress), [legs, stress]);
  const tolerance = useMemo(() => stressTolerance(legs), [legs]);
  const warnings = useMemo(() => correlationWarnings(legs), [legs]);

  const stake = base ? quarterKellyStake(SHOWCASE_BANKROLL, base.kelly) : 0;
  const retained = base && stressed && base.edge > 0
    ? Math.round(Math.max(0, stressed.edge) / base.edge * 100) : 0;

  return (
    <div className="px-4 sm:px-6 pb-12 sm:pb-16 max-w-[1180px] mx-auto">
      <Panel
        id="parlay"
        title="Parlay builder"
        icon="target"
        tone="accent"
        className="scroll-mt-20"
        meta={
          <span className="flex items-center gap-3">
            <span className="font-mono tabular-nums">
              {legs.length === 0 ? "no legs" : `${legs.length} leg${legs.length > 1 ? "s" : ""}`}
            </span>
            {legs.length > 0 && <Button variant="ghost" size="sm" onClick={onClear}>Clear</Button>}
          </span>
        }>
        {legs.length === 0 || !base || !stressed ? (
          <EmptyState
            title="Build a ticket from the board above"
            body="Add any US Open row as a leg. Legs are priced together against the same market, so you can see what the combined ticket is really worth before you leg into it." />
        ) : (
          <>
            {/* ── Legs ── */}
            <ul className="border-b border-border">
              {legs.map((l, i) => (
                <li key={l.matchId}
                  className="flex items-center gap-3 px-4 py-2 border-b border-border last:border-b-0">
                  <span className="w-4 shrink-0 font-mono tabular-nums text-xs text-content-muted">{i + 1}</span>
                  <span className="flex-1 min-w-0">
                    <span className="flex items-center gap-1.5 min-w-0">
                      <span className="text-base font-semibold text-content-strong truncate">{l.player}</span>
                      {l.live && <Badge tone="primary" icon="live">Live</Badge>}
                    </span>
                    <span className="block text-xs text-content-muted truncate">vs {l.opponent}</span>
                  </span>
                  <span className="w-12 shrink-0 text-right">
                    <span className="block font-mono tabular-nums text-sm text-content">{pct(l.trueP, 0)}</span>
                    <span className="block text-micro uppercase text-content-muted">true p</span>
                  </span>
                  <span className="w-12 shrink-0 text-right">
                    <span className="block font-mono tabular-nums text-sm text-content">{fmtOdds(l.odds)}</span>
                    <span className="block text-micro uppercase text-content-muted">odds</span>
                  </span>
                  <IconButton name="close" label={`Remove ${l.player} from the ticket`}
                    size="sm" onClick={() => onRemove(l.matchId)} />
                </li>
              ))}
            </ul>

            {/* ── The ticket ── */}
            <div className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-border border-b border-border">
              <Stat label="Combined odds" value={fmtOdds(base.odds)} className="px-4 py-3" />
              <Stat label="Model says" value={pct(base.trueP)} className="px-4 py-3" />
              <Stat label="Market says" value={pct(base.marketP)} className="px-4 py-3" />
              <Stat label="Model edge" value={signedPct(base.edge)} className="px-4 py-3"
                tone={base.edge >= EDGE_FLOOR ? "primary" : base.edge > 0 ? "accent" : "danger"} />
            </div>

            {/* ── Reality check — beside the headline, not beneath it ── */}
            <div className="px-4 py-4 border-b border-border bg-bg/40">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-3">
                <h3 className="flex items-center gap-1.5 text-micro uppercase font-semibold text-warning">
                  <Icon name="scale" size={13} /> Reality check
                </h3>
                <label htmlFor="parlay-stress"
                  className="flex items-center gap-2 text-xs text-content-muted flex-1 min-w-[240px]">
                  Assume model is
                  <input
                    id="parlay-stress"
                    type="range" min={0} max={Math.round(MAX_STRESS_POINTS * 100)}
                    value={Math.round(stress * 100)}
                    onChange={e => setStress(Number(e.target.value) / 100)}
                    className="flex-1 accent-warning"
                    aria-valuetext={`${Math.round(stress * 100)} points too high per leg`} />
                  <span className="font-mono tabular-nums text-content w-16 text-right">
                    {Math.round(stress * 100)}pp hot
                  </span>
                </label>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                <Stat label={`Edge if ${Math.round(stress * 100)}pp hot`} value={signedPct(stressed.edge)}
                  tone={stressed.edge >= EDGE_FLOOR ? "primary" : stressed.edge > 0 ? "accent" : "danger"} />
                <Stat label="Adjusted win chance" value={pct(stressed.trueP)} />
                <Stat
                  label="Survives error up to"
                  value={tolerance === null || tolerance === 0
                    ? "no edge"
                    : tolerance >= MAX_STRESS_POINTS
                      ? `>${Math.round(MAX_STRESS_POINTS * 100)}pp`
                      : `${(tolerance * 100).toFixed(1)}pp`}
                  tone={tolerance !== null && tolerance >= MEASURED_GAP_PP ? "primary" : "danger"} />
              </div>

              <p className="mt-3 text-xs text-content-muted leading-relaxed max-w-[80ch]">
                The slider assumes the model is that many points too high on <em>every</em> leg. Tolerance
                measures per-leg model error, so it does not move with leg count — but the share of edge a
                ticket keeps under that error does: at {Math.round(MEASURED_GAP_PP * 100)}pp this ticket
                retains{" "}
                <span className="text-content font-semibold font-mono tabular-nums">
                  {base.edge > 0 ? `${retained}%` : "none"}
                </span>{" "}
                of its headline edge, and a longer ticket keeps less than a shorter one.{" "}
                {tolerance === null || tolerance === 0
                  ? "This ticket is already priced against you."
                  : tolerance >= MAX_STRESS_POINTS
                    ? `It stays positive even at a ${Math.round(MAX_STRESS_POINTS * 100)}-point-per-leg haircut.`
                    : tolerance < MEASURED_GAP_PP
                      ? `Past ${(tolerance * 100).toFixed(1)}pp per leg it turns negative — less than the ~${Math.round(MEASURED_GAP_PP * 100)}pp the model currently runs against the market on this card.`
                      : `Past ${(tolerance * 100).toFixed(1)}pp per leg it turns negative, more than the ~${Math.round(MEASURED_GAP_PP * 100)}pp gap measured on this card.`}
              </p>
            </div>

            {/* ── Stake ── */}
            <div className="flex flex-wrap items-center gap-x-8 gap-y-3 px-4 py-4">
              <div>
                <div className="text-micro uppercase text-content-muted">¼ Kelly per $1k</div>
                {isPro ? (
                  <div className="font-mono tabular-nums text-lg font-semibold text-warning">{money(stake)}</div>
                ) : (
                  <Button variant="ghost" size="sm" icon="lock" onClick={onUpgrade} className="-ml-2.5">
                    Unlock stake
                  </Button>
                )}
              </div>
              <Stat label="Returns on $100" value={money(base.payout * 100)} />
              {base.edge < EDGE_FLOOR && (
                <p className="text-xs text-content-muted">
                  Below the {Math.round(EDGE_FLOOR * 100)}% edge floor — not a bet.
                </p>
              )}
            </div>

            {(warnings.length > 0 || legs.length >= LEG_WARNING_THRESHOLD) && (
              <div className="px-4 py-3 border-t border-danger/30 bg-danger/5">
                <h3 className="flex items-center gap-1.5 text-micro uppercase font-semibold text-danger mb-1.5">
                  <Icon name="alert" size={13} /> Before you bet this
                </h3>
                <ul className="flex flex-col gap-1">
                  {legs.length >= LEG_WARNING_THRESHOLD && (
                    <li className="text-xs text-content-muted">
                      {legs.length} legs: the win probability collapses as legs are added, and the ticket keeps
                      less of its stated edge under the same per-leg error.
                    </li>
                  )}
                  {warnings.map((w, i) => (
                    <li key={i} className="text-xs text-content-muted">{w}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        <footer className="px-4 py-3 text-xs text-content-muted leading-relaxed border-t border-border">
          Legs are multiplied assuming independence, and priced against the same de-vigged market as the board
          above. Polymarket has no parlay product — this is a synthetic ticket you build by legging in
          separately, so the combined price assumes every leg fills at the shown odds. Stake is ¼ Kelly on the
          combined price, capped at {Math.round(MAX_BANKROLL_FRACTION * 100)}% of bankroll.
        </footer>
      </Panel>
    </div>
  );
}
