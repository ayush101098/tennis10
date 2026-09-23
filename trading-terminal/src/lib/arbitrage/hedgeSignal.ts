import type { ArbitrageOpportunity } from "./types";

/**
 * HOLD / HEDGE / STOP for each arbitrage opportunity, tracked across scan
 * cycles — same vocabulary as the live game-state decision tree
 * (gameStateGraph.ts's PositionAction), applied here to price drift instead
 * of point-by-point win probability.
 *
 * WHY THIS EXISTS
 *   An opportunity's ROI at detection is not its ROI a scan cycle later —
 *   Polymarket's book moves between when the screener sees a price and when
 *   a person could actually act on it. Nothing before this tracked that: two
 *   scans just produced two independent, unrelated-looking opportunity lists
 *   (classify.ts previously minted a fresh id with Date.now() every single
 *   scan, so "the same real comparison" had no continuity across cycles at
 *   all — fixed alongside this by making opportunity ids deterministic).
 *
 * HOW IT WORKS
 *   The caller (ArbitrageScreener.tsx) keeps a small map of
 *   opportunityId -> the ROI it first saw for that id. Each new scan, this
 *   compares the fresh ROI against that entry reference:
 *     STOP  — edge has gone to zero or reversed since detection. Whatever
 *             made this look like an opportunity is gone.
 *     HEDGE — edge has shrunk materially (>40% relative drop) but is still
 *             positive. Worth covering the position rather than assuming the
 *             entry price is still available.
 *     HOLD  — edge has held roughly steady or improved.
 *   A model_positive_ev opportunity gets the same treatment but softer
 *   language — it was never a guaranteed payoff, so "the edge narrowed" reads
 *   differently than it does for an arbitrage leg pair.
 */

export type PositionAction = "HOLD" | "HEDGE" | "STOP";

export interface HedgeSignal {
  action: PositionAction;
  reason: string;
  entryRoiPct: number;
  currentRoiPct: number;
  driftPct: number; // currentRoiPct - entryRoiPct; negative means the edge shrank
}

const HEDGE_DROP_FRACTION = 0.4; // a >40% relative drop in ROI is material, not noise

export function evaluateArbHedge(
  opp: Pick<ArbitrageOpportunity, "classification">,
  entryRoiPct: number,
  currentRoiPct: number,
): HedgeSignal {
  const driftPct = currentRoiPct - entryRoiPct;
  const isModel = opp.classification === "model_positive_ev";

  if (currentRoiPct <= 0) {
    return {
      action: "STOP",
      reason: isModel
        ? "The model's edge against this price has gone to zero or reversed — the case for this side is gone."
        : "The arbitrage spread has closed or reversed since detection — this is no longer a covered position.",
      entryRoiPct, currentRoiPct, driftPct,
    };
  }

  if (entryRoiPct > 0 && currentRoiPct < entryRoiPct * (1 - HEDGE_DROP_FRACTION)) {
    return {
      action: "HEDGE",
      reason: isModel
        ? `Model edge has narrowed from ${entryRoiPct.toFixed(1)}% to ${currentRoiPct.toFixed(1)}% since detection — the market has moved toward the model's view.`
        : `Spread has narrowed from ${entryRoiPct.toFixed(1)}% to ${currentRoiPct.toFixed(1)}% since detection — confirm current prices before sizing up, or cover what's already on.`,
      entryRoiPct, currentRoiPct, driftPct,
    };
  }

  return {
    action: "HOLD",
    reason: driftPct >= 0
      ? "Edge has held or improved since detection."
      : "Edge has drifted slightly but remains within normal quote noise.",
    entryRoiPct, currentRoiPct, driftPct,
  };
}

/** An opportunity that was showing last scan and is entirely absent this
 *  scan — its legs moved enough to be reclassified "invalid" or one side
 *  vanished. Same conclusion as evaluateArbHedge's STOP case, from the
 *  outside: there is nothing left to hold.
 */
export function vanishedHedgeSignal(entryRoiPct: number): HedgeSignal {
  return {
    action: "STOP",
    reason: "This opportunity is no longer present in the latest scan — one or both legs moved past the point of being an edge.",
    entryRoiPct, currentRoiPct: 0, driftPct: -entryRoiPct,
  };
}
