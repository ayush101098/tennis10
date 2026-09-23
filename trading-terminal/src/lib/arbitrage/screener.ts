import type { ScheduledMatch } from "@/lib/scheduleService";
import type { ArbitrageOpportunity, NormalizedMarket } from "./types";
import { toCanonicalEvent } from "./eventMatcher";
import { classifyTwoWay } from "./classify";
import { polymarketAdapter } from "./providers/polymarketAdapter";
import { modelMarketsFor } from "./providers/modelAdapter";

const DEFAULT_STAKE = 1000;

/**
 * Run one full screener pass over the matches the terminal already has
 * loaded. No independent polling of Polymarket beyond what this pass needs —
 * callers control cadence (see ArbitrageScreener.tsx), same discipline as
 * the rest of the app's "no per-client provider polling" rule.
 */
export async function scanForOpportunities(
  matches: ScheduledMatch[],
  totalStake: number = DEFAULT_STAKE,
): Promise<ArbitrageOpportunity[]> {
  // The caller passes [...today, ...tomorrow]; a match starting near midnight
  // can legitimately appear in both date-bucketed lists (same root cause as
  // the live-board duplicate-match issue), which would otherwise scan and
  // display it twice. Dedupe by id defensively rather than trust every
  // upstream caller to have already deduped across its own date boundaries.
  const seen = new Set<string>();
  const singles = matches.filter(m => {
    if (m.status !== "live" && m.status !== "scheduled") return false;
    if (seen.has(m.id)) return false;
    seen.add(m.id);
    return true;
  });
  const events = singles.map(toCanonicalEvent);

  const pmLegs = await polymarketAdapter.fetchMatchWinnerMarkets(events);
  const legsByEvent = new Map<string, NormalizedMarket[]>();
  for (const leg of pmLegs) {
    const arr = legsByEvent.get(leg.eventId) ?? [];
    arr.push(leg);
    legsByEvent.set(leg.eventId, arr);
  }

  const opportunities: ArbitrageOpportunity[] = [];

  for (const match of singles) {
    const event = toCanonicalEvent(match);
    const pm = legsByEvent.get(match.id) ?? [];
    const pm1 = pm.find(l => l.selection === "player1");
    const pm2 = pm.find(l => l.selection === "player2");

    // Polymarket's own two sides against each other — the one genuinely
    // tradeable arbitrage pattern this data supports (see types.ts).
    if (pm1 && pm2) {
      const opp = classifyTwoWay(event, "match_winner", "full_match", pm1, pm2, totalStake);
      if (opp) opportunities.push(opp);
    }

    // Model vs Polymarket: pair the model's pick on ONE player against the
    // market's price on the OTHER player — that's what makes it a coverage
    // pair classifyTwoWay can even evaluate (it requires opposite selections;
    // pairing the model's player1 leg against the market's own player1 leg
    // was rejected outright by the "not mutually exclusive" guard, which is
    // why this produced zero rows before). Concretely: "the model likes
    // player1 more than the market's player2 price implies it should" is
    // legs = [model backs player1, market backs player2] — always lands in
    // "model_positive_ev", never an arbitrage bucket (see classify.ts).
    const modelLegs = modelMarketsFor(match);
    const model1 = modelLegs.find(l => l.selection === "player1");
    const model2 = modelLegs.find(l => l.selection === "player2");
    if (model1 && pm2) {
      const opp = classifyTwoWay(event, "match_winner", "full_match", model1, pm2, totalStake);
      if (opp) opportunities.push(opp);
    }
    if (model2 && pm1) {
      const opp = classifyTwoWay(event, "match_winner", "full_match", model2, pm1, totalStake);
      if (opp) opportunities.push(opp);
    }
  }

  return opportunities;
}

/** Ranking: verified arbitrage first, then by net/theoretical ROI descending. */
export function rankOpportunities(list: ArbitrageOpportunity[]): ArbitrageOpportunity[] {
  const order: Record<ArbitrageOpportunity["classification"], number> = {
    verified_arbitrage: 0, conditional_arbitrage: 1, model_positive_ev: 2, invalid: 3,
  };
  return list.slice().sort((a, b) => {
    const c = order[a.classification] - order[b.classification];
    if (c !== 0) return c;
    const roiA = a.netRoiPct ?? a.theoreticalRoiPct;
    const roiB = b.netRoiPct ?? b.theoreticalRoiPct;
    return roiB - roiA;
  });
}
