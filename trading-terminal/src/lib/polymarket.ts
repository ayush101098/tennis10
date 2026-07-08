/**
 * Polymarket market discovery for the Value Board.
 *
 * Queries the public Gamma API straight from the browser (CORS is open),
 * indexes open tennis fixtures by surname pair, and exposes lookups so a
 * Value Board row can show the live Polymarket price + edge for the model's
 * pick and deep-link to the market. Mirrors execution/polymarket.py.
 */

const GAMMA_URL = "https://gamma-api.polymarket.com";
const CACHE_TTL_MS = 60_000;

export type PmMarketType = "match" | "set1" | "set2" | "set3";

export interface PmMarket {
  eventTitle: string;
  eventSlug: string;
  question: string;
  marketType: PmMarketType;
  conditionId: string;
  outcomes: string[];
  prices: number[];      // gamma snapshot, aligned with outcomes
}

export interface PmFixture {
  match?: PmMarket;
  sets: PmMarket[];
}

const EXCLUDED = ["o/u", "over", "under", "completed match", "total sets",
  "tie break", "tiebreak", "any set", "win a set"];

export function norm(text: string): string {
  return (text || "")
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9 ]/g, " ")
    .trim();
}

export function surname(fullName: string): string {
  const parts = norm(fullName).split(/\s+/);
  return parts[parts.length - 1] || "";
}

export function fixtureKey(p1: string, p2: string): string {
  return [surname(p1), surname(p2)].sort().join("|");
}

function classify(question: string, eventTitle: string): PmMarketType | null {
  const qn = norm(question);
  if (EXCLUDED.some(p => qn.includes(p))) return null;
  const set = /set\s+([123])\s+winner/i.exec(question);
  if (set) return `set${set[1]}` as PmMarketType;
  if (!qn.includes("set") && norm(eventTitle).includes(" vs") && qn === norm(eventTitle)) {
    return "match";
  }
  return null;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
function parseEvent(event: any): PmMarket[] {
  const out: PmMarket[] = [];
  for (const m of event?.markets ?? []) {
    if (!m?.active || m?.closed) continue;
    const marketType = classify(m.question || "", event.title || "");
    if (!marketType) continue;
    let outcomes: string[] = [];
    let prices: number[] = [];
    try {
      outcomes = JSON.parse(m.outcomes || "[]");
      prices = JSON.parse(m.outcomePrices || "[]").map(Number);
    } catch {
      continue;
    }
    if (outcomes.length !== 2) continue;
    out.push({
      eventTitle: event.title || "",
      eventSlug: event.slug || "",
      question: m.question || "",
      marketType,
      conditionId: m.conditionId || "",
      outcomes,
      prices,
    });
  }
  return out;
}

let cache: { at: number; index: Map<string, PmFixture> } | null = null;
let inflight: Promise<Map<string, PmFixture>> | null = null;

async function fetchIndex(): Promise<Map<string, PmFixture>> {
  const index = new Map<string, PmFixture>();
  let offset = 0;
  for (let page = 0; page < 6; page++) {
    const res = await fetch(
      `${GAMMA_URL}/events?tag_slug=tennis&closed=false&limit=100&offset=${offset}`,
    );
    if (!res.ok) break;
    const events: any[] = await res.json();
    for (const event of events) {
      const title: string = event?.title || "";
      const vs = norm(title).split(" vs ");
      if (vs.length !== 2) continue;
      const key = [surname(vs[0]), surname(vs[1])].sort().join("|");
      const markets = parseEvent(event);
      if (!markets.length) continue;
      const fixture = index.get(key) ?? { sets: [] };
      for (const mkt of markets) {
        if (mkt.marketType === "match") fixture.match = mkt;
        else fixture.sets.push(mkt);
      }
      index.set(key, fixture);
    }
    if (events.length < 100) break;
    offset += 100;
  }
  return index;
}

/** Cached index of open Polymarket tennis fixtures keyed by fixtureKey(). */
export async function getPolymarketIndex(): Promise<Map<string, PmFixture>> {
  if (cache && Date.now() - cache.at < CACHE_TTL_MS) return cache.index;
  if (!inflight) {
    inflight = fetchIndex()
      .then(index => {
        cache = { at: Date.now(), index };
        return index;
      })
      .finally(() => { inflight = null; });
  }
  return inflight;
}

/** Price of a named player's outcome in a market (0..1), or null. */
export function outcomePrice(market: PmMarket, playerName: string): number | null {
  const sn = surname(playerName);
  const i = market.outcomes.findIndex(o => sn && norm(o).includes(sn));
  if (i < 0) return null;
  const p = market.prices[i];
  return p > 0 && p < 1 ? p : null;
}

export function eventUrl(market: PmMarket): string {
  return `https://polymarket.com/event/${market.eventSlug}`;
}
