import { describe, it, expect } from "vitest";
import { lookupEntry, type RankMap } from "../scheduleService";

/**
 * SofaScore is the only feed that reaches production (ESPN 403s servers), and
 * it abbreviates given names: "Borges N.", not "Nuno Borges". Before these
 * cases passed, every SofaScore match resolved to rank 0 → prob_method
 * "unknown" → no value assessment, which is why the US Open board was priced
 * on nothing.
 */

// Same normalisation the loader applies when it builds the map.
const norm = (n: string) =>
  n.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "")
    .replace(/-/g, " ").replace(/\s+/g, " ").trim();

function mapOf(names: Record<string, number>): RankMap {
  const m: RankMap = new Map();
  for (const [name, rank] of Object.entries(names)) m.set(norm(name), { rank, points: 0 });
  return m;
}

const RANKS = mapOf({
  "Nuno Borges": 40,
  "Jan-Lennard Struff": 91,
  "Tomas Martin Etcheverry": 55,
  "Juan Manuel Cerundolo": 96,
  "Francisco Cerundolo": 21,
  "Xinyu Wang": 39,
  "Xiyu Wang": 86,
  "Darwin Blanch": 224,
  "Dali Blanch": 309,
  "Emma Navarro": 11,
});

describe("lookupEntry", () => {
  it("still resolves the plain and swapped full-name forms", () => {
    expect(lookupEntry(RANKS, "Emma Navarro")?.rank).toBe(11);
    expect(lookupEntry(RANKS, "Navarro Emma")?.rank).toBe(11);
  });

  it("resolves the SofaScore 'Surname F.' form", () => {
    expect(lookupEntry(RANKS, "Borges N.")?.rank).toBe(40);
  });

  it("resolves hyphenated given names ('Struff J-L.')", () => {
    expect(lookupEntry(RANKS, "Struff J-L.")?.rank).toBe(91);
  });

  it("resolves two-part given names from two initials", () => {
    expect(lookupEntry(RANKS, "Etcheverry T. M.")?.rank).toBe(55);
    // Two Cerundolos: the initials are what separate them.
    expect(lookupEntry(RANKS, "Cerundolo J. M.")?.rank).toBe(96);
    expect(lookupEntry(RANKS, "Cerundolo F.")?.rank).toBe(21);
  });

  it("uses the full abbreviation prefix to separate same-initial players", () => {
    expect(lookupEntry(RANKS, "Wang Xin.")?.rank).toBe(39);
    expect(lookupEntry(RANKS, "Wang Xiy.")?.rank).toBe(86);
    expect(lookupEntry(RANKS, "Blanch Dar.")?.rank).toBe(224);
    expect(lookupEntry(RANKS, "Blanch Dal.")?.rank).toBe(309);
  });

  it("returns null rather than guessing when the abbreviation is ambiguous", () => {
    // "Wang X." could be either Wang — a confident prior on the wrong player
    // is worse than no prior, because it produces a bet.
    expect(lookupEntry(RANKS, "Wang X.")).toBeNull();
    expect(lookupEntry(RANKS, "Blanch D.")).toBeNull();
  });

  it("returns null for players outside the rankings file", () => {
    expect(lookupEntry(RANKS, "Gorzny S.")).toBeNull();
    expect(lookupEntry(RANKS, "Sebastian Gorzny")).toBeNull();
  });
});
