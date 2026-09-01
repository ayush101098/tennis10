import { describe, it, expect } from "vitest";
import { surname, fixtureKey } from "../polymarket";

/**
 * The schedule feed writes "Sabalenka A."; Polymarket writes "Aryna Sabalenka".
 * Both must key to the same fixture or the board can never find a price.
 */
describe("surname / fixtureKey", () => {
  it("reads Polymarket's full-name form", () => {
    expect(surname("Aryna Sabalenka")).toBe("sabalenka");
    expect(surname("Camila Osorio")).toBe("osorio");
  });

  it("reads the feed's abbreviated surname-first form", () => {
    expect(surname("Sabalenka A.")).toBe("sabalenka");
    expect(surname("Cerundolo J. M.")).toBe("cerundolo");
    expect(surname("Wang Xin.")).toBe("wang");
    expect(surname("Struff J-L.")).toBe("struff");
  });

  it("keeps compound surnames reachable by their final word", () => {
    expect(surname("Merida Aguilar D.")).toBe("aguilar");
    expect(surname("Davidovich Fokina A.")).toBe("fokina");
  });

  it("never strips a short name down to nothing", () => {
    expect(surname("Lee")).toBe("lee");
    expect(surname("Wu Y.")).toBe("wu");
  });

  it("keys both name forms of the same fixture identically", () => {
    expect(fixtureKey("Sabalenka A.", "Osorio C."))
      .toBe(fixtureKey("Aryna Sabalenka", "Camila Osorio"));
  });
});
