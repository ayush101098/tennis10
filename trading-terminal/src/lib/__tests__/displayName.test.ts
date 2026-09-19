import { describe, it, expect } from "vitest";
import { displayName } from "../scheduleService";

/**
 * displayName exists because `.split(" ").pop()` was reading this feed's
 * "Surname Initial." names backwards — "Kecmanovic M." shortened to "M.", the
 * initial, not the surname. It surfaced when two players in one match shared
 * a first initial and both displayed as the same label.
 */
describe("displayName", () => {
  it("drops a trailing initial to reveal the surname", () => {
    expect(displayName("Kecmanovic M.")).toBe("Kecmanovic");
    expect(displayName("Tsitsipas S.")).toBe("Tsitsipas");
  });

  it("keeps a multi-word surname whole", () => {
    expect(displayName("Van De Zandschulp B.")).toBe("Van De Zandschulp");
    expect(displayName("de Carvalho Damazio L. E.")).toBe("de Carvalho Damazio");
  });

  it("two players sharing a first initial no longer display identically", () => {
    const a = displayName("Dlimi Y.");
    const b = displayName("Yusuf A.");
    expect(a).not.toBe(b);
    expect(a).toBe("Dlimi");
    expect(b).toBe("Yusuf");
  });

  it("leaves a name with no trailing initial alone", () => {
    expect(displayName("Kecmanovic")).toBe("Kecmanovic");
    expect(displayName("Novak Djokovic")).toBe("Novak Djokovic");
  });

  it("handles a bare initial-less single name", () => {
    expect(displayName("Nadal")).toBe("Nadal");
  });
});
