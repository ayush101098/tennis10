import { test, expect } from "@playwright/test";
import { stubNetwork, settle, volatile } from "./fixtures";

/**
 * Homepage — the whole free product, and the page every visitor judges the
 * product by. These assertions cover layout regressions, the states the board
 * can be in, and the interaction contracts the design system promises.
 */

test.describe("homepage", () => {
  /**
   * Element screenshots, not a full-page one.
   *
   * The landing page is ~6,800px tall and its match list is sized in `dvh`.
   * Playwright resizes the viewport to capture a full page, so that container
   * changes height mid-capture and the page height oscillates by 2px forever —
   * the capture never stabilises. Beyond that, a 6,800px baseline is
   * low-signal: any copy edit anywhere invalidates it, so it gets re-baselined
   * without being read, which is the same as not having it.
   *
   * These assert the surfaces that carry the product instead.
   */
  test("hero matches the visual baseline", async ({ page }) => {
    await stubNetwork(page, { mode: "populated" });
    await page.goto("/");
    await settle(page);
    await expect(page.locator("section").first()).toHaveScreenshot("hero.png", {
      mask: volatile(page),
    });
  });

  test("US Open value board matches the visual baseline", async ({ page }) => {
    await stubNetwork(page, { mode: "populated" });
    await page.goto("/");
    await settle(page);
    await expect(page.locator("#us-open")).toHaveScreenshot("us-open-board.png");
  });

  test("parlay builder matches the visual baseline, empty and filled", async ({ page }) => {
    await stubNetwork(page, { mode: "populated" });
    await page.goto("/");
    await settle(page);

    const parlay = page.locator("#parlay");
    await expect(parlay).toHaveScreenshot("parlay-empty.png");

    await page.getByTestId("parlay-toggle").first().click();
    await expect(parlay).toHaveScreenshot("parlay-filled.png");
  });

  test("value board and parlay builder are present and labelled", async ({ page }) => {
    await stubNetwork(page, { mode: "populated" });
    await page.goto("/");
    await settle(page);

    // Landmarks, not text: this asserts the section exists as a real region
    // with an accessible name, which is also what a screen reader navigates by.
    await expect(page.getByRole("heading", { name: /US Open — value board/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Parlay builder/i })).toBeVisible();
  });

  test("empty board explains itself instead of rendering blank", async ({ page }) => {
    await stubNetwork(page, { mode: "empty" });
    await page.goto("/");
    await settle(page);

    // An empty state must say what would be here and why it is not — a blank
    // panel is indistinguishable from a broken one.
    const empty = page.getByText(/No US Open matches on today.s card/i);
    await expect(empty).toBeVisible();
    await expect(page.getByText(/Build a ticket from the board above/i)).toBeVisible();
  });

  test("page never scrolls sideways", async ({ page }) => {
    await stubNetwork(page);
    await page.goto("/");
    await settle(page);
    // The classic mobile break. Wide content must scroll inside its own
    // container, never take the page with it.
    const overflow = await page.evaluate(() =>
      document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  });
});

test.describe("interaction contracts", () => {
  test("keyboard focus is always visible", async ({ page }) => {
    await stubNetwork(page);
    await page.goto("/");
    await settle(page);

    // The single largest accessibility gap this redesign closed: 92 buttons
    // and 28 inputs previously had no focus-visible style at all.
    await page.keyboard.press("Tab");
    const outline = await page.evaluate(() => {
      const el = document.activeElement as HTMLElement | null;
      if (!el || el === document.body) return null;
      const s = getComputedStyle(el);
      return { width: s.outlineWidth, style: s.outlineStyle };
    });
    expect(outline).not.toBeNull();
    expect(outline!.style).not.toBe("none");
    expect(parseFloat(outline!.width)).toBeGreaterThanOrEqual(2);
  });

  test("every interactive control has an accessible name", async ({ page }) => {
    await stubNetwork(page, { mode: "populated" });
    await page.goto("/");
    await settle(page);

    const unnamed = await page.evaluate(() => {
      const bad: string[] = [];
      document.querySelectorAll("button, a[href], input, select").forEach(el => {
        const e = el as HTMLElement;
        if (e.offsetParent === null) return;                 // not visible
        const name = (e.getAttribute("aria-label") ||
          e.getAttribute("title") ||
          e.textContent?.trim() ||
          (e as HTMLInputElement).labels?.[0]?.textContent?.trim() ||
          "").trim();
        if (!name) bad.push(e.outerHTML.slice(0, 120));
      });
      return bad;
    });
    expect(unnamed, `controls with no accessible name:\n${unnamed.join("\n")}`).toEqual([]);
  });

  test("the browser never talks to Polymarket directly", async ({ page }) => {
    // The economics of the whole product rest on this: one upstream consumer
    // (our backend), not one per visitor. Before /api/pm existed in
    // production, every open tab refetched the Polymarket fixture index every
    // 60s straight from gamma-api, so upstream load scaled with user count.
    const direct: string[] = [];
    page.on("request", r => {
      if (/(^|\.)polymarket\.com/.test(new URL(r.url()).hostname)) direct.push(r.url());
    });

    await stubNetwork(page, { mode: "populated" });
    await page.goto("/");
    await settle(page);
    // Open a trade ticket path too — the order-book poll was the second direct caller.
    await page.getByTestId("parlay-toggle").first().click();
    await page.waitForTimeout(500);

    expect(direct, `direct Polymarket requests from the page:\n${direct.join("\n")}`).toEqual([]);
  });

  test("adding a parlay leg updates the ticket", async ({ page }) => {
    await stubNetwork(page, { mode: "populated" });
    await page.goto("/");
    await settle(page);

    const add = page.getByTestId("parlay-toggle").first();
    await expect(add).toBeVisible();
    await expect(add).toHaveAttribute("aria-pressed", "false");

    await add.click();
    // aria-pressed is the contract the button advertises; assert that, not colour.
    await expect(add).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByText(/Combined odds/i)).toBeVisible();

    // And it toggles back off, leaving the ticket empty again.
    await add.click();
    await expect(add).toHaveAttribute("aria-pressed", "false");
    await expect(page.getByText(/Build a ticket from the board above/i)).toBeVisible();
  });
});
