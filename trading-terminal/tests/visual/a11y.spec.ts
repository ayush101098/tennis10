import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { stubNetwork, settle } from "./fixtures";

/**
 * Automated accessibility checks.
 *
 * Axe catches roughly a third of real accessibility problems — it is a floor,
 * not a certificate. The manual checklist in /docs/ui-qa.md covers what it
 * cannot: focus order, whether a live region actually says something useful,
 * and whether the primary action is findable.
 *
 * Contrast is included deliberately: the pre-redesign palette used a muted
 * grey at 2.55:1 for most small labels, and this is the test that stops it
 * coming back.
 */

const ROUTES = ["/", "/manual", "/calculator"];

for (const route of ROUTES) {
  test(`${route} has no critical or serious accessibility violations`, async ({ page }) => {
    await stubNetwork(page, { mode: "populated" });
    await page.goto(route);
    await settle(page);

    const { violations } = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();

    const serious = violations.filter(v => v.impact === "critical" || v.impact === "serious");
    const report = serious
      .map(v => `${v.impact} · ${v.id} · ${v.help}\n    ${v.nodes.slice(0, 3).map(n => n.target.join(" ")).join("\n    ")}`)
      .join("\n");

    expect(serious, `accessibility violations on ${route}:\n${report}`).toEqual([]);
  });
}

test("headings form a coherent outline", async ({ page }) => {
  await stubNetwork(page, { mode: "populated" });
  await page.goto("/");
  await settle(page);

  const levels = await page.$$eval("h1,h2,h3,h4,h5,h6", els =>
    els.filter(e => (e as HTMLElement).offsetParent !== null)
       .map(e => Number(e.tagName[1])));

  // Exactly one h1 — the page's subject.
  expect(levels.filter(l => l === 1)).toHaveLength(1);

  // No skipped levels. A jump from h2 to h4 makes the document outline
  // meaningless to anyone navigating by heading.
  for (let i = 1; i < levels.length; i++) {
    expect(levels[i] - levels[i - 1]).toBeLessThanOrEqual(1);
  }
});
