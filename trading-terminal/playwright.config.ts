import { defineConfig, devices } from "@playwright/test";

/**
 * Visual + interaction QA.
 *
 * Two viewports only, on purpose: a phone and a laptop. Baselines are a
 * maintenance cost, and a matrix of five widths mostly catches the same
 * regression five times.
 *
 * `animations: "disabled"` and a frozen clock matter more here than usual —
 * this app has a blinking live indicator and polls a live schedule, so
 * without both, every screenshot differs from the last for reasons that have
 * nothing to do with the UI.
 */
export default defineConfig({
  testDir: "./tests/visual",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : [["list"]],

  expect: {
    toHaveScreenshot: {
      // Font rasterisation differs a little between machines; a hard zero
      // makes the suite fail for reasons no one can act on.
      maxDiffPixelRatio: 0.01,
      animations: "disabled",
    },
  },

  use: {
    baseURL: process.env.PW_BASE_URL ?? "http://127.0.0.1:3100",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 800 } } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],

  // A production build, not `next dev`: dev serves unminified CSS through a
  // different pipeline and its overlay can appear in screenshots.
  webServer: {
    command: "npm run build && npx next start --port 3100",
    url: "http://127.0.0.1:3100",
    reuseExistingServer: !process.env.CI,
    timeout: 240_000,
  },
});
