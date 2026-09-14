import { defineConfig } from "vitest/config";
import path from "path";

/**
 * Vitest needs the same `@/*` -> `src/*` alias tsconfig gives Next, otherwise
 * any module under test that imports by alias fails to resolve and the suite
 * reports "0 test" rather than a missing-import error. Existing tests dodged
 * this with relative imports; new library code should not have to.
 */
export default defineConfig({
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  test: {
    // tests/visual/* are PLAYWRIGHT specs and belong to a different runner.
    // Collected here they throw "test() was called here" and `vitest run`
    // exits non-zero with two failed files while every actual test passes —
    // a red suite that means nothing, which is worse than no suite at all
    // because it trains you to ignore the result.
    exclude: ["**/node_modules/**", "**/dist/**", "tests/visual/**"],
  },
});
