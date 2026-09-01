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
});
