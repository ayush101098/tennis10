"use client";

import { useCallback, useEffect, useState } from "react";

/**
 * Portfolio bankroll ($), persisted to localStorage and synced across every
 * component/tab so a stake set once drives sizing everywhere. Defaults to
 * $1,000 — the manual's "100 units" maps cleanly onto any bankroll.
 */
const LS_KEY = "tt.portfolio";
const DEFAULT = 1000;
const EVT = "tt-portfolio-changed";

function read(): number {
  if (typeof window === "undefined") return DEFAULT;
  const raw = Number(localStorage.getItem(LS_KEY));
  return Number.isFinite(raw) && raw > 0 ? raw : DEFAULT;
}

export function usePortfolio(): [number, (v: number) => void] {
  const [portfolio, setState] = useState<number>(DEFAULT);

  // Read the persisted value after mount (avoids SSR/hydration mismatch).
  useEffect(() => setState(read()), []);

  useEffect(() => {
    const sync = () => setState(read());
    window.addEventListener(EVT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(EVT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  const setPortfolio = useCallback((v: number) => {
    const clean = Number.isFinite(v) && v > 0 ? v : DEFAULT;
    localStorage.setItem(LS_KEY, String(clean));
    setState(clean);
    window.dispatchEvent(new Event(EVT));
  }, []);

  return [portfolio, setPortfolio];
}
