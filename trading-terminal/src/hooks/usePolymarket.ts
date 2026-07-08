"use client";

import { useEffect, useState } from "react";
import { getPolymarketIndex, type PmFixture } from "@/lib/polymarket";

const REFRESH_MS = 60_000;

/** Live-refreshing index of open Polymarket tennis fixtures. */
export function usePolymarket(): Map<string, PmFixture> {
  const [index, setIndex] = useState<Map<string, PmFixture>>(new Map());

  useEffect(() => {
    let alive = true;
    const load = () =>
      getPolymarketIndex()
        .then(i => { if (alive) setIndex(i); })
        .catch(() => { /* PM column just shows "—" until next refresh */ });
    load();
    const t = setInterval(load, REFRESH_MS);
    return () => { alive = false; clearInterval(t); };
  }, []);

  return index;
}
