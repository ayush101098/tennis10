"use client";

import { useEffect, useState } from "react";

/**
 * Shows how many people currently have the terminal open, by heartbeating a
 * per-tab id to /api/presence every 15s. Hides itself if the endpoint isn't
 * available (e.g. a static export with no server).
 */
export default function LiveUsers() {
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    // Stable id for the lifetime of this tab
    let id = sessionStorage.getItem("tt_presence_id");
    if (!id) {
      id = Math.random().toString(36).slice(2) + Date.now().toString(36);
      sessionStorage.setItem("tt_presence_id", id);
    }

    let alive = true;
    const beat = async () => {
      try {
        const res = await fetch("/api/presence", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id }),
          cache: "no-store",
        });
        if (!res.ok) throw new Error(String(res.status));
        const data = await res.json();
        if (alive && typeof data.count === "number") setCount(data.count);
      } catch {
        if (alive) setCount(null); // endpoint gone → hide
      }
    };

    beat();
    const iv = setInterval(beat, 15_000);
    return () => { alive = false; clearInterval(iv); };
  }, []);

  if (count === null || count < 1) return null;

  return (
    <span
      className="flex items-center gap-1 text-terminal-green"
      title={`${count} ${count === 1 ? "person" : "people"} viewing the terminal right now`}
    >
      <span className="relative flex h-1.5 w-1.5">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-terminal-green opacity-75" />
        <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-terminal-green" />
      </span>
      <span className="font-mono font-bold">{count}</span>
      <span className="text-terminal-muted">live</span>
    </span>
  );
}
