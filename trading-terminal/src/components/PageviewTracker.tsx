"use client";

import { useEffect } from "react";
import { usePathname } from "next/navigation";

/**
 * First-party pageview beacon. Fires once per route change with the path, the
 * referrer, and an anonymous per-browser visitor id (so the admin dashboard can
 * count unique visitors). No cookies, no PII. Failures are swallowed — tracking
 * must never affect the page.
 */
function visitorId(): string {
  try {
    let v = localStorage.getItem("tt_vid");
    if (!v) {
      v = (crypto.randomUUID?.() || Math.random().toString(36).slice(2)) as string;
      localStorage.setItem("tt_vid", v);
    }
    return v;
  } catch {
    return "anon";
  }
}

export default function PageviewTracker() {
  const pathname = usePathname();
  useEffect(() => {
    try {
      const body = JSON.stringify({
        path: pathname || "/",
        ref: document.referrer || "",
        vid: visitorId(),
      });
      // sendBeacon survives navigation; fall back to fetch keepalive.
      if (navigator.sendBeacon) {
        navigator.sendBeacon("/api/track", new Blob([body], { type: "application/json" }));
      } else {
        fetch("/api/track", { method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true }).catch(() => {});
      }
    } catch {
      /* never break the page */
    }
  }, [pathname]);
  return null;
}
