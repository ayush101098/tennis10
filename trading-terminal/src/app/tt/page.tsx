"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/** The TT terminal now lives inside the unified terminal — keep old links working. */
export default function TtRedirect() {
  const router = useRouter();
  useEffect(() => { router.replace("/terminal?tab=tt"); }, [router]);
  return (
    <div className="h-screen flex items-center justify-center text-[11px] text-terminal-muted">
      Redirecting to the unified terminal…
    </div>
  );
}
