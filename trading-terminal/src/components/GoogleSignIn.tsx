"use client";

import { useEffect, useRef, useState } from "react";

/**
 * "Continue with Google".
 *
 * Renders nothing unless NEXT_PUBLIC_GOOGLE_CLIENT_ID is set, so a
 * half-configured deploy never shows a button that cannot work — the worst
 * possible thing to put on a signup screen.
 *
 * The credential Google hands back is NOT trusted here. It goes to
 * /api/google-auth, which asks Google whether the token is genuine and whether
 * it was issued for this site; only the email that comes back from there is
 * used. Anything decided in this file could be forged by the person using it.
 */

const CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || "";
const GSI_SRC = "https://accounts.google.com/gsi/client";

interface GsiWindow extends Window {
  google?: {
    accounts: {
      id: {
        initialize: (o: Record<string, unknown>) => void;
        renderButton: (el: HTMLElement, o: Record<string, unknown>) => void;
      };
    };
  };
}

function loadGsi(): Promise<boolean> {
  return new Promise(resolve => {
    if (typeof window === "undefined") return resolve(false);
    if ((window as GsiWindow).google?.accounts?.id) return resolve(true);
    const existing = document.querySelector(`script[src="${GSI_SRC}"]`);
    if (existing) {
      existing.addEventListener("load", () => resolve(true));
      existing.addEventListener("error", () => resolve(false));
      return;
    }
    const el = document.createElement("script");
    el.src = GSI_SRC;
    el.async = true;
    el.defer = true;
    el.onload = () => resolve(true);
    el.onerror = () => resolve(false);   // blocked or offline — email still works
    document.head.appendChild(el);
  });
}

export default function GoogleSignIn({ onEmail, onError }: {
  onEmail: (email: string) => void;
  onError?: (reason: string) => void;
}) {
  const holder = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!CLIENT_ID) return;
    let alive = true;

    loadGsi().then(ok => {
      if (!ok || !alive || !holder.current) return;
      const g = (window as GsiWindow).google;
      if (!g) return;

      g.accounts.id.initialize({
        client_id: CLIENT_ID,
        callback: async (resp: { credential?: string }) => {
          try {
            const r = await fetch("/api/google-auth", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ credential: resp.credential }),
            });
            const data = await r.json();
            if (data.ok && data.email) onEmail(data.email);
            else onError?.(data.reason || "Google sign-in failed.");
          } catch {
            onError?.("Couldn't reach the server. Try the email option.");
          }
        },
      });

      g.accounts.id.renderButton(holder.current, {
        theme: "filled_black",
        size: "large",
        text: "continue_with",
        shape: "rectangular",
        width: 320,
      });
      setReady(true);
    });

    return () => { alive = false; };
  }, [onEmail, onError]);

  if (!CLIENT_ID) return null;

  return (
    <div className="flex flex-col items-center">
      <div ref={holder} className="min-h-[40px]" />
      {!ready && (
        <div className="text-[10px] text-terminal-muted py-2">loading Google sign-in…</div>
      )}
    </div>
  );
}

export const GOOGLE_ENABLED = !!CLIENT_ID;
