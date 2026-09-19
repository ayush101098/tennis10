/**
 * Client helpers for the email-capture + payment ledger endpoint
 * (/api/subscribe -> Netlify function `subscribe` in production).
 */

/**
 * `stored` tells you whether the address actually reached Blobs.
 *
 * netlify/functions/subscribe.js falls back to a per-CONTAINER in-memory list
 * when the blob write fails — the server keeps answering `ok: true` either
 * way, because a memory-only save is still a genuine capture at that instant.
 * But that memory evaporates the moment the function's container recycles,
 * which on a low-traffic serverless deploy can be within the hour. `ok: true,
 * stored: "memory"` is the signal that an address was captured but is not yet
 * SAFE — see lib/leadQueue, which exists specifically to catch this case and
 * keep retrying until a later attempt lands in Blobs for real.
 */
export type SubscribeResult = { ok: boolean; error?: string; stored?: "blobs" | "memory" };

export async function captureLead(email: string, source = "cta"): Promise<SubscribeResult> {
  try {
    const res = await fetch("/api/subscribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, source }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return { ok: false, error: data.error || "Something went wrong. Try again." };
    return { ok: true, stored: data.stored };
  } catch {
    return { ok: false, error: "Network error. Check your connection and retry." };
  }
}

/**
 * Link a verified on-chain payment to an email so "who paid" is answerable
 * server-side. Fire-and-forget from the payment flow — never blocks the grant.
 */
export function recordPayment(email: string, txHash: string, amount?: string, from?: string): void {
  try {
    fetch("/api/subscribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, txHash, amount, from, source: "payment" }),
      keepalive: true,
    }).catch(() => {});
  } catch {
    /* best-effort */
  }
}
