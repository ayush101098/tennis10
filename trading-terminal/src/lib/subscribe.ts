/**
 * Client helpers for the email-capture + payment ledger endpoint
 * (/api/subscribe -> Netlify function `subscribe` in production).
 */

export type SubscribeResult = { ok: boolean; error?: string };

export async function captureLead(email: string, source = "cta"): Promise<SubscribeResult> {
  try {
    const res = await fetch("/api/subscribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, source }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return { ok: false, error: data.error || "Something went wrong. Try again." };
    return { ok: true };
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
