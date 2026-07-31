/**
 * Client bridge to the AUTHORITATIVE server verifier (/api/verify).
 *
 * The client never decides entitlement on its own anymore: it asks the server,
 * which checks the chain and its own records. verifyPaymentTx in auth.tsx stays
 * only as instant UX feedback — the *grant* comes from serverVerifyPayment, and
 * on every load syncEntitlement reconciles localStorage against serverEntitlement
 * so a hand-edited paidUntil is overwritten by the server's truth.
 */

export interface ServerVerify {
  ok: boolean;
  reason: string;
  paidUntil?: number;
  amountUsd?: number;
}

/** Authoritative verification + grant. Fail-closed: no ok unless the server says so. */
export async function serverVerifyPayment(email: string, txHash: string): Promise<ServerVerify> {
  try {
    const res = await fetch("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, txHash }),
    });
    const data = await res.json().catch(() => ({}));
    return {
      ok: !!data.ok,
      reason: data.reason || (res.ok ? "Verification failed." : "Verification service unavailable — try again."),
      paidUntil: data.paidUntil,
      amountUsd: data.amountUsd,
    };
  } catch {
    return { ok: false, reason: "Couldn't reach the verification service. Check your connection and retry." };
  }
}

/** The server's current entitlement for an email, or null if it can't be reached. */
export async function serverEntitlement(email: string): Promise<{ active: boolean; paidUntil: number } | null> {
  try {
    const res = await fetch(`/api/verify?email=${encodeURIComponent(email)}`, { cache: "no-store" });
    if (!res.ok) return null;
    const d = await res.json();
    return { active: !!d.active, paidUntil: Number(d.paidUntil) || 0 };
  } catch {
    return null;
  }
}
