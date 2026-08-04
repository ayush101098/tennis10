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

/** Start a Stripe Checkout session; returns the hosted URL to redirect to. */
export async function startStripeCheckout(email: string): Promise<{ ok: boolean; url?: string; reason: string }> {
  try {
    const res = await fetch("/api/stripe/checkout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });
    const d = await res.json().catch(() => ({}));
    return {
      ok: !!d.ok && !!d.url,
      url: d.url,
      reason: d.reason || (res.ok ? "Could not start checkout." : "Checkout is unavailable right now."),
    };
  } catch {
    return { ok: false, reason: "Couldn't reach the payment service. Check your connection and retry." };
  }
}

/** Confirm a returned Checkout session with Stripe (authoritative — the server
 *  asks Stripe, so this works even if the webhook hasn't landed yet). */
export async function confirmStripeSession(sessionId: string): Promise<ServerVerify & { email?: string }> {
  try {
    const res = await fetch(`/api/stripe/confirm?session_id=${encodeURIComponent(sessionId)}`, { cache: "no-store" });
    const d = await res.json().catch(() => ({}));
    return {
      ok: !!d.ok,
      reason: d.reason || "Could not confirm the payment.",
      paidUntil: d.paidUntil,
      amountUsd: d.amountUsd,
      email: d.email,
    };
  } catch {
    return { ok: false, reason: "Couldn't reach the payment service to confirm." };
  }
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

/** The server's current entitlement for an email, or null if it can't be reached.
 *
 * Checks BOTH sources and takes the later expiry: /api/verify (on-chain payment
 * entitlements) and /api/account (the unified account DB, which also carries
 * manual grants for off-platform payments). Querying only the verifier meant a
 * granted account still read as unpaid. Null only if neither can be reached, so
 * a single outage can't silently revoke access. */
export async function serverEntitlement(email: string): Promise<{ active: boolean; paidUntil: number } | null> {
  const one = async (url: string) => {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) return null;
      const d = await res.json();
      return Number(d.paidUntil) || 0;
    } catch {
      return null;
    }
  };
  const q = encodeURIComponent(email);
  const [paid, acct] = await Promise.all([
    one(`/api/verify?email=${q}`),
    one(`/api/account?email=${q}`),
  ]);
  if (paid === null && acct === null) return null;
  const paidUntil = Math.max(paid || 0, acct || 0);
  return { active: paidUntil > Date.now(), paidUntil };
}
