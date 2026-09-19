/**
 * The safety net under every "join the waitlist" / sign-in email.
 *
 * WHY THIS EXISTS
 * captureLead() posts to /api/subscribe, which is supposed to land in Netlify
 * Blobs. Two ways that silently doesn't happen:
 *
 *   1. The request never completes — a network blip, or the call was made
 *      fire-and-forget (see auth.tsx's recordLogin, which must never block
 *      sign-in on a network round trip) and its .catch() swallowed the error.
 *   2. It DOES complete with `ok: true` — but the blob write itself failed
 *      server-side, and subscribe.js fell back to a per-CONTAINER in-memory
 *      list to avoid returning an error for what is, at that instant, a real
 *      capture. That memory is gone the moment the function's container
 *      recycles. Confirmed live 2026-09-20: the production blob store is
 *      SUSPENDED (every write 403s), which means EVERY capture right now is
 *      landing in that doomed memory fallback, `ok: true` and all — nothing
 *      before this file told the browser, so nothing retried, and the address
 *      was gone within the hour with the UI having said "you're in."
 *
 * WHAT THIS DOES ABOUT IT
 * Every capture that is not confirmed `stored: "blobs"` is queued here, in the
 * one place guaranteed to survive a serverless container recycling: the
 * visitor's own browser. flushPendingLeads() retries the queue — on load, and
 * after every future capture — until the address is durably confirmed, however
 * long the backend takes to recover. This does not fix a suspended paid
 * service (that needs a human on the Vercel dashboard); it makes sure no
 * address is lost to the outage while it lasts.
 */

import { captureLead } from "./subscribe";

const KEY = "tt_pending_leads";
const MAX_QUEUE = 200;   // a runaway queue is itself a symptom, not a fix

interface PendingLead {
  email: string;
  source: string;
  queuedAt: number;
  attempts: number;
}

function readQueue(): PendingLead[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];   // a corrupted queue blocks nothing; it just starts empty
  }
}

function writeQueue(q: PendingLead[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(q.slice(-MAX_QUEUE)));
  } catch {
    /* localStorage full or unavailable — the capture already happened once;
       losing the retry record is a lesser failure than throwing here. */
  }
}

/** Add an email to the durability queue, unless it's already waiting. */
export function queueLead(email: string, source: string): void {
  const q = readQueue();
  const e = email.trim().toLowerCase();
  if (q.some(p => p.email === e)) return;
  q.push({ email: e, source, queuedAt: Date.now(), attempts: 0 });
  writeQueue(q);
}

/**
 * Capture an email, and queue it for retry unless it is CONFIRMED durable.
 * This is the function every sign-up path should call instead of talking to
 * captureLead directly — it is the same call, with the safety net attached.
 */
export async function captureLeadDurably(email: string, source: string): Promise<void> {
  try {
    const res = await captureLead(email, source);
    if (!res.ok || res.stored !== "blobs") queueLead(email, source);
  } catch {
    queueLead(email, source);
  }
}

let flushing = false;

/**
 * Retry every queued email. Safe to call often — a queue of zero returns
 * immediately, and a concurrent call is a no-op rather than a double-send.
 */
export async function flushPendingLeads(): Promise<void> {
  if (flushing || typeof window === "undefined") return;
  const q = readQueue();
  if (!q.length) return;
  flushing = true;
  try {
    const remaining: PendingLead[] = [];
    for (const lead of q) {
      try {
        const res = await captureLead(lead.email, lead.source);
        if (res.ok && res.stored === "blobs") continue;   // confirmed durable — drop it
      } catch {
        /* still not durable — falls through to requeue */
      }
      // Give up after a while rather than retrying a bad address forever —
      // ~2 weeks of once-a-session attempts is generous for a real outage.
      if (lead.attempts + 1 < 50) {
        remaining.push({ ...lead, attempts: lead.attempts + 1 });
      }
    }
    writeQueue(remaining);
  } finally {
    flushing = false;
  }
}

/** For diagnostics — how many addresses are currently unconfirmed. */
export function pendingLeadCount(): number {
  return readQueue().length;
}
