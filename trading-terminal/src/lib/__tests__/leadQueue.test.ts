import { describe, it, expect, beforeEach, vi } from "vitest";

/**
 * No jsdom in this project — a minimal in-memory localStorage is enough for
 * what leadQueue actually calls (getItem/setItem), and avoids adding a new
 * test dependency for one file's worth of coverage.
 */
function fakeLocalStorage() {
  const data = new Map<string, string>();
  return {
    getItem: (k: string) => (data.has(k) ? data.get(k)! : null),
    setItem: (k: string, v: string) => { data.set(k, v); },
    removeItem: (k: string) => { data.delete(k); },
    clear: () => data.clear(),
  };
}

beforeEach(() => {
  vi.resetModules();
  (globalThis as unknown as { window: unknown }).window = globalThis;
  (globalThis as unknown as { localStorage: unknown }).localStorage = fakeLocalStorage();
});

describe("queueLead / pendingLeadCount", () => {
  it("queues an email and reports it pending", async () => {
    const { queueLead, pendingLeadCount } = await import("../leadQueue");
    queueLead("a@example.com", "hero");
    expect(pendingLeadCount()).toBe(1);
  });

  it("does not queue the same email twice", async () => {
    const { queueLead, pendingLeadCount } = await import("../leadQueue");
    queueLead("a@example.com", "hero");
    queueLead("A@Example.com", "signin"); // same address, different case/source
    expect(pendingLeadCount()).toBe(1);
  });

  it("normalises case before deduping and storing", async () => {
    const { queueLead, pendingLeadCount } = await import("../leadQueue");
    queueLead("Mixed@Case.com", "hero");
    queueLead("mixed@case.com", "hero");
    expect(pendingLeadCount()).toBe(1);
  });
});

describe("captureLeadDurably", () => {
  it("does NOT queue when the server confirms it reached blobs", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true, json: async () => ({ ok: true, stored: "blobs" }),
    }) as unknown as typeof fetch;
    const { captureLeadDurably } = await import("../leadQueue");
    const { pendingLeadCount } = await import("../leadQueue");
    await captureLeadDurably("safe@example.com", "hero");
    expect(pendingLeadCount()).toBe(0);
  });

  it("queues when the server falls back to memory — the actual production bug", async () => {
    // This is exactly what netlify/functions/subscribe.js returns when the
    // blob store rejects the write: HTTP 200, ok:true, stored:"memory" — a
    // capture that LOOKS successful and is about to evaporate.
    global.fetch = vi.fn().mockResolvedValue({
      ok: true, json: async () => ({ ok: true, stored: "memory" }),
    }) as unknown as typeof fetch;
    const { captureLeadDurably, pendingLeadCount } = await import("../leadQueue");
    await captureLeadDurably("atrisk@example.com", "hero");
    expect(pendingLeadCount()).toBe(1);
  });

  it("queues when the request fails outright", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("network down")) as unknown as typeof fetch;
    const { captureLeadDurably, pendingLeadCount } = await import("../leadQueue");
    await captureLeadDurably("offline@example.com", "hero");
    expect(pendingLeadCount()).toBe(1);
  });

  it("queues when the server returns ok:false", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false, json: async () => ({ error: "Enter a valid email address." }),
    }) as unknown as typeof fetch;
    const { captureLeadDurably, pendingLeadCount } = await import("../leadQueue");
    await captureLeadDurably("bad@example.com", "hero");
    expect(pendingLeadCount()).toBe(1);
  });
});

describe("flushPendingLeads", () => {
  it("drops an entry once a retry confirms it reached blobs", async () => {
    const { queueLead, flushPendingLeads, pendingLeadCount } = await import("../leadQueue");
    queueLead("recovers@example.com", "hero");
    global.fetch = vi.fn().mockResolvedValue({
      ok: true, json: async () => ({ ok: true, stored: "blobs" }),
    }) as unknown as typeof fetch;
    await flushPendingLeads();
    expect(pendingLeadCount()).toBe(0);
  });

  it("keeps retrying while the backend is still not durable", async () => {
    const { queueLead, flushPendingLeads, pendingLeadCount } = await import("../leadQueue");
    queueLead("stillstuck@example.com", "hero");
    global.fetch = vi.fn().mockResolvedValue({
      ok: true, json: async () => ({ ok: true, stored: "memory" }),
    }) as unknown as typeof fetch;
    await flushPendingLeads();
    expect(pendingLeadCount()).toBe(1); // still queued, not lost
  });

  it("does nothing when the queue is empty — no needless network calls", async () => {
    const { flushPendingLeads } = await import("../leadQueue");
    const fetchSpy = vi.fn();
    global.fetch = fetchSpy as unknown as typeof fetch;
    await flushPendingLeads();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("eventually gives up on an address that never confirms, rather than retrying forever", async () => {
    const { queueLead, flushPendingLeads, pendingLeadCount } = await import("../leadQueue");
    queueLead("neverworks@example.com", "hero");
    global.fetch = vi.fn().mockResolvedValue({
      ok: true, json: async () => ({ ok: true, stored: "memory" }),
    }) as unknown as typeof fetch;
    for (let i = 0; i < 60; i++) await flushPendingLeads();
    expect(pendingLeadCount()).toBe(0);
  });
});
