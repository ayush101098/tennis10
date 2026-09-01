"use client";

/**
 * Admin-only analytics dashboard.
 *
 * Access: gated on the client-side admin flag (ADMIN_EMAILS). Because the site
 * is a static export with no server session, the leads/payments readout is
 * additionally protected server-side by LEADS_ADMIN_TOKEN — the admin pastes it
 * once (kept in localStorage), and it's sent as x-admin-token. We never bake the
 * token into the shipped bundle.
 *
 * Shows: first-party lead capture + conversions (from /api/subscribe) and the
 * on-chain "who paid" ledger (queried live from Blockscout for PAYMENT_ADDRESS).
 */

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useTier, PAYMENT_ADDRESS } from "@/lib/auth";

type Lead = { email: string; ts: number; lastSeen: number; source: string; paid: boolean };
type Payment = { email: string; txHash: string; amount: string | null; from: string | null; ts: number };
type OnchainTx = { hash: string; from: string; when: number; amount: number; symbol: string; usd: number | null };
type AccountRow = {
  email: string; firstSeen: number; lastLogin: number; loginCount: number;
  source: string; active: boolean; paidUntil: number; daysLeft: number;
  totalPaidUsd: number; payments: number; grants: number;
  pending?: PendingClaim[];
};
type PendingClaim = { method: string; note: string; amountUsd: number; ts: number; status: string };
type AccountCounts = { accounts: number; active: number; paying: number; comped: number; revenueUsd: number; pendingClaims?: number };
type KV = { k: string; v: number };
type Traffic = {
  views: number; uniques: number;
  byPath: KV[]; byRef: KV[];
  byDay: { day: string; count: number }[];
  recent: { ts: number; path: string; ref: string; vid: string }[];
};

const TOKEN_KEY = "tt_admin_token";
const fmtDate = (ms: number) => new Date(ms).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
const short = (s: string) => (s.length > 14 ? `${s.slice(0, 8)}…${s.slice(-4)}` : s);

export default function AdminPage() {
  const { session } = useTier();
  const [token, setToken] = useState("");
  const [tokenInput, setTokenInput] = useState("");
  const [leads, setLeads] = useState<Lead[] | null>(null);
  const [payments, setPayments] = useState<Payment[]>([]);
  const [traffic, setTraffic] = useState<Traffic | null>(null);
  const [onchain, setOnchain] = useState<OnchainTx[] | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [accounts, setAccounts] = useState<AccountRow[] | null>(null);
  const [accCounts, setAccCounts] = useState<AccountCounts | null>(null);
  const [granting, setGranting] = useState<string | null>(null);
  const [grantEmail, setGrantEmail] = useState("");
  const [grantDays, setGrantDays] = useState(30);
  const [grantMsg, setGrantMsg] = useState<{ ok: boolean; text: string } | null>(null);

  /**
   * Confirm a payment claim: grant 30 days and clear the pending flag.
   * Only ever driven by a human who has just seen the money land — the claim
   * itself proves nothing.
   */
  const grant = async (email: string, days: number, reason: string) => {
    setGranting(email);
    setGrantMsg(null);
    try {
      const res = await fetch("/api/account", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, action: "grant", days, reason, adminToken: token }),
      });
      const data = await res.json();
      if (data.ok) {
        setAccounts(prev => prev && prev.map(a => a.email === email
          ? { ...a, active: true, paidUntil: data.paidUntil, pending: [] } : a));
        setGrantMsg({ ok: true, text: `${email} has access until ${fmtDate(data.paidUntil)} (${days} days).` });
        return true;
      }
      setGrantMsg({ ok: false, text: data.reason || "Grant failed." });
    } catch {
      setGrantMsg({ ok: false, text: "Grant failed — network error." });
    } finally {
      setGranting(null);
    }
    return false;
  };

  const confirmClaim = (email: string) => grant(email, 30, "paypal.me confirmed");

  useEffect(() => { setToken(localStorage.getItem(TOKEN_KEY) || ""); }, []);

  // The account database — logins, payments and grants in one roster.
  useEffect(() => {
    if (!token) return;
    (async () => {
      try {
        const res = await fetch("/api/account", { headers: { "x-admin-token": token } });
        if (!res.ok) return;
        const d = await res.json();
        setAccounts(d.rows || []); setAccCounts(d.counts || null);
      } catch { /* leave null — section shows its own empty state */ }
    })();
  }, [token]);

  const loadLeads = useCallback(async (t: string) => {
    setLoading(true); setErr("");
    try {
      const res = await fetch("/api/subscribe", { headers: { "x-admin-token": t } });
      if (res.status === 401) { setErr("Token rejected. Check LEADS_ADMIN_TOKEN."); setLeads(null); return; }
      const data = await res.json();
      setLeads(data.leads || []); setPayments(data.payments || []);
      setTraffic(data.traffic || null);
    } catch {
      setErr("Could not reach the leads endpoint.");
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { if (token) loadLeads(token); }, [token, loadLeads]);

  // On-chain "who paid" — incoming transfers to the access address.
  useEffect(() => {
    if (!session?.isAdmin) return;
    (async () => {
      const A = PAYMENT_ADDRESS.toLowerCase();
      try {
        let ethUsd: number | null = null;
        try {
          const p = await fetch("https://api.coinbase.com/v2/prices/ETH-USD/spot").then(r => r.json());
          ethUsd = parseFloat(p.data.amount);
        } catch { /* price optional */ }
        const base = "https://eth.blockscout.com/api?module=account&address=" + A + "&sort=desc";
        const [nat, tok] = await Promise.all([
          fetch(base + "&action=txlist").then(r => r.json()),
          fetch(base + "&action=tokentx").then(r => r.json()),
        ]);
        const out: OnchainTx[] = [];
        for (const t of (nat.result || [])) {
          if (t.to?.toLowerCase() === A && t.value !== "0") {
            const amt = Number(t.value) / 1e18;
            out.push({ hash: t.hash, from: t.from, when: Number(t.timeStamp) * 1000, amount: amt, symbol: "ETH", usd: ethUsd ? amt * ethUsd : null });
          }
        }
        for (const t of (tok.result || [])) {
          if (t.to?.toLowerCase() === A) {
            const dec = Number(t.tokenDecimal || 18);
            const amt = Number(t.value) / 10 ** dec;
            const sym = t.tokenSymbol || "TOKEN";
            const usd = /^(usdt|usdc|dai)$/i.test(sym) ? amt : null;
            out.push({ hash: t.hash, from: t.from, when: Number(t.timeStamp) * 1000, amount: amt, symbol: sym, usd });
          }
        }
        out.sort((a, b) => b.when - a.when);
        setOnchain(out);
      } catch {
        setOnchain([]); // CORS / offline — fall back to the explorer link in the UI
      }
    })();
  }, [session?.isAdmin]);

  if (!session) {
    return <Shell><p className="text-terminal-muted">Sign in to continue.</p></Shell>;
  }
  if (!session.isAdmin) {
    return <Shell><p className="text-red-400">Not authorized. This page is admin-only.</p>
      <Link href="/" className="text-primary text-sm underline underline-offset-2 decoration-1 decoration-current/40 hover:decoration-current">← back to site</Link></Shell>;
  }

  const paidLeads = (leads || []).filter(l => l.paid).length;
  const conv = leads && leads.length ? (paidLeads / leads.length) * 100 : 0;
  const onchainUsd = (onchain || []).reduce((s, t) => s + (t.usd || 0), 0);
  const realPayers = (onchain || []).filter(t => (t.usd ?? 0) >= 90).length;

  return (
    <Shell>
      <div className="flex items-baseline justify-between flex-wrap gap-2 mb-6">
        <h1 className="text-xl font-bold text-slate-100">Admin · Analytics</h1>
        <Link href="/" className="text-terminal-muted text-xs hover:text-slate-200">← site</Link>
      </div>

      {!token ? (
        <div className="bg-terminal-panel border border-terminal-border rounded p-4 max-w-md">
          <p className="text-sm text-slate-200 mb-2">Enter the leads admin token</p>
          <p className="text-[11px] text-terminal-muted mb-3">The <code>LEADS_ADMIN_TOKEN</code> you set in Netlify env. Stored locally, sent as a header.</p>
          <div className="flex gap-2">
            <input type="password" value={tokenInput} onChange={e => setTokenInput(e.target.value)}
              placeholder="token" className="flex-1 px-3 py-2 rounded bg-terminal-bg border border-terminal-border text-sm text-slate-100" />
            <button onClick={() => { localStorage.setItem(TOKEN_KEY, tokenInput); setToken(tokenInput); }}
              className="px-4 py-2 rounded bg-terminal-green text-black text-xs font-bold">Unlock</button>
          </div>
          {err && <p className="text-xs text-red-400 mt-2">{err}</p>}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-8">
            <Kpi label="Page views" value={traffic ? traffic.views : "…"} />
            <Kpi label="Unique visitors" value={traffic ? traffic.uniques : "…"} />
            <Kpi label="Leads captured" value={leads ? leads.length : "…"} />
            <Kpi label="Paid" value={paidLeads} />
            <Kpi label="Lead→paid" value={`${conv.toFixed(0)}%`} />
            <Kpi label="On-chain payers (≥$90)" value={onchain ? realPayers : "…"} />
          </div>

          {err && <p className="text-xs text-red-400 mb-4">{err} <button className="underline" onClick={() => { localStorage.removeItem(TOKEN_KEY); setToken(""); }}>re-enter token</button></p>}

          {/* Comp an account directly — a reviewer, a trial, someone who paid by
              a route the site never saw. Writes a grant, same as confirming a
              claim; access lapses on its own when the window ends. */}
          <Section title="Grant access">
            <div className="flex flex-wrap items-end gap-2">
              <label className="flex flex-col gap-1">
                <span className="text-[10px] text-terminal-muted">Email</span>
                <input value={grantEmail} onChange={e => setGrantEmail(e.target.value)}
                  placeholder="someone@example.com" type="email"
                  className="min-h-[36px] w-[260px] max-w-full bg-terminal-bg border border-terminal-border rounded px-3 text-[12px] text-slate-100 focus:border-terminal-green outline-none" />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-[10px] text-terminal-muted">Days</span>
                <input value={grantDays} onChange={e => setGrantDays(Math.max(1, Number(e.target.value) || 1))}
                  type="number" min={1} max={3650}
                  className="min-h-[36px] w-[90px] bg-terminal-bg border border-terminal-border rounded px-3 text-[12px] text-slate-100 focus:border-terminal-green outline-none" />
              </label>
              <button
                disabled={!/\S+@\S+\.\S+/.test(grantEmail) || granting === grantEmail.trim().toLowerCase()}
                onClick={async () => {
                  const e = grantEmail.trim().toLowerCase();
                  if (await grant(e, grantDays, "granted from admin")) setGrantEmail("");
                }}
                className="min-h-[36px] px-4 rounded bg-terminal-green text-black text-[11px] font-bold hover:opacity-90 disabled:opacity-40">
                {granting === grantEmail.trim().toLowerCase() ? "GRANTING…" : `GRANT ${grantDays}d`}
              </button>
              {[7, 30, 90].map(d => (
                <button key={d} onClick={() => setGrantDays(d)}
                  className={`min-h-[36px] px-2.5 rounded border text-[10px] font-bold ${grantDays === d ? "border-terminal-green text-terminal-green" : "border-terminal-border text-terminal-muted hover:text-slate-300"}`}>
                  {d}d
                </button>
              ))}
            </div>
            {grantMsg && (
              <p className={`mt-2 text-[11px] ${grantMsg.ok ? "text-terminal-green" : "text-terminal-red"}`}>{grantMsg.text}</p>
            )}
            <p className="mt-2 text-[10px] text-terminal-muted">
              The account does not need to exist yet — it is created, and the grant applies when they sign in
              with that address. Access expires on its own; nothing to undo.
            </p>
          </Section>

          {/* Claims come from payment methods with no callback (PayPal.me), so
              they are queued rather than trusted — this is where a human turns
              money that actually arrived into access. */}
          {accounts && accounts.some(a => (a.pending || []).length > 0) && (
            <Section title={`Payment claims awaiting confirmation (${accCounts?.pendingClaims ?? accounts.reduce((n, a) => n + (a.pending || []).length, 0)})`}>
              <p className="text-[11px] text-terminal-muted mb-2">
                Check the payment actually landed in PayPal before confirming — a claim is only the
                customer&apos;s word for it. Confirming grants 30 days.
              </p>
              <Table head={["Claimed", "Email", "Method", "Their reference", "Amount", ""]}>
                {accounts.flatMap(a => (a.pending || []).map((c, i) => (
                  <tr key={`${a.email}-${i}`} className="border-t border-terminal-border">
                    <td className="py-1.5 pr-3 text-terminal-muted">{fmtDate(c.ts)}</td>
                    <td className="pr-3 text-slate-200">{a.email}</td>
                    <td className="pr-3 text-terminal-muted">{c.method}</td>
                    <td className="pr-3 text-slate-300">{c.note || "—"}</td>
                    <td className="pr-3 tabular-nums">{c.amountUsd ? `$${c.amountUsd}` : "—"}</td>
                    <td className="pr-3">
                      <button onClick={() => confirmClaim(a.email)} disabled={granting === a.email}
                        className="min-h-[32px] px-2 rounded bg-terminal-green text-black text-[10px] font-bold hover:opacity-90 disabled:opacity-40">
                        {granting === a.email ? "GRANTING…" : "CONFIRM & GRANT 30d"}
                      </button>
                    </td>
                  </tr>
                )))}
              </Table>
            </Section>
          )}

          <Section title={`Accounts — logins, payments & grants${accCounts ? ` (${accCounts.accounts})` : ""}`}>
            {accounts === null ? <Muted>Loading account database…</Muted> :
              accounts.length === 0 ? <Muted>No accounts yet. Every sign-in is recorded here from now on.</Muted> :
              <>
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-3">
                  <Kpi label="Accounts" value={accCounts?.accounts ?? "…"} />
                  <Kpi label="Active now" value={accCounts?.active ?? "…"} />
                  <Kpi label="Paying" value={accCounts?.paying ?? "…"} />
                  <Kpi label="Comped" value={accCounts?.comped ?? "…"} />
                  <Kpi label="Revenue" value={`$${(accCounts?.revenueUsd ?? 0).toFixed(0)}`} />
                </div>
                <Table head={["Status", "Email", "Last login", "Logins", "Access until", "Days left", "Paid", "Source"]}>
                  {accounts.map(a => (
                    <tr key={a.email} className="border-t border-terminal-border">
                      <td className="py-1.5 pr-3">
                        <span className={a.active ? "text-terminal-green font-bold" : "text-terminal-muted"}>
                          {a.active ? (a.payments > 0 ? "● PAID" : "● COMP") : "○ none"}
                        </span>
                      </td>
                      <td className="pr-3 text-slate-200">{a.email}</td>
                      <td className="pr-3 text-terminal-muted">{fmtDate(a.lastLogin)}</td>
                      <td className="pr-3 tabular-nums">{a.loginCount}</td>
                      <td className="pr-3 text-terminal-muted">{a.paidUntil ? fmtDate(a.paidUntil) : "—"}</td>
                      <td className="pr-3 tabular-nums">{a.daysLeft || "—"}</td>
                      <td className="pr-3 tabular-nums">{a.totalPaidUsd ? `$${a.totalPaidUsd.toFixed(0)}` : "—"}</td>
                      <td className="pr-3 text-terminal-muted">{a.source || "—"}</td>
                    </tr>
                  ))}
                </Table>
                <p className="text-[11px] text-terminal-muted mt-2">
                  One row per email — the single source of truth for who signed in, who paid and until when.
                  ● PAID = on-chain payment · ● COMP = manual grant. Grant access with{" "}
                  <code>POST /api/account {`{email, action:"grant", days, adminToken}`}</code>.
                </p>
              </>}
          </Section>

          <Section title="On-chain payments — who actually paid">
            {onchain === null ? <Muted>Loading chain…</Muted> :
              onchain.length === 0 ? <Muted>No on-chain transfers found (or blocked locally). <a className="text-primary underline underline-offset-2 decoration-1 decoration-current/40 hover:decoration-current" target="_blank" rel="noreferrer" href={`https://etherscan.io/address/${PAYMENT_ADDRESS}`}>View on Etherscan ↗</a></Muted> :
              <Table head={["When", "Amount", "≈USD", "From", "Tx"]}>
                {onchain.map(t => (
                  <tr key={t.hash} className="border-t border-terminal-border">
                    <td className="py-1.5 pr-3 text-terminal-muted">{fmtDate(t.when)}</td>
                    <td className="pr-3 tabular-nums">{t.amount.toFixed(t.symbol === "ETH" ? 4 : 2)} {t.symbol}</td>
                    <td className="pr-3 tabular-nums">{t.usd != null ? `$${t.usd.toFixed(0)}` : "—"}</td>
                    <td className="pr-3"><a className="text-primary underline underline-offset-2 decoration-1 decoration-current/40 hover:decoration-current" target="_blank" rel="noreferrer" href={`https://etherscan.io/address/${t.from}`}>{short(t.from)}</a></td>
                    <td><a className="text-primary underline underline-offset-2 decoration-1 decoration-current/40 hover:decoration-current" target="_blank" rel="noreferrer" href={`https://etherscan.io/tx/${t.hash}`}>{short(t.hash)}</a></td>
                  </tr>
                ))}
              </Table>}
            {onchain && onchain.length > 0 && <p className="text-[11px] text-terminal-muted mt-2">Total inbound ≈ <b className="text-slate-200">${onchainUsd.toFixed(0)}</b> (stablecoins + ETH). Amount isn&apos;t verified at grant time — small/spam transfers can appear here.</p>}
          </Section>

          <Section title="Email-linked payments">
            {payments.length === 0 ? <Muted>None yet. New payments link email↔tx automatically from now on.</Muted> :
              <Table head={["When", "Email", "Amount", "Tx"]}>
                {payments.map(p => (
                  <tr key={p.txHash} className="border-t border-terminal-border">
                    <td className="py-1.5 pr-3 text-terminal-muted">{fmtDate(p.ts)}</td>
                    <td className="pr-3 text-slate-200">{p.email}</td>
                    <td className="pr-3 tabular-nums">{p.amount ?? "—"}</td>
                    <td><a className="text-primary underline underline-offset-2 decoration-1 decoration-current/40 hover:decoration-current" target="_blank" rel="noreferrer" href={`https://etherscan.io/tx/${p.txHash}`}>{short(p.txHash)}</a></td>
                  </tr>
                ))}
              </Table>}
          </Section>

          <Section title={`Leads${leads ? ` (${leads.length})` : ""}`}>
            {loading && !leads ? <Muted>Loading…</Muted> :
              !leads || leads.length === 0 ? <Muted>No leads captured yet.</Muted> :
              <Table head={["Captured", "Email", "Source", "Paid"]}>
                {[...leads].sort((a, b) => b.ts - a.ts).map(l => (
                  <tr key={l.email} className="border-t border-terminal-border">
                    <td className="py-1.5 pr-3 text-terminal-muted">{fmtDate(l.ts)}</td>
                    <td className="pr-3 text-slate-200">{l.email}</td>
                    <td className="pr-3 text-terminal-muted">{l.source}</td>
                    <td>{l.paid ? <span className="text-terminal-green">✓</span> : <span className="text-terminal-muted">—</span>}</td>
                  </tr>
                ))}
              </Table>}
          </Section>

          <Section title="Website traffic — every visit (first-party)">
            {!traffic || traffic.views === 0 ? (
              <Muted>No page views recorded yet. Every visit to any page is tracked here automatically (anonymous, no cookies).</Muted>
            ) : (
              <div className="space-y-4">
                <div className="bg-terminal-panel border border-terminal-border rounded p-3">
                  <div className="text-[10px] uppercase tracking-wide text-terminal-muted mb-2">Views · last 14 days</div>
                  <div className="flex items-end gap-1 h-16">
                    {traffic.byDay.map(d => {
                      const max = Math.max(1, ...traffic.byDay.map(x => x.count));
                      return <div key={d.day} title={`${d.day}: ${d.count}`} className="flex-1 bg-terminal-green/70 rounded-t"
                        style={{ height: `${Math.max(3, (d.count / max) * 100)}%` }} />;
                    })}
                  </div>
                </div>
                <div className="grid sm:grid-cols-2 gap-4">
                  <div>
                    <div className="text-[10px] uppercase tracking-wide text-terminal-muted mb-2">Top pages</div>
                    <Table head={["Path", "Views"]}>
                      {traffic.byPath.map(p => (
                        <tr key={p.k} className="border-t border-terminal-border">
                          <td className="py-1.5 pr-3 text-slate-200">{p.k}</td>
                          <td className="tabular-nums">{p.v}</td>
                        </tr>
                      ))}
                    </Table>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wide text-terminal-muted mb-2">Referrers</div>
                    <Table head={["Source", "Views"]}>
                      {traffic.byRef.map(r => (
                        <tr key={r.k} className="border-t border-terminal-border">
                          <td className="py-1.5 pr-3 text-slate-200">{r.k}</td>
                          <td className="tabular-nums">{r.v}</td>
                        </tr>
                      ))}
                    </Table>
                  </div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wide text-terminal-muted mb-2">Recent visits</div>
                  <Table head={["When", "Path", "Referrer", "Visitor"]}>
                    {traffic.recent.map((v, i) => (
                      <tr key={i} className="border-t border-terminal-border">
                        <td className="py-1.5 pr-3 text-terminal-muted">{fmtDate(v.ts)}</td>
                        <td className="pr-3 text-slate-200">{v.path}</td>
                        <td className="pr-3 text-terminal-muted">{v.ref ? short(v.ref) : "direct"}</td>
                        <td className="text-terminal-muted">{short(v.vid)}</td>
                      </tr>
                    ))}
                  </Table>
                </div>
                <Muted>For richer traffic analytics (geo, devices, funnels) also set <code>NEXT_PUBLIC_PLAUSIBLE_DOMAIN</code> or <code>NEXT_PUBLIC_GA_ID</code>. Signup/Payment events already fire to that provider.</Muted>
              </div>
            )}
          </Section>
        </>
      )}
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <main className="min-h-screen bg-terminal-bg text-slate-200 font-mono px-4 sm:px-8 py-8">
    <div className="max-w-4xl mx-auto">{children}</div></main>;
}
function Kpi({ label, value }: { label: string; value: React.ReactNode }) {
  return <div className="bg-terminal-panel border border-terminal-border rounded p-3">
    <div className="text-2xl font-bold text-terminal-green tabular-nums">{value}</div>
    <div className="text-[10px] uppercase tracking-wide text-terminal-muted mt-1">{label}</div></div>;
}
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="mb-8">
    <h2 className="text-[11px] uppercase tracking-widest text-terminal-muted mb-3">{title}</h2>{children}</section>;
}
function Table({ head, children }: { head: string[]; children: React.ReactNode }) {
  return <div className="overflow-x-auto bg-terminal-panel border border-terminal-border rounded">
    <table className="w-full text-xs"><thead><tr className="text-left text-terminal-muted">
      {head.map(h => <th key={h} className="font-normal py-2 px-3 text-[10px] uppercase tracking-wide">{h}</th>)}
    </tr></thead><tbody className="px-3">{children}</tbody></table></div>;
}
function Muted({ children }: { children: React.ReactNode }) {
  return <p className="text-xs text-terminal-muted">{children}</p>;
}
