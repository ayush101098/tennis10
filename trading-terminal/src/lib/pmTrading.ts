/**
 * Direct Polymarket trading from the browser — no SDK, no extra deps.
 *
 * Connect flow: MetaMask (or any injected wallet) on Polygon → derive CLOB
 * API credentials with one EIP-712 signature → place signed orders straight
 * against the CLOB. Credentials are stored per signed-in terminal user so
 * each user trades (and journals) under their own account.
 *
 * Two Polymarket account shapes are supported:
 *   - EOA accounts (funds live in the wallet itself)          → sigType 0
 *   - Polymarket web profiles created with a browser wallet
 *     (funds live in the proxy shown on your PM profile)      → sigType 2,
 *     set the proxy as the funder address when connecting.
 *
 * Mirrors execution/polymarket.py on the Python side.
 */

import { CLOB_URL, type PmMarket } from "@/lib/polymarket";

const POLYGON_CHAIN_HEX = "0x89";
const POLYGON_CHAIN_ID = 137;
const CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E";
const NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a";

/* ── Connection storage (per terminal user) ──────────────────────────────── */

export interface PmConnection {
  address: string;      // signing wallet (EOA)
  funder: string;       // address holding the USDC (EOA or PM proxy)
  sigType: 0 | 2;       // 0 = EOA, 2 = Polymarket browser-wallet proxy
  apiKey: string;
  secret: string;
  passphrase: string;
  connectedAt: number;
}

const CONN_PREFIX = "tt_pm_conn_v1_";
export const PM_CHANGED_EVENT = "tt-pm-changed";

function connKey(email: string | null | undefined): string {
  return CONN_PREFIX + (email ? email.trim().toLowerCase() : "guest");
}

export function loadPmConnection(email: string | null | undefined): PmConnection | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(connKey(email));
    return raw ? (JSON.parse(raw) as PmConnection) : null;
  } catch {
    return null;
  }
}

export function disconnectPolymarket(email: string | null | undefined): void {
  localStorage.removeItem(connKey(email));
  window.dispatchEvent(new Event(PM_CHANGED_EVENT));
}

/* ── Wallet plumbing ─────────────────────────────────────────────────────── */

type Eth = { request: (args: { method: string; params?: unknown[] }) => Promise<unknown> };

export function hasWallet(): boolean {
  return typeof window !== "undefined" && !!(window as unknown as { ethereum?: Eth }).ethereum;
}

function eth(): Eth {
  const e = (window as unknown as { ethereum?: Eth }).ethereum;
  if (!e) throw new Error("No wallet found — install MetaMask (or use paper trading).");
  return e;
}

async function ensurePolygon(): Promise<void> {
  const chain = (await eth().request({ method: "eth_chainId" })) as string;
  if (chain === POLYGON_CHAIN_HEX) return;
  try {
    await eth().request({
      method: "wallet_switchEthereumChain",
      params: [{ chainId: POLYGON_CHAIN_HEX }],
    });
  } catch (err) {
    const code = (err as { code?: number }).code;
    if (code !== 4902) throw new Error("Switch your wallet to Polygon to trade on Polymarket.");
    await eth().request({
      method: "wallet_addEthereumChain",
      params: [{
        chainId: POLYGON_CHAIN_HEX,
        chainName: "Polygon",
        nativeCurrency: { name: "POL", symbol: "POL", decimals: 18 },
        rpcUrls: ["https://polygon-rpc.com"],
        blockExplorerUrls: ["https://polygonscan.com"],
      }],
    });
  }
}

async function signTypedData(address: string, typedData: object): Promise<string> {
  return (await eth().request({
    method: "eth_signTypedData_v4",
    params: [address, JSON.stringify(typedData)],
  })) as string;
}

/* ── CLOB L1 auth: one signature derives durable API credentials ─────────── */

const CLOB_AUTH_MSG = "This message attests that I control the given wallet";

async function l1Headers(address: string): Promise<Record<string, string>> {
  const timestamp = String(Math.floor(Date.now() / 1000));
  const nonce = 0;
  const signature = await signTypedData(address, {
    primaryType: "ClobAuth",
    domain: { name: "ClobAuthDomain", version: "1", chainId: POLYGON_CHAIN_ID },
    types: {
      EIP712Domain: [
        { name: "name", type: "string" },
        { name: "version", type: "string" },
        { name: "chainId", type: "uint256" },
      ],
      ClobAuth: [
        { name: "address", type: "address" },
        { name: "timestamp", type: "string" },
        { name: "nonce", type: "uint256" },
        { name: "message", type: "string" },
      ],
    },
    message: { address, timestamp, nonce, message: CLOB_AUTH_MSG },
  });
  return {
    POLY_ADDRESS: address,
    POLY_SIGNATURE: signature,
    POLY_TIMESTAMP: timestamp,
    POLY_NONCE: String(nonce),
  };
}

async function deriveOrCreateApiCreds(address: string): Promise<Pick<PmConnection, "apiKey" | "secret" | "passphrase">> {
  const headers = await l1Headers(address);
  // Existing accounts derive the same creds deterministically; new ones create.
  for (const [method, path] of [["GET", "/auth/derive-api-key"], ["POST", "/auth/api-key"]] as const) {
    const res = await fetch(CLOB_URL + path, { method, headers });
    if (!res.ok) continue;
    const json = await res.json();
    if (json.apiKey && json.secret && json.passphrase) {
      return { apiKey: json.apiKey, secret: json.secret, passphrase: json.passphrase };
    }
  }
  throw new Error("Could not obtain Polymarket API credentials for this wallet.");
}

/**
 * Connect a wallet and store per-user trading credentials.
 * `funder` — optional Polymarket profile (proxy) address for web-created
 * accounts; leave empty when the USDC sits in the wallet itself.
 */
export async function connectPolymarket(email: string | null | undefined, funder?: string): Promise<PmConnection> {
  const accounts = (await eth().request({ method: "eth_requestAccounts" })) as string[];
  const address = accounts?.[0];
  if (!address) throw new Error("Wallet returned no account.");
  await ensurePolygon();
  const creds = await deriveOrCreateApiCreds(address);
  const proxied = !!funder && funder.toLowerCase() !== address.toLowerCase();
  const conn: PmConnection = {
    address,
    funder: proxied ? funder!.trim() : address,
    sigType: proxied ? 2 : 0,
    ...creds,
    connectedAt: Date.now(),
  };
  localStorage.setItem(connKey(email), JSON.stringify(conn));
  window.dispatchEvent(new Event(PM_CHANGED_EVENT));
  return conn;
}

/* ── CLOB L2 auth: HMAC request signing with the derived secret ──────────── */

function b64urlToBytes(s: string): Uint8Array {
  const b64 = s.replace(/-/g, "+").replace(/_/g, "/");
  const bin = atob(b64 + "=".repeat((4 - (b64.length % 4)) % 4));
  return Uint8Array.from(bin, c => c.charCodeAt(0));
}

function bytesToB64url(bytes: ArrayBuffer): string {
  let bin = "";
  for (const b of new Uint8Array(bytes)) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_");
}

async function l2Headers(conn: PmConnection, method: string, path: string, body: string): Promise<Record<string, string>> {
  const timestamp = String(Math.floor(Date.now() / 1000));
  const key = await crypto.subtle.importKey(
    "raw", b64urlToBytes(conn.secret) as unknown as ArrayBuffer,
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const mac = await crypto.subtle.sign(
    "HMAC", key, new TextEncoder().encode(timestamp + method + path + body) as unknown as ArrayBuffer);
  return {
    POLY_ADDRESS: conn.address,
    POLY_SIGNATURE: bytesToB64url(mac),
    POLY_TIMESTAMP: timestamp,
    POLY_API_KEY: conn.apiKey,
    POLY_PASSPHRASE: conn.passphrase,
    "Content-Type": "application/json",
  };
}

/* ── Order build + sign + submit ─────────────────────────────────────────── */

export interface PlacedOrder {
  orderId: string;
  status: string;        // matched | live | delayed ...
  priceLimit: number;    // marketable limit price signed
  shares: number;
  costUsd: number;
}

function randomSalt(): string {
  // Unique per order; the CLOB only needs it to differ between orders
  return String(Math.floor(Math.random() * Number.MAX_SAFE_INTEGER));
}

/**
 * Buy `stakeUsd` of one outcome as a fill-and-kill marketable limit at
 * `askPrice` (best ask, padded a tick by the caller if desired): fills what
 * the book offers at or better, cancels the rest — behaves like a market buy.
 */
export async function placeBuyOrder(
  conn: PmConnection, market: PmMarket, outcomeIdx: number,
  askPrice: number, stakeUsd: number,
): Promise<PlacedOrder> {
  const tokenId = market.tokenIds[outcomeIdx];
  if (!tokenId) throw new Error("Market has no CLOB token for that outcome.");
  const tick = market.tickSize || 0.01;
  const price = Math.min(1 - tick, Math.round(askPrice / tick) * tick);
  if (!(price > 0 && price < 1)) throw new Error("No tradeable price.");

  const shares = Math.floor((stakeUsd / price) * 100) / 100;
  if (shares < 5) throw new Error(`Stake too small — Polymarket minimum is 5 shares (≈ $${(5 * price).toFixed(2)}).`);
  const takerAmount = Math.round(shares * 1e6);           // shares out
  const makerAmount = Math.round(takerAmount * price);    // USDC in

  const order = {
    salt: randomSalt(),
    maker: conn.funder,
    signer: conn.address,
    taker: "0x0000000000000000000000000000000000000000",
    tokenId,
    makerAmount: String(makerAmount),
    takerAmount: String(takerAmount),
    expiration: "0",
    nonce: "0",
    feeRateBps: "0",
    side: 0, // BUY
    signatureType: conn.sigType,
  };

  const signature = await signTypedData(conn.address, {
    primaryType: "Order",
    domain: {
      name: "Polymarket CTF Exchange",
      version: "1",
      chainId: POLYGON_CHAIN_ID,
      verifyingContract: market.negRisk ? NEG_RISK_EXCHANGE : CTF_EXCHANGE,
    },
    types: {
      EIP712Domain: [
        { name: "name", type: "string" },
        { name: "version", type: "string" },
        { name: "chainId", type: "uint256" },
        { name: "verifyingContract", type: "address" },
      ],
      Order: [
        { name: "salt", type: "uint256" },
        { name: "maker", type: "address" },
        { name: "signer", type: "address" },
        { name: "taker", type: "address" },
        { name: "tokenId", type: "uint256" },
        { name: "makerAmount", type: "uint256" },
        { name: "takerAmount", type: "uint256" },
        { name: "expiration", type: "uint256" },
        { name: "nonce", type: "uint256" },
        { name: "feeRateBps", type: "uint256" },
        { name: "side", type: "uint8" },
        { name: "signatureType", type: "uint8" },
      ],
    },
    message: order,
  });

  const body = JSON.stringify({
    order: { ...order, side: "BUY", signature },
    owner: conn.apiKey,
    orderType: "FAK",
  });
  const res = await fetch(`${CLOB_URL}/order`, {
    method: "POST",
    headers: await l2Headers(conn, "POST", "/order", body),
    body,
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok || json.success === false || json.error) {
    throw new Error(json.errorMsg || json.error || `Order rejected (HTTP ${res.status}).`);
  }
  return {
    orderId: json.orderID || json.orderId || "",
    status: json.status || "live",
    priceLimit: price,
    shares,
    costUsd: makerAmount / 1e6,
  };
}
