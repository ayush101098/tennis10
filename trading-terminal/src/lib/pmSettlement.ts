/**
 * Polymarket settlement helpers — balance, resolution, redemption.
 *
 *  - readUsdcBalance:  live USDC (Polygon) balance of a funder, via the wallet RPC.
 *  - fetchResolution:  did a market resolve, and which token won (CLOB is
 *                      authoritative — it flags the winning token directly).
 *  - redeemWinnings:   redeem a resolved position's USDC through the Conditional
 *                      Tokens contract (EOA accounts only; Polymarket proxy
 *                      accounts auto-redeem, and neg-risk markets must be
 *                      redeemed on Polymarket).
 *
 * All reads use the injected wallet's RPC (no external RPC host, no extra deps).
 */

import { CLOB_URL } from "@/lib/polymarket";
import type { PmConnection } from "@/lib/pmTrading";

// USDC.e on Polygon PoS — Polymarket's collateral (6 decimals).
export const USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174";
// Gnosis Conditional Tokens Framework on Polygon.
const CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045";
const ZERO_BYTES32 = "0x" + "0".repeat(64);
const POLYGON_CHAIN_HEX = "0x89";

type Eth = { request: (a: { method: string; params?: unknown[] }) => Promise<unknown> };
function eth(): Eth | null {
  if (typeof window === "undefined") return null;
  return (window as unknown as { ethereum?: Eth }).ethereum || null;
}

function pad32(hexNo0x: string): string {
  return hexNo0x.toLowerCase().replace(/^0x/, "").padStart(64, "0");
}

/** Funder USDC balance in whole USDC (e.g. 42.5), or null if unavailable. */
export async function readUsdcBalance(address: string): Promise<number | null> {
  const e = eth();
  if (!e || !address) return null;
  try {
    const data = "0x70a08231" + pad32(address); // balanceOf(address)
    const res = (await e.request({
      method: "eth_call",
      params: [{ to: USDC_ADDRESS, data }, "latest"],
    })) as string;
    if (!res || res === "0x") return null;
    return Number(BigInt(res)) / 1e6;
  } catch {
    return null;
  }
}

export interface Resolution {
  closed: boolean;
  winnerTokenId: string | null; // CLOB token id of the winning outcome
}

/**
 * Market resolution from the CLOB — the token carrying `winner: true` is the
 * outcome that paid out. Returns null on any lookup failure (caller keeps the
 * bet open and retries next poll).
 */
export async function fetchResolution(conditionId: string): Promise<Resolution | null> {
  if (!conditionId) return null;
  try {
    const res = await fetch(`${CLOB_URL}/markets/${conditionId}`);
    if (!res.ok) return null;
    const m = await res.json();
    const tokens: { token_id?: string | number; winner?: boolean }[] = m.tokens || [];
    const win = tokens.find(t => t.winner);
    return {
      closed: !!m.closed,
      winnerTokenId: win?.token_id != null ? String(win.token_id) : null,
    };
  } catch {
    return null;
  }
}

/** True when this position can be redeemed straight from the wallet here. */
export function canRedeem(conn: PmConnection | null, negRisk?: boolean): boolean {
  // Proxy accounts (sigType 2) auto-redeem on Polymarket; neg-risk markets use a
  // different adapter we don't drive here — send users to Polymarket for those.
  return !!conn && conn.sigType === 0 && !negRisk;
}

/**
 * Redeem a resolved binary position's USDC via ConditionalTokens.redeemPositions
 * (collateral=USDC, parentCollectionId=0x0, indexSets=[1,2] covers both slots).
 * Returns the tx hash. Throws with a readable message on failure.
 */
export async function redeemWinnings(conn: PmConnection, conditionId: string): Promise<string> {
  const e = eth();
  if (!e) throw new Error("No wallet found.");
  if (conn.sigType !== 0) {
    throw new Error("This is a Polymarket proxy account — winnings auto-redeem to your profile; check Polymarket.");
  }
  if (!/^0x[0-9a-fA-F]{64}$/.test(conditionId)) {
    throw new Error("Missing market id for this bet — redeem it on Polymarket.");
  }
  const chain = (await e.request({ method: "eth_chainId" })) as string;
  if (chain !== POLYGON_CHAIN_HEX) {
    throw new Error("Switch your wallet to Polygon to redeem.");
  }

  // redeemPositions(address collateral, bytes32 parentCollectionId, bytes32 conditionId, uint256[] indexSets)
  const selector = "0x01b7037c";
  const offset = pad32((4 * 32).toString(16)); // dynamic array starts after 4 head words
  const arrLen = pad32((2).toString(16));
  const idx1 = pad32((1).toString(16));
  const idx2 = pad32((2).toString(16));
  const data =
    selector +
    pad32(USDC_ADDRESS) +
    pad32(ZERO_BYTES32) +
    pad32(conditionId) +
    offset +
    arrLen + idx1 + idx2;

  const txHash = (await e.request({
    method: "eth_sendTransaction",
    params: [{ from: conn.address, to: CTF_ADDRESS, data }],
  })) as string;
  return txHash;
}
