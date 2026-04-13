"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { TradeBoxFrame } from "@/lib/types";

const WS_BASE =
  (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8888").replace("http", "ws");

export function useTradeStream(matchId: string | null) {
  const [frame, setFrame] = useState<TradeBoxFrame | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const pingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const connect = useCallback(() => {
    if (!matchId) return;
    const ws = new WebSocket(`${WS_BASE}/ws/${matchId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      // keep-alive
      pingRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send("ping");
      }, 15_000);
    };

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === "pong") return;
        setFrame(data as TradeBoxFrame);
      } catch {
        /* ignore */
      }
    };

    ws.onclose = () => {
      setConnected(false);
      if (pingRef.current) clearInterval(pingRef.current);
      // auto-reconnect after 2s
      setTimeout(() => connect(), 2000);
    };

    ws.onerror = () => ws.close();
  }, [matchId]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      if (pingRef.current) clearInterval(pingRef.current);
    };
  }, [connect]);

  return { frame, connected };
}
