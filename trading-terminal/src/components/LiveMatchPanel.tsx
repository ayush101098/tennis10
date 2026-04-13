"use client";

import { useEffect, useState, useCallback } from "react";
import { fetchLiveMatches, attachLiveFeed, detachLiveFeed, fetchFeedStatus } from "@/lib/api";
import type { LiveMatch } from "@/lib/api";

interface Props {
  activeMatchId: string | null;
  onAutoSetup?: (player1: string, player2: string, tournament: string) => void;
}

export default function LiveMatchPanel({ activeMatchId, onAutoSetup }: Props) {
  const [matches, setMatches] = useState<LiveMatch[]>([]);
  const [sources, setSources] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [feedActive, setFeedActive] = useState(false);
  const [feedSource, setFeedSource] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchLiveMatches();
      setMatches(data.matches || []);
      setSources(data.sources || []);
    } catch (e) {
      setError("Failed to fetch live matches");
    } finally {
      setLoading(false);
    }
  }, []);

  // Refresh on mount and every 30 seconds
  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 30000);
    return () => clearInterval(interval);
  }, [refresh]);

  // Check feed status for active match
  useEffect(() => {
    if (!activeMatchId) return;
    let cancelled = false;
    const check = async () => {
      try {
        const status = await fetchFeedStatus(activeMatchId);
        if (!cancelled) {
          setFeedActive(status.feed_active);
          setFeedSource(status.current_match?.source || null);
        }
      } catch {
        // ignore
      }
    };
    check();
    const interval = setInterval(check, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [activeMatchId]);

  const handleAttach = async (match: LiveMatch) => {
    if (!activeMatchId) return;
    setError(null);
    const result = await attachLiveFeed(activeMatchId, match.player1, match.player2);
    if (result.error) {
      setError(result.error);
    } else {
      setFeedActive(true);
      setFeedSource(result.source || match.source);
    }
  };

  const handleDetach = async () => {
    if (!activeMatchId) return;
    await detachLiveFeed(activeMatchId);
    setFeedActive(false);
    setFeedSource(null);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border">
        <span className="text-[10px] font-bold text-terminal-muted uppercase tracking-wider">
          📡 LIVE MATCHES
        </span>
        <div className="flex items-center gap-2">
          {feedActive && (
            <span className="flex items-center gap-1 text-[9px] text-terminal-green">
              <span className="w-1.5 h-1.5 rounded-full bg-terminal-green live-dot" />
              FEED: {feedSource}
            </span>
          )}
          <button
            onClick={refresh}
            disabled={loading}
            className="text-[10px] bg-terminal-blue/20 text-terminal-blue border border-terminal-blue/40 rounded px-2 py-0.5 hover:bg-terminal-blue/30 disabled:opacity-50"
          >
            {loading ? "..." : "↻"}
          </button>
        </div>
      </div>

      {/* Feed control for active match */}
      {activeMatchId && feedActive && (
        <div className="px-3 py-1.5 bg-terminal-green/5 border-b border-terminal-border flex items-center justify-between">
          <span className="text-[10px] text-terminal-green">
            Auto-scoring active via {feedSource}
          </span>
          <button
            onClick={handleDetach}
            className="text-[9px] bg-terminal-red/20 text-terminal-red border border-terminal-red/40 rounded px-2 py-0.5 hover:bg-terminal-red/30"
          >
            DETACH
          </button>
        </div>
      )}

      {error && (
        <div className="px-3 py-1.5 bg-terminal-red/10 text-terminal-red text-[10px]">
          {error}
        </div>
      )}

      {/* Sources indicator */}
      <div className="px-3 py-1 border-b border-terminal-border">
        <span className="text-[9px] text-terminal-muted">
          Sources: {sources.join(", ") || "none"}
        </span>
      </div>

      {/* Match list */}
      <div className="flex-1 overflow-y-auto">
        {matches.map((m) => (
          <div
            key={`${m.source}-${m.id}`}
            className="px-3 py-2 border-b border-terminal-border hover:bg-terminal-panel/80 transition"
          >
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-terminal-green live-dot" />
                  <span className="text-xs text-slate-200 truncate font-medium">
                    {m.player1} vs {m.player2}
                  </span>
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-[9px] text-terminal-muted truncate">
                    {m.tournament}
                    {m.round ? ` · ${m.round}` : ""}
                  </span>
                  <TourBadge tour={m.tour} />
                  <span className="text-[8px] text-terminal-blue bg-terminal-blue/10 px-1 rounded">
                    {m.source}
                  </span>
                </div>
                {m.score && (
                  <div className="text-[10px] text-terminal-cyan mt-0.5">
                    Sets: {m.score.sets_p1}-{m.score.sets_p2} | Games: {m.score.games_p1}-{m.score.games_p2}
                    {m.score.point_text ? ` | ${m.score.point_text}` : ""}
                  </div>
                )}
              </div>
              <div className="flex flex-col gap-1 ml-2">
                {activeMatchId && !feedActive && (
                  <button
                    onClick={() => handleAttach(m)}
                    className="text-[9px] bg-terminal-green/20 text-terminal-green border border-terminal-green/40 rounded px-2 py-0.5 hover:bg-terminal-green/30 whitespace-nowrap"
                  >
                    ATTACH
                  </button>
                )}
                {onAutoSetup && (
                  <button
                    onClick={() => onAutoSetup(m.player1, m.player2, m.tournament)}
                    className="text-[9px] bg-terminal-cyan/20 text-terminal-cyan border border-terminal-cyan/40 rounded px-2 py-0.5 hover:bg-terminal-cyan/30 whitespace-nowrap"
                  >
                    SETUP
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
        {!loading && matches.length === 0 && (
          <div className="text-terminal-muted text-[10px] text-center py-6">
            No live tennis matches found
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Tour Badge ── */

const TOUR_COLORS: Record<string, string> = {
  ATP: "text-blue-400 bg-blue-400/10",
  WTA: "text-pink-400 bg-pink-400/10",
  "ITF-M": "text-orange-400 bg-orange-400/10",
  "ITF-W": "text-orange-300 bg-orange-300/10",
  Challenger: "text-yellow-400 bg-yellow-400/10",
};

function TourBadge({ tour }: { tour: string }) {
  const color = TOUR_COLORS[tour] || "text-terminal-muted bg-terminal-muted/10";
  return (
    <span className={`text-[8px] font-bold px-1 rounded ${color}`}>
      {tour}
    </span>
  );
}
