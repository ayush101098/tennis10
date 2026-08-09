import { todaysBoard } from "@/lib/archive";
import LandingClient from "@/components/LandingClient";
import type { SsrMatch } from "@/components/SsrMatchList";

/**
 * Landing route — a SERVER component now.
 *
 * It was a client component that fetched the whole board after hydration, so
 * the HTML Google received contained the words "Loading live schedule…" and no
 * matches at all: the page's entire subject matter was invisible to crawlers.
 * This reads the committed archive at build time and hands the board down as
 * real markup; the client swaps in live data as soon as it arrives.
 */
export default async function Page() {
  const board = await todaysBoard();
  const initialMatches: SsrMatch[] = board.map(m => ({
    id: m.id,
    tour: m.tour,
    tournament: m.tournament,
    player1: m.player1,
    player2: m.player2,
    p1Prob: m.p1Prob,
    p2Prob: m.p2Prob,
    startTs: m.startTs,
    round: m.round,
    surface: m.surface,
  }));
  return <LandingClient initialMatches={initialMatches} />;
}
