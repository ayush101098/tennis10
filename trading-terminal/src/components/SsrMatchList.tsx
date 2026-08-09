/**
 * The match board as it exists in the exported HTML.
 *
 * A server component: no hooks, no fetching. It exists so a crawler (and a
 * reader on a slow connection) gets the actual fixtures and model
 * probabilities instead of a loading spinner. The live client board replaces
 * it the moment real data arrives.
 */

export interface SsrMatch {
  id: string;
  tour: string;
  tournament: string;
  player1: string;
  player2: string;
  p1Prob: number;
  p2Prob: number;
  startTs: number;
  round: string;
  surface: string;
}

const pct = (p: number) => `${Math.round(p * 100)}%`;

export default function SsrMatchList({ matches }: { matches: SsrMatch[] }) {
  return (
    <div>
      {matches.map(m => (
        <div key={m.id}
          className="flex items-center gap-2 px-3 py-2 border-b border-terminal-border/60 text-[11px]">
          <span className="shrink-0 w-[52px] text-[9px] text-terminal-muted">{m.tour}</span>
          <span className="flex-1 min-w-0">
            <span className="block truncate text-slate-200">{m.player1}</span>
            <span className="block truncate text-slate-200">{m.player2}</span>
          </span>
          <span className="shrink-0 w-[44px] text-right font-mono text-[10px]">
            <span className={`block ${m.p1Prob >= 0.5 ? "text-terminal-green font-bold" : "text-slate-500"}`}>{pct(m.p1Prob)}</span>
            <span className={`block ${m.p1Prob < 0.5 ? "text-terminal-green font-bold" : "text-slate-500"}`}>{pct(m.p2Prob)}</span>
          </span>
          <span className="shrink-0 w-[96px] text-[9px] text-terminal-muted truncate" title={m.tournament}>
            {m.tournament}
          </span>
        </div>
      ))}
    </div>
  );
}
