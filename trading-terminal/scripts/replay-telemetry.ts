/** Replay real recorded score polls through the telemetry engine. */
import { readFileSync } from "node:fs";
import { observeTelemetry, momentum, workload, resetTelemetry } from "../src/lib/liveTelemetry";

const data: Record<string, number[][]> = JSON.parse(readFileSync(process.argv[2], "utf8"));
for (const [mid, seq] of Object.entries(data)) {
  resetTelemetry();
  let t: ReturnType<typeof observeTelemetry> | null = null;
  for (const [a, b, g1, g2, si] of seq) {
    t = observeTelemetry({ matchId: mid, p1Pts: a, p2Pts: b, gamesP1: g1, gamesP2: g2, setIndex: si });
  }
  const m = momentum(t!), w = workload(t!);
  console.log(`\n${mid}  (${seq.length} polls)`);
  console.log(`  points derived : ${t!.observedPoints}   P1 ${t!.p1Points} / P2 ${t!.p2Points}`);
  console.log(`  games seen     : ${t!.gamesPlayed}   deuce ${t!.deuceGames}   tiebreaks ${t!.tiebreaks}`);
  console.log(`  unattributable : ${t!.gaps} gaps`);
  console.log(`  momentum       : ${m ? `P1 ${(m.p1*100).toFixed(1)}%  P2 ${(m.p2*100).toFixed(1)}%  (recent share ${(m.fastP1*100).toFixed(0)}%)` : "withheld — too few points"}`);
  console.log(`  workload       : ${w ? `${w.points} pts, ${w.deuceGames} deuce, load ${(w.load*100).toFixed(0)}%${w.incomplete ? " [incomplete]" : ""}` : "n/a"}`);
}
