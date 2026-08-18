"""Tennis trading agent — a toggle you flip on/off.

When ON it: (1) generates fresh model signals for the open Polymarket fixtures,
(2) paper-trades match + set markets that clear the Kelly/edge gate, and
(3) auto-hedges (full-lock) any position whose price moves against it.

    python -m execution.agent on            # generate signals + start the loop (background)
    python -m execution.agent on --fg       # same, but run in the foreground
    python -m execution.agent status        # is it running? + journal summary
    python -m execution.agent off           # stop it
    python -m execution.agent regen         # just refresh signals.auto.json

Everything is paper unless the three live guards are open (see EXECUTION_PIPELINE.md):
--live flag, TRADING_DRY_RUN=false, and POLYMARKET_PRIVATE_KEY set. This agent
never passes --live, so it stays in paper/dry-run mode.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PID_FILE = REPO_ROOT / "execution" / ".agent.pid"
LOG_FILE = REPO_ROOT / "execution" / "agent.log"
SIGNALS = REPO_ROOT / "signals.auto.json"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _running_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError:
        return None
    if _pid_alive(pid):
        return pid
    PID_FILE.unlink(missing_ok=True)
    return None


def regen(min_edge: float = 0.0) -> int:
    from execution.signals_gen import generate
    signals = generate(min_edge=min_edge)
    SIGNALS.write_text(json.dumps(signals, indent=2))
    print(f"Wrote {len(signals)} signals -> {SIGNALS.name}")
    return len(signals)


def on(interval: int, foreground: bool) -> None:
    if _running_pid():
        print(f"Agent already ON (pid {_running_pid()}). Use 'off' first to restart.")
        return
    print("Agent turning ON — generating fresh signals...")
    n = regen()
    if n == 0:
        print("No signals produced (no open fixtures the model can price). Not starting loop.")
        return

    cmd = [sys.executable, "-m", "execution.pipeline",
           "--signals", str(SIGNALS), "--watch", str(interval)]
    if foreground:
        print(f"Running watch+hedge loop in foreground (every {interval}s). Ctrl-C to stop.\n")
        os.execv(sys.executable, cmd)  # replaces this process
        return

    log = open(LOG_FILE, "a")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=str(REPO_ROOT), start_new_session=True)
    PID_FILE.write_text(str(proc.pid))
    time.sleep(2)  # let it boot / surface immediate errors
    if _pid_alive(proc.pid):
        print(f"Agent ON (pid {proc.pid}, every {interval}s). Log: {LOG_FILE}")
        print("  live view:  tail -f execution/agent.log")
        print("  journal:    python -m execution.pipeline --log")
        print("  stop:       python -m execution.agent off")
    else:
        print(f"Agent failed to start — check {LOG_FILE}")


def off() -> None:
    pid = _running_pid()
    if not pid:
        print("Agent is already OFF.")
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    PID_FILE.unlink(missing_ok=True)
    print(f"Agent OFF (stopped pid {pid}).")


def status() -> None:
    from execution import trade_log
    pid = _running_pid()
    print(f"Agent: {'ON (pid ' + str(pid) + ')' if pid else 'OFF'}")
    if SIGNALS.exists():
        try:
            n = len(json.loads(SIGNALS.read_text()))
            print(f"Signals loaded: {n} ({SIGNALS.name})")
        except json.JSONDecodeError:
            pass
    s = trade_log.summary()
    print(f"Journal: {s['trades']} bets | open stake ${s['open_stake_usd']:.2f} | "
          f"settled PnL ${s['settled_pnl_usd']:+.2f} ({s['wins']}W-{s['losses']}L)")


def settle() -> None:
    from execution.settle import settle_open
    settle_open()


def report() -> None:
    from execution.report import build_report
    print(build_report())


def cancel_open() -> None:
    from execution import trade_log
    n = trade_log.cancel_all_open()
    print(f"Cancelled {n} open bet(s).")


def calibration() -> None:
    from execution.calibration import build_report
    print(build_report())


def calibrate() -> None:
    from execution.calibrate import fit_and_save
    fit_and_save()


def main() -> None:
    ap = argparse.ArgumentParser(description="Tennis paper-trading agent toggle")
    ap.add_argument("command",
                    choices=["on", "off", "status", "regen", "settle", "report",
                             "cancel-open", "calibration", "calibrate"])
    ap.add_argument("--interval", type=int, default=45,
                    help="watch loop poll interval seconds (default 45)")
    ap.add_argument("--fg", action="store_true", help="run the loop in the foreground")
    args = ap.parse_args()

    if args.command == "on":
        on(args.interval, args.fg)
    elif args.command == "off":
        off()
    elif args.command == "status":
        status()
    elif args.command == "regen":
        regen()
    elif args.command == "settle":
        settle()
    elif args.command == "report":
        report()
    elif args.command == "cancel-open":
        cancel_open()
    elif args.command == "calibration":
        calibration()
    elif args.command == "calibrate":
        calibrate()


if __name__ == "__main__":
    main()
