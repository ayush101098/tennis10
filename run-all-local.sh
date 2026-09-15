#!/usr/bin/env bash
# Every TennisAlpha component on this machine, with every parameter stated.
#
#   ./run-all-local.sh          start everything, then tail the board
#   ./run-all-local.sh status   what is up, what is not
#   ./run-all-local.sh stop     stop what this script started
#
# WHY EVERY PARAMETER IS WRITTEN OUT
#   Each of these has a default somewhere in the code. Spread across three
#   files, the effective configuration of a running system was not readable
#   from any one place -- so "why is the board empty" meant grepping for
#   getenv. Everything that changes behaviour is set here, explicitly, even
#   where the value equals the default.
#
# WHAT THIS DELIBERATELY DOES NOT START
#   execution.agent and execution.sx_breakbot. They place orders. They are
#   left to an explicit, separate decision rather than folded into "run all",
#   and they need keys this machine does not have.

set -uo pipefail
cd "$(dirname "$0")"
mkdir -p .run logs

# ── data feed ───────────────────────────────────────────────────────────────
export SOFA_PROXY="${SOFA_PROXY:-http://127.0.0.1:3001}"
export SOFA_LANES="${SOFA_LANES:-2}"          # parallel egress lanes
export SOFA_FALLBACK="${SOFA_FALLBACK:-1}"    # Flashscore when SofaScore 403s

# ── live board / engine ─────────────────────────────────────────────────────
export LIVE_PROVIDER="${LIVE_PROVIDER:-sofaproxy}"  # local proxy: has the server
export LIVE_HOST="${LIVE_HOST:-127.0.0.1}"          # loopback, not the LAN
export LIVE_PORT="${LIVE_PORT:-8080}"
export BOARD_POLL_S="${BOARD_POLL_S:-2}"            # scoreboard cadence
export BOARD_PRICE_S="${BOARD_PRICE_S:-10}"         # market cadence
export BOARD_MIN_EDGE="${BOARD_MIN_EDGE:-0.02}"
export BOARD_CATEGORIES="${BOARD_CATEGORIES:-all}"  # ATP+WTA+Challenger+ITF

# ── corpus ──────────────────────────────────────────────────────────────────
export POINTSTORE_DB="${POINTSTORE_DB:-$PWD/tennis_points.db}"
export POINTSTORE_MAX_GAP="${POINTSTORE_MAX_GAP:-45}"

# ── model / sizing (read-only here; nothing places an order) ────────────────
export TRADING_DRY_RUN="${TRADING_DRY_RUN:-true}"
export TRADING_BANKROLL="${TRADING_BANKROLL:-1000}"
export TRADING_CALIBRATION="${TRADING_CALIBRATION:-false}"
export TRADING_EDGESCORE_GREEN="${TRADING_EDGESCORE_GREEN:-2.0}"
export TRADING_EDGESCORE_AMBER="${TRADING_EDGESCORE_AMBER:-1.0}"

PY="$PWD/.venv/bin/python3"; [ -x "$PY" ] || PY="python3"

up()   { curl -s -o /dev/null -m 5 "$1"; }
pidof_run() { [ -f ".run/$1.pid" ] && kill -0 "$(cat ".run/$1.pid")" 2>/dev/null; }

start_bg() {  # name, command...
  local name="$1"; shift
  if pidof_run "$name"; then echo "  $name already running ($(cat .run/$name.pid))"; return; fi
  "$@" >>"logs/$name.log" 2>&1 &
  echo $! > ".run/$name.pid"
  echo "  $name started (pid $!) -> logs/$name.log"
}

case "${1:-start}" in
status)
  echo "component        where                     state"
  up "$SOFA_PROXY/sport/tennis/events/live" && s=UP || s=DOWN
  printf "  %-14s %-25s %s\n" "sofa_proxy" "$SOFA_PROXY" "$s"
  up "http://127.0.0.1:3000/" && s=UP || s=DOWN
  printf "  %-14s %-25s %s\n" "web UI" "http://localhost:3000" "$s"
  up "http://$LIVE_HOST:$LIVE_PORT/health" && s=UP || s=DOWN
  printf "  %-14s %-25s %s\n" "gateway" "http://$LIVE_HOST:$LIVE_PORT" "$s"
  pgrep -f "execution.pointstore --collect" >/dev/null && s=UP || s=DOWN
  printf "  %-14s %-25s %s\n" "pointstore" "$POINTSTORE_DB" "$s"
  launchctl list 2>/dev/null | grep -i tennisalpha | awk '{printf "  launchd %-28s pid %s\n", $3, $1}'
  exit 0 ;;
stop)
  for f in .run/*.pid; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .pid); p=$(cat "$f")
    kill "$p" 2>/dev/null && echo "  stopped $n ($p)"
    rm -f "$f"
  done
  echo "  launchd agents left running (launchctl bootout to stop those)"
  exit 0 ;;
esac

echo "TennisAlpha — starting everything locally"
echo

# 1. the feed must answer before anything downstream is worth starting
if ! up "$SOFA_PROXY/sport/tennis/events/live"; then
  echo "  sofa_proxy is not answering at $SOFA_PROXY — starting it"
  launchctl kickstart -k "gui/$(id -u)/in.tennisalpha.sofa-proxy" 2>/dev/null \
    || start_bg sofa-proxy "$PY" sofa_proxy.py
  for _ in $(seq 20); do up "$SOFA_PROXY/sport/tennis/events/live" && break; sleep 1; done
fi
up "$SOFA_PROXY/sport/tennis/events/live" \
  && echo "  feed        UP    $SOFA_PROXY" \
  || { echo "  feed        DOWN  $SOFA_PROXY  (nothing downstream will have data)"; }

# 2. corpus collector
pgrep -f "execution.pointstore --collect" >/dev/null \
  && echo "  pointstore  UP    $POINTSTORE_DB" \
  || start_bg pointstore "$PY" -m execution.pointstore --collect --source proxy --interval 8

# 3. gateway
up "http://$LIVE_HOST:$LIVE_PORT/health" \
  && echo "  gateway     UP    http://$LIVE_HOST:$LIVE_PORT" \
  || start_bg gateway "$PY" -m execution.live serve --port "$LIVE_PORT"

# 4. web UI
up "http://127.0.0.1:3000/" \
  && echo "  web UI      UP    http://localhost:3000" \
  || ( cd trading-terminal && start_bg web npx next dev --port 3000 )

echo
echo "  parameters:  poll ${BOARD_POLL_S}s · prices ${BOARD_PRICE_S}s · min edge ${BOARD_MIN_EDGE} · categories ${BOARD_CATEGORIES}"
echo "  web http://localhost:3000    gateway http://$LIVE_HOST:$LIVE_PORT    ./run-all-local.sh status|stop"
echo
exec "$PY" -m execution.live board
