#!/usr/bin/env bash
# Start TennisAlpha on this machine, for personal use.
#
#   ./run-local.sh          web UI  -> http://localhost:3000
#   ./run-local.sh board    dense terminal board (EdgeScore, market reaction)
#
# Nothing here talks to a host you pay for. The data comes from sofa_proxy,
# which is already running as a launchd agent on this Mac.

set -euo pipefail
cd "$(dirname "$0")"

PROXY="${SOFA_PROXY:-http://127.0.0.1:3001}"

# The single most common reason for an empty board, so it is checked first and
# named explicitly rather than left to look like "no matches today".
if ! curl -s -o /dev/null -m 5 "$PROXY/sport/tennis/events/live"; then
  echo "sofa_proxy is not answering at $PROXY" >&2
  echo >&2
  echo "  launchctl kickstart -k gui/\$(id -u)/in.tennisalpha.sofa-proxy" >&2
  echo "  # or, in a spare terminal:  python sofa_proxy.py" >&2
  exit 1
fi

if [ "${1:-web}" = "board" ]; then
  exec python -m execution.live board
fi

# SOFA_PROXY_URL is what makes the app read the LOCAL proxy instead of the
# cloud blob cache. Without it the page loads but serves stale data and looks
# like the feed is broken.
cd trading-terminal
echo "web UI  -> http://localhost:3000     (data: $PROXY)"
SOFA_PROXY_URL="$PROXY" exec npx next dev --port 3000
