#!/bin/bash
#
# Install LaunchAgents for the production data feeds.
#
# Without these the site's match board is empty — the tennis and table-tennis
# data both come from this machine. Previously they were started by hand and
# died on reboot.
#
#   bash deploy/launchd/install.sh              install + start
#   bash deploy/launchd/install.sh --uninstall  stop + remove
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
LOGS="$HOME/Library/Logs/tennisalpha"
PREFIX="in.tennisalpha"

PYTHON="$(command -v python3)"
[ -x "$PYTHON" ] || { echo "python3 not found on PATH"; exit 1; }

# label|working dir|args…
JOBS=(
  "sofa-proxy|$REPO|$PYTHON sofa_proxy.py"
  "push-sofa|$REPO|$PYTHON push_sofa.py"
  "tt-live|$REPO|$PYTHON -m tabletennis.live"
  "tt-push|$REPO|$PYTHON -m tabletennis.push"
  "tt-refresh|$REPO|$PYTHON -m tabletennis.refresh"
  # Not a daemon: a periodic job (see INTERVALS below). Keeps the match archive
  # — the source for the server-rendered board — from going stale.
  "archive|$REPO|/bin/bash $REPO/deploy/launchd/refresh-archive.sh"
)

# Jobs that run on a timer instead of staying resident, in seconds. A case
# rather than an associative array: macOS still ships bash 3.2, which has none.
interval_for() {
  case "$1" in
    archive) echo 10800 ;;    # every 3h
    *)       echo "" ;;
  esac
}

uninstall() {
  for job in "${JOBS[@]}"; do
    name="${job%%|*}"
    label="$PREFIX.$name"
    plist="$AGENTS/$label.plist"
    launchctl unload "$plist" 2>/dev/null || true
    rm -f "$plist"
    echo "removed $label"
  done
  echo "done — nothing is feeding the site now."
}

if [ "${1:-}" = "--uninstall" ]; then uninstall; exit 0; fi

mkdir -p "$AGENTS" "$LOGS"

for job in "${JOBS[@]}"; do
  IFS='|' read -r name workdir cmd <<< "$job"
  label="$PREFIX.$name"
  plist="$AGENTS/$label.plist"

  # ProgramArguments, one <string> per token
  args=""
  for tok in $cmd; do args="$args
    <string>$tok</string>"; done

  # A resident daemon is kept alive; a periodic job runs on an interval and is
  # expected to exit. KeepAlive on a job that exits would spin it forever.
  every="$(interval_for "$name")"
  if [ -n "$every" ]; then
    SCHEDULE="  <key>StartInterval</key><integer>$every</integer>"
  else
    SCHEDULE="  <key>KeepAlive</key><true/>"
  fi

  cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array>$args
  </array>
  <key>WorkingDirectory</key><string>$workdir</string>
  <!-- come back after a reboot, and after a crash -->
  <key>RunAtLoad</key><true/>
$SCHEDULE
  <!-- do not hammer on a tight crash loop -->
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$LOGS/$name.log</string>
  <key>StandardErrorPath</key><string>$LOGS/$name.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    <key>PYTHONUNBUFFERED</key><string>1</string>
  </dict>
</dict>
</plist>
PLIST

  launchctl unload "$plist" 2>/dev/null || true
  launchctl load "$plist"
  echo "installed $label"
done

echo
echo "logs:   $LOGS"
echo "status: launchctl list | grep $PREFIX"
