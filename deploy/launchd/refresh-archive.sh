#!/bin/bash
#
# Refresh the match archive and publish it.
#
# The archive is what the server-rendered board is built from. A static export
# renders whatever exists at build time, so if nothing refreshes this the
# homepage ships an EMPTY board to crawlers — silently, because the client
# still fills it in for humans. It went two days stale before this existed.
#
# Runs under launchd beside the feed daemons. Commits only data/matches, so it
# can never pick up unrelated work in progress, and pushes only when something
# actually changed (each push triggers a Netlify build).
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/trading-terminal" || exit 1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "archiving today's matches"
if ! npx tsx scripts/archive-matches.ts; then
  log "archive failed — leaving the previous data in place"
  exit 1
fi

cd "$REPO" || exit 1

# Only the archive. Never `git add -A`: this runs unattended and must not
# publish whatever else happens to be in the working tree.
git add trading-terminal/data/matches || exit 1

if git diff --cached --quiet -- trading-terminal/data/matches; then
  log "no change — nothing to publish"
  git reset --quiet -- trading-terminal/data/matches
  exit 0
fi

DAYS=$(git diff --cached --name-only -- trading-terminal/data/matches | wc -l | tr -d ' ')
git commit -q -m "chore: refresh match archive ($DAYS day file(s))

Automated by deploy/launchd/refresh-archive.sh. The archive is the source for
the server-rendered board; without a refresh the static build ships an empty
board to crawlers." || { log "commit failed"; exit 1; }

# Rebase on anything pushed meanwhile so an unattended run never creates a
# merge commit or clobbers work. --autostash because the repo routinely has
# unrelated modified files (the pipeline writes to the databases and CSVs);
# without it the rebase refuses and the archive silently stops publishing.
if ! git pull --rebase --autostash --quiet origin main; then
  log "rebase failed — leaving the commit local for a human to sort out"
  exit 1
fi

if git push --quiet origin main; then
  log "published — Netlify will rebuild"
else
  log "push failed — commit is local, will retry next run"
  exit 1
fi
