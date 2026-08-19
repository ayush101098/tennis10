#!/usr/bin/env bash
# Grant tennisalpha.in subscription access to an email, with no payment recorded.
#
# The admin token lives only in the Netlify site env — deliberately, so a copy of
# the repo is not a copy of the keys. Export it here rather than putting it in
# .env, and it stays out of the working tree entirely:
#
#     export LEADS_ADMIN_TOKEN='…'          # from Netlify → Site config → Env vars
#     ./scripts/grant_access.sh someone@example.com 30 "friend comp"
#
# Grants stack with payments: paidUntil is recomputed as max(payments, grants),
# so comping someone who later pays does not shorten either.

set -euo pipefail

EMAIL="${1:?usage: grant_access.sh <email> [days] [reason]}"
DAYS="${2:-30}"
REASON="${3:-manual grant}"
SITE="${TT_SITE_URL:-https://tennisalpha.in}"

if [[ -z "${LEADS_ADMIN_TOKEN:-}" ]]; then
  echo "ERROR: LEADS_ADMIN_TOKEN is not set." >&2
  echo "       Netlify → Site config → Environment variables → LEADS_ADMIN_TOKEN" >&2
  echo "       Then:  export LEADS_ADMIN_TOKEN='…'" >&2
  exit 2
fi

echo "granting ${DAYS}d to ${EMAIL} on ${SITE}"

# --fail-with-body so a 401 surfaces the reason instead of a bare exit code.
curl -sS --fail-with-body --max-time 30 \
  -X POST "${SITE}/api/account" \
  -H "Content-Type: application/json" \
  -d "$(python3 - "$EMAIL" "$DAYS" "$REASON" <<'PY'
import json, sys
# Built with json.dumps so an apostrophe in the reason cannot break the payload.
email, days, reason = sys.argv[1], int(sys.argv[2]), sys.argv[3]
print(json.dumps({
    "email": email, "action": "grant", "days": days,
    "reason": reason, "by": "operator",
    "adminToken": __import__("os").environ["LEADS_ADMIN_TOKEN"],
}))
PY
)" | python3 -c '
import json, sys, datetime
try:
    d = json.load(sys.stdin)
except Exception:
    print("unexpected response:", sys.stdin.read()[:200]); raise SystemExit(1)
if not d.get("ok"):
    print("FAILED:", d.get("reason") or d); raise SystemExit(1)
until = datetime.datetime.fromtimestamp(d["paidUntil"] / 1000)
print(f"  ok — {d[\"email\"]} has access for {d[\"days\"]}d, until {until:%Y-%m-%d %H:%M}")
'
