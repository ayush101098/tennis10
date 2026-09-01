# Live edge — Cloudflare Worker + Durable Objects

Fan-out layer for the live engine. One Durable Object per match acts as the
room; the Python runtime pushes updates in, the DO broadcasts to every viewer.
The upstream provider is never touched from here.

```
runtime ──POST /match/:id/push──▶ MatchRoom (DO) ──WebSocket──▶ viewers
```

## Deploy

```bash
cd edge
npx wrangler secret put PUSH_TOKEN     # same value as EDGE_PUSH_TOKEN below
npx wrangler deploy
```

Then point the engine at it:

```bash
export EDGE_BASE_URL=https://tennisalpha-live.<subdomain>.workers.dev
export EDGE_PUSH_TOKEN=<the secret you just set>
python -m execution.live serve
```

## Routes

| Route | Purpose |
|---|---|
| `GET /match/:id` | WebSocket upgrade — a viewer joins the room |
| `POST /match/:id/push` | Runtime pushes an update (requires `x-push-token`) |
| `GET /match/:id/state` | Last known state, for a cold client |
| `GET /health` | Liveness; deliberately does not wake any room |

## Why it is written this way

**Hibernation is not optional.** The room uses `state.acceptWebSocket()` (the
Hibernation API) rather than `server.accept()`. Tennis is mostly silence — a
point every ~30 s, longer between games — and a room pinned in memory bills
duration through all of it. With hibernation an idle room costs nothing;
incoming messages bill at 20:1 and outgoing are free.

**Stale pushes are dropped.** Pushes can arrive out of order across retries. A
late one would roll the scoreboard backwards in front of every viewer, so a
payload whose `sequence` is behind the one already held is ignored.

**Last state is persisted.** A viewer joining mid-match gets the current state
immediately instead of staring at an empty panel until the next point.

**Pushes are authenticated, reads are not.** Anyone who can push can put
arbitrary prices on a trading screen. The token is compared in constant time so
it cannot be probed byte by byte.

## Status

Written and syntax-checked; **not deployed** — that needs a Cloudflare account.
`RoomRegistry` in the Python gateway is the same fan-out logic and works today
without any of this, which is what the tests exercise.
