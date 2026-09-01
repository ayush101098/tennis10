/**
 * TennisAlpha live edge — Worker + Durable Objects (PRD §19, §20, §25).
 *
 * ROUTES
 *   GET  /match/:id            WebSocket upgrade — a viewer joins the room
 *   POST /match/:id/push       the Python runtime pushes an update (auth'd)
 *   GET  /match/:id/state      last known state, for a cold client
 *   GET  /health               rooms and viewer counts
 *
 * WHY A DURABLE OBJECT PER MATCH
 *   A DO is a single-threaded actor with an identity, which is exactly what a
 *   match room is: one authoritative state, many watchers, and updates that
 *   must not interleave. Broadcasting from a stateless Worker would need an
 *   external pub/sub and would reintroduce the ordering problem the sequence
 *   tracker exists to solve.
 *
 * WHY HIBERNATION IS NOT OPTIONAL
 *   `acceptWebSocket` (the Hibernation API) lets the runtime evict the object
 *   from memory between messages while keeping sockets open. Tennis is mostly
 *   silence — a point every ~30s, and long gaps between games — so a room held
 *   in memory bills duration for time in which nothing happens. With
 *   hibernation an idle room costs nothing and only messages are billed
 *   (incoming at 20:1, outgoing free). Using the older addEventListener style
 *   would make 20 idle rooms a standing charge.
 */

const encoder = new TextEncoder();

/** Constant-time compare so the push token cannot be probed byte by byte. */
function safeEqual(a, b) {
  const x = encoder.encode(a || "");
  const y = encoder.encode(b || "");
  if (x.length !== y.length) return false;
  let diff = 0;
  for (let i = 0; i < x.length; i++) diff |= x[i] ^ y[i];
  return diff === 0;
}

export class MatchRoom {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    // Survives hibernation; a viewer joining mid-match gets the last state
    // immediately rather than staring at an empty panel until the next point,
    // which in tennis can be 30+ seconds.
    this.last = null;
    this.state.blockConcurrencyWhile(async () => {
      this.last = (await this.state.storage.get("last")) || null;
    });
  }

  sockets() {
    return this.state.getWebSockets();
  }

  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname.endsWith("/push")) return this.handlePush(request);
    if (url.pathname.endsWith("/state")) {
      return Response.json({ state: this.last, viewers: this.sockets().length });
    }

    if (request.headers.get("Upgrade") !== "websocket") {
      return new Response("expected websocket", { status: 426 });
    }

    const cap = Number(this.env.MAX_VIEWERS_PER_ROOM || 2000);
    if (this.sockets().length >= cap) {
      // Refuse rather than degrade everyone already watching.
      return new Response("room full", { status: 503 });
    }

    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);

    // Hibernation API. NOT server.accept() — that pins the object in memory
    // for the life of the connection and bills duration through every silence.
    this.state.acceptWebSocket(server);

    if (this.last) {
      try {
        server.send(JSON.stringify(this.last));
      } catch {
        /* a socket that dies on the first write is handled by webSocketClose */
      }
    }

    return new Response(null, { status: 101, webSocket: client });
  }

  async handlePush(request) {
    if (request.method !== "POST") {
      return new Response("method not allowed", { status: 405 });
    }
    // Anyone who can push can put arbitrary prices on a trading screen, so
    // this is authenticated even though reads are public.
    const token = request.headers.get("x-push-token");
    if (!this.env.PUSH_TOKEN || !safeEqual(token, this.env.PUSH_TOKEN)) {
      return new Response("unauthorized", { status: 401 });
    }

    let payload;
    try {
      payload = await request.json();
    } catch {
      return new Response("bad json", { status: 400 });
    }

    // Drop anything older than what we already hold. Pushes can arrive out of
    // order across retries, and a late update would otherwise roll the
    // scoreboard backwards in front of every viewer.
    const incoming = Number(payload?.sequence ?? -1);
    const held = Number(this.last?.sequence ?? -1);
    if (incoming >= 0 && held >= 0 && incoming < held) {
      return Response.json({ ok: true, ignored: "stale sequence", held });
    }

    this.last = payload;
    await this.state.storage.put("last", payload);

    const body = JSON.stringify(payload);
    let sent = 0;
    for (const ws of this.sockets()) {
      try {
        ws.send(body);
        sent++;
      } catch {
        try { ws.close(1011, "send failed"); } catch { /* already gone */ }
      }
    }
    return Response.json({ ok: true, viewers: this.sockets().length, sent });
  }

  /** Hibernation callbacks — the object may have been evicted between these. */
  async webSocketMessage(ws, message) {
    // Viewers are read-only. The only accepted message is a keepalive, and
    // replying keeps intermediaries from closing an idle connection.
    if (message === "ping") {
      try { ws.send("pong"); } catch { /* closing */ }
    }
  }

  async webSocketClose(ws, code, reason, wasClean) {
    try { ws.close(code, reason); } catch { /* already closed */ }
  }

  async webSocketError(ws) {
    try { ws.close(1011, "socket error"); } catch { /* already closed */ }
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      // Deliberately cheap: it must not wake every room to answer.
      return Response.json({ ok: true, service: "tennisalpha-live" });
    }

    const m = url.pathname.match(/^\/match\/([^/]+)(\/push|\/state)?$/);
    if (!m) return new Response("not found", { status: 404 });

    const matchId = decodeURIComponent(m[1]);
    // idFromName is deterministic: the same match always routes to the same
    // object, from any colo, which is what makes the room authoritative.
    const id = env.MATCH_ROOM.idFromName(matchId);
    return env.MATCH_ROOM.get(id).fetch(request);
  },
};
