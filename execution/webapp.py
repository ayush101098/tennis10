"""Local dashboard for the tennis trading agent — localhost only.

    python -m execution.webapp            # serves http://127.0.0.1:8899
    python -m execution.webapp --port 9000

Shows the live journal (trades the agent takes, plus the full log), the agent
on/off state, and lets you SELL (liquidate at the live bid) or CANCEL any open
position. Bound to 127.0.0.1 — not reachable from other machines.
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

from execution.pipeline import load_env  # noqa: E402
from execution import trade_log, agent as agent_mod  # noqa: E402
from execution.polymarket import PolymarketClient  # noqa: E402
from execution.sxbet import SXBetClient  # noqa: E402

_client = PolymarketClient()
_sx = SXBetClient()

OPEN_STATUSES = ("dry_run", "placed")

# ── background price cache (mark-to-market open positions) ────────────────────
# The CLOB is polled off the request path so /api/state stays instant. Each open
# token's best bid (what you could sell into) is refreshed every ~20s.
_price_cache: dict[str, float] = {}   # token_id -> best bid
_price_lock = threading.Lock()
_PRICE_TTL = 20  # seconds between refreshes


def _refresh_prices_once() -> None:
    tokens = {r["token_id"] for r in trade_log.unsettled_trades() if r.get("token_id")}
    if not tokens:
        return
    client = PolymarketClient()  # own session for this thread

    def _bid(tok):
        return tok, client.best_bid(tok)

    fresh = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for tok, bid in ex.map(_bid, tokens):
            if bid is not None and 0.0 < bid < 1.0:
                fresh[tok] = bid
    with _price_lock:
        _price_cache.clear()
        _price_cache.update(fresh)


def _price_loop() -> None:
    while True:
        try:
            _refresh_prices_once()
        except Exception:
            pass
        time.sleep(_PRICE_TTL)


# ── background intelligence table (all live fixtures) ────────────────────────
_intel_cache: dict = {"ts": 0.0, "rows": []}
_intel_lock = threading.Lock()
_INTEL_TTL = 30


def _intel_loop() -> None:
    from execution.intel import compute_intel
    while True:
        try:
            rows = compute_intel()
            with _intel_lock:
                _intel_cache["rows"] = rows
                _intel_cache["ts"] = time.time()
        except Exception:
            pass
        time.sleep(_INTEL_TTL)


# ── background LIVE BOARD (win prob + break watch for every live match) ───────
_live_cache: dict = {"ts": 0.0, "board": []}
_live_lock = threading.Lock()
_LIVE_TTL = 20


def _live_loop() -> None:
    from execution.live_odds import SofascoreOdds
    from execution.inplay import InPlayModel
    from execution import sx_breakbot
    sofa = SofascoreOdds()
    ip = InPlayModel(sofa=sofa)
    while True:
        try:
            board = sx_breakbot.scan_board(sofa, ip)
            with _live_lock:
                _live_cache["board"] = board
                _live_cache["ts"] = time.time()
        except Exception:
            pass
        time.sleep(_LIVE_TTL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_price_loop, daemon=True).start()
    threading.Thread(target=_live_loop, daemon=True).start()
    yield


app = FastAPI(title="Tennis Agent — Local", lifespan=lifespan)


def _is_hedge(r: dict) -> bool:
    return (r.get("detail") or "").startswith("HEDGE") or r.get("side") == "hedge"


@app.get("/api/state")
def state():
    load_env()
    rows = trade_log.all_trades()
    with _price_lock:
        prices = dict(_price_cache)
    trades = []
    unrealized = 0.0
    for r in reversed(rows):  # newest first
        is_open = r["status"] in OPEN_STATUSES
        cur = upnl = None
        if is_open:
            bid = prices.get(r["token_id"])
            if bid is not None:
                cur = bid
                upnl = round(float(r["shares"] or 0) * bid - float(r["stake_usd"] or 0), 2)
                unrealized += upnl
        trades.append({
            "id": r["trade_id"], "ts": (r["ts_utc"] or "")[:19],
            "match": r["match_name"], "market": r["market_type"],
            "outcome": r["outcome"], "side": r["side"],
            "true_p": r["true_p"], "price": r["market_price"], "edge": r["edge"],
            "kelly": r["kelly_frac"], "stake": r["stake_usd"], "shares": r["shares"],
            "status": r["status"], "pnl": r["pnl_usd"], "detail": r["detail"],
            "token_id": r["token_id"], "open": is_open, "hedge": _is_hedge(r),
            "current": cur, "upnl": upnl,
        })
    pid = agent_mod._running_pid()
    return {
        "agent": {"on": bool(pid), "pid": pid},
        "summary": trade_log.summary(),
        "unrealized_pnl": round(unrealized, 2),
        "priced": len(prices),
        "trades": trades,
    }


@app.get("/api/intel")
def intel():
    with _intel_lock:
        return {"ts": _intel_cache["ts"], "rows": _intel_cache["rows"]}


@app.get("/api/live")
def live_board():
    with _live_lock:
        return {"ts": _live_cache["ts"], "board": _live_cache["board"]}


@app.post("/api/sxbet/place")
def sxbet_place(payload: dict):
    load_env()
    p1, p2 = payload.get("player1"), payload.get("player2")
    target = payload.get("target")
    stake = float(payload.get("stake") or 25)
    if not (p1 and p2 and target):
        raise HTTPException(status_code=400, detail="player1, player2, target required")
    dry = (os.getenv("TRADING_DRY_RUN", "true").lower() != "false") or not _sx.can_trade_live
    res = _sx.place_bet(p1, p2, target, stake, dry_run=dry)
    q = _sx.quote(p1, p2)
    implied = None
    if q is not None:
        implied = q["fair_p1"] if _norm_eq(target, p1) else round(1 - q["fair_p1"], 4)
    trade_log.record_trade({
        "venue": "sxbet", "match_name": f"{p1} vs {p2}", "market_type": "match",
        "question": q.get("league") if q else "SX Bet", "condition_id": res.get("fill", {}).get("marketHash"),
        "token_id": None, "outcome": target, "side": "sxbet",
        "true_p": implied, "market_price": implied, "edge": 0.0, "kelly_frac": 0.0,
        "stake_usd": round(stake, 2), "shares": None, "order_id": None,
        "status": res["status"], "detail": f"SX {'LIVE' if not dry else 'paper'}: {res.get('detail', '')[:200]}",
    })
    return {"ok": res["status"] in ("dry_run", "placed"), "mode": "paper" if dry else "live", **res}


def _norm_eq(a: str, b: str) -> bool:
    return (a or "").strip().split()[-1].lower() == (b or "").strip().split()[-1].lower()


@app.get("/api/price/{token_id}")
def price(token_id: str):
    bid, ask = _client.best_bid(token_id), _client.best_ask(token_id)
    return {"bid": bid, "ask": ask}


@app.post("/api/agent/on")
def agent_on():
    agent_mod.on(interval=60, foreground=False)
    pid = agent_mod._running_pid()
    return {"ok": bool(pid), "pid": pid}


@app.post("/api/agent/off")
def agent_off():
    agent_mod.off()
    return {"ok": True}


@app.post("/api/agent/settle")
def agent_settle():
    from execution.settle import settle_open
    return settle_open(verbose=False)


@app.post("/api/cancel-open")
def cancel_open():
    return {"ok": True, "cancelled": trade_log.cancel_all_open()}


@app.post("/api/trade/{trade_id}/cancel")
def cancel(trade_id: int):
    try:
        trade_log.cancel_trade(trade_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.post("/api/trade/{trade_id}/sell")
def sell(trade_id: int):
    row = next((r for r in trade_log.all_trades() if r["trade_id"] == trade_id), None)
    if not row:
        raise HTTPException(status_code=404, detail="trade not found")
    if row["status"] not in OPEN_STATUSES:
        raise HTTPException(status_code=400, detail=f"trade is {row['status']}, not open")
    token = row["token_id"]
    exit_price = _client.best_bid(token) or _client.best_ask(token)
    if not exit_price or not 0.0 < exit_price < 1.0:
        raise HTTPException(status_code=409,
                            detail="no live price to sell into right now")
    pnl = trade_log.close_trade(trade_id, exit_price)
    return {"ok": True, "exit_price": exit_price, "pnl": pnl}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/intel", response_class=HTMLResponse)
def intel_page():
    return INTEL_HTML


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Tennis Agent — Local</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#8b949e;
   --grn:#3fb950;--red:#f85149;--amb:#d29922;--blu:#388bfd;}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
 header{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);
   padding:12px 18px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;z-index:5}
 h1{font-size:16px;margin:0;font-weight:600} .pill{padding:3px 10px;border-radius:999px;
   font-weight:600;font-size:12px} .on{background:rgba(63,185,80,.15);color:var(--grn)}
 .off{background:rgba(248,81,73,.15);color:var(--red)}
 button{background:#21262d;color:var(--fg);border:1px solid var(--line);border-radius:6px;
   padding:6px 12px;cursor:pointer;font-size:13px} button:hover{border-color:#8b949e}
 button.p{background:var(--grn);border-color:var(--grn);color:#04160a;font-weight:600}
 button.d{background:transparent;border-color:var(--red);color:var(--red);padding:3px 9px}
 button.s{border-color:var(--blu);color:var(--blu);padding:3px 9px}
 .tiles{display:flex;gap:14px;margin-left:auto;flex-wrap:wrap}
 .tile{text-align:right} .tile .v{font-size:16px;font-weight:700} .tile .k{font-size:11px;color:var(--mut)}
 main{padding:18px;max-width:1400px;margin:0 auto} h2{font-size:13px;color:var(--mut);
   text-transform:uppercase;letter-spacing:.5px;margin:22px 0 8px}
 .wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;white-space:nowrap}
 th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
 th{background:var(--panel);color:var(--mut);font-weight:600;font-size:11px;position:sticky;top:0}
 td.l,th.l{text-align:left} tr:last-child td{border-bottom:0}
 .pos{color:var(--grn)} .neg{color:var(--red)} .mut{color:var(--mut)}
 .bdg{font-size:11px;padding:1px 7px;border-radius:4px;font-weight:600}
 .b-open{background:rgba(56,139,253,.15);color:var(--blu)}
 .b-win{background:rgba(63,185,80,.15);color:var(--grn)}
 .b-loss{background:rgba(248,81,73,.15);color:var(--red)}
 .b-x{background:rgba(139,148,158,.15);color:var(--mut)}
 .b-hedge{background:rgba(210,153,34,.15);color:var(--amb)}
 .flash{animation:fl 1.5s ease-out} @keyframes fl{from{background:rgba(56,139,253,.25)}to{background:transparent}}
 #err{color:var(--red);font-size:12px;margin-left:8px}
</style></head><body>
<header>
 <h1>🎾 Tennis Agent</h1>
 <a href="/intel" style="color:#388bfd;text-decoration:none;font-size:13px">Intelligence →</a>
 <span id="agentPill" class="pill off">…</span>
 <button class="p" id="btnOn">Turn ON</button>
 <button id="btnOff">Turn OFF</button>
 <button id="btnSettle">Settle finished</button>
 <button class="d" id="btnCancelAll" style="padding:6px 12px">Cancel all open</button>
 <span id="err"></span>
 <div class="tiles">
  <div class="tile"><div class="v" id="tBets">–</div><div class="k">bets</div></div>
  <div class="tile"><div class="v" id="tOpen">–</div><div class="k">open exposure</div></div>
  <div class="tile"><div class="v" id="tUpnl">–</div><div class="k">unrealized PnL</div></div>
  <div class="tile"><div class="v" id="tPnl">–</div><div class="k">realized PnL</div></div>
  <div class="tile"><div class="v" id="tWL">–</div><div class="k">W–L</div></div>
 </div>
</header>
<main>
 <h2>Open positions <span class="mut" id="openCount"></span></h2>
 <div class="wrap"><table id="openTbl"><thead><tr>
  <th>#</th><th class="l">match</th><th>mkt</th><th class="l">outcome</th>
  <th>trueP</th><th>entry</th><th>current</th><th>edge</th><th>shares</th><th>stake</th>
  <th>uPnL</th><th>actions</th>
 </tr></thead><tbody></tbody></table></div>

 <h2>Full log</h2>
 <div class="wrap"><table id="logTbl"><thead><tr>
  <th>#</th><th class="l">time (utc)</th><th class="l">match</th><th>mkt</th>
  <th class="l">outcome</th><th>price</th><th>edge</th><th>stake</th><th>status</th><th>PnL</th>
 </tr></thead><tbody></tbody></table></div>
</main>
<script>
const $=s=>document.querySelector(s), fmt=(x,d=2)=>x==null?'–':(+x).toFixed(d);
const money=x=>x==null?'–':(x<0?'-$':'$')+Math.abs(x).toFixed(2);
const cls=x=>x==null?'':(x>0?'pos':x<0?'neg':'mut');
let seen=new Set(), first=true, err=$('#err');
function badge(s,hedge){
 if(hedge && (s==='dry_run'||s==='placed')) return '<span class="bdg b-hedge">hedge</span>';
 const m={dry_run:'b-open',placed:'b-open',settled_win:'b-win',settled_loss:'b-loss',
   closed:'b-win',cancelled:'b-x'}; const t={dry_run:'open',placed:'open'};
 return `<span class="bdg ${m[s]||'b-x'}">${t[s]||s.replace('settled_','')}</span>`;
}
async function post(u){err.textContent='';try{const r=await fetch(u,{method:'POST'});
 const j=await r.json(); if(!r.ok) throw new Error(j.detail||'error'); return j;}
 catch(e){err.textContent=e.message;throw e;} }
async function refresh(){
 let d; try{ d=await (await fetch('/api/state')).json(); }catch(e){return;}
 const a=d.agent, s=d.summary;
 $('#agentPill').className='pill '+(a.on?'on':'off');
 $('#agentPill').textContent=a.on?('AGENT ON · pid '+a.pid):'AGENT OFF';
 $('#tBets').textContent=s.trades;
 $('#tOpen').textContent=money(s.open_stake_usd);
 $('#tPnl').textContent=money(s.settled_pnl_usd); $('#tPnl').className='v '+cls(s.settled_pnl_usd);
 $('#tUpnl').textContent=money(d.unrealized_pnl); $('#tUpnl').className='v '+cls(d.unrealized_pnl);
 $('#tWL').textContent=s.wins+'–'+s.losses;
 const open=d.trades.filter(t=>t.open);
 $('#openCount').textContent='· '+open.length+(d.priced?' · '+d.priced+' priced':' · pricing…');
 $('#openTbl tbody').innerHTML = open.map(t=>`<tr>
   <td>${t.id}</td><td class="l">${t.match||''}</td><td>${t.market}</td>
   <td class="l">${(t.outcome||'')}${t.hedge?' <span class="bdg b-hedge">H</span>':''}</td>
   <td>${fmt(t.true_p,3)}</td><td>${fmt(t.price,3)}</td>
   <td>${t.current==null?'<span class="mut">…</span>':fmt(t.current,3)}</td>
   <td class="${cls(t.edge)}">${t.edge==null?'–':(t.edge>0?'+':'')+fmt(t.edge,3)}</td>
   <td>${fmt(t.shares)}</td><td>${money(t.stake)}</td>
   <td class="${cls(t.upnl)}">${t.upnl==null?'<span class="mut">…</span>':money(t.upnl)}</td>
   <td><button class="s" onclick="sell(${t.id})">Sell</button>
       <button class="d" onclick="cancel(${t.id})">Cancel</button></td></tr>`).join('')
   || '<tr><td colspan="12" class="mut" style="text-align:center;padding:16px">no open positions</td></tr>';
 $('#logTbl tbody').innerHTML = d.trades.map(t=>{
   const isNew=!first&&!seen.has(t.id); seen.add(t.id);
   return `<tr class="${isNew?'flash':''}">
   <td>${t.id}</td><td class="l mut">${t.ts}</td><td class="l">${t.match||''}</td>
   <td>${t.market}</td><td class="l">${t.outcome||''}</td><td>${fmt(t.price,3)}</td>
   <td class="${cls(t.edge)}">${t.edge==null?'–':(t.edge>0?'+':'')+fmt(t.edge,3)}</td>
   <td>${money(t.stake)}</td><td>${badge(t.status,t.hedge)}</td>
   <td class="${cls(t.pnl)}">${t.pnl==null?'–':money(t.pnl)}</td></tr>`; }).join('');
 first=false;
}
async function sell(id){ if(!confirm('Sell position #'+id+' at the live bid?'))return;
 const j=await post('/api/trade/'+id+'/sell');
 err.textContent='Sold #'+id+' @ '+fmt(j.exit_price,3)+' → PnL '+money(j.pnl);
 refresh(); }
async function cancel(id){ if(!confirm('Cancel (void) position #'+id+'?'))return;
 await post('/api/trade/'+id+'/cancel'); refresh(); }
$('#btnOn').onclick=async()=>{$('#btnOn').textContent='starting…';await post('/api/agent/on');
 $('#btnOn').textContent='Turn ON';refresh();};
$('#btnOff').onclick=async()=>{await post('/api/agent/off');refresh();};
$('#btnSettle').onclick=async()=>{const j=await post('/api/agent/settle');
 err.textContent='Settled '+j.settled+' ('+j.wins+'W-'+j.losses+'L) '+money(j.pnl);refresh();};
$('#btnCancelAll').onclick=async()=>{if(!confirm('Cancel (void) ALL open positions?'))return;
 const j=await post('/api/cancel-open');err.textContent='Cancelled '+j.cancelled+' open bet(s)';refresh();};
refresh(); setInterval(refresh,3000);
</script></body></html>"""


INTEL_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Live Intelligence</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#8b949e;
   --grn:#3fb950;--red:#f85149;--amb:#d29922;--blu:#388bfd;--cy:#39c5cf;}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
 header{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);
   padding:11px 18px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;z-index:5}
 h1{font-size:15px;margin:0;font-weight:600} a{color:var(--blu);text-decoration:none;font-size:13px}
 main{padding:16px;max-width:1200px;margin:0 auto}
 h2{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--mut);
   margin:20px 0 7px;font-weight:700}
 .wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;white-space:nowrap}
 th,td{padding:8px 11px;text-align:right;border-bottom:1px solid var(--line)}
 th{background:var(--panel);color:var(--mut);font-weight:600;font-size:10.5px}
 td.l,th.l{text-align:left} tr:last-child td{border-bottom:0}
 .pos{color:var(--grn)} .neg{color:var(--red)} .mut{color:var(--mut)}
 .big{font-size:15px;font-weight:800;font-family:ui-monospace,Menlo,monospace}
 .bdg{font-size:10px;padding:1px 7px;border-radius:5px;font-weight:700}
 .sure{background:rgba(63,185,80,.16);color:var(--grn)}
 .brk{background:rgba(248,81,73,.16);color:var(--red)}
 .dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--cy);margin-right:5px;vertical-align:middle}
 button.s{background:#21262d;color:var(--blu);border:1px solid var(--line);border-radius:6px;
   padding:3px 9px;cursor:pointer;font-size:12px} button.s:hover{border-color:var(--blu)}
 .sub{font-size:10px;color:var(--mut)}
</style></head><body>
<header>
 <h1>🎾 Live Intelligence</h1><a href="/">← Agent</a>
 <span class="mut" id="meta" style="margin-left:auto;font-size:12px"></span>
</header>
<main>
 <h2>Live matches — win probability &amp; breaks now</h2>
 <div class="wrap"><table id="live"><thead><tr>
  <th class="l">match</th><th class="l">score</th><th>live win</th>
  <th class="l">break watch</th><th>mom</th><th class="l"></th>
 </tr></thead><tbody></tbody></table></div>

 <h2>Open positions</h2>
 <div class="wrap"><table id="open"><thead><tr>
  <th>#</th><th class="l">match</th><th class="l">backing</th><th>venue</th>
  <th>entry</th><th>now</th><th>P&amp;L</th><th></th>
 </tr></thead><tbody></tbody></table></div>

 <h2>Hedges</h2>
 <div class="wrap"><table id="hedge"><thead><tr>
  <th>#</th><th class="l">match</th><th class="l">hedge leg</th><th>entry</th><th>now</th><th>result</th>
 </tr></thead><tbody></tbody></table></div>

 <p class="mut" id="pnl" style="font-size:12px;margin-top:12px"></p>
 <div id="toast" style="position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
   background:var(--grn);color:#04160a;padding:8px 16px;border-radius:6px;font-weight:600;
   opacity:0;transition:opacity .2s;pointer-events:none">ok</div>
</main>
<script>
const $=s=>document.querySelector(s);
const pct=x=>x==null?'-':(100*x).toFixed(1)+'%';
const money=x=>x==null?'-':(x<0?'-$':'$')+Math.abs(x).toFixed(2);
let toastT; function toast(m){const t=$('#toast');t.textContent=m;t.style.opacity=1;
 clearTimeout(toastT);toastT=setTimeout(()=>t.style.opacity=0,1400);}
async function post(u){try{const r=await fetch(u,{method:'POST'});return await r.json();}catch(e){return{};}}

function favored(r){ // returns {name, p, side}
 const p=r.inplay_p1; if(p==null) return null;
 return p>=0.5 ? {name:r.p1, p:p, side:1} : {name:r.p2, p:1-p, side:2};
}
function momArrow(r,side){
 const m=r.momentum; if(!m||!m.has_signal) return '<span class="mut">-</span>';
 const mv = side===1 ? m.momentum_p1 : -m.momentum_p1;
 const a = mv>0.05?'▲':mv<-0.05?'▼':'▬', c = mv>0.05?'pos':mv<-0.05?'neg':'mut';
 return `<span class="${c}">${a} ${mv>=0?'+':''}${mv.toFixed(2)}</span>`;
}
function breakCell(r){
 const b=r.break; if(!b) return '<span class="mut">-</span>';
 const bp=b.prob, who=b.returner.split(' ').pop();
 if(bp>=0.5){ const strong=bp>=0.70;
  return `<span class="${strong?'neg':'amb'}" style="font-weight:${strong?700:600}">`
    +`⚡ ${who} ${(bp*100).toFixed(0)}%</span> <span class="sub">(${b.pts})</span>`; }
 return `<span class="mut">hold ${b.server.split(' ').pop()} ${(100-bp*100).toFixed(0)}% (${b.pts})</span>`;
}
function tag(r,fav){
 const b=r.break;
 if(b && b.prob>=0.70) return '<span class="bdg brk">⚡ BREAK</span>';
 if(fav && fav.p>=0.80) return '<span class="bdg sure">🔒 SURE</span>';
 return '';
}
function scoreCell(sc){ if(!sc) return '<span class="mut">-</span>';
 const pt=sc.point?` <span class="mut">· ${sc.point}</span>`:'';
 return `<span class="dot"></span><b>${sc.sets}</b> <span class="mut">${sc.games}</span>${pt}`; }

async function refresh(){
 let L,S; try{ [L,S]=await Promise.all([fetch('/api/live').then(r=>r.json()),fetch('/api/state').then(r=>r.json())]); }catch(e){return;}
 // LIVE MATCHES: every in-progress match; server sorts breaks & sure first
 const board=L.board||[];
 const age=L.ts?Math.round(Date.now()/1000-L.ts):null;
 $('#meta').textContent=`${board.length} live · updated ${age!=null?age+'s':'…'} ago`;
 $('#live tbody').innerHTML = board.map(r=>{
   const fav={name:r.favored, p:r.win, side:r.side};
   return `<tr>
   <td class="l">${r.p1.split(' ').pop()} v ${r.p2.split(' ').pop()}
     <div class="sub">${(r.league||'')}</div></td>
   <td class="l">${scoreCell(r.score)}</td>
   <td class="l"><span class="big ${fav.p>=0.80?'pos':''}">${pct(fav.p)}</span>
     <span class="sub"> ${fav.name.split(' ').pop()}</span></td>
   <td class="l">${breakCell(r)}</td>
   <td>${momArrow(r,fav.side)}</td>
   <td class="l">${tag(r,fav)}</td></tr>`; }).join('')
   || '<tr><td colspan="6" class="mut" style="text-align:center;padding:16px">no live matches right now</td></tr>';

 // POSITIONS split: open (non-hedge) vs hedge
 const tr=(S.trades||[]);
 const venue=t=>(t.detail||'').match(/HEDGE/)?'':(t.token_id?'PM':'SX');
 const open=tr.filter(t=>t.open&&!t.hedge);
 const hedge=tr.filter(t=>t.hedge);
 const CAP=30;  // newest first; keep the tables scannable
 $('#open tbody').innerHTML = open.slice(0,CAP).map(t=>`<tr>
   <td>${t.id}</td><td class="l">${(t.match||'').slice(0,28)}</td>
   <td class="l">${t.outcome||''} <span class="sub">${t.market}</span></td>
   <td>${t.token_id?'PM':'SX'}</td>
   <td>${pct(t.price)}</td><td>${t.current==null?'<span class=mut>-</span>':pct(t.current)}</td>
   <td class="${t.upnl>0?'pos':t.upnl<0?'neg':'mut'}">${t.upnl==null?'-':money(t.upnl)}</td>
   <td>${t.token_id?`<button class="s" onclick="sell(${t.id})">sell</button>`:''}</td></tr>`).join('')
   || '<tr><td colspan="8" class="mut" style="text-align:center;padding:14px">no open positions</td></tr>';
 $('#hedge tbody').innerHTML = hedge.slice(0,CAP).map(t=>`<tr>
   <td>${t.id}</td><td class="l">${(t.match||'').slice(0,28)}</td>
   <td class="l"><span class="amb" style="color:var(--amb)">HEDGE</span> ${t.outcome||''}</td>
   <td>${pct(t.price)}</td><td>${t.current==null?'<span class=mut>-</span>':pct(t.current)}</td>
   <td class="${(t.pnl??t.upnl)>0?'pos':(t.pnl??t.upnl)<0?'neg':'mut'}">${(t.pnl??t.upnl)==null?'open':money(t.pnl??t.upnl)}</td></tr>`).join('')
   || '<tr><td colspan="6" class="mut" style="text-align:center;padding:14px">no hedges</td></tr>';

 const su=S.summary||{}; const rp=su.settled_pnl_usd||0;
 $('#pnl').innerHTML = `realized <b class="${rp>=0?'pos':'neg'}">${money(rp)}</b>`
   +` · unrealized <b class="${(S.unrealized_pnl||0)>=0?'pos':'neg'}">${money(S.unrealized_pnl||0)}</b>`
   +` · record ${su.wins||0}W-${su.losses||0}L · showing newest ${Math.min(CAP,open.length)}/${open.length} open · ${Math.min(CAP,hedge.length)}/${hedge.length} hedge`;
}
async function sell(id){ if(!confirm('Sell #'+id+' at live bid?'))return;
 const j=await post('/api/trade/'+id+'/sell'); toast('sold #'+id); refresh(); }
refresh(); setInterval(refresh,5000);
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Local agent dashboard (localhost only)")
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()
    import uvicorn
    print(f"Tennis Agent dashboard → http://127.0.0.1:{args.port}  (localhost only)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
