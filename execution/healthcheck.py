"""Health check for the execution agent — verifies every stage end-to-end.

    python -m execution.healthcheck

Exercises: env/config, Gamma discovery, CLOB pricing, model prediction,
signal generation, trade journal, and the hedge decision path. Read-only for
markets; the only write is one temporary journal row (rolled back).
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from execution.pipeline import load_env  # noqa: E402


def _ok(label, detail=""):
    print(f"  \033[32mPASS\033[0m  {label}" + (f" — {detail}" if detail else ""))


def _fail(label, detail=""):
    print(f"  \033[31mFAIL\033[0m  {label}" + (f" — {detail}" if detail else ""))


def main() -> int:
    load_env()
    failures = 0
    print("execution agent health check\n" + "=" * 50)

    # 1. Config / mode
    from execution.polymarket import PolymarketClient
    client = PolymarketClient()
    dry_env = os.getenv("TRADING_DRY_RUN", "true").lower() != "false"
    key = bool(os.getenv("POLYMARKET_PRIVATE_KEY"))
    mode = "PAPER (dry-run)" if (dry_env or not key) else "LIVE-CAPABLE"
    print(f"\n[config]")
    _ok("endpoints", f"gamma={client.gamma_url}  clob={client.clob_url}  chain={client.chain_id}")
    _ok("trade mode", f"{mode}  (dry_run_env={dry_env}, key={'set' if key else 'unset'})")

    # 2. Gamma discovery
    print(f"\n[market discovery]")
    try:
        events = client.fetch_tennis_events()
        (_ok if events else _fail)("gamma tennis events", f"{len(events)} open")
        failures += 0 if events else 1
    except Exception as e:
        _fail("gamma tennis events", f"{type(e).__name__}: {e}"); events = []; failures += 1

    # 3. CLOB pricing (find any market token and price it)
    print(f"\n[pricing]")
    token = None
    for ev in events:
        for m in ev.get("markets", []):
            import json
            try:
                toks = json.loads(m.get("clobTokenIds") or "[]")
            except Exception:
                toks = []
            if toks:
                token = toks[0]; break
        if token:
            break
    if token:
        price = client.best_ask(token)
        (_ok if price is not None else _fail)("clob best_ask", f"token…{token[-6:]} -> {price}")
        failures += 0 if price is not None else 1
    else:
        _fail("clob best_ask", "no token found to price"); failures += 1

    # 4. Model prediction
    print(f"\n[model]")
    try:
        from execution.model_predict import MatchModel
        mm = MatchModel()
        p = mm.predict_match("Cameron Norrie", "Mariano Navone", "Clay")
        if p is not None:
            _ok("match prediction", f"P(p1)={p:.3f}  set_prob={mm.set_prob(p):.3f}")
        else:
            _fail("match prediction", "returned None for a known fixture"); failures += 1
        mm.close()
    except Exception as e:
        _fail("match prediction", f"{type(e).__name__}: {e}"); failures += 1

    # 5. Signal generation
    print(f"\n[signals]")
    try:
        from execution.signals_gen import generate
        sigs = generate(verbose=False)
        (_ok if sigs else _fail)("signal generation", f"{len(sigs)} signals")
        failures += 0 if sigs else 1
    except Exception as e:
        _fail("signal generation", f"{type(e).__name__}: {e}"); failures += 1

    # 6. Journal + hedge decision path (mocked, no market write)
    print(f"\n[journal + hedge]")
    try:
        from execution import trade_log, watch
        s = trade_log.summary()
        _ok("trade journal", f"{s['trades']} bets, open ${s['open_stake_usd']:.2f}")
        # hedge trigger arithmetic via mocks (no real trade recorded)
        pos = {"trade_id": -1, "match_name": "HC", "market_type": "match",
               "question": "q", "condition_id": "C", "token_id": "A", "outcome": "A",
               "side": "player1", "true_p": 0.7, "market_price": 0.70,
               "stake_usd": 25.0, "shares": 35.71}
        ev = {"markets": [{"conditionId": "C", "outcomes": '["A","B"]',
                           "clobTokenIds": '["A","B"]'}]}
        captured = {}
        _orig_open, _orig_traded, _orig_rec = (
            trade_log.open_positions, trade_log.already_traded, trade_log.record_trade)
        trade_log.open_positions = lambda *a, **k: [pos]
        trade_log.already_traded = lambda *a, **k: False
        trade_log.record_trade = lambda row: captured.update(row) or -1

        class FC:
            def best_ask(self, t): return {"A": 0.50, "B": 0.50}[t]  # 0.20 drop
            def place_buy(self, *a, **k): return {"status": "dry_run"}
        notes = watch._hedge_positions(FC(), [ev], dry_run=True, drop=0.15)
        trade_log.open_positions, trade_log.already_traded, trade_log.record_trade = (
            _orig_open, _orig_traded, _orig_rec)
        if notes and abs(captured.get("shares", 0) - 35.71) < 0.01:
            _ok("hedge full-lock", notes[0].split(":", 1)[-1].strip())
        else:
            _fail("hedge full-lock", "did not trigger / wrong size"); failures += 1
    except Exception as e:
        _fail("journal + hedge", f"{type(e).__name__}: {e}"); failures += 1

    print("\n" + "=" * 50)
    if failures == 0:
        print("\033[32mALL SYSTEMS GO\033[0m — agent is running smoothly.")
        return 0
    print(f"\033[31m{failures} check(s) failed.\033[0m")
    return 1


if __name__ == "__main__":
    sys.exit(main())
