import os
import sqlite3

from execution import trade_log


def test_trade_log_records_user_name_and_filters_by_user(tmp_path, monkeypatch):
    db_path = tmp_path / "tennis_betting.db"
    csv_path = tmp_path / "trades_log.csv"
    monkeypatch.setattr(trade_log, "DB_PATH", db_path)
    monkeypatch.setattr(trade_log, "CSV_PATH", csv_path)

    trade_id = trade_log.record_trade({
        "match_name": "Test vs Demo",
        "market_type": "match",
        "outcome": "Test",
        "side": "player1",
        "true_p": 0.6,
        "market_price": 0.6,
        "edge": 0.1,
        "kelly_frac": 0.2,
        "stake_usd": 10.0,
        "shares": 16.7,
        "status": "dry_run",
        "user_name": "alice",
    })

    assert trade_id > 0

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT user_name FROM trade_log WHERE trade_id = ?", (trade_id,)).fetchall()
    assert rows[0][0] == "alice"

    alice_rows = trade_log.recent_trades(limit=10, user_name="alice")
    bob_rows = trade_log.recent_trades(limit=10, user_name="bob")
    assert len(alice_rows) == 1
    assert len(bob_rows) == 0
