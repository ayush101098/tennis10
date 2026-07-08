"""Persistent bet journal: every intended or placed trade, one row.

Writes to the existing tennis_betting.db (table: trade_log) and mirrors to
trades_log.csv so the user can open the full history in a spreadsheet.
"""

import csv
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "tennis_betting.db"
CSV_PATH = REPO_ROOT / "trades_log.csv"

SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_log (
    trade_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc       TEXT NOT NULL,
    venue        TEXT NOT NULL DEFAULT 'polymarket',
    match_name   TEXT NOT NULL,
    market_type  TEXT NOT NULL,           -- match | set1 | set2 | set3
    question     TEXT,
    condition_id TEXT,
    token_id     TEXT,
    outcome      TEXT,                    -- outcome bought (player name)
    side         TEXT,                    -- player1 | player2 (signal frame)
    true_p       REAL,
    market_price REAL,                    -- price paid / best ask at decision
    edge         REAL,
    kelly_frac   REAL,
    stake_usd    REAL,
    shares       REAL,
    order_id     TEXT,
    status       TEXT NOT NULL,           -- dry_run | placed | failed | skipped | settled_win | settled_loss
    detail       TEXT,
    settled_at   TEXT,
    pnl_usd      REAL
);
"""

CSV_FIELDS = ["trade_id", "ts_utc", "venue", "match_name", "market_type",
              "question", "outcome", "side", "true_p", "market_price", "edge",
              "kelly_frac", "stake_usd", "shares", "order_id", "status",
              "detail", "condition_id", "token_id"]


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def record_trade(row: dict) -> int:
    """Insert one trade row; returns trade_id. Also appends to the CSV mirror."""
    row = dict(row)
    row.setdefault("ts_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    row.setdefault("venue", "polymarket")
    cols = [c for c in row if c in {f for f in CSV_FIELDS} | {"settled_at", "pnl_usd"}]
    with _conn() as conn:
        cur = conn.execute(
            f"INSERT INTO trade_log ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
            [row[c] for c in cols],
        )
        trade_id = cur.lastrowid
    row["trade_id"] = trade_id
    _append_csv(row)
    return trade_id


def _append_csv(row: dict) -> None:
    new_file = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def already_traded(token_id: str) -> bool:
    """True if we already hold an open (non-failed, non-skipped) bet on this token."""
    with _conn() as conn:
        cur = conn.execute(
            "SELECT 1 FROM trade_log WHERE token_id = ? "
            "AND status IN ('dry_run', 'placed') LIMIT 1", (token_id,))
        return cur.fetchone() is not None


def recent_trades(limit: int = 25) -> list[dict]:
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM trade_log ORDER BY trade_id DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


def settle_trade(trade_id: int, won: bool) -> None:
    """Mark a trade settled and compute PnL (binary market payout $1/share)."""
    with _conn() as conn:
        cur = conn.execute(
            "SELECT stake_usd, shares FROM trade_log WHERE trade_id = ?", (trade_id,))
        found = cur.fetchone()
        if not found:
            raise ValueError(f"trade {trade_id} not found")
        stake, shares = found
        pnl = (shares - stake) if won else -stake
        conn.execute(
            "UPDATE trade_log SET status = ?, settled_at = ?, pnl_usd = ? WHERE trade_id = ?",
            ("settled_win" if won else "settled_loss",
             datetime.now(timezone.utc).isoformat(timespec="seconds"), pnl, trade_id))


def summary() -> dict:
    with _conn() as conn:
        cur = conn.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN status IN ('dry_run','placed') THEN stake_usd ELSE 0 END),
                   SUM(CASE WHEN status LIKE 'settled%' THEN pnl_usd ELSE 0 END),
                   SUM(CASE WHEN status = 'settled_win' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN status = 'settled_loss' THEN 1 ELSE 0 END)
            FROM trade_log""")
        n, open_stake, pnl, wins, losses = cur.fetchone()
    return {"trades": n or 0, "open_stake_usd": open_stake or 0.0,
            "settled_pnl_usd": pnl or 0.0, "wins": wins or 0, "losses": losses or 0}
