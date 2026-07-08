"""Signal -> Polymarket execution pipeline.

Modules:
    polymarket  - Gamma market discovery, CLOB pricing, order placement
    trade_log   - persistent bet journal (SQLite + CSV mirror)
    pipeline    - orchestrator CLI: signals -> mapping -> sizing -> orders -> log
"""
