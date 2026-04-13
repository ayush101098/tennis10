# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY trading_server/ trading_server/
COPY betfair/ betfair/
COPY api/ api/
COPY features.py .
COPY data_pipeline.py .
COPY data_pipeline_enhanced.py .
COPY hierarchical_model.py .

# Copy ML models (if they exist as .pkl / .joblib)
COPY *.pkl ./
COPY *.joblib ./
COPY *.csv ./

# Create certs directory (mount at runtime)
RUN mkdir -p /app/certs

# ── Trading Server ────────────────────────────────────────────────────────────
FROM base AS trading-server

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

CMD ["uvicorn", "trading_server.main:app", "--host", "0.0.0.0", "--port", "8888"]


# ── Betfair Live Loop ────────────────────────────────────────────────────────
FROM base AS betfair-loop

# The live loop is started via command line args
# Override CMD when running:
#   docker run ... betfair-loop --player1 "X" --player2 "Y" --dry-run
ENTRYPOINT ["python", "-m", "betfair.live_loop"]
CMD ["--help"]
