#!/usr/bin/env bash
# ============================================================================
# Deploy to GCP VM
# ============================================================================
# Builds Docker images, pushes to Artifact Registry, and deploys to the VM.
#
# Usage:
#   ./deploy/deploy.sh [--live]    # --live enables real trading
# ============================================================================

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-tennis-trading-prod}"
REGION="europe-west1"
ZONE="europe-west1-b"
INSTANCE_NAME="tennis-trading-vm"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/tennis-trading"

echo "═══════════════════════════════════════════════════════════════"
echo "  Deploying Tennis Trading System to GCP"
echo "  Registry: $REGISTRY"
echo "═══════════════════════════════════════════════════════════════"

# ── 1. Configure Docker for Artifact Registry ────────────────────────────────
echo ""
echo "▶ Configuring Docker auth..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ── 2. Build and push images ─────────────────────────────────────────────────
echo ""
echo "▶ Building trading server..."
docker build --target trading-server -t "${REGISTRY}/trading-server:latest" .
docker push "${REGISTRY}/trading-server:latest"

echo ""
echo "▶ Building betfair loop..."
docker build --target betfair-loop -t "${REGISTRY}/betfair-loop:latest" .
docker push "${REGISTRY}/betfair-loop:latest"

echo ""
echo "▶ Building frontend..."
docker build -t "${REGISTRY}/frontend:latest" ./trading-terminal/
docker push "${REGISTRY}/frontend:latest"

# ── 3. Create deployment script for the VM ────────────────────────────────────
cat > /tmp/vm_deploy.sh << 'VMSCRIPT'
#!/bin/bash
set -euo pipefail

REGISTRY="__REGISTRY__"

# Pull latest images
docker pull "${REGISTRY}/trading-server:latest"
docker pull "${REGISTRY}/betfair-loop:latest"
docker pull "${REGISTRY}/frontend:latest"

# Stop existing containers
docker stop trading-server betfair-loop frontend 2>/dev/null || true
docker rm trading-server betfair-loop frontend 2>/dev/null || true

# Fetch secrets from Secret Manager
export BETFAIR_USERNAME=$(gcloud secrets versions access latest --secret=betfair-username)
export BETFAIR_PASSWORD=$(gcloud secrets versions access latest --secret=betfair-password)
export BETFAIR_APP_KEY=$(gcloud secrets versions access latest --secret=betfair-app-key)

# Start trading server
docker run -d \
    --name trading-server \
    --restart unless-stopped \
    -p 8888:8888 \
    -e BETFAIR_USERNAME \
    -e BETFAIR_PASSWORD \
    -e BETFAIR_APP_KEY \
    -e BETFAIR_CERT_PATH=/app/certs/client-2048.crt \
    -e BETFAIR_KEY_PATH=/app/certs/client-2048.key \
    -v /home/certs:/app/certs:ro \
    "${REGISTRY}/trading-server:latest"

# Start frontend
docker run -d \
    --name frontend \
    --restart unless-stopped \
    -p 3000:3000 \
    -e NEXT_PUBLIC_API_URL=http://localhost:8888 \
    "${REGISTRY}/frontend:latest"

echo "✅ Deployment complete"
docker ps
VMSCRIPT

# Replace registry placeholder
sed -i.bak "s|__REGISTRY__|${REGISTRY}|g" /tmp/vm_deploy.sh

# ── 4. Copy and execute on VM ─────────────────────────────────────────────────
echo ""
echo "▶ Deploying to VM..."
gcloud compute scp /tmp/vm_deploy.sh "${INSTANCE_NAME}:~/deploy.sh" --zone="$ZONE"
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="bash ~/deploy.sh"

echo ""
echo "═══════════════════════════════════════════════════════════════"

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --zone="$ZONE" --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo "  ✅ Deployment Complete!"
echo ""
echo "  Trading Server: http://${EXTERNAL_IP}:8888/health"
echo "  Frontend:       http://${EXTERNAL_IP}:3000"
echo ""
echo "  To start live trading on the VM:"
echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "    docker run -it --rm \\"
echo "      -e BETFAIR_USERNAME -e BETFAIR_PASSWORD -e BETFAIR_APP_KEY \\"
echo "      -e BETFAIR_CERT_PATH=/app/certs/client-2048.crt \\"
echo "      -e BETFAIR_KEY_PATH=/app/certs/client-2048.key \\"
echo "      -v /home/certs:/app/certs:ro \\"
echo "      ${REGISTRY}/betfair-loop:latest \\"
echo "      --player1 \"Djokovic\" --player2 \"Alcaraz\" --dry-run"
echo "═══════════════════════════════════════════════════════════════"
