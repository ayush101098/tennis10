#!/usr/bin/env bash
# ============================================================================
# GCP Setup Script for Tennis Trading System
# ============================================================================
# This script sets up everything on Google Cloud for EU-based deployment.
#
# Prerequisites:
#   - Google Cloud SDK installed (gcloud)
#   - A GCP project created
#   - Billing enabled
#
# Usage:
#   chmod +x deploy/gcp_setup.sh
#   ./deploy/gcp_setup.sh
# ============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
# Change these to match your setup:

PROJECT_ID="${GCP_PROJECT_ID:-tennis-trading-prod}"
REGION="europe-west1"            # Belgium — closest to Betfair servers
ZONE="europe-west1-b"
MACHINE_TYPE="e2-medium"         # 2 vCPU, 4 GB RAM (good for trading)
INSTANCE_NAME="tennis-trading-vm"
SERVICE_ACCOUNT_NAME="tennis-trading-sa"

echo "═══════════════════════════════════════════════════════════════"
echo "  GCP Setup: Tennis Trading System"
echo "  Project:   $PROJECT_ID"
echo "  Region:    $REGION (EU — close to Betfair)"
echo "  Zone:      $ZONE"
echo "═══════════════════════════════════════════════════════════════"

# ── 1. Set project ───────────────────────────────────────────────────────────
echo ""
echo "▶ Step 1: Setting GCP project..."
gcloud config set project "$PROJECT_ID"
gcloud config set compute/region "$REGION"
gcloud config set compute/zone "$ZONE"

# ── 2. Enable required APIs ──────────────────────────────────────────────────
echo ""
echo "▶ Step 2: Enabling APIs..."
gcloud services enable \
    compute.googleapis.com \
    containerregistry.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com

# ── 3. Create service account ────────────────────────────────────────────────
echo ""
echo "▶ Step 3: Creating service account..."
gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
    --display-name="Tennis Trading Service Account" \
    --description="Service account for tennis trading VM" \
    2>/dev/null || echo "  (already exists)"

SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Grant necessary roles
for role in \
    roles/secretmanager.secretAccessor \
    roles/logging.logWriter \
    roles/monitoring.metricWriter \
    roles/artifactregistry.reader; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$role" \
        --quiet 2>/dev/null || true
done

# ── 4. Store Betfair credentials in Secret Manager ───────────────────────────
echo ""
echo "▶ Step 4: Storing secrets..."
echo "  You'll be prompted to enter your Betfair credentials."

read -rp "  Betfair Username: " BF_USER
read -rsp "  Betfair Password: " BF_PASS
echo ""
read -rp "  Betfair App Key: " BF_APPKEY

# Create secrets
for secret in betfair-username betfair-password betfair-app-key; do
    gcloud secrets create "$secret" --replication-policy="user-managed" \
        --locations="$REGION" 2>/dev/null || true
done

echo -n "$BF_USER" | gcloud secrets versions add betfair-username --data-file=-
echo -n "$BF_PASS" | gcloud secrets versions add betfair-password --data-file=-
echo -n "$BF_APPKEY" | gcloud secrets versions add betfair-app-key --data-file=-

echo "  ✅ Secrets stored"

# ── 5. Create Artifact Registry repo ─────────────────────────────────────────
echo ""
echo "▶ Step 5: Creating Artifact Registry..."
gcloud artifacts repositories create tennis-trading \
    --repository-format=docker \
    --location="$REGION" \
    --description="Tennis trading Docker images" \
    2>/dev/null || echo "  (already exists)"

# ── 6. Create firewall rules ─────────────────────────────────────────────────
echo ""
echo "▶ Step 6: Creating firewall rules..."
gcloud compute firewall-rules create allow-trading-server \
    --allow=tcp:8888 \
    --target-tags=trading-server \
    --description="Allow trading server access" \
    2>/dev/null || echo "  (already exists)"

gcloud compute firewall-rules create allow-frontend \
    --allow=tcp:3000 \
    --target-tags=trading-server \
    --description="Allow frontend access" \
    2>/dev/null || echo "  (already exists)"

# ── 7. Create VM instance ────────────────────────────────────────────────────
echo ""
echo "▶ Step 7: Creating Compute Engine VM in EU..."

gcloud compute instances create "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=cos-stable \
    --image-project=cos-cloud \
    --boot-disk-size=30GB \
    --boot-disk-type=pd-ssd \
    --tags=trading-server \
    --service-account="$SA_EMAIL" \
    --scopes=cloud-platform \
    --metadata=startup-script='#!/bin/bash
# Install Docker Compose
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    docker/compose:latest version 2>/dev/null || true
echo "VM ready for deployment"
' \
    2>/dev/null || echo "  (already exists)"

# ── 8. Reserve static IP ─────────────────────────────────────────────────────
echo ""
echo "▶ Step 8: Reserving static IP..."
gcloud compute addresses create trading-ip \
    --region="$REGION" \
    2>/dev/null || echo "  (already exists)"

STATIC_IP=$(gcloud compute addresses describe trading-ip --region="$REGION" --format="get(address)" 2>/dev/null || echo "pending")
echo "  Static IP: $STATIC_IP"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ GCP Setup Complete!"
echo ""
echo "  VM:        $INSTANCE_NAME ($ZONE)"
echo "  Region:    $REGION (EU)"
echo "  Static IP: $STATIC_IP"
echo ""
echo "  Next steps:"
echo "  1. Upload Betfair SSL certs to the VM"
echo "  2. Run: ./deploy/deploy.sh"
echo "═══════════════════════════════════════════════════════════════"
