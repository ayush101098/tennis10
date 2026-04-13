#!/usr/bin/env bash
# ============================================================================
# Generate Betfair SSL Certificates
# ============================================================================
# Betfair requires SSL certificates for non-interactive API login.
# This script generates a self-signed cert pair.
#
# Usage:
#   chmod +x deploy/generate_betfair_certs.sh
#   ./deploy/generate_betfair_certs.sh
#
# Then upload the .crt content to your Betfair account at:
#   https://myaccount.betfair.com/accountdetails/mysecurity
# ============================================================================

set -euo pipefail

CERT_DIR="certs"
mkdir -p "$CERT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  Generating Betfair SSL Certificates"
echo "═══════════════════════════════════════════════════════════════"

# Generate 2048-bit RSA key
openssl genrsa -out "${CERT_DIR}/client-2048.key" 2048

# Generate self-signed certificate (valid for 10 years)
openssl req -new -x509 \
    -days 3650 \
    -key "${CERT_DIR}/client-2048.key" \
    -out "${CERT_DIR}/client-2048.crt" \
    -subj "/CN=TennisTradingBot/O=Trading/C=GB"

echo ""
echo "  ✅ Certificates generated:"
echo "     ${CERT_DIR}/client-2048.key  (KEEP SECRET)"
echo "     ${CERT_DIR}/client-2048.crt  (upload to Betfair)"
echo ""
echo "  Next steps:"
echo "  1. Go to: https://myaccount.betfair.com/accountdetails/mysecurity"
echo "  2. Click 'Edit' next to 'Automated Betting Program Access'"
echo "  3. Upload ${CERT_DIR}/client-2048.crt"
echo "  4. Copy the .crt and .key files to your GCP VM:"
echo "     gcloud compute scp ${CERT_DIR}/* tennis-trading-vm:/home/certs/"
echo "═══════════════════════════════════════════════════════════════"
