# GCP + Betfair Deployment Guide

Complete guide to deploying the Tennis Trading System on Google Cloud (EU region) with live Betfair execution.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  GCP Compute Engine VM  (europe-west1-b, Belgium)            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Trading     │  │  Betfair     │  │  Next.js         │   │
│  │  Server      │◄─│  Live Loop   │  │  Frontend        │   │
│  │  :8888       │  │  (executor)  │  │  :3000           │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────┘   │
│         │                 │                                   │
│         │    ┌────────────▼──────────┐                       │
│         │    │  Betfair Exchange API  │                       │
│         │    │  (streaming + orders)  │                       │
│         │    └───────────────────────┘                        │
│         │                                                    │
│  ┌──────▼───────┐                                            │
│  │  ML Models   │                                            │
│  │  (Markov +   │                                            │
│  │   Ensemble)  │                                            │
│  └──────────────┘                                            │
└──────────────────────────────────────────────────────────────┘
```

---

## Step 1: Google Cloud Setup

### 1.1 Install Google Cloud SDK

```bash
# macOS
brew install google-cloud-sdk

# Or download from https://cloud.google.com/sdk/docs/install
```

### 1.2 Create a GCP Project

```bash
# Login
gcloud auth login

# Create project
gcloud projects create tennis-trading-prod --name="Tennis Trading"

# Set as active
gcloud config set project tennis-trading-prod

# Enable billing (required - do this in the Console)
# https://console.cloud.google.com/billing
```

### 1.3 Set EU Region

**Why EU?** Betfair's exchange servers are in the UK/EU. Running your trading VM in `europe-west1` (Belgium) or `europe-west2` (London) gives you **~5ms latency** vs **~150ms from US**.

```bash
# Set default region to EU
gcloud config set compute/region europe-west1
gcloud config set compute/zone europe-west1-b
```

### 1.4 Run the Setup Script

```bash
chmod +x deploy/gcp_setup.sh
./deploy/gcp_setup.sh
```

This will:
- Enable required GCP APIs
- Create a service account
- Store Betfair credentials in Secret Manager
- Create an Artifact Registry for Docker images
- Create firewall rules
- Spin up a VM in `europe-west1-b`
- Reserve a static IP

---

## Step 2: Betfair API Setup

### 2.1 Create Betfair Account

1. Go to [betfair.com](https://www.betfair.com) and create an account
2. **Important**: Use a UK or EU-based account for exchange access
3. Deposit funds (minimum ~£20 to start)

### 2.2 Get API App Key

1. Go to [developer.betfair.com](https://developer.betfair.com/)
2. Click **My Account** → **Manage App Keys**
3. Create a new application:
   - Name: `TennisTradingBot`
   - Choose **"Delayed API Access"** (free) or apply for **"Live API Access"**
4. Copy your **App Key**

### 2.3 Generate SSL Certificates

Betfair requires SSL certs for non-interactive (automated) login:

```bash
# Generate certs
./deploy/generate_betfair_certs.sh
```

This creates:
- `certs/client-2048.key` — Private key (keep secret!)
- `certs/client-2048.crt` — Certificate (upload to Betfair)

### 2.4 Upload Certificate to Betfair

1. Go to [myaccount.betfair.com/accountdetails/mysecurity](https://myaccount.betfair.com/accountdetails/mysecurity)
2. Find **"Automated Betting Program Access"**
3. Click **"Edit"**
4. Upload `certs/client-2048.crt`
5. Click **"Save"**

### 2.5 Set Up Environment

```bash
cp .env.example .env

# Edit with your actual credentials:
nano .env
```

Fill in:
```env
BETFAIR_USERNAME=your_username
BETFAIR_PASSWORD=your_password
BETFAIR_APP_KEY=your_app_key
BETFAIR_CERT_PATH=./certs/client-2048.crt
BETFAIR_KEY_PATH=./certs/client-2048.key
```

---

## Step 3: Test Locally (Paper Trading)

### 3.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.2 Test Betfair Connection

```python
from betfair.client import BetfairClient

client = BetfairClient.from_env()
client.login()

# Check account
print(client.get_account_funds())

# Find tennis markets
markets = client.find_tennis_markets()
for m in markets[:5]:
    print(f"{m.event_name} — {m.market_id}")

client.logout()
```

### 3.3 Run Paper Trading

```bash
# Dry run (no real money)
python -m betfair.live_loop \
    --player1 "Djokovic" \
    --player2 "Alcaraz" \
    --p1-serve 68 --p2-serve 65 \
    --p1-rank 2 --p2-rank 3 \
    --bankroll 5000 \
    --max-stake 25 \
    --dry-run
```

This will:
- Login to Betfair
- Find the Djokovic vs Alcaraz market
- Stream live odds
- Run your full ML pipeline on every odds change
- Log what trades it **would** make (but not actually place them)

---

## Step 4: Deploy to GCP

### 4.1 Upload Certificates to VM

```bash
# Create certs directory on VM
gcloud compute ssh tennis-trading-vm --zone=europe-west1-b \
    --command="mkdir -p /home/certs"

# Upload certs
gcloud compute scp certs/client-2048.crt certs/client-2048.key \
    tennis-trading-vm:/home/certs/ --zone=europe-west1-b
```

### 4.2 Build & Deploy

```bash
chmod +x deploy/deploy.sh
./deploy/deploy.sh
```

This builds Docker images, pushes to Artifact Registry, and deploys to your VM.

### 4.3 Verify Deployment

```bash
# Get VM IP
gcloud compute instances describe tennis-trading-vm \
    --zone=europe-west1-b \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)"

# Check health
curl http://<VM_IP>:8888/health
```

---

## Step 5: Start Live Trading

### 5.1 SSH into VM

```bash
gcloud compute ssh tennis-trading-vm --zone=europe-west1-b
```

### 5.2 Start Paper Trading (recommended first)

```bash
# Fetch secrets
export BETFAIR_USERNAME=$(gcloud secrets versions access latest --secret=betfair-username)
export BETFAIR_PASSWORD=$(gcloud secrets versions access latest --secret=betfair-password)
export BETFAIR_APP_KEY=$(gcloud secrets versions access latest --secret=betfair-app-key)

# Run paper trading
docker run -it --rm \
    -e BETFAIR_USERNAME \
    -e BETFAIR_PASSWORD \
    -e BETFAIR_APP_KEY \
    -e BETFAIR_CERT_PATH=/app/certs/client-2048.crt \
    -e BETFAIR_KEY_PATH=/app/certs/client-2048.key \
    -v /home/certs:/app/certs:ro \
    europe-west1-docker.pkg.dev/tennis-trading-prod/tennis-trading/betfair-loop:latest \
    --player1 "Sinner" --player2 "Alcaraz" \
    --p1-serve 67 --p2-serve 65 \
    --bankroll 5000 --max-stake 25 \
    --dry-run
```

### 5.3 Enable Live Trading

⚠️ **Only do this after thorough paper trading testing!**

```bash
# Replace --dry-run with --live
docker run -it --rm \
    -e BETFAIR_USERNAME \
    -e BETFAIR_PASSWORD \
    -e BETFAIR_APP_KEY \
    -e BETFAIR_CERT_PATH=/app/certs/client-2048.crt \
    -e BETFAIR_KEY_PATH=/app/certs/client-2048.key \
    -v /home/certs:/app/certs:ro \
    europe-west1-docker.pkg.dev/tennis-trading-prod/tennis-trading/betfair-loop:latest \
    --player1 "Sinner" --player2 "Alcaraz" \
    --p1-serve 67 --p2-serve 65 \
    --bankroll 5000 --max-stake 25 \
    --live
```

You'll be prompted to type `YES` to confirm.

---

## Risk Controls

The system has multiple safety layers:

| Control | Default | Description |
|---------|---------|-------------|
| `--dry-run` | **ON** | Paper trading mode (no real orders) |
| `--max-stake` | £50 | Maximum single bet size |
| `min_odds` | 1.10 | Won't back below this |
| `max_odds` | 20.0 | Won't back above this |
| `min_liquidity` | £20 | Minimum available at best price |
| Risk Manager | Built-in | Stops after consecutive losses |
| Kelly criterion | Capped | Position sizing never exceeds % of bankroll |

---

## Monitoring

### Check logs on the VM

```bash
gcloud compute ssh tennis-trading-vm --zone=europe-west1-b

# View trading server logs
docker logs -f trading-server

# View betfair loop logs
docker logs -f betfair-loop
```

### GCP Cloud Logging

```bash
# View logs from your machine
gcloud logging read "resource.type=gce_instance" --limit=50
```

---

## Cost Estimate

| Component | Monthly Cost |
|-----------|-------------|
| `e2-medium` VM (EU) | ~$25 |
| 30GB SSD disk | ~$5 |
| Artifact Registry | ~$1 |
| Network egress | ~$2 |
| **Total** | **~$33/month** |

To save money when not trading:
```bash
# Stop VM (keeps disk, stops billing for compute)
gcloud compute instances stop tennis-trading-vm --zone=europe-west1-b

# Start when needed
gcloud compute instances start tennis-trading-vm --zone=europe-west1-b
```

---

## Troubleshooting

### "Login failed"
- Check `BETFAIR_USERNAME`, `BETFAIR_PASSWORD`, `BETFAIR_APP_KEY` are correct
- Verify SSL cert is uploaded to Betfair account
- Ensure cert files are in the `certs/` directory

### "No markets found"
- Check the match has started or is scheduled on Betfair
- Try specifying `--market-id` directly from the Betfair website
- Tennis markets appear ~24h before the match

### "Insufficient liquidity"
- Lower `min_liquidity` in `ExecutionConfig`
- In-play markets have much more liquidity than pre-match
- Wait for the match to go in-play

### High latency
- Verify VM is in `europe-west1` (Belgium) or `europe-west2` (London)
- Use streaming API (default) not REST polling
- Check: `ping exchange.betfair.com` from the VM should be <10ms
