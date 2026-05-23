#!/usr/bin/env bash
# One-shot server setup for Oracle Cloud Ubuntu 22.04 aarch64.
# Run once after cloning the repo. Idempotent — safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON=python3
SERVICE_USER="${SUDO_USER:-ubuntu}"

echo "========================================"
echo "  TradingAgents Server Setup"
echo "========================================"

# ── System packages ────────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
  python3-pip python3-venv python3-dev \
  build-essential git curl \
  sqlite3 libsqlite3-dev
# nodejs/npm: install via nodesource only if not already present
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y -qq nodejs
else
  echo "  node $(node --version) / npm $(npm --version) already present — skipping nodesource"
fi

# ── Python venv ────────────────────────────────────────────────────────────────
echo "[2/6] Setting up Python virtual environment..."
if [[ ! -d "$ROOT/.venv" ]]; then
  $PYTHON -m venv "$ROOT/.venv"
fi
source "$ROOT/.venv/bin/activate"
pip install --upgrade pip -q
pip install -e . -q
pip install uvicorn fastapi -q   # ensure API extras present

# ── Frontend ───────────────────────────────────────────────────────────────────
echo "[3/6] Building frontend..."
(cd "$ROOT/frontend" && npm install --silent && npm run build)
echo "Frontend built."

# ── Required directories ───────────────────────────────────────────────────────
echo "[4/6] Creating required directories..."
mkdir -p "$ROOT/data"
mkdir -p "$ROOT/eval_results"

# ── .env check ─────────────────────────────────────────────────────────────────
echo "[5/7] Checking .env..."
if [[ ! -f "$ROOT/.env" ]]; then
  echo ""
  echo "  WARNING: No .env file found."
  echo "  Copy your .env to $ROOT/.env before starting services."
  echo "  Example: scp .env ubuntu@<server-ip>:~/TradingAgents/.env"
  echo ""
fi

# ── systemd: FastAPI backend ───────────────────────────────────────────────────
echo "[6/7] Installing systemd service: tradingagents-api..."
sudo tee /etc/systemd/system/tradingagents-api.service > /dev/null <<EOF
[Unit]
Description=TradingAgents FastAPI Backend
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$ROOT
EnvironmentFile=$ROOT/.env
ExecStart=$ROOT/.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# ── systemd: Telegram bot ──────────────────────────────────────────────────────
echo "[7/7] Installing systemd service: tradingagents-bot..."
sudo tee /etc/systemd/system/tradingagents-bot.service > /dev/null <<EOF
[Unit]
Description=TradingAgents Telegram Bot
After=network.target tradingagents-api.service

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$ROOT
EnvironmentFile=$ROOT/.env
ExecStart=$ROOT/.venv/bin/python -m telegram_bot
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# ── OS firewall: allow port 8000 (must be BEFORE the REJECT rule) ─────────────
sudo iptables -D INPUT -p tcp -m state --state NEW -m tcp --dport 8000 -j ACCEPT 2>/dev/null || true
sudo iptables -I INPUT 5 -p tcp -m state --state NEW -m tcp --dport 8000 -j ACCEPT 2>/dev/null || true

# ── Enable & start services ────────────────────────────────────────────────────
sudo systemctl daemon-reload
sudo systemctl enable tradingagents-api tradingagents-bot
sudo systemctl start tradingagents-api tradingagents-bot
# Persist iptables rules across reboots
if command -v netfilter-persistent &>/dev/null; then
  sudo netfilter-persistent save
elif command -v iptables-save &>/dev/null; then
  sudo mkdir -p /etc/iptables
  sudo iptables-save | sudo tee /etc/iptables/rules.v4 > /dev/null
fi

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  API:         http://$(curl -s ifconfig.me 2>/dev/null || echo '<your-ip>'):8000"
echo "  API docs:    http://$(curl -s ifconfig.me 2>/dev/null || echo '<your-ip>'):8000/docs"
echo ""
echo "  Service status:"
sudo systemctl is-active tradingagents-api  && echo "  tradingagents-api  ✓ running" || echo "  tradingagents-api  ✗ check: sudo journalctl -u tradingagents-api -n 50"
sudo systemctl is-active tradingagents-bot  && echo "  tradingagents-bot  ✓ running" || echo "  tradingagents-bot  ✗ check: sudo journalctl -u tradingagents-bot -n 50"
echo ""
echo "  Useful commands:"
echo "    sudo systemctl status tradingagents-api"
echo "    sudo systemctl status tradingagents-bot"
echo "    sudo journalctl -u tradingagents-api -f"
echo "    sudo journalctl -u tradingagents-bot -f"
echo "========================================"
