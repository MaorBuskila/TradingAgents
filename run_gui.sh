#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "========================================"
echo "   Starting TradingAgents GUI Stack"
echo "========================================"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  PY="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  PY="python"
fi

echo "Using Python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown))"

if ! "$PY" -c "import uvicorn" 2>/dev/null; then
  echo ""
  echo "Missing API dependencies (uvicorn). From the project root, run:"
  echo "  pip install -e ."
  echo "That installs the same package as \`tradingagents\` plus the GUI server."
  echo ""
  exit 1
fi

# Fail fast: catch TypeScript/build errors before starting services
echo "Checking frontend for build errors..."
if ! (cd "$ROOT/frontend" && npm run build -- --mode development 2>&1); then
  echo ""
  echo "ERROR: Frontend build failed. Fix the errors above before running."
  exit 1
fi
echo "Frontend OK."
echo ""

echo "Starting FastAPI backend on http://127.0.0.1:8000 ..."
# Kill any stale process holding port 8000 before starting
if lsof -ti tcp:8000 >/dev/null 2>&1; then
  echo "Port 8000 in use — killing stale process..."
  lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true
  sleep 0.5
fi
"$PY" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

for _ in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:8000/docs" >/dev/null 2>&1; then
    echo "Backend is up."
    break
  fi
  sleep 0.25
done

if ! curl -sf "http://127.0.0.1:8000/docs" >/dev/null 2>&1; then
  echo "ERROR: Backend did not start (check errors above). Is port 8000 free?"
  kill "$BACKEND_PID" 2>/dev/null || true
  exit 1
fi

echo "Starting Vite frontend (proxies /api -> backend) ..."
(cd "$ROOT/frontend" && npm run dev 2>&1) &
FRONTEND_PID=$!

# Start Telegram bot with auto-restart if token is configured
BOT_PID=""
_BOT_STOP=0
if grep -q "^TELEGRAM_BOT_TOKEN=.\+" "$ROOT/.env" 2>/dev/null; then
  _bot_watchdog() {
    while [[ "$_BOT_STOP" -eq 0 ]]; do
      echo "[bot] Starting Telegram bot ..."
      "$PY" -m telegram_bot &
      BOT_PID=$!
      wait "$BOT_PID" || true
      [[ "$_BOT_STOP" -eq 1 ]] && break
      echo "[bot] Crashed or exited — restarting in 5s ..."
      sleep 5
    done
  }
  _bot_watchdog &
  BOT_WATCHDOG_PID=$!
else
  echo "TELEGRAM_BOT_TOKEN not set in .env — skipping Telegram bot."
  BOT_WATCHDOG_PID=""
fi

cleanup() {
  echo ""
  echo "Shutting down services..."
  _BOT_STOP=1
  kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" "${BOT_PID:-}" "${BOT_WATCHDOG_PID:-}" 2>/dev/null || true
  exit
}
trap cleanup SIGINT SIGTERM

echo "========================================"
echo "   GUI is running!"
echo "   Open:     http://localhost:5173"
echo "   API docs: http://127.0.0.1:8000/docs"
echo "   Press Ctrl+C to stop all services."
echo "========================================"

wait "$BACKEND_PID" "$FRONTEND_PID" ${BOT_WATCHDOG_PID:+"$BOT_WATCHDOG_PID"}
