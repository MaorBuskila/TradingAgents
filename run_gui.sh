#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "========================================"
echo "   Starting TradingAgents GUI Stack"
echo "========================================"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
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

cleanup() {
  echo ""
  echo "Shutting down services..."
  kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
  exit
}

trap cleanup SIGINT SIGTERM

echo "Starting FastAPI backend on http://127.0.0.1:8000 ..."
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
(cd "$ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

echo "========================================"
echo "   GUI is running!"
echo "   Open:     http://localhost:5173"
echo "   API docs: http://127.0.0.1:8000/docs"
echo "   Press Ctrl+C to stop both services."
echo "========================================"

wait "$BACKEND_PID" "$FRONTEND_PID"
