#!/bin/bash

echo "🎯 Starting Touring Agent Demo"
echo "🔇 Silencing WebSocket debug logs for clean output"
echo "📞 Important logs will show: TRANSCRIPT, AGENT THINKING, AGENT RESPONSE"
echo "=" * 60

# Start uvicorn with minimal logging
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level warning \
  --access-log \
  --no-server-header \
  2>&1 | grep -v "DEBUG:" | grep -v "connection open" | grep -v "connection closed"
