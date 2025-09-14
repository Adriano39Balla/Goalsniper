#!/usr/bin/env bash
set -euo pipefail

# Railway sets PORT, but default to 8080 if it doesn't
export PORT="${PORT:-8080}"

echo "Starting gunicorn on port ${PORT}..."
exec gunicorn -w 2 -k gthread -t 120 --bind "0.0.0.0:${PORT}" main:app
