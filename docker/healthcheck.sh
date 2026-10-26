#!/bin/sh
# Container health: GET /health (no API key needed). Healthy only when the server answers and says it is ok.
# While the model downloads, packs, loads (hours the first time) or warms up (LLMSERVE_WARMUP: the server opens its port only
# when the warm-up requests are done) the container is "starting" (see HEALTHCHECK --start-period in the Dockerfile), not
# "unhealthy".
url="http://127.0.0.1:${LLMSERVE_PORT:-8080}/health"
body=$(curl -fsS --max-time 5 "$url" 2>/dev/null) || { echo "not ready: $url does not answer (preparing, loading or warming up the model)" >&2; exit 1; }
case "$body" in
  *'"status": "ok"'*|*'"status":"ok"'*) exit 0 ;;
  *) echo "not healthy: $body" >&2; exit 1 ;;
esac
