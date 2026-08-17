#!/usr/bin/env bash
# Bring the whole system back up. Everything is resumable, so this is safe to re-run.
#
#   ./scripts/start_all.sh          reader + crawl + embed
#   ./scripts/start_all.sh --llm    also start vLLM (needs working NVML)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LARA_VENV:-/home/user/Desktop/Learned-Data-Selection/venv}"
LOGS="$ROOT/data/logs"
mkdir -p "$LOGS"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

running() { pgrep -f "bin/[l]ara $1" >/dev/null; }

launch() {
  local name="$1"; shift
  if running "$name"; then echo "  $name already running"; return; fi
  PYTHONUNBUFFERED=1 setsid nohup "$@" > "$LOGS/$name.log" 2>&1 < /dev/null &
  disown
  echo "  $name started -> $LOGS/$name.log"
}

echo "== preflight =="
lara preflight || echo "  (preflight reported problems; continuing)"

if [[ "${1:-}" == "--llm" ]]; then
  echo "== vLLM =="
  if python -c "import pynvml; pynvml.nvmlInit()" 2>/dev/null; then
    launch "serve-llm" lara serve-llm
    echo "  waiting for vLLM..."
    for _ in $(seq 1 60); do
      curl -s --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q '"id"' && { echo "  vLLM ready"; break; }
      sleep 10
    done
  else
    echo "  SKIPPED: NVML still broken (pynvml.nvmlInit failed)."
    echo "  vLLM detects devices via NVML, so it cannot start. Reboot to load the"
    echo "  matching kernel module; torch/embedding/reranking are unaffected."
  fi
fi

echo "== ingest =="
launch crawl lara crawl
launch embed lara embed --device cuda:2

echo "== reader =="
launch serve lara serve
for _ in $(seq 1 40); do
  curl -s --max-time 3 http://127.0.0.1:8080/api/health 2>/dev/null | grep -q '"ready": *true' && break
  sleep 10
done
curl -s http://127.0.0.1:8080/api/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  (server still warming)"

echo
echo "reader:  http://127.0.0.1:8080"
echo "status:  lara status"
