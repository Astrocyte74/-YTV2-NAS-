#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENABLE_TTS_QUEUE_WORKER="${ENABLE_TTS_QUEUE_WORKER:-1}"
TTS_QUEUE_INTERVAL="${TTS_QUEUE_INTERVAL:-30}"

echo "[entrypoint] Starting services (ENABLE_TTS_QUEUE_WORKER=${ENABLE_TTS_QUEUE_WORKER}, INTERVAL=${TTS_QUEUE_INTERVAL}s)"

# Start Telegram bot
${PYTHON_BIN} telegram_bot.py &
BOT_PID=$!
echo "[entrypoint] Telegram bot PID=${BOT_PID}"

# Optionally start the TTS queue worker in watch mode
if [ "${ENABLE_TTS_QUEUE_WORKER}" = "1" ]; then
  ${PYTHON_BIN} tools/drain_tts_queue.py --watch --interval "${TTS_QUEUE_INTERVAL}" &
  WORKER_PID=$!
  echo "[entrypoint] TTS queue worker PID=${WORKER_PID}"
else
  WORKER_PID=""
fi

term() {
  echo "[entrypoint] Caught signal, shutting down..."
  kill -TERM "${BOT_PID}" 2>/dev/null || true
  if [ -n "${WORKER_PID}" ]; then
    kill -TERM "${WORKER_PID}" 2>/dev/null || true
  fi
  wait || true
}
trap term INT TERM

# Wait for either process to exit, then terminate the other (POSIX sh, no wait -n)
if [ -n "${WORKER_PID}" ]; then
  # Poll until one of the PIDs exits
  while :; do
    if ! kill -0 "${BOT_PID}" 2>/dev/null; then
      echo "[entrypoint] Bot process exited"
      break
    fi
    if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
      echo "[entrypoint] Worker process exited"
      break
    fi
    sleep 1
  done
else
  # Only the bot is running; wait for it
  wait "${BOT_PID}" || true
fi

echo "[entrypoint] Process exited; stopping remaining services"
term
echo "[entrypoint] Exit"
