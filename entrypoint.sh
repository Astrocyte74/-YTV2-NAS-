#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENABLE_TTS_QUEUE_WORKER="${ENABLE_TTS_QUEUE_WORKER:-1}"
TTS_QUEUE_INTERVAL="${TTS_QUEUE_INTERVAL:-30}"
ENABLE_IMAGE_QUEUE_WORKER="${ENABLE_IMAGE_QUEUE_WORKER:-1}"
IMAGE_QUEUE_INTERVAL="${IMAGE_QUEUE_INTERVAL:-30}"
ENABLE_ZIMAGE_QUEUE_WORKER="${ENABLE_ZIMAGE_QUEUE_WORKER:-1}"
ZIMAGE_QUEUE_INTERVAL="${ZIMAGE_QUEUE_INTERVAL:-30}"
YTV2_API_ENABLED="${YTV2_API_ENABLED:-false}"
YTV2_API_PORT="${YTV2_API_PORT:-6453}"
YTV2_API_HOST="${YTV2_API_HOST:-127.0.0.1}"

echo "[entrypoint] Starting services (ENABLE_TTS_QUEUE_WORKER=${ENABLE_TTS_QUEUE_WORKER}, TTS_INTERVAL=${TTS_QUEUE_INTERVAL}s, ENABLE_IMAGE_QUEUE_WORKER=${ENABLE_IMAGE_QUEUE_WORKER}, IMG_INTERVAL=${IMAGE_QUEUE_INTERVAL}s, ENABLE_ZIMAGE_QUEUE_WORKER=${ENABLE_ZIMAGE_QUEUE_WORKER}, ZIMG_INTERVAL=${ZIMAGE_QUEUE_INTERVAL}s, YTV2_API_ENABLED=${YTV2_API_ENABLED})"

is_enabled() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

# Start Telegram bot
${PYTHON_BIN} telegram_bot.py &
BOT_PID=$!
echo "[entrypoint] Telegram bot PID=${BOT_PID}"

WORKER_PID=""
IMG_WORKER_PID=""
ZIMG_WORKER_PID=""
API_PID=""

# Optionally start the YTV2 API server
if is_enabled "${YTV2_API_ENABLED}"; then
  echo "[entrypoint] Starting YTV2 API server on ${YTV2_API_HOST}:${YTV2_API_PORT}..."
  ${PYTHON_BIN} -m ytv2_api.main &
  API_PID=$!
  echo "[entrypoint] YTV2 API PID=${API_PID}"
fi

# Optionally start the TTS queue worker in watch mode
if [ "${ENABLE_TTS_QUEUE_WORKER}" = "1" ]; then
  ${PYTHON_BIN} tools/drain_tts_queue.py --watch --interval "${TTS_QUEUE_INTERVAL}" &
  WORKER_PID=$!
  echo "[entrypoint] TTS queue worker PID=${WORKER_PID}"
fi

# Optionally start the Image queue worker in watch mode
if is_enabled "${ENABLE_IMAGE_QUEUE_WORKER}"; then
  ${PYTHON_BIN} tools/drain_image_queue.py --watch --interval "${IMAGE_QUEUE_INTERVAL}" &
  IMG_WORKER_PID=$!
  echo "[entrypoint] Image queue worker PID=${IMG_WORKER_PID}"
fi

# Optionally start the Z-Image queue worker in watch mode
if is_enabled "${ENABLE_ZIMAGE_QUEUE_WORKER}"; then
  ${PYTHON_BIN} tools/drain_zimage_queue.py --watch --interval "${ZIMAGE_QUEUE_INTERVAL}" &
  ZIMG_WORKER_PID=$!
  echo "[entrypoint] Z-Image queue worker PID=${ZIMG_WORKER_PID}"
fi

term() {
  echo "[entrypoint] Caught signal, shutting down..."
  kill -TERM "${BOT_PID}" 2>/dev/null || true
  [ -n "${WORKER_PID}" ] && kill -TERM "${WORKER_PID}" 2>/dev/null || true
  [ -n "${IMG_WORKER_PID}" ] && kill -TERM "${IMG_WORKER_PID}" 2>/dev/null || true
  [ -n "${ZIMG_WORKER_PID}" ] && kill -TERM "${ZIMG_WORKER_PID}" 2>/dev/null || true
  [ -n "${API_PID}" ] && kill -TERM "${API_PID}" 2>/dev/null || true
  wait || true
}
trap term INT TERM

# Wait for either process to exit, then terminate the other (POSIX sh, no wait -n)
if [ -n "${WORKER_PID}" ] || [ -n "${IMG_WORKER_PID}" ] || [ -n "${ZIMG_WORKER_PID}" ] || [ -n "${API_PID}" ]; then
  # Poll until one of the PIDs exits
  while :; do
    if ! kill -0 "${BOT_PID}" 2>/dev/null; then
      echo "[entrypoint] Bot process exited"
      break
    fi
    if [ -n "${WORKER_PID}" ] && ! kill -0 "${WORKER_PID}" 2>/dev/null; then
      echo "[entrypoint] TTS worker exited"
      break
    fi
    if [ -n "${IMG_WORKER_PID}" ] && ! kill -0 "${IMG_WORKER_PID}" 2>/dev/null; then
      echo "[entrypoint] Image worker exited"
      break
    fi
    if [ -n "${ZIMG_WORKER_PID}" ] && ! kill -0 "${ZIMG_WORKER_PID}" 2>/dev/null; then
      echo "[entrypoint] Z-Image worker exited"
      break
    fi
    if [ -n "${API_PID}" ] && ! kill -0 "${API_PID}" 2>/dev/null; then
      echo "[entrypoint] YTV2 API server exited"
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
