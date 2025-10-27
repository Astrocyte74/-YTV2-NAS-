#!/usr/bin/env sh
set -eu

# Simple snapshot of container env + compose/env files into backups/
# Usage: CONTAINER=youtube-summarizer-bot ./tools/backup_env.sh [backup_dir]

CONTAINER="${CONTAINER:-youtube-summarizer-bot}"
OUT_DIR="${1:-backups}"
TS="$(date +%F_%H%M%S)"

mkdir -p "$OUT_DIR"

echo "[backup_env] Target container: $CONTAINER"
echo "[backup_env] Output dir: $OUT_DIR (timestamp=$TS)"

# 1) Save the container's configured env (sorted)
ENV_OUT="$OUT_DIR/env_${CONTAINER}_${TS}.env"
echo "[backup_env] Saving container env → $ENV_OUT"
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" | sort > "$ENV_OUT"

# 2) Save compose/env files if present
if [ -f docker-compose.yml ]; then
  cp docker-compose.yml "$OUT_DIR/docker-compose_${TS}.yml"
  echo "[backup_env] Saved: $OUT_DIR/docker-compose_${TS}.yml"
fi
for f in .env.nas stack.env runtime.env; do
  if [ -f "$f" ]; then
    cp "$f" "$OUT_DIR/$(basename "$f")_${TS}"
    echo "[backup_env] Saved: $OUT_DIR/$(basename "$f")_${TS}"
  fi
done

# 3) Optional: runtime env from inside the container (best-effort)
RUN_OUT="$OUT_DIR/container_env_runtime_${TS}.txt"
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "[backup_env] Capturing runtime env → $RUN_OUT"
  docker exec "$CONTAINER" sh -lc 'env | sort' > "$RUN_OUT" 2>/dev/null || true
fi

echo "[backup_env] Done."

