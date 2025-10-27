#!/usr/bin/env sh
set -eu

# Backup Portainer configuration/state (either named volume or bind mount).
# Usage: PORTAINER_CNAME=PortainerCE ./tools/backup_portainer.sh [backup_dir]

PORTAINER_CNAME="${PORTAINER_CNAME:-PortainerCE}"
OUT_DIR="${1:-backups}"
TS="$(date +%F_%H%M%S)"

mkdir -p "$OUT_DIR"

echo "[backup_portainer] Container: $PORTAINER_CNAME"

if ! docker ps --format '{{.Names}}' | grep -qx "$PORTAINER_CNAME"; then
  echo "[backup_portainer] Container '$PORTAINER_CNAME' not running or not found." >&2
  exit 1
fi

# Inspect mounts (no jq dependency). Format: TYPE NAME SOURCE DEST
MOUNTS=$(docker inspect "$PORTAINER_CNAME" --format '{{range .Mounts}}{{.Type}}|{{.Name}}|{{.Source}}|{{.Destination}}{{printf "\n"}}{{end}}')

VOL_NAME=""
BIND_SRC=""
echo "$MOUNTS" | while IFS='|' read -r TYPE NAME SOURCE DEST; do
  [ -z "$TYPE" ] && continue
  if [ "$DEST" = "/data" ]; then
    if [ "$TYPE" = "volume" ] && [ -n "$NAME" ]; then
      VOL_NAME="$NAME"
      echo "volume:$VOL_NAME"
      exit 0
    fi
    if [ "$TYPE" = "bind" ] && [ -n "$SOURCE" ]; then
      BIND_SRC="$SOURCE"
      echo "bind:$BIND_SRC"
      exit 0
    fi
  fi
done | {
  MODE=""
  TARGET=""
  read MODE TARGET || true
  case "$MODE" in
    volume:*)
      VOL_NAME=${TARGET#volume:}
      TAR="$OUT_DIR/portainer_data_${TS}.tar.gz"
      echo "[backup_portainer] Backing up volume '$VOL_NAME' → $TAR"
      docker run --rm -v "$VOL_NAME":/data -v "$PWD/$OUT_DIR":/backup alpine sh -c "cd / && tar czf /backup/$(basename \"$TAR\") data"
      echo "[backup_portainer] Saved: $TAR"
      ;;
    bind:*)
      SRC=${TARGET#bind:}
      TAR="$OUT_DIR/portainerCE_data_${TS}.tar.gz"
      echo "[backup_portainer] Backing up bind directory '$SRC' → $TAR"
      tar czf "$TAR" -C "$(dirname "$SRC")" "$(basename "$SRC")"
      echo "[backup_portainer] Saved: $TAR"
      ;;
    *)
      echo "[backup_portainer] Could not locate a /data mount on $PORTAINER_CNAME" >&2
      exit 1
      ;;
  esac
}

echo "[backup_portainer] Done."

