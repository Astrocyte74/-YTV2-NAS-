# Backup Guide (NAS + Portainer)

This doc summarizes what we backed up and how to repeat it quickly.

## What’s already backed up (this repo, backups/)

Files (timestamps will vary):
- Container env snapshots
  - `backups/env_youtube-summarizer-bot_YYYY-MM-DD_HHMMSS.env`
  - `backups/container_env_runtime_YYYY-MM-DD_HHMMSS.txt`
- Compose + env files
  - `backups/docker-compose_YYYY-MM-DD_HHMMSS.yml`
  - `backups/.env.nas_YYYY-MM-DD_HHMMSS`
  - `backups/stack.env_YYYY-MM-DD_HHMMSS` (if present)
  - `backups/runtime.env_YYYY-MM-DD_HHMMSS` (if present)
- Portainer data archive
  - If Portainer uses a bind mount: `backups/portainerCE_data_YYYY-MM-DD_HHMMSS.tar.gz`
  - If Portainer uses a volume: `backups/portainer_data_YYYY-MM-DD_HHMMSS.tar.gz`

These are sufficient to reconstruct the stack and (optionally) Portainer’s UI state.

## Quick snapshot (env + compose)

Use the helper script to capture the current container env and local config files:

```
CONTAINER=youtube-summarizer-bot ./tools/backup_env.sh
```

Outputs land in `backups/` with a timestamp.

## Full Portainer state backup

Back up the `/data` mount of the Portainer container (either a named volume or a bind directory). The script auto-detects which one is used:

```
PORTAINER_CNAME=PortainerCE ./tools/backup_portainer.sh
```

The tarball will be saved to `backups/`.

Security note: Portainer archives contain secrets (API keys, endpoint creds). Store securely.

## Restore

### From Compose + env
1. Copy `docker-compose_*.yml` and `.env.nas_*` from backups to a working directory
2. Run:
   - `docker compose -f docker-compose_*.yml --env-file .env.nas_* up -d`
3. Verify with `/status` and `/diag`

### Portainer state
If Portainer uses a bind dir (example `/share/Docker/PortainerCE/data`):
1. Stop `PortainerCE`
2. Extract the archive:
   - `tar xzf backups/portainerCE_data_*.tar.gz -C /share/Docker/PortainerCE`
3. Start `PortainerCE` and log in

If Portainer uses a named volume:
1. Create the volume (if needed):
   - `docker volume create portainer_data`
2. Restore:
   - `docker run --rm -v portainer_data:/data -v "$PWD/backups":/backup alpine sh -c 'cd / && tar xzf /backup/portainer_data_*.tar.gz'`
3. Start Portainer and log in

## Notes

- You can also export a Portainer stack’s Compose YAML via the UI (Stacks → Duplicate/Edit → Copy as text) and save it to `backups/`.
- For regular backups, consider a cron job calling `./tools/backup_env.sh` daily and `./tools/backup_portainer.sh` weekly.

