#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[devcontainer-initialize] $1"
}

# When running inside WSL, VS Code may attempt to bind mount the Wayland socket
# from /run/user/${UID}/wayland-0 even though the real socket lives under
# /mnt/wslg/runtime-dir. Creating a symlink ahead of time prevents the docker run
# from failing when the mount source is missing.
if grep -qi microsoft /proc/version 2>/dev/null && [ -d "/mnt/wslg/runtime-dir" ]; then
  uid_dir="/run/user/${UID:-1000}"
  source_socket="/mnt/wslg/runtime-dir/wayland-0"
  target_socket="${uid_dir}/wayland-0"

  if [ -S "${source_socket}" ]; then
    mkdir -p "${uid_dir}"
    if [ ! -e "${target_socket}" ]; then
      ln -s "${source_socket}" "${target_socket}"
      log "Created symlink ${target_socket} -> ${source_socket} for Wayland socket."
    fi
  else
    log "WSL Wayland socket not found at ${source_socket}; skipping symlink."
  fi
else
  log "Non-WSL environment detected; no host adjustments required."
fi
