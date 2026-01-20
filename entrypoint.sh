#!/bin/sh
set -e

PUID=${PUID:-1000}
PGID=${PGID:-1000}

if getent group "$PGID" >/dev/null 2>&1; then
  GROUP_NAME=$(getent group "$PGID" | cut -d: -f1)
elif getent group "appgroup" >/dev/null 2>&1; then
  groupmod -g "$PGID" appgroup 2>/dev/null || true
  GROUP_NAME="appgroup"
else
  if command -v addgroup >/dev/null 2>&1; then
    addgroup --gid "$PGID" appgroup 2>/dev/null || true
  else
    groupadd -g "$PGID" appgroup 2>/dev/null || true
  fi
  GROUP_NAME="appgroup"
fi

if id -u appuser >/dev/null 2>&1; then
  CURRENT_UID=$(id -u appuser)
  CURRENT_GID=$(id -g appuser)
  if [ "$CURRENT_GID" != "$PGID" ]; then
    usermod -g "$GROUP_NAME" appuser 2>/dev/null || true
  fi
  if [ "$CURRENT_UID" != "$PUID" ]; then
    usermod -u "$PUID" appuser 2>/dev/null || true
  fi
else
  if command -v adduser >/dev/null 2>&1; then
    adduser --disabled-password --gecos "" --uid "$PUID" --ingroup "$GROUP_NAME" appuser 2>/dev/null || true
  else
    useradd -m -u "$PUID" -g "$GROUP_NAME" appuser 2>/dev/null || true
  fi
fi

mkdir -p /tmp/temp_uploads /app/vectorstore /app/cache /app/db_data 2>/dev/null || true

chown "$PUID:$PGID" /app /tmp/temp_uploads /app/vectorstore /app/cache /app/db_data /app/pdfs 2>/dev/null || true

exec gosu appuser "$@"
