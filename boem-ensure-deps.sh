#!/usr/bin/env bash
set -euo pipefail

REQS="/pgsql/boem-webapp/BOEM-webapp/requirements.txt"
STAMP="/var/lib/boem-webapp/requirements.sha256"
ENV_PIP="/home/retrieverdash/miniconda3/envs/boem-webapp/bin/pip"

# No requirements file? Nothing to do.
[ -f "$REQS" ] || exit 0

mkdir -p "$(dirname "$STAMP")"

CUR_HASH="$(sha256sum "$REQS" | awk '{print $1}')"
OLD_HASH="$(cat "$STAMP" 2>/dev/null || true)"

if [ "$CUR_HASH" != "$OLD_HASH" ]; then
  "$ENV_PIP" install -r "$REQS"
  echo "$CUR_HASH" > "$STAMP"
fi

