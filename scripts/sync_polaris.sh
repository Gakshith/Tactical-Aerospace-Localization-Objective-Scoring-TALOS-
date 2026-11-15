#!/usr/bin/env bash
# Copy the working tree to Polaris and stamp it with the source commit, so
# results produced there can name the exact code that made them.
#   POLARIS=<user>@polaris.sfsu.edu scripts/sync_polaris.sh
set -euo pipefail
: "${POLARIS:?set POLARIS=<user>@polaris.sfsu.edu}"
cd "$(git rev-parse --show-toplevel)"
{
  echo "git_sha=$(git rev-parse HEAD)"
  echo "git_dirty_files=$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
  echo "synced_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > .talos_source
rsync -az --exclude .git --exclude .venv --exclude artifacts --exclude source_video \
  --exclude docs/media --exclude __pycache__ --exclude logs --exclude results \
  ./ "$POLARIS:talos/repo/"
echo "synced $(cut -d= -f2 <<<"$(head -1 .talos_source)" | cut -c1-7) to $POLARIS"
