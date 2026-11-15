#!/usr/bin/env bash
# TrackEval (git master, 12c8791) still uses np.float / np.int / np.bool,
# which were removed in NumPy 1.24+. Rewrite them to the Python builtins in
# the installed package so it runs on NumPy 2.x. Idempotent.
# Usage: bash env/patch_trackeval.sh [venv]   (default ~/talos/.venv)
set -euo pipefail
VENV="${1:-$HOME/talos/.venv}"
PKG="$("$VENV/bin/python" -c 'import os, importlib.util as u; print(os.path.dirname(u.find_spec("trackeval").origin))')"
grep -rlE '\bnp\.(float|int|bool|object|str)\b' --include='*.py' "$PKG" \
  | xargs -r perl -pi -e 's/\bnp\.(float|int|bool|object|str)\b/$1/g'
echo "patched: $PKG"
grep -rnE '\bnp\.(float|int|bool|object|str)\b' --include='*.py' "$PKG" && exit 1 || echo "no deprecated aliases left"
