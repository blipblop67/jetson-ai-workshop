#!/bin/bash
# =============================================================================
# ENVIRONMENT GUARD - keep the workshop Python environment safe on Day 2
# =============================================================================
#   bash setup/env_guard.sh lock      # save a baseline + make the venv READ-ONLY (pip install fails harmlessly)
#   bash setup/env_guard.sh check     # what changed since the baseline? + health check
#   bash setup/env_guard.sh unlock    # allow changes again (trainer only)
#   bash setup/env_guard.sh restore   # put back the exact baseline versions (needs internet), then lock again
#
# Run it from anywhere; it works on ~/ai_workshop/env.
# Note: this protects the workshop venv. It can't stop "sudo pip install", so tell participants not to use sudo.
# =============================================================================
set -e
WS="$HOME/ai_workshop"
VENV="$WS/env"
PIP="$VENV/bin/pip"
BASE="$WS/.env_baseline.txt"
SITE="$("$VENV/bin/python3" -c 'import site; print(site.getsitepackages()[0])')"
JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/"

[ -x "$PIP" ] || { echo "No venv at $VENV - create it first (see ~/ai_workshop/README.md)"; exit 1; }

case "$1" in
  lock)
    if [ ! -f "$BASE" ] || [ "$2" == "--new-baseline" ]; then
        "$PIP" freeze --local > "$BASE"
        echo "Baseline saved: $BASE ($(wc -l < "$BASE") packages)"
    else
        echo "Using existing baseline: $BASE   (add --new-baseline to replace it)"
    fi
    chmod -R a-w "$SITE" "$VENV/bin"
    echo "LOCKED: the workshop environment is now read-only."
    echo "  pip install / uninstall inside the venv will fail with 'Permission denied' - nothing breaks."
    ;;
  unlock)
    chmod -R u+w "$SITE" "$VENV/bin"
    echo "UNLOCKED: the environment can be changed again. Run 'lock' when you're done."
    ;;
  check)
    if [ -f "$BASE" ]; then
        NOW=$(mktemp); "$PIP" freeze --local > "$NOW"
        if diff -q "$BASE" "$NOW" > /dev/null; then
            echo "Packages: identical to the baseline."
        else
            echo "Packages changed since the baseline:"
            diff "$BASE" "$NOW" | grep '^[<>]' | sed 's/^</  removed\/old:/; s/^>/  added\/new:  /'
            echo "Fix with:  bash setup/env_guard.sh restore"
        fi
        rm -f "$NOW"
    else
        echo "No baseline yet - run 'lock' first."
    fi
    [ -w "$SITE" ] && echo "State: UNLOCKED" || echo "State: LOCKED (read-only)"
    echo
    "$VENV/bin/python3" "$WS/setup/jetson_check.py"
    ;;
  restore)
    [ -f "$BASE" ] || { echo "No baseline found ($BASE)."; exit 1; }
    chmod -R u+w "$SITE" "$VENV/bin"
    NOW=$(mktemp); "$PIP" freeze --local > "$NOW"
    EXTRA=$(comm -13 <(sed 's/[=@ ].*//' "$BASE" | sort -u) <(sed 's/[=@ ].*//' "$NOW" | sort -u))
    if [ -n "$EXTRA" ]; then
        echo "Removing packages that weren't in the baseline: $EXTRA"
        "$PIP" uninstall -y $EXTRA
    fi
    echo "Re-installing the exact baseline versions..."
    "$PIP" install --no-deps -r "$BASE" --extra-index-url "$JETSON_INDEX"
    rm -f "$NOW"
    chmod -R a-w "$SITE" "$VENV/bin"
    echo "RESTORED and LOCKED."
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1 ;;
esac
