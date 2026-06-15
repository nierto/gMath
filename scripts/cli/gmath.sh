#!/bin/bash
# Thin CLI shim: `geodineum gmath <op> <args>` → the real g_math FASC binary.
# The canonical implementation lives in the Rust binary (src/bin/gmath.rs); this
# resolves the compiled binary and delegates. This is NOT a bc demo — it runs
# the genuine deterministic fixed-point engine.
set -euo pipefail

GEODINEUM_ROOT="${GEODINEUM_ROOT:-/opt/geodineum}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(dirname "$(dirname "$SELF_DIR")")"   # scripts/cli/ → crate root

for cand in \
    "${CRATE_DIR}/target/release/gmath" \
    "${GEODINEUM_ROOT}/gMath/target/release/gmath" \
    "${GEODINEUM_ROOT}/gMath/target/debug/gmath" \
    /usr/local/bin/gmath; do
    if [[ -x "$cand" ]]; then
        exec "$cand" "$@"
    fi
done

cat >&2 <<EOF
gmath: compiled binary not found. Build it:
  (cd ${GEODINEUM_ROOT}/gMath && cargo build --release --bin gmath)
or let the deploy orchestrator build it (gMath is in the deploy set).
EOF
exit 1
