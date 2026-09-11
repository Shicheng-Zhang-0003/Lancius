#!/usr/bin/env bash
# Compile gate: builds the library after each phase script.
set -u
cd "$(dirname "$0")/../.."   # repo root

echo "[verify_build] make liblancius.a"
make liblancius.a 2>&1
rc=$?

if [ $rc -ne 0 ]; then
    echo "[verify_build] BUILD FAILED (exit $rc)"
else
    echo "[verify_build] BUILD OK"
fi
exit $rc
