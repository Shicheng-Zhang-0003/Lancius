#!/usr/bin/env bash
# physics_gate.sh -- the reproducible entry point for `make check-physics`.
#
# Two arms must agree before this gate is green:
#
#   1. tools/audit/physics_probe.c   -- compiled with the project's own flags
#      (-O3 -mavx2 -mfma -fopenmp -std=c11) and run against liblancius.a. It
#      records what THIS machine's CPU and C library actually produce for the
#      kernels, constants, and file format Lancius depends on.
#
#   2. tools/audit/physics_oracle.py -- re-derives every one of those numbers
#      from sources that do not know Lancius exists: IEEE 754-2019, ISO/IEC
#      60929 decimal arithmetic, zlib.crc32 / IEEE 802.3, and the papers the
#      equations come from. It never imports or executes Lancius.
#
# The gate is green only when the machine's behaviour matches the independent
# reference on every check. No network, no torch, no numpy -- python3 stdlib
# and the project's own toolchain are enough, so it runs in CI and offline.
set -euo pipefail
cd "$(dirname "$0")/../.."

BUILD=temp/audit/physics
OUT=temp/audit/physics_probe_out.txt
mkdir -p temp/audit

# ---------------------------------------------------------------- build probe
# Same flag set the project builds its own objects with, so the probe measures
# the code that actually ships (including any -ffp-contract behaviour).
CC=${CC:-gcc}
CFLAGS="-Wall -Wextra -g -Werror -O3 -mavx2 -mfma -fopenmp -std=c11"
if ! $CC $CFLAGS -I./include -o "$BUILD" \
        tools/audit/physics_probe.c build/lib/liblancius.a -lm -fopenmp -lpthread; then
    echo "[physics-gate] FAIL: probe did not compile" >&2
    exit 1
fi

# ---------------------------------------------------------------- run probe
if ! "./$BUILD" > "$OUT"; then
    echo "[physics-gate] FAIL: probe crashed or returned nonzero" >&2
    exit 1
fi

# ---------------------------------------------------------------- oracle
python3 tools/audit/physics_oracle.py "$OUT"
rc=$?

if [ "$rc" -eq 0 ]; then
    echo "[physics-gate] PASS: machine behaviour matches the independent reference"
else
    echo "[physics-gate] FAIL: a physics/numerics claim is false (see above)" >&2
fi
exit "$rc"
