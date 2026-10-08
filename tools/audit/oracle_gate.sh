#!/usr/bin/env bash
# External-oracle gate.
#
# Recomputes every Lancius kernel and every graph-level op from an INDEPENDENT
# source -- NumPy, PyTorch autograd, hand-derived closed form, and central
# differences on the definition of the forward op -- then compares. Nothing on
# the oracle side links or imports Lancius; the C probes only produce numbers.
#
# Fails nonzero on any divergence. Requires numpy; torch is used when present
# as a second independent engine.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/temp/oracle"
mkdir -p "$OUT/build" "$OUT/dumps"

CC="${CC:-gcc}"
CFLAGS="-Wall -Wextra -O2 -fopenmp -std=c11 -I${ROOT}/include"
LDFLAGS="-fopenmp -lm -lpthread"

echo "=== building the dump probes against the real library ==="
make -C "$ROOT" liblancius.a >/dev/null
for p in kernel_oracle_dump graph_oracle_dump autodiff_leak_probe trainlib_dump; do
  $CC $CFLAGS -c "$ROOT/tools/audit/$p.c" -o "$OUT/build/$p.o"
  $CC -o "$OUT/build/$p" "$OUT/build/$p.o" "$ROOT/liblancius.a" $LDFLAGS
done

echo "=== producing dumps ==="
rm -f "$OUT/dumps"/*
"$OUT/build/kernel_oracle_dump" "$OUT/dumps" >/dev/null
"$OUT/build/graph_oracle_dump"  "$OUT/dumps" >/dev/null
# train-lib layer: the four entry points section 17 never checked against a
# primary source (sgd, sgdm, clip_grad_norm, lr_warmup_cosine)
make -C "$ROOT" trainlib_dump >/dev/null
"$ROOT/trainlib_dump" "$OUT/dumps/trainlib" >/dev/null

echo "=== independent NumPy / PyTorch / finite-difference comparison ==="
rc=0
python3 "$ROOT/tools/audit/kernel_oracle.py" "$OUT/dumps" || rc=1
python3 "$ROOT/tools/audit/graph_oracle.py"  "$OUT/dumps" || rc=1
python3 "$ROOT/tools/audit/trainlib_oracle.py" "$OUT/dumps/trainlib" || rc=1

if [ "$rc" -ne 0 ]; then
  echo "EXTERNAL ORACLE GATE FAILED: the library disagrees with NumPy/PyTorch/closed form."
  exit 1
fi
echo "External oracle gate green."