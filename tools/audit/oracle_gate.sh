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
LOG="$OUT/oracle_gate.log"
rm -f "$LOG"
python3 "$ROOT/tools/audit/kernel_oracle.py" "$OUT/dumps" 2>&1 | tee -a "$LOG" || rc=1
python3 "$ROOT/tools/audit/graph_oracle.py"  "$OUT/dumps" 2>&1 | tee -a "$LOG" || rc=1
python3 "$ROOT/tools/audit/trainlib_oracle.py" "$OUT/dumps/trainlib" 2>&1 | tee -a "$LOG" || rc=1
# conv2d and the reduction/softmax conventions, against torch specifically:
# a self-authored NumPy reference shares the author with the kernel.
python3 "$ROOT/tools/audit/conv_oracle.py"  "$OUT/dumps" 2>&1 | tee -a "$LOG" || rc=1
python3 "$ROOT/tools/audit/math_oracle.py"  "$OUT/dumps" 2>&1 | tee -a "$LOG" || rc=1

# Despot V9: a SKIP is not a pass. kernel/graph degrade to NumPy+closed form
# when torch is absent (loud SKIP); conv/math/trainlib hard-require torch.
# CI installs torch, so any skip there means weaker evidence and must fail.
# Local runs without torch stay green but report skips loudly.
_nskips=$(grep -cE "^  SKIP" "$LOG" || true)
if [ "${_nskips:-0}" -gt 0 ]; then
  if python3 -c "import torch" 2>/dev/null; then
    echo "EXTERNAL ORACLE GATE FAILED: ${_nskips} check(s) skipped despite torch being present."
    echo "A skipped check is a check that did not happen; with torch installed 0 skips are required."
    exit 1
  else
    echo "WARNING: ${_nskips} check(s) skipped (torch absent). Gate ran on weaker evidence."
  fi
fi

if [ "$rc" -ne 0 ]; then
  echo "EXTERNAL ORACLE GATE FAILED: the library disagrees with NumPy/PyTorch/closed form."
  exit 1
fi
echo "External oracle gate green."