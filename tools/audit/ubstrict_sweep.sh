#!/usr/bin/env bash
# Strict UndefinedBehaviorSanitizer pass.
#
# The combined check-sanitizers gate runs ASan+UBSan together, where a UB
# report is a warning rather than a failure. This pass runs UBSan ALONE with
# -fno-sanitize-recover=all (abort on the first finding) and the checks that
# only exist in standalone mode: signed-integer-overflow, shift, bool, enum,
# float-cast-overflow, integer-divide-by-zero, object-size, bounds.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/temp/ubstrict"
mkdir -p "$OUT/obj" "$OUT/logs"

CC="${CC:-gcc}"
SAN="-fsanitize=undefined,integer-divide-by-zero,float-cast-overflow,bounds,alignment,null,vptr,object-size,shift,signed-integer-overflow,bool,enum -fno-sanitize-recover=all"
CFLAGS="-Wall -Wextra -g -O1 -fno-omit-frame-pointer $SAN -fopenmp -std=c11 -I${ROOT}/include -fPIC"
LDFLAGS="$SAN -fopenmp -lm -lpthread"

echo "=== instrumenting the library for strict UBSan ==="
rm -rf "$OUT/obj"; mkdir -p "$OUT/obj"
for f in "$ROOT"/src/*/*.c; do
  $CC $CFLAGS -c "$f" -o "$OUT/obj/$(basename "$f" .c).o" || {
    echo "BUILD FAILED: $f"; grep -m3 error "$OUT/logs/$(basename "$f" .c).log"; exit 2; }
done
ar rcs "$OUT/libub.a" "$OUT"/obj/*.o

AUDITS="stress_test test_torture fuzz_lancius test_path_bg test_grad_check audit_internals audit_ffi \
audit_threadpool_parity audit_nan_injection audit_memory_pool test_diamond_memory \
audit_flash_attention audit_modern_llm audit_known_answer audit_regression_13c \
audit_transformer_known_answer audit_fp32_path audit_fault_injection \
audit_despot_probe audit_train_lib audit_sum_axis_nd audit_train_bwd \
audit_train_converge audit_sandbox audit_v7_hardening train_micromodel \
eval_verifier train_verifier_head"

echo "=== running under -fno-sanitize-recover=all ==="
pass=0; fails=0
for a in $AUDITS; do
  $CC $CFLAGS -o "$OUT/$a" "$ROOT/examples/$a.c" "$OUT/libub.a" $LDFLAGS 2>"$OUT/logs/$a.build" || {
    echo "BUILD FAILED: $a"; fails=$((fails+1)); continue; }
  ( cd "$ROOT" && UBSAN_OPTIONS=print_stacktrace=1 "$OUT/$a" ) >"$OUT/logs/$a.run" 2>&1
  rc=$?
  n=$(grep -c "runtime error" "$OUT/logs/$a.run" 2>/dev/null | head -1); n=${n:-0}
  if [ "$rc" -ne 0 ] || [ "$n" -ne 0 ]; then
    echo "FAIL $a (exit=$rc ubsan_reports=$n)"
    grep -m3 "runtime error" "$OUT/logs/$a.run" | sed 's/^/     /'
    fails=$((fails+1))
  else
    pass=$((pass+1))
  fi
done
echo
echo "STRICT UBSAN: $pass clean, $fails with findings"
[ "$fails" -eq 0 ] || exit 1