#!/usr/bin/env bash
# Sanitizer sweep over EVERY gate binary.
#
# Despot V9: header corrected (was stale: claimed makefile only rebuilds three
# binaries with an uninstrumented library -- true pre-V7, false since V7
# instruments the library in makefile:238-242). This sweep is the redundant
# second opinion: it rebuilds the library AND every binary including the CLI
# and parity runners that make check-sanitizers omits by design (distill is
# standalone, parity/soak are manual runners).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ROOT}/temp/sanitizers"
LOG="${OUT}/logs"
mkdir -p "$OUT/obj" "$LOG"

CC="${CC:-gcc}"
BASE="-Wall -Wextra -g -O1 -fno-omit-frame-pointer -fsanitize=address,undefined -fopenmp -std=c11 -I${ROOT}/include -fPIC"
LINK="-fsanitize=address,undefined -fopenmp -lm -lpthread"
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=0:detect_stack_use_after_return=1:strict_string_checks=1:check_initialization_order=1:detect_invalid_pointer_pairs=2"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=0"
export LSAN_OPTIONS="report_objects=1"

echo "=== building instrumented library ==="
for f in "${ROOT}"/src/*/*.c; do
  o="$OUT/obj/$(basename "$f" .c).o"
  $CC $BASE -c "$f" -o "$o" || { echo "BUILD FAIL $f"; exit 2; }
done
ar rcs "$OUT/liblancius_san.a" "$OUT"/obj/*.o || exit 2

echo "=== building every gate binary ==="
BINS="stress_test test_torture fuzz_lancius test_path_bg test_grad_check \
parity_runner soak_fuzz \
audit_internals audit_ffi audit_threadpool_parity audit_nan_injection \
audit_memory_pool test_diamond_memory audit_flash_attention \
audit_modern_llm audit_known_answer audit_regression_13c \
audit_transformer_known_answer audit_fp32_path audit_fault_injection \
audit_despot_probe train_verifier_head distill_prm800k audit_train_lib \
audit_sum_axis_nd audit_train_bwd audit_train_converge audit_sandbox \
train_micromodel eval_verifier lancius"
BUILT=""
for b in $BINS; do
  case "$b" in
    lancius) src="${ROOT}/examples/lancius_cli.c"; outbin="lancius" ;;
    *)       src="${ROOT}/examples/${b}.c"; outbin="$b" ;;
  esac
  [ -f "$src" ] || { echo "  (skip $b: no source)"; continue; }
  if ! $CC $BASE -o "$OUT/$outbin" "$src" "$OUT/liblancius_san.a" $LINK 2>"$LOG/$b.build"; then
    echo "  BUILD FAIL $b"; sed -n '1,6p' "$LOG/$b.build"; continue
  fi
  BUILT="$BUILT $b"
done
echo "built:$BUILT"

echo
echo "=== running every binary under ASan+UBSan+LSan ==="
fails=0
pass=0
for b in $BUILT; do
  case "$b" in lancius) outbin="lancius" ;; *) outbin="$b" ;; esac
  out="$LOG/$b.run"
  # a bounded run: 180s cap so a deadlock is a failure not a hang
  # run from the repo root so CWD-relative data paths (data_text/) resolve
  # exactly as `make check` runs them
  case "$b" in
    lancius)
      # drive the CLI exactly as `make check` does, then its own subcommands
      { timeout 120 "$OUT/lancius" info test_model.lancius &&
        timeout 120 "$OUT/lancius" run test_model.lancius --mode static --fill zero &&
        timeout 120 "$OUT/lancius" eval test_model.lancius --mode static &&
        timeout 120 "$OUT/lancius" status &&
        timeout 120 "$OUT/lancius" models --check ; } >"$out" 2>&1
      rc=$?
      ;;
    *)
      ( cd "$ROOT" && timeout 300 "$OUT/$outbin" >"$out" 2>&1 )
      rc=$?
      ;;
  esac
  rc=$?
  san=$(grep -cE "runtime error:|ERROR: (AddressSanitizer|LeakSanitizer)|SUMMARY: (AddressSanitizer|UndefinedBehaviorSanitizer)" "$out" || true)
  if [ "$rc" -ne 0 ] || [ "$san" -ne 0 ]; then
    echo "FAIL  $b  (exit=$rc sanitizer_reports=$san)"
    grep -E "runtime error:|ERROR: (AddressSanitizer|LeakSanitizer)" "$out" | head -4 | sed 's/^/        /'
    fails=$((fails+1))
  else
    pass=$((pass+1))
  fi
done
echo
echo "SANITIZER SWEEP: $pass clean, $fails with findings"
[ "$fails" -eq 0 ] || exit 1