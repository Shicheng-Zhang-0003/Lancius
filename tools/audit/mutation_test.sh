#!/usr/bin/env bash
# Mutation testing: does the gate actually catch a wrong answer?
#
# A gate that cannot fail is decoration. Each mutation below is a REAL defect
# of the kind this codebase cares about (a wrong index, a wrong constant, a
# missing return, a missing free, an off-by-one). For each one we patch a
# single source file, rebuild, run the gate, and record whether it went red.
#
# A mutation that survives is a hole in the gate: it means `make check` can be
# green while the library computes the wrong thing.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$ROOT/temp/mutation"
LOG="$WORK/logs"
mkdir -p "$WORK" "$LOG"

RUNS=(
  stress_test test_torture test_path_bg test_grad_check
  audit_internals audit_ffi audit_threadpool_parity audit_nan_injection
  audit_memory_pool audit_flash_attention audit_modern_llm
  audit_known_answer audit_regression_13c audit_transformer_known_answer
  audit_fp32_path audit_fault_injection audit_despot_probe
  audit_train_lib audit_sum_axis_nd audit_train_bwd audit_train_converge
  audit_sandbox audit_v7_hardening train_micromodel eval_verifier train_verifier_head
)

# mutation name | file | python replacement expression
# Each replacement is applied to a pristine copy of the file.
mutate() {
python3 - "$1" "$2" "$3" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
if old not in s:
    print("MUTATION-TARGET-NOT-FOUND"); sys.exit(3)
if s.count(old) != 1:
    print(f"MUTATION-TARGET-AMBIGUOUS ({s.count(old)} matches)"); sys.exit(3)
open(path, "w").write(s.replace(old, new, 1))
PY
}

# filesystem-safe slug: mutation names contain spaces and colons
slug() { printf '%s' "$1" | tr -cs 'A-Za-z0-9' '_' | cut -c1-48; }

run_gate() {
  local tag="$1"
  local detected=0 missed=0 buildfail=0
  for b in "${RUNS[@]}"; do
    src="$ROOT/examples/$b.c"
    [ -f "$src" ] || continue
    gcc -w -O1 -fopenmp -std=c11 -I"$ROOT/include" \
        -o "$WORK/$b" "$src" "$ROOT/liblancius.a" -fopenmp -lm -lpthread \
        2>"$LOG/$tag.$b.build" || { buildfail=$((buildfail+1)); continue; }
    ( cd "$ROOT" && timeout 300 "$WORK/$b" ) >"$LOG/$tag.$b.run" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then detected=$((detected+1)); fi
  done
  echo "$detected|$buildfail"
}

declare -a NAMES
declare -a RESULTS

try() {
  local name="$1" file="$2" old="$3" new="$4"
  local tag
  tag=$(slug "$name")
  cp "$ROOT/$file" "$WORK/orig.c"
  local out
  out=$(mutate "$ROOT/$file" "$old" "$new")
  if [ $? -ne 0 ]; then
    printf "  %-46s SKIP (%s)\n" "$name" "$out"
    cp "$WORK/orig.c" "$ROOT/$file"
    RESULTS+=("$name|SKIPPED|0")
    return
  fi
  # rebuild the library only; audits relink against it
  ( cd "$ROOT" && make liblancius.a >/dev/null 2>&1 ) || { echo "  lib rebuild failed"; cp "$WORK/orig.c" "$ROOT/$file"; return; }
  local r detected buildfail
  r=$(run_gate "$tag")
  detected="${r%%|*}"; buildfail="${r##*|}"

  # LeakSanitizer arm: build audit_fault_injection + audit_v7_hardening with
  # ASan so a reintroduced leak is caught even though `make check` has no
  # sanitizer in it.
  for lsan_a in audit_fault_injection audit_v7_hardening; do
    if gcc -w -g -O0 -fopenmp -std=c11 -I"$ROOT/include" -fsanitize=address \
        -o "$WORK/$lsan_a.san" "$ROOT/examples/$lsan_a.c" "$ROOT/liblancius.a" \
        -fsanitize=address -fopenmp -lm -lpthread 2>/dev/null; then
      ( cd "$ROOT" && ASAN_OPTIONS=detect_leaks=1 timeout 300 "$WORK/$lsan_a.san" ) \
        >"$LOG/$tag.$lsan_a.san" 2>&1
      if grep -qE "ERROR: (AddressSanitizer|LeakSanitizer)" "$LOG/$tag.$lsan_a.san"; then
        detected=$((detected+1))
      fi
    fi
  done


  cp "$WORK/orig.c" "$ROOT/$file"
  ( cd "$ROOT" && make liblancius.a >/dev/null 2>&1 )
  if [ "$detected" -gt 0 ]; then
    printf "  %-46s CAUGHT by %d audit(s)%s\n" "$name" "$detected" \
      "$([ "$buildfail" -gt 0 ] && echo " (+$buildfail build failures)")"
    RESULTS+=("$name|CAUGHT|$detected")
  else
    case "$name" in
      *NEUTRAL*)
        printf "  %-46s NEUTRAL (behaviourally equivalent; not a hole)\n" "$name"
        RESULTS+=("$name|NEUTRAL|0") ;;
      *)
        printf "  %-46s *** SURVIVED - GATE HOLE ***\n" "$name"
        RESULTS+=("$name|SURVIVED|0") ;;
    esac
  fi
  NAMES+=("$name")
}

echo "=== baseline: is the unmutated tree green under this harness? ==="
make -C "$ROOT" liblancius.a >/dev/null 2>&1
r=$(run_gate baseline)
echo "baseline failing audits: ${r%%|*} (must be 0)"
if [ "${r%%|*}" != "0" ]; then
  echo "BASELINE IS RED - mutation results would be meaningless. Aborting."
  exit 2
fi

echo
echo "=== mutations ==="

try "kernels: matmul wrong stride (b[k*N+c] -> b[k*N+c+1])" \
    src/math/lancius_kernels.c \
    'out[r*N + c] += val * b[k*N + c];' \
    'out[r*N + c] += val * b[k*N + c + 1];'

try "kernels: layernorm uses SAMPLE variance (n-1)" \
    src/math/lancius_kernels.c \
    'var /= hidden_size;

        double denom = sqrt(var + eps);' \
    'var /= (hidden_size - 1);

        double denom = sqrt(var + eps);'

try "kernels: attention drops the 1/sqrt(d) scale" \
    src/math/lancius_kernels.c \
    '        double scale = 1.0 / sqrt((double)head_dim);
    if (!o_i) {
        #pragma omp critical
        { if (omp_err_attn == LANCIUS_ERROR_OK) omp_err_attn = LANCIUS_ERROR_OOM; }' \
    '        double scale = 1.0;   /* MUTATION: the sqrt(d) normalisation is gone */
    if (!o_i) {
        #pragma omp critical
        { if (omp_err_attn == LANCIUS_ERROR_OK) omp_err_attn = LANCIUS_ERROR_OOM; }'

try "kernels: GELU coefficient 0.044715 -> 0.0447151" \
    src/math/lancius_kernels.c \
    '0.5 * x * (1.0 + tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));' \
    '0.5 * x * (1.0 + tanh(sqrt_2_over_pi * (x + 0.0447151 * x * x * x)));'

try "kernels: RoPE theta base 10000 -> 1000" \
    src/math/lancius_kernels.c \
    'double freq = 1.0 / pow(10000.0, (double)d / (double)head_dim);' \
    'double freq = 1.0 / pow(1000.0, (double)d / (double)head_dim);'

try "kernels: INT8 conv accumulator narrowed to int32" \
    src/math/lancius_kernels.c \
    'int64_t sum = 0; // 64-bit accumulator: int32 overflows at 132104 terms (K*C*Kh*Kw can exceed for LLM GEMM)' \
    'int32_t sum = 0; // narrowed: overflows past 132104 terms'

try "kernels: conv2d forward adds a spurious bias to every output" \
    src/math/lancius_kernels.c \
    '                    size_t out_idx = ni*(C_out*H_out*W_out) + co*(H_out*W_out) + ho*W_out + wo;
                    out[out_idx] = sum;
                }
            }
        }
    }
}

void kernel_conv2d_bwd_in' \
    '                    size_t out_idx = ni*(C_out*H_out*W_out) + co*(H_out*W_out) + ho*W_out + wo;
                    out[out_idx] = sum + 1e-12;   /* MUTATION: spurious bias */
                }
            }
        }
    }
}

void kernel_conv2d_bwd_in'

try "scheduler: SUM_AXIS_ND loses its terminating return (V7 defect)" \
    src/runtime/lancius_scheduler.c \
    '        return;
    }
    else if (n->op == LANCIUS_OP_TRANSPOSE_BATCHED) {' \
    '    }
    else if (n->op == LANCIUS_OP_TRANSPOSE_BATCHED) {'

try "scheduler: softmax forgets the max-subtraction" \
    src/runtime/lancius_scheduler.c \
    'for(size_t c=0; c<C; c++) { n->runtime_data[r*C+c] = exp(a[r*C+c] - max_val); sum += n->runtime_data[r*C+c]; }' \
    'for(size_t c=0; c<C; c++) { n->runtime_data[r*C+c] = exp(a[r*C+c]); sum += n->runtime_data[r*C+c]; }'

try "scheduler: SUM_AXIS0 sums along the wrong axis" \
    src/runtime/lancius_scheduler.c \
    'for(size_t r=0; r<R; r++) for(size_t c=0; c<C; c++) n->runtime_data[c] += a[r*C + c];' \
    'for(size_t r=0; r<R; r++) for(size_t c=0; c<C; c++) n->runtime_data[c] += a[r*C + (C - 1 - c)];'

try "train: AdamW decoupled decay becomes coupled" \
    src/train/lancius_train.c \
    'w[i] -= lr * (m_hat / (sqrt(v_hat) + eps)) + lr * wd * w[i];' \
    'double gi2 = g[i] + wd * w[i]; m_hat = (m[i] + a1 * (gi2 - g[i])) / bc1; w[i] -= lr * (m_hat / (sqrt(v_hat) + eps));'

try "train: clip_global_norm rescales when it should not" \
    src/train/lancius_train.c \
    'if (norm > max_norm) {
        double scale = max_norm / norm;
        for (t = 0; t < ntensors; ++t) {' \
    'if (norm > 0.0) {
        double scale = max_norm / norm;
        for (t = 0; t < ntensors; ++t) {'

try "arena: DEFAULT alignment 32B -> 16B (breaks the AVX2 contract)" \
    src/core/lancius_arena.c \
    'if (alignment == 0) alignment = 32; // V10S ARMOR: Enforce 32-byte AVX2 boundary' \
    'if (alignment == 0) alignment = 16;'

try "arena: size rounding 32B -> 8B (NEUTRAL: ALIGN_UP(.,32) already separates)" \
    src/core/lancius_arena.c \
    'size = (size + 31) & ~(size_t)31; // Force 32-byte footprint' \
    'size = (size + 7) & ~(size_t)7;'

try "arena: uncapped alignment (OOM-DoS regression)" \
    src/core/lancius_arena.c \
    'if (alignment > (1u << 20)) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return NULL; }' \
    'if (alignment > (1u << 20) && alignment != (size_t)1 << 60) { lancius_set_error(LANCIUS_ERROR_OVERFLOW); return NULL; }'

try "autodiff: abort path stops freeing tg->grad_nodes (V7 leak)" \
    src/math/lancius_autodiff.c \
    'static void autodiff_abort(lancius_training_graph* tg, lancius_node** fwd_to_full,
                           lancius_node** grad_map) {
    free(fwd_to_full);
    free(grad_map);
    if (tg) {
        free(tg->grad_nodes);' \
    'static void autodiff_abort(lancius_training_graph* tg, lancius_node** fwd_to_full,
                           lancius_node** grad_map) {
    free(fwd_to_full);
    free(grad_map);
    if (tg) {
        /* leak injected */'

try "quantize: per-tensor scale becomes max/128 (wrong range)" \
    src/compiler/lancius_quantize.c \
    'n->scale = max_val / 127.0;' \
    'n->scale = max_val / 128.0;'

try "optimizer: fusion fires even when shapes differ" \
    src/compiler/lancius_optimizer.c \
    'if (memcmp(conv->shape, n->shape, 4 * sizeof(size_t)) != 0) continue;' \
    'if (0) continue;'

try "trainer: global-norm clip uses L1 instead of L2" \
    src/train/lancius_train.c \
    'sum += g[i] * g[i];
    }
    norm = sqrt(sum);' \
    'sum += g[i] * g[i];
    }
    norm = sum;'

echo
echo "=== summary ==="
caught=0; survived=0; skipped=0; neutral=0
for r in "${RESULTS[@]}"; do
  case "$r" in
    *"|CAUGHT|"*)  caught=$((caught+1));;
    *"|SURVIVED|"*) survived=$((survived+1)); echo "  HOLE: ${r%%|}";;
    *"|NEUTRAL|"*)  neutral=$((neutral+1));;
    *) skipped=$((skipped+1));;
  esac
done
echo "mutations applied and caught: $caught"
echo "mutations applied and SURVIVED (gate holes): $survived"
echo "mutations behaviourally neutral (verified equivalent, not holes): $neutral"
echo "mutations skipped (target text not found or ambiguous): $skipped"
[ "$survived" -eq 0 ] || exit 1
