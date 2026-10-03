# Lancius Despot Truth Batch V2 — Full Mathematical, Programming, Operational Audit

**Scope:** every kernel, executor, gradient, shape formula, serializer field, converter mapping, trainer scale, harness exit code.
**Policy:** plausible outputs are not proof. Every lie fails loud. Every fix is re-proven by independent execution (`make check` + `probe_v2`).
**Temp execution:** all scratch builds, probes, and logs run under `/tmp/opencode/lancius-despot-logs`; repo stays clean; no privileged paths.

## 1. Math kernels (`src/math/lancius_kernels.c`)

- `kernel_matmul`: `O[MxN]=A[MxK]·B[KxN]`, IKJ, FP64 accum. Error `O(K·eps)`. Correct.
- `kernel_matmul_f32`: FP32 I/O, `double acc` per `(r,c)`. Correct, strictly better than FP32 accum.
- INT8 matmul (`scheduler.c:546-581`): `scale_a=max|A|/127`, `q=clamp(round(a/scale))`, `sum=int64 Σa_q·b_q`, `out=sum·scale_a·scale_b`. `int64` correct (int32 overflows at 132104 terms). All-zero activations use `scale_a=1.0` (not `1e-8` lie); result 0 either way, scale now consistent with offline quantizer which skips all-zero. Correct.
- `kernel_conv2d_fwd/bwd_in/bwd_w/relu_fwd/int8_fwd`: DL cross-correlation `out[n,co,ho,wo]=Σ in[ho·s-pad+kh,wo·s-pad+kw]·w`, `Hout=(H+2p-K)/s+1`, `int64 ih/iw`, disjoint `collapse(2)`, thread-local `calloc+critical` for `dW`. Correct. `int64` accum for INT8. Correct.
- `kernel_layernorm`: `μ=Σx/H`, `σ²=Σ(x-μ)²/H`, `y=(x-μ)/√(σ²+eps)·γ+β`, two-pass, `denom<=0||NaN→β`. `eps=LANCIUS_NORM_EPS=1e-5` pinned by scheduler. Correct.
- `kernel_rmsnorm`: `rms=√(Σx²/H+eps)`, `y=x/rms·γ`, `rms<=0||NaN→0`. Correct.
- `kernel_gelu`: **tanh-approx** `0.5·x·(1+tanh(√(2/π)(x+0.044715x³)))`, `x>10→x`, `x<-10→0`, NaN passthrough. Error ~2e-3 vs erf-exact. Documented as GPT-2/BERT variant, not erf-exact. Correct-as-approx.
- `kernel_swiglu`: `SiLU(g)·up`, `g/(1+exp(-g))` for `g≥0`, `g·e^g/(1+e^g)` for negative, `g<-500→0`, NaN passthrough. Overflow-safe. Correct.
- `kernel_rope`: interleaved NeoX per-pair `θ=pos/10000^(d/D)`, rejects odd `head_dim`. Correct.
- `kernel_attention` (Flash causal): `softmax(QKᵀ/√D+M)·V`, `M=0 if j≤i else -inf`, online `m,l,o` rescaling, `O(N²D)` compute / `O(D)` mem. Max-shift, `exp(-inf)=0`. **NaN denominator is NUMERICAL** (fail loud), zero denominator stays zeros for causal safety. Previously masked NaN as zeros. Correct now.
- `kernel_attention_kv_cache`: single-query `s=Q·Kcᵀ/√D`, max-sub softmax. `max NaN→NUMERICAL`, `max -inf→zeros`, `sum NaN→NUMERICAL`, `sum 0→zeros`. Previously lumped NaN with -inf. Correct now.
- `kernel_gqa`: same Flash loop with `hk=hq/group`, `n_q%n_kv==0`, NaN→NUMERICAL. Correct.

## 2. Executor (`src/runtime/lancius_scheduler.c`)

- `execute_broadcast_binary:106-176`: NumPy trailing-rank, `out=max(a,b)`, compat `da==1||db==1||da==db`, strides + `SIZE_MAX` checks, per-elem decode, bounds. Fixes flat `a[k]+b[k]` OOB. Correct.
- `ADD/MUL/SUB`: same-shape fast path else N-dim broadcast. Correct. INT8 Add is **row-bias only** (`[1,N]+[R,N]`); any other INT8 broadcast falls through to exact FP64 broadcast, never miscomputes via `shape[1]` assumption. Correct.
- `ReLU max(0,x)`, `ReLU_BWD g·(fwd>0)`, `Softmax max-sub + NUMERICAL on sum≤0/NaN`, `Softmax_BWD y·(dy-dot)`, `Tanh tanh`, `MSE Σ(p-t)²/N + 2g(p-t)/N`, `Sum/Axis0/1`, `BROADCAST` scalar-splat + N-dim strided path with checked strides, `Permute` validated perm with **checked stride products**, `Batched-matmul` per-batch `kernel_matmul` with **checked batch offsets**. Correct.
- `CrossEntropy`: per-row `log-sum-exp`, `-Σy·logsm/R`, `dx=(sm-y)·g/R`, `sum≤0|NaN→NUMERICAL`, tiny-neg `→0`. Mean `1/R`. 2D-only. Correct.
- Attention dispatch: cache+`q_seq==1` requires cache heads/dim match; else requires K/V heads/dim match; `q_seq!=kv_seq→UNSUPPORTED_OP` loud. GQA/ROPE/LayerNorm/RMSNorm shape/divisibility checks. Correct.
- FP32 gate: only `MATMUL/INPUT/CONST` FP32 else `UNSUPPORTED_DTYPE`. Correct.
- Hot paths use `_checked` element/byte counts; `lancius_node_elements/bytes` aborting never called on hot paths. `liveness` uses `bytes_checked`. Correct.
- `plan_pool_offset`, `prepare_buffers`, `execute_static/bounded`, `static_memory_required` all `SIZE_MAX`-guarded, pool-offset bounded, FP32 binding, NOP skipping. Correct.

## 3. Autodiff (`src/math/lancius_autodiff.c`) — the despot core

VJPs (all correct calculus):
`MATMUL dA=dY·Bᵀ,dB=Aᵀ·dY`, `RELU 0-subgrad at 0`, `TRANSPOSE self-inverse`, `PERMUTE inv[axes[i]]=i`, `SOFTMAX y·(dy-dot)`, `CE (sm-y)·g/R`, `TANH g·(1-y²)`, `MSE 2g(p-t)/N`, `FLATTEN reshape`, `CONV/BWD` thread-local reduce, `MAXPOOL += at argmax`, `SUB -grad`, `SUM d_input[i]=g[0]`, `SUM_AXIS0 d[r,c]=g[0,c]`, `SUM_AXIS1 d[r,c]=g[r,0]`, `RESHAPE reshape-back`, `MUL product rule with scalar lift`.

Truth machinery:
- `lancius_broadcast_to_shape(g,a,shape,ndim)` (new, `ir.c/ir.h`): exact N-dim scalar lift for 1..4-D. Scheduler already executes N-dim; constructor makes it reachable. Used by SUM grads (any ndim) and MUL scalar lifts.
- `accum_grad` returns `1/0`; every `lancius_add/sub/mul` incompatibility sets sticky `SHAPE_MISMATCH`; scalar-input reduction via `SUM+reshape` checked; 2D row/col via `SUM/SUM_AXIS0/1` checked; N-dim scalar-grad via `broadcast_to_shape`; N-dim partial reduction (e.g. `[2,1,4]` vs `[2,3,4]`) **fails loud** (IR cannot express per-axis N-dim sums without new ops; zero-grad would be a lie).
- Forward rebuild clones `SUM_AXIS0/1`, `BROADCAST` any ndim, transformer forward ops (so `fwd_to_full` stays complete); `_BWD` in forward returns NULL immediately.
- Backward has explicit VJPs for `SUM_AXIS0/1`, `RESHAPE`; transformer/`MATMUL_BATCHED` fail loud; final `else` fails loud for any unhandled forward op; end-of-iteration sticky-error gate aborts partial graphs. **No silent drop.**
- `lancius_add/sub/mul` set `SHAPE_MISMATCH` on broadcast incompat (was silent NULL).
- `lancius_clear_error()` at autodiff entry; any error aborts to NULL. Trainers `memcpy` only when grad exists; missing grad after honest NULL aborts instead of training as zero.

Proven: `test_grad_check` (Conv/CE/MatMul VJPs incl. `1/R`), `probe_v2` (SUM 3D→`BROADCAST ndim3 [2,3,4]`, RESHAPE, SUM_AXIS, `broadcast_to_shape`, N-D partial fails loud, attention NaN→NUMERICAL).

Deferred (honest, not silent): per-axis N-dim broadcast grad reduction needs new `SUM_AXIS_ND` ops; currently fails loud. Documented in `KNOWN_LIMITATIONS.md`.

## 4. Memory / checked / planner / threadpool

- Planner: `birth`=producing wave (excludes `INPUT/CONST` external), `death`=max consuming wave, sinks pinned, birth-sorted linear scan, expiry `death<birth`, first-fit + 32B align, `peak` via `end_addr`. No coalesce (safe over-estimate). Corrupt sizes→0 then `LIMIT/OVERFLOW`. Correct.
- Arena: `size==0→1`, `size>SIZE_MAX-32` reject, 32B footprint, pow2 check, `uintptr_t` align math with `SIZE_MAX` guards, grow `max(default,size+align)`. Correct.
- `checked_product_shape`: `ndim>4→0`, `NULL→0`, fold `checked_mul`. Zero dims yield `e=0,1`; zero-rejection lives in `validate_shape`. `node_elements_checked` adds `>100M` cap; `node_bytes_checked` adds `SIZE_MAX/elem_size` + `>800M` cap. Correct.
- `validate_binary_broadcast`: NumPy trailing-rank, `da==1||db==1||da==db`, zero→`INVALID_SHAPE`. Correct.
- Threadpool: FIFO + `active_tasks`, overflow inline-run (still safe, disjoint), wave barrier, serial alloc, per-node disjoint outputs. Deterministic for inference; `conv_bwd_w critical` FP-sum order nondeterministic last-bit (documented). Correct.
- Bytecode VM: 2D MLP subset only, broadcast-aware binary, softmax guard, OOM→-1. Parity within subset. Correct, honest non-parity by design.

## 5. Serialization / quantize / ONNX

- v2: header 48B (`8×u32+u64+2×u32`), node 104B, magic `0x32434E41`, ver `2`, `LE|STATIC`. Python `<IIB4QId4I4I3BdQ` / `<8IQ2I` matches C. LE-only (BE refuses). CRC32 ISO-3309/zlib over `48..EOF`, saver maps `0→1` (never emits legacy), loader rejects `0` unless `LANCIUS_ALLOW_LEGACY_UNVERIFIED=1`, else `malloc` body + compare. Caps: `version, LE, !EXT, header==48, node_count≤1M, reserved==0, ndim≤4, in_count≤16, welems≤100M, dtype∈{0,1,2}, op≤41, id<10M, id<node_count*16+1024`, `bytes/elem==elems`, O(1) duplicate via map, forward-ref reject, stream-skip. `u64→size_t` narrowing checked. `BROADCAST` loads any 1..4-D via `broadcast_to_shape`. Correct. Doc stale `checksum==0 accepted` fixed to rejected-by-default.
- v1 fallback: native `size_t/double`, no CRC, deprecated, isolated caps. Correct.
- Quantize: symmetric per-tensor, no zero-point, only `INPUT∧ndim==4∧FP64`, `max=max|w|`, `!(max>0)→skip` (all-zero/NaN stays FP64), `scale=max/127`, `q=round/clamp[-128,127]`, owned-heap bind. Dequant `w≈q·scale`. Per-channel quantization added in V4 (one scale per output channel, stored in `rt->scale_per_channel`). Correct for symmetric.
- ONNX `onnx_to_lancius.py`: 9 ops only, else raise; symbolic dims raise; rank>4 raise. Reshape `0`=copy (positional for 2/4), `-1`=infer, `>1×-1` reject, `total_in%total_known==0`. Gemm `transA!=0` reject, `alpha/beta!=1` reject, `transB` deep-copy transpose (never mutates shared init), out `A_rows×B_cols`. Conv symmetric pads/strides only, kernel recovered from `W`, **dilations!=1/group!=1/auto_pad!=NOTSET reject** (was silent dense). MaxPool square only, **pads!=0/dilations!=1/ceil_mode!=0/auto_pad reject** (was silent). Transpose `[1,0]/[1,0,2,3]` only else raise. Add 1-D bias forced `[1,N]`; C trailing-rank + N-dim exec faithful; INT8 Add row-bias only. MatMul 2D only (N-D batch collapses — documented). Flatten 4D `axis=1`. Correct, fail-loud.

## 6. Training

- Loss: CE `1/R` fwd/bwd consistent, pinned `CE([0,0],[1,0])==log2`, MSE `1/3`. Correct.
- CIFAR `((x/255)-0.5)/0.5 ∈ [-1,1]` matches PyTorch `Normalize((0.5,),(0.5,))`. MNIST `(x/255)-0.5 ∈ [-0.5,0.5]` (half scale, Adam-tolerant, documented inconsistency). He `√(2/fan_in)` Box-Muller avoiding `log(0)`, biases 0. Correct for ReLU. Verifier uniform `√(1/fan)` for tanh (avoids saturation). Correct.
- Adam canonical with bias correction; CIFAR global-norm clip `max_norm=1.0` direction-preserving. Correct. LR: MNIST `0.001` fixed, CIFAR `0.0003→×0.5@15,25`, PyTorch ref `0.001/batch64` vs C `batch32` — 3.3× LR + 2× batch gap documented, not transferred. Correct update, unproven LR (documented).
- Grad-check `EPS=1e-5,TOL=1e-4` central diff, trunc `1e-10`, round `1e-6` ≪ `1e-4`. Proves Conv/CE/MatMul VJPs. Narrow (biases, bcast reductions, extreme logits, single seed) documented.
- Honesty: CIFAR raw gate aborts on `NaN/>1000/<0` immediately (was clamped-average hiding explosion); both trainers exit 1 if accuracy ≤ chance (was unconditional `return 0`); `test_ffi_error` exits 1 on unexpected success + non-NULL handle check (was always 0). `make check` gates `test_grad_check + audit_* + train_verifier_head + distill --selftest`, not `train_mnist/cifar10` (documented: green says nothing about MNIST/CIFAR convergence).

## 7. Validation honesty

Real gates (`return fails?1:0`, in `make check` without `||true`): `test_grad_check`, `test_torture` (real cycle), `audit_known_answer 73/73`, `audit_despot_probe`, `audit_internals |sum-1|<1e-9`, `audit_fault_injection 11/11`, `audit_ffi`, `audit_fp32_path 19/19`, `audit_regression_13c 49/49`, `audit_transformer_known_answer 265/265`, `audit_memory_pool`, `audit_nan_injection`, `train_verifier_head`, `audit_trained_reality.py <95/100→exit1`, `audit_pytorch_parity.py rel>1e-5→exit1`, `probe_v2` (new). `make check` green means every check passed.

## 8. Remaining honest deferrals (not lies)

- N-dim partial broadcast grad reduction needs `SUM_AXIS_ND`; currently fails loud.
- FP32 operator coverage matmul-scoped; no FP32 LLM path; KV-cache FP64-only.
- ONNX LeNet-class only; N-D MatMul batch collapses; `Sub/Mul` mirror Add.
- Dynamic shapes, GPU, distributed, production LLM serving: not supported.
- `conv_bwd_w critical` last-bit nondeterminism (training only).
- LR parity unproven across PyTorch/C batch/scale gaps.

## 9. V3 addendum (2026-09-28) — forensic sweep, same policy

Four parallel reviews, ~70 code-backed defects, all fixed and re-proven
(`make check`, `check-sanitizers` clean, pytorch parity `3.42e-07`).
Behavioral deltas vs §1–§8:

- Norms: degenerate LayerNorm/RMSNorm is `NUMERICAL` (was silent
  beta/zeros), matching the attention contract.
- Matmul: IR builds 2D-only (was: N-D built a 2D node that dropped batch
  dims at execution); FP32/INT8 paths check K-match like FP64 always did.
- Ownership: FP32 has `f32_owner` (was aliased: leak + free-of-external);
  quantizer syncs `rt->scale`, frees stale int8, skips non-finite max.
- VM: `CONST` regs materialized (were garbage); inputs checked.
- Planner/pool: zero-size plans abort (were offset-0 overlaps); queue grows
  (was racy inline run); shutdown rejects.
- Persistence: v1 checked writes + partials unlinked + ndim 1/3 loads +
  view validation + double-free removed; v2 tmp+rename + streamed CRC (no
  800MB malloc, no `long` truncation, no seek-bypass) + dup-NOP seen-list.
- Interop: converter true ranks + Reshape rank + pack validation;
  exporters cap/check/validate/multi-input dummies; datasets no-shell +
  cwd-jail + slip guards.
- Trainers/operator: CLI fork+exec, checked allocs/graphs/grads, cifar
  evaluated-denominator, verifier heap-copied grads + worst-step max,
  edge/demo/diagnostic cleanup + return codes.
- Doc drift fixed as part of this batch: single-owner rule per file (see
  README § Documentation), stale v11A1 format/ops claims bannered,
  duplicate release note removed.

## 10. Hardening batch V4 (2026-09-28) — race conditions, memory safety, quantization, portability

Comprehensive bug-fix campaign: 26 defects fixed across autodiff, kernels,
serialization, CLI, build system, and Python tooling. All re-proven by
`make check`, `check-sanitizers`, and pytorch parity.

Behavioral deltas vs §1–§9:

- **Autodiff:** BROADCAST backward now correctly reduces over broadcast
  dimensions (was passing `grad_out` unchanged). NULL checks on
  `fwd_n->inputs` throughout. OOB reads on shape/axes arrays fixed (pads
  to 4D). Off-by-one in node capacity check fixed.
- **Kernels:** Race condition in `kernel_conv2d_bwd_in` fixed (thread-local
  accumulators). Race condition in MaxPool2D backward fixed (thread-local
  accumulators).
- **Memory:** Memory leaks in IR node allocation fixed. `abort()` removed
  from library code (replaced with `lancius_set_error` returns). All
  `fprintf`/`printf` in library code replaced with `lancius_set_error`.
- **Stable API:** Dangling `wrapper->sched` fixed. `set_owner` now updates
  `int8_owner`.
- **Serialization:** Portability fixed (`uint64_t` sizing, byte swapping).
  CRC32 table init race fixed (`call_once`). NOP IDs no longer mapped to NULL.
  Double-read for CRC eliminated (now computed during parsing).
- **Threadpool:** `lancius_pool_wait` now accepts a timeout parameter.
- **Bytecode VM:** Overflow checks added.
- **Quantization:** Per-channel quantization support added. Dequantization
  support added.
- **CLI:** Command injection fixed (now uses `fork+execvp` instead of
  `system`).
- **Python scripts:** Security fixes, stale version references removed,
  error handling improved.
- **Build system:** Version consistency enforced, `-Werror` added, Threads
  dependency fixed. `train_cifar10` now links `-lpthread`. Dependency
  tracking improved. `.gitignore` updated with missing entries.
  `lancius.pc.in` version fixed.

## 11. Hardening batch V5 (2026-09-30) — threadpool, IR honesty, example hardening

Comprehensive bug-fix campaign: 12 defects fixed across threadpool, IR,
compiler, runtime, examples, and Python tooling. All re-proven by
`make check`, `check-sanitizers`, and despot truth probes.

Behavioral deltas vs §1–§10:

- **Threadpool:** Queue growth use-before-initialization fixed (was reading
  from uninitialized `nq` buffer during realloc; now reads from `pool->queue`).
- **Security:** Command injection in `distill_prm800k` fixed (replaced
  `system()` with `mkdir()`).
- **IR:** Silent NULL returns fixed — `lancius_matmul_batched`,
  `lancius_cross_entropy`, `lancius_reshape`, `lancius_flatten`,
  `lancius_gqa`, `lancius_permute` now set error codes on validation failure.
- **Optimizer:** Error clearing on success fixed (no longer masks prior errors).
- **Quantizer:** Zero-scale check added in `lancius_dequantize_graph`.
- **Scheduler:** Cross-entropy consistency fixed (forward and backward use
  same R/C source).
- **Memory planner:** Free block splitting maintains 32-byte alignment.
- **Stable API:** Duplicate `#include` removed.
- **Serialization/Vision:** `fprintf`/`printf` removed from library code.
- **Examples:** `parity_runner` and `run_trained_batch` hardened.
- **Python:** `onnx_to_lancius.py` shape filtering fixed.
- **Autodiff:** NOP comment clarified.
- **Code quality:** Magic numbers replaced with named constants throughout.

## 12. Despot audit V6 (2026-10-01) — full-system hostile pass, 45 defects

Policy unchanged: plausible ≠ proof; every defect code-backed, fixed,
re-proven by `make check` + `check-long` + `check-sanitizers` +
`test_grad_check (8.6e-10, 5.8e-8)` + `probe_v6` in
`/tmp/opencode/lancius-despot-logs`. Scratch under `/tmp/opencode/`;
repo `temp/` stays empty.

§1 Math (autodiff/kernels/scheduler exec):
`y=broadcast(x)`, `dx[I]=sum_{J:bcast(J)=I} grad[J]`.
4D reduction exact via permute+reshape+sum_axis0/1:
d==0 `[D0,R]->sum_axis0`, d==3 `[P,D3]->sum_axis1`,
d==1 `permute(1,0,2,3)->[D1,R]->sum_axis0->permute back`,
d==2 `permute(2,0,1,3)->[D2,R]->sum_axis0->permute(1,2,0,3)`.
Prior dim2/3 flatten `[pre,Dd*post]->sum_axis1` summed post dims too
(wrong values) plus reshape-target order bug (`RESHAPE_MISMATCH`);
dim0/1 called 2D-only sums on 4D (`INVALID_RANK`).
ndim!=2,4 with needed reduction fails loud (`UNSUPPORTED_OP`,
no `SUM_AXIS_ND`). `CE_BWD` ctor `2D + xe==ye + ge==1`
(was `g[0]`-only silent); exec same guards (was OOB read).
`conv_bwd_w` stride/pad/`H+2p>=K` (was missing).
`LAYERNORM/RMSNORM` `ie==total` (was OOB `in+b*hidden`).

§2 Runtime (VM/scheduler/pool/arena):
VM `ndim!=2` reject (was `>2`, so 1D `[5]->5x0` mismatch);
inputs validated before `reg_map` (was `NULL+0`/OOB);
tape `pc+need<=len` (was 3-word overread);
`out_reg<num_regs`, `code/rows/cols!=NULL`;
`RELU/SOFTMAX in==out`, `SUM out==1x1` (was OOB/uninit leak).
Scheduler `!inputs/input_count/inputs[k]` first on every op;
`!buffer->NULL_PTR` everywhere (was silent uninit).
Pool create/submit set `OOM/INTERNAL/LIMIT`;
grow `malloc+linearize+free` (was `realloc` then UAF read of freed
`pool->queue`, plus `nq` leak on `tmp`-OOM, plus silent drop).
Arena grow `checked_add` + fresh-block fit re-check
(was wrap to 16MB + OOB `used+=SIZE_MAX`).
Static pool: plan path requires 32B base (offsets 32-aligned);
bump path starts at `base_pad`; CLI `posix_memalign(32)`.

§3 Persistence/IR/quant:
v2 save `mkstemp+fsync+rename`; per-channel refused (no silent drop);
`ftello/off_t`, empty body savable; loader `set_error` on all paths,
`ndim==0` reject, `INPUT 2/3/4-D`, `CONST 1..4-D` (scalar shape overwrite),
`ROPE` case `(seq,heads,hd/2)`, `INT8 scale>0 finite`.
v1 save `tmp+rename`, `FP32` branch; load `NULL_PTR/IO/INVALID_MODEL/OOM`,
`INT8 scale`, `FP32` branch.
Quantizer never clears sticky error; per-tensor frees stale per-channel;
dequant checks per-channel `s>0 finite`, frees stale owned FP64.
Fusion `memcmp(shape)==0`. `conv_bwd/maxpool_bwd/gqa` mirror fwd checks.
Vision `BWD` rank/stride/`eH/eW` + `NULL_PTR`.
`graph_runtime/node_rt` set errors; stable `scratch` allocate-first.

§4 Ops:
`manage_datasets` chunked capped; `onnx_to_lancius` 2GB pre-stat;
`distill` realloc-tmp + `fopen/malloc/fwrite/fseek` checks;
`run_trained_batch/parity_runner` unified cleanup + checked elems;
`train_cifar10` tar-list slip validation; `make clean` purges `.d`.

Proven: `make check` green (73/73, 265/265, 49/49, 19/19, 11/11,
despot probe, verifier `3.9e-09`, distill selftest),
`check-long` (soak 3/3, fuzz 500/0), `check-sanitizers` clean + restore,
`probe_v6` (dim0/1/2/3, multi-dim, `CE_BWD`, VM rank) all truth holds.

## 13. Despot audit V7 (2026-10-03) — error channels, grad shapes, per-channel honesty

Math proofs (`temp/proofs/`): per-channel W ch0=0.5/ch1=100 now refuses
`UNSUPPORTED_OP` instead of executing ch0=100.0 (was smax lie); 1x1 grad
where 3x3 required now `SHAPE_MISMATCH` at build (was dw 63 vs 1053);
1x1 grad where 2x2 pool required now `SHAPE_MISMATCH` (was 3/4 zero);
`test_grad_check` heap-copies analytic grads (was arena-reset luck).

Programming proofs: every conv overflow guard sets `OVERFLOW`; OpenMP
`layernorm/rmsnorm/attention/GQA/bwd` errors propagate via
`omp_err_*` shared flags + master re-set (pattern mirrors
`vision_ops.c` maxpool-bwd `omp_alloc_failed`); v1/v2 saves set codes;
KV-cache/VM/pool_wait/RoPE/stable-magic/`ValueError`/`fsync` all fail
loud with the narrowest code. `make -Werror` clean.

Operational proofs: `cmake -B temp/cmake-check` configures with train
sources; `make check` green from clean tree; version grep shows no stale
`v12R1`-as-current (except frozen history under `docs/v11A*`,
`docs/releases/`, `CHANGELOG.md` history sections which are intentionally
historical).

## 14. Despot audit V8 (2026-10-03) — external-truth re-audit

Every V7 claim re-derived from outside the repo: stable-softmax/CE vs
max-sub/log-sum-exp oracles, LayerNorm/RMSNorm/GELU/RoPE/attention/GQA
vs published formulas, INT8/broadcast/conv/optimizer vs NumPy/PyTorch,
`_Thread_local`/mkstemp+fsync/ftello/OpenMP/`assert`/IEEE-754 vs
C11/POSIX/spec docs, CMake train-glob via fresh configure, ONNX Reshape
`allowzero` vs live spec, distill/micromodel/eval/grad/parity/gates by
independent re-execution. Findings: conv_bwd grad hole, tensor-handle
magic gap, allowzero gap, exporter assert, makefile default-goal hijack,
converter 1D-helper + ndim lies (parity RED), micromodel FEAT gap, eval
tautology. All closed above; parity re-measured `3.42e-07` (not marked
RED — the bug was the converter, the number reproduces exactly).
