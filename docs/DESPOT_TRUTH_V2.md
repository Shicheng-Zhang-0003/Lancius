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
- `kernel_gelu`: **tanh-approx** `0.5·x·(1+tanh(√(2/π)(x+0.044715x³)))`, `x>10→x`, `x<-10→0`, NaN passthrough. Error vs erf-exact measured at **4.74e-04** (rms 1.42e-04, extremum at x=2.699); this comment previously said ~2e-3, which was never checked against anything. Documented as GPT-2/BERT variant, not erf-exact. Correct-as-approx.
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

Historical (superseded by section 15, then corrected in section 16): per-axis
N-dim broadcast grad reduction needed new `SUM_AXIS_ND` ops and failed loud until
R3-1. It now trains, and section 16 records that its VJP was unreachable until
V7 fixed `lancius_broadcast_to_shape`.

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

- N-dim partial broadcast grad reduction needs `SUM_AXIS_ND`; failed loud until R3-1 landed. See sections 15 and 16.
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

## 15. R3 training-wrap (in progress) — N-dim, batched, norm exactness

Closed VJPs (all re-proven by independent execution + finite differences;
oracles in `temp/proofs/r3_probe_nd.c`, `temp/proofs/r3_probe_bwd.c`):

- `SUM_AXIS_ND` (id 42): `y[i]=sum_{k} x[i,k]` along one axis, rank kept.
  VJP: `dx=broadcast_to_shape(dy)`. **This VJP was unreachable until V7**:
  `broadcast_to_shape` then demanded exact shape equality, so the `[R,1] ->
  [R,C]` expansion it is defined in terms of could not build and any training
  graph containing a `SUM_AXIS_ND` forward node aborted with `INTERNAL`.
  Section 16.4 fixes it and pins the gradient against central differences at
  ~1e-9. N-dim partial broadcast backward:
  align trailing ranks; per axis with input-dim 1 sum via `SUM_AXIS_ND`;
  drop reduced leading dims by exact reshape. Old ([2,1,4] vs [2,3,4])
  fail-loud now trains with grads `-0.0625` exact by hand.
- `TRANSPOSE_BATCHED` (id 43): `out[b,i,j]=in[b,j,i]`. Self-inverse VJP.
  `MATMUL_BATCHED` VJP: `dA=dY@Bt, dB=At@dY` (finite-diff ~1e-10/1e-11).
- GELU tanh-approx `G=0.5x(1+T)`: `G'=0.5(1+T)+0.5x(1-T^2)C(1+3ax^2)`,
  clamps mirror fwd (`>10 -> 1`, `<-10 -> 0`), NaN propagates
  (finite-diff 1.25e-09).
- LayerNorm `y=(x-mu)/sig*g+b`: `dx=(d-mean(d)-xhat*mean(d*xhat))/sig`
  with `d=g*g` (NOT `g` alone — first cut missed the gamma weighting and
  finite-diff caught it at 0.77); `dg=sum_b(g*xhat)`; `db=sum_b(g)`.
  Degenerate sig -> NUMERICAL + zeros, mirroring fwd (finite-diff
  2e-09/1.7e-11/6e-11).
- RMSNorm `y=x/rms*g`: `dx=(g*g-x*mean(g*y)/rms)/rms`;
  `dg=sum_b(g*x/rms)`. Degenerate rms -> NUMERICAL + zeros
  (finite-diff 1.1e-10/1.9e-10).
- Executor placement: new ids sort after `CONV2D`, so all new cases run
  ahead of the vision-op router (first cut sat after it and the router
  rejected op 42 — caught by execution, fixed by relocation).
- Optimizer: global-norm clip `s=min(1,max/norm)` over tensor lists
  (per-tensor clip unchanged); AdamW bias correction already exact.
- Convergence: 2-4-1 tanh XOR through graph+autodiff+SGD solves 4/4 in
  28ms; same-seed rerun agrees 1e-12; v2 checkpoint at half-time resumes
  to the same loss 1e-9 (moments stay caller-owned by stateless-lib
  contract; weights persist via v2).
- Scope held: attention/GQA/SwiGLU/RoPE backward stays fail-loud;
  per-channel INT8 still refuses execution (dequantize first).

## 16. V7 truth: the gate audited against itself (mutation + external oracle)

Everything above was written by the same person who wrote the code, which is
the standing weakness of an external-oracle discipline: the oracle can share
the author's blind spot. V7 closes that with two gates that do not ask the
author to be right.

### 16.1 External oracle (`make check-oracle`)

Every kernel and every graph-level op is recomputed from a source that does not
link or import Lancius:

| Quantity | Independent source |
|---|---|
| `kernel_matmul`, `kernel_matmul_f32` | NumPy `@`, FP64-accumulate reference cast to FP32 |
| `kernel_conv2d_fwd` (stride 1/2, pad 0/1) | explicit NCHW zero-padded correlation |
| `kernel_conv2d_int8_fwd` | int64 correlation times `scale_in*scale_w` |
| `kernel_layernorm` / `_bwd` / `_bwd_gamma` / `_bwd_beta` | closed form, then **torch autograd** |
| `kernel_rmsnorm` / `_bwd` / `_bwd_gamma` | closed form, then **torch autograd** |
| `kernel_gelu` | Hendrycks-Gimpel tanh form; deviation from erf-exact **measured and pinned** at 4.74e-04 (rms 1.42e-04) |
| `kernel_gelu_bwd` | central difference of the forward (interior); clamp branches asserted directly |
| `kernel_swiglu` | `silu(gate)*up` |
| `kernel_rope` | explicit rotation at `theta=10000`, plus L2-norm preservation |
| `kernel_attention` | explicit causal softmax, then a second torch pass |
| `kernel_gqa` | grouped causal attention; group mapping forced by construction (all kv heads equal) |
| `kernel_attention_kv_cache` | `softmax(qK^T/sqrt d)V`, and equality with the last causal row of full attention |
| `lancius_sgd/sgdm/adamw_step` | hand-derived recurrence, then **torch.optim.AdamW** |
| `lancius_clip_global_norm` | post-clip global norm is exactly `max_norm`; directions preserved |
| `lancius_lr_cosine`, `_warmup_cosine` | closed form incl. endpoints and the zero-warmup identity |
| `SOFTMAX`, `CROSS_ENTROPY`, `MSE` | NumPy, then **torch.nn.functional** |
| `ADD/SUB/MUL` N-dim broadcast | NumPy trailing-rank broadcast at 2-D, 3-D and 4-D |
| `SUM_AXIS_ND` axes 0..3 over 2-D/3-D/4-D | `numpy.sum(axis, keepdims)` |
| `CONV2D_BWD`, `CONV2D_BWD_W` | central differences on the forward definition (~1e-9) |
| `MAXPOOL2D`, `MAXPOOL2D_BWD` | NumPy block max; gradient routed to the argmax only |
| `CONV2D_RELU_FUSED` | `max(numpy conv, 0)`, and bit-identity with separate conv->relu |
| whole MLP autodiff | **torch autograd** on the identical objective, plus central differences |

Tally: **41/41** kernel checks, **37/37** graph-level checks. `torch` is a
second independent engine where present; the oracle degrades to NumPy plus
closed form rather than skipping.

### 16.2 Mutation gate (`make check-mutation`)

A gate that cannot fail is decoration. `temp/audit/mutation_test.sh` injects
real defects into a pristine tree, rebuilds, and records which audit goes red.
Baseline must be green or the run is meaningless.

**First run: 8 caught, 7 survived.** Those 7 survivors were holes in the gate,
not decoration, and each became a check in `examples/audit_v7_hardening.c`:

| Mutation | Why the old gate missed it | Closure |
|---|---|---|
| `SUM_AXIS_ND` loses its terminating `return` | correct values, then a false `UNSUPPORTED_OP` nobody asserted against | H1 sticky-error contract |
| softmax drops the max-subtraction | every test logit was in `[-3,3]`, where `exp` cannot overflow | H2 logits at `1e5`, CE at `900`, attention at 200 |
| arena default alignment `32 -> 16` | one arena; glibc's block happened to be 32-aligned | H3 over 64 fresh arenas |
| arena alignment cap removed | no check on absurd alignment | H4 `1<<60` and `1<<40` refused, `1<<20` still allowed |
| autodiff abort path stops freeing `grad_nodes` | `make check` has no sanitizer in it | LeakSanitizer arm in the mutation harness + instrumented `check-sanitizers` |
| quantizer scale `max/128` | no assertion on the exact scale or the error bound | H5 `scale == max/127` exactly, round-trip `<= max/254` |
| fusion ignores the shape mismatch | the memcmp guard had no test | H6 shape-mismatched RELU refused, shape left intact |

**Second run: 18/18 caught, 1 behaviourally neutral, 0 holes.** The neutral one
(`arena` size rounding `32B -> 8B`) was verified equivalent rather than assumed
so: `ALIGN_UP(ptr, 32)` already separates consecutive allocations, and 4096
allocations of varying sizes cross no 32-byte boundary under either rounding.
It is reported as NEUTRAL, not counted as a pass.

### 16.3 Sanitizer gate (`make check-sanitizers`)

The old gate rebuilt three binaries (`stress_test`, `test_torture`,
`fuzz_lancius`) against a `liblancius.a` that was **not instrumented**, so
nothing inside the library was ever checked, and 25 of the 28 audits never ran
under a sanitizer at all. `test_diamond_memory` linked `-fsanitize=address`
against an uninstrumented archive: ASan intercepts `malloc` globally, so it
caught heap errors at allocation boundaries, but saw nothing inside the library.

V7 instruments the **library** and every audit, and adds a standalone
`-fno-sanitize-recover=all` UBSan pass (`make check-ubstrict`) for the checks
that only exist outside the combined mode: signed-integer-overflow, shift,
bool, enum, float-cast-overflow, integer-divide-by-zero, object-size, bounds.

### 16.4 V7 defects found and fixed

Four defects, each with the evidence that found it:

1. **`SUM_AXIS_ND` fell through to the vision router.** The branch at
   `lancius_scheduler.c:529` computed correct values and had no terminating
   `return`, so control reached `if (n->op >= LANCIUS_OP_CONV2D)` and
   `lancius_execute_vision_op` rejected opcode 42 with `UNSUPPORTED_OP`. The op
   computed the right answer and then poisoned the thread-local error, which
   autodiff reads as "any sticky error aborts the whole training graph".
   Found by executing the op through the public API and checking the error
   state, which no existing audit did.

2. **45 autodiff abort paths leaked `tg->grad_nodes`.** `lancius_ir_autodiff`
   allocates `grad_nodes` once; 45 of its abort paths freed `grad_map` and
   `fwd_to_full` and then called `lancius_graph_destroy(tg->graph); free(tg);`.
   ASan: `Direct leak of 40000 byte(s) in 1000 object(s)`, 72 bytes per failing
   call, and `audit_fault_injection` — a gate member — hit it on every run.
   Fixed by routing every abort through one `autodiff_abort()` helper, so the
   next op cannot reintroduce it.

3. **`lancius_broadcast_to_shape` was stricter than the operation it builds.**
   Its 4-D sibling `lancius_broadcast_4d` and the BROADCAST executor both
   accept trailing-rank alignment with per-dim `da == 1 || da == out[i]`;
   `broadcast_to_shape` demanded exact shape equality. Consequences:
   the documented `SUM_AXIS_ND` VJP (`dx = broadcast_to_shape(dy)`, with `dy`
   of shape `[R,1]` and `x` of shape `[R,C]`) **could never build**, so every
   training graph containing a `SUM_AXIS_ND` forward node aborted with
   `INTERNAL` — the R3-1 headline feature was not trainable end to end; and
   1-D and 3-D expanding broadcasts were unbuildable while their 2-D and 4-D
   equivalents worked, so the v2 loader's `ndim >= 1` branch could only ever see
   exact-match shapes. Fixed to the NumPy rule; `SUM_AXIS_ND` gradients are now
   exact against central differences at `~1e-9` on both axes (H1).

4. **Two gate members leaked their own buffers.** `test_path_bg` leaked 960
   bytes per run and `audit_modern_llm` leaked 640+ bytes per run, both by
   assigning `node->runtime_data` directly instead of binding ownership, so
   `lancius_graph_destroy` correctly refused to free memory it did not own.
   Fixed with `lancius_node_bind_owned_heap` and an explicit release.

## 17. V8 truth: every constant and equation checked against its primary source

Sections 1-16 proved the implementation against *itself* (finite differences,
self-consistency, mutation). This section checks the claims against the
**published literature**, because a wrong constant is perfectly self-consistent:
if GELU's coefficient were 0.0447 instead of 0.044715, every finite difference
would still pass and every known-answer test would still be green.

Each row names the primary source, the exact equation as published, and what
this codebase computes.

### 17.1 Constants and formulas

| Quantity | Primary source | Published | Lancius | Verdict |
|---|---|---|---|---|
| GELU tanh-approx | Hendrycks & Gimpel 2016, arXiv:1606.08415 §2 | `0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))` | `kernel_gelu`, `C=0.7978845608028654`, `A=0.044715` | **exact match** |
| GELU vs erf-exact | same, §2 (exact is `x*Phi(x)`) | max deviation | **4.74e-04** at x=2.699, rms 1.42e-04 | correct as approx; **the old "~2e-3" comment was 4x wrong and is now fixed** |
| LayerNorm | Ba, Kiros & Hinton 2016, arXiv:1607.06450; PyTorch docs | `(x-E[x])/sqrt(Var[x]+eps)*g+b`, variance **biased (1/n)**, eps default 1e-5 | `var /= hidden_size`, `LANCIUS_NORM_EPS 1e-5` | **exact match** (agrees with `torch.nn.functional.layer_norm` to 6.7e-16) |
| RMSNorm | Zhang & Sennrich 2019, arXiv:1910.07467 eq. 4 | `a_i/RMS(a) * g_i`, `RMS = sqrt(1/n * sum a_i^2)`, **no mean subtraction** | `kernel_rmsnorm` | **exact match** |
| Attention scale | Vaswani et al. 2017, arXiv:1706.03762 §3.2.1 eq. 1 | `softmax(QK^T/sqrt(d_k))V` | `scale = 1.0/sqrt((double)head_dim)` | **exact match** |
| RoPE frequencies | Su et al. 2021, arXiv:2104.09864 §3.2.2 | `theta_i = 10000^(-2(i-1)/d)`, i=1..d/2 | `freq = 1/pow(10000, d/head_dim)` with `d = 0,2,4,..` | **identical** — the code's `d` is the element index, equal to `2*(pair index)`, so `d/D == 2i/D` |
| RoPE defining property | same, eq. 16 | `<q_m, k_n>` depends only on `n-m` | verified: `<RoPE(q,0),RoPE(k,3)> == <RoPE(q,5),RoPE(k,8)>` to 4.4e-16 | **holds** |
| SiLU | Hendrycks & Gimpel 2016 (named SiLU); Ramachandran et al. 2017 (swish) | `silu(x) = x*sigmoid(x)` | `kernel_swiglu` = `silu(gate)*up`, three numerically-stable branches | **exact match** |
| SwiGLU | Shazeer 2020 (GLU variants); Dauphin et al. 2017 | `Swish(w_g) * w_v` | `silu * up` | **exact match** |
| AdamW decoupled decay | Loshchilov & Hutter 2019, arXiv:1711.05101 eq. 2 | `theta_t = (1 - lr*lambda)*theta_{t-1} - lr*g~` | `w -= lr*(m_hat/(sqrt(v_hat)+eps)) + lr*wd*w`, i.e. `w(1-lr*wd) - lr*g~` | **algebraically identical**; agrees with `torch.optim.AdamW` to 5.6e-16 over 20 steps |
| AdamW epsilon placement | PyTorch | `denom = sqrt(v_hat) + eps` (**outside** the sqrt) | same | **match** — the common `sqrt(v_hat + eps)` bug is absent |
| Cosine LR schedule | SGDR, Loshchilov & Hutter 2016 eq. 6 | `eta_min + 0.5(eta_max-eta_min)(1+cos(pi*T_cur/T_i))` | `lancius_lr_cosine` | **exact match**, endpoints verified |
| Gradient norm clipping | Pascanu, Mikolov & Bengio 2013, ICML, Algorithm 1 | `if ||g|| >= threshold: g <- (threshold/||g||)*g` | `if (norm > max_norm) g *= max_norm/norm` | **equivalent**; `>` vs `>=` differs only at exact equality where the scale is 1.0 |
| GQA grouping | Ainslie et al. 2023, EMNLP, §2.2 | query heads split into G groups, each group shares ONE kv head | `hk = hq / (n_heads_q/n_heads_kv)` | **exact match**; the H-oracle forces all kv heads equal and proves heads in a group produce identical rows |
| He/Kaiming init | He et al. 2015, arXiv:1502.01852 §2.2; `torch.nn.init` | `std = sqrt(2/fan_in)` for ReLU | `he_init: std_dev = sqrt(2.0/fan_in)` with `fan_in = C_in*K_h*K_w` for conv | **exact match**, and `fan_in` is the conv receptive-field product torch uses |
| INT32 accumulator width | ONNX `ConvInteger` spec; oneDNN int8 docs | int8 x int8 accumulates in **int32** | `int64_t sum` | **stricter than the spec, and correct**; the in-code threshold "132104 terms" verified exactly: int8 range [-128,127] gives max product 16256, and `2147483647/16256 = 132104.06` |
| INT8 symmetric quantization | TFLite quantization spec; TensorRT | `q = clamp(roundWithTiesToEven(x/s), -128, 127)`, `s = max_abs/127`, zero-point 0 | `scale = max/127`, `round`, clamp to [-128,127] | **matches the scheme**; see §17.2 for the one divergence |
| CRC-32 | IEEE 802.3 / PKZIP | reflected poly `0xEDB88320`, init/final `~0`, check value `crc32("123456789") = 0xCBF43926` | `lancius_crc32` | **byte-identical to `zlib.crc32`** on both real model files |
| Online (streaming) softmax | Milakov & Gimelshein 2018; FlashAttention-2 | running max `m`, rescale `exp(m_old-m_new)` applied to both numerator and denominator | `kernel_attention` | **exact**, verified numerically identical to direct softmax to 2.2e-16 |
| CIFAR-10 record layout | Krizhevsky, `cs.toronto.edu/~kriz/cifar.html` | binary = 1 label byte + 3072 pixel bytes = **3073**; 5x10000 train + 10000 test; test set exactly 1000 per class | all 6 batch files are 30,730,000 bytes; test balance is exactly `[1000]*10` | **exact match** |
| MNIST IDX layout | LeCun; `torchvision.datasets.mnist` | images magic 2051, 60000/10000 records, 28x28; labels magic 2049, values 0-9 | all four headers and counts correct; class balance `[5923,6742,5958,...]` matches the published train balance exactly | **exact match** |

### 17.2 The one divergence, stated plainly

Lancius's INT8 quantizer clamps to **[-128, 127]**. The TFLite spec and
TensorRT both specify **[-127, 127]** for symmetric weight quantization, using
`-127` precisely so that negation is exact and `-128` is never relied upon.

This is not a correctness bug: the clamp is inert, because `scale = max_abs/127`
guarantees `x/scale` never exceeds 127 in magnitude, so -128 is unreachable.
It is verified: the H5 oracle asserts both `+127` and `-127` saturation are
reached and never `-128`.

It is recorded here rather than "fixed", because changing the clamp to -127 would
be a no-op numerically and would alter a format whose round-trip is already
pinned. If a future consumer ever writes -128 through another path, the clamp
must become -127.

### 17.3 Claims that were WRONG and are now corrected

1. **GELU max error vs erf-exact.** The comment in `lancius_kernels.c` claimed
   `~2e-3`. Measured over a dense scan of `[-12,12]` the true maximum is
   **4.74e-04**, at x = 2.699. The bound was never checked against anything --
   the external oracle even asserted a loose `2.1e-3`, inheriting the same
   unverified number. The oracle now **measures and pins** 4.732e-04 (and the
   1.4239e-04 rms), so the next drift is caught rather than inherited.

This is the general lesson the V8 pass exists to record: **a self-consistent
implementation of a wrong constant passes every internal test.** Finite
differences prove the derivative of whatever function you implemented, not that
you implemented the intended function. Only reading the paper distinguishes the
two, which is why the citation table above is a deliverable and not a footnote.

### 17.4 What remains unchecked

- The GELU clamps at `+-10` are **not** from the paper; they are a local
  numerical guard. Verified exact in the limit: `|GELU(x)-x| < 1e-9` for
  `x > 10` and `GELU(x) < 1e-9` for `x < -10`, so they do not change any
  representable result the oracle compares.
- CIFAR-10 normalisation here is `(x/255 - 0.5)/0.5`, i.e. `[-1,1]`, **not**
  PyTorch's default per-channel `(0.4914,0.4822,0.4465)/(0.2470,0.2435,0.2616)`.
  This is a deliberate, documented choice in the code and is a legitimate
  alternative, but it means CIFAR-10 numbers here are not directly comparable to
  torchvision-trained baselines without accounting for the scaling.
- The `MD5` of a `.tar.gz` cannot be compared across mirrors because gzip output
  is not byte-reproducible; dataset verification is therefore by **content**
  invariants (magic numbers, record counts, class balance) rather than digest.
  The one digest the tree does carry, `c32a1d4a...` for `cifar-10-binary.tar.gz`,
  was wrongly compared here against `c58f3010...`, which is
  `cifar-10-python.tar.gz` -- a different file entirely. The binary tarball's own
  published digest is not asserted by any authoritative source this pass could
  find, so it is left unverified by digest and verified by content instead.

## 18. PRM800K verifier stack (2026-10-07) — what the model can and cannot read

A verifier built on `Lancius` autodiff, and an honest account of its reach.

### 18.1 The shipped distiller is not a verifier task

`examples/distill_prm800k.c` trains on `chosen_completion` alone. `chosen` carries
rating `+1` in **12961/12961** train rows and **1603/1603** test rows, so the
task as shipped is a constant label and the best achievable "accuracy" is
reproducing a prior. Parsing every completion instead gives 48,672 train and
5,082 test steps with real spread (train `-1/0/+1` = 19656/10234/18782, test =
1970/867/2245). The test majority-class baseline is `2245/5082 = 0.4418`, and
that — not 1.0000 — is the number a model must clear to have shown anything.

### 18.2 The 3-class verifier, and four defects that would each have printed a plausible number

`examples/train_prm_verifier.c`: `logits = X@W1 + b1 -> tanh -> @W2 + b2`,
minimising `mean((logits - onehot)^2)`. Final held-out result **acc 0.4957,
macro-F1 0.3932, CE 1.0432**.

1. Multiplying logits by `rows/2` "to make MSE `0.5*||.||^2`". MSE is a
   *mean*, so `d/dz mean((s*z - T)^2) = (2/pe)*s^2*(z-T)`: the multiplier enters
   twice, scaling the gradient by `s^2 = 1024`, not `s`. Diverged to NaN.
2. `softmax_ce` returned `mean log p`, i.e. **negative** cross-entropy, so a
   working run reported `-1.04`.
3. The graph's MSE VJP is `(2/pe)*(z-T)`; softmax CE gives `(1/rows)*(z-T)`.
   The relationship is the constant `2/NCLASS`, not `2/pe`.
4. The held-out baseline evaluated `(model_t){0}`, whose `W1/b1/W2/b2` are all
   NULL — segfault. Separately, one `metrics_t` was reused for both splits, so
   the `train-acc` column printed test accuracy.

### 18.3 Gradients verified independently, not asserted

`tools/prm/ref_prm.c` is a pure-C trainer with hand-derived softmax-CE backprop
and no framework in it, mirroring every hyperparameter, the RNG stream, the He
init and the effective step. It reaches **test acc 0.4959** against the
framework's **0.4957**, trajectories tracking. The graph path trains correctly.

The self-contained gradient check finite-differences the **graph's own
mean-MSE objective**, not cross-entropy, and pins at **9.5e-09**. Three false
alarms preceded it, two of which were my harness rather than the framework:
`MSE_BWD` was exact to `0e+00` in isolation and a hand-derived CE gradient
matched central differences to all printed digits, so the residual came from
converting mean-MSE to CE with the wrong prefactor (twice) and from a metric
that divides by gradients at the `1e-12` level where a central difference is
pure float noise. An RMS-relative denominator was required.

### 18.4 What this model cannot do: a proof, not a measurement

`tools/prm/bench_complexity.py` scores the trained model across ten complexity
levels, integer addition through matrix products. **Balanced accuracy is
0.5000 at every level**, with `P(+1|correct) = P(+1|wrong)` to four decimals.

The decisive result: **1150 minimal pairs whose 16-dim feature vectors are
bit-identical** — e.g. `2 + 2 = 4` vs `2 + 2 = 7` — receive **bit-identical
predictions in 1150/1150 cases**. Same length, same digit count, same
everything the featurizer sees. The capability is **absent, not weak**.

The control trains the same architecture directly on this data and reaches
**0.5215 on the training half** — it cannot fit what it is shown. The failure is
in the representation, not the optimisation.

This does not contradict 0.4957 on real PRM data; it explains it. PRM800K
ratings correlate with the *surface shape* of the writing (length, digit
density, position), which is the entire content of this model.

### 18.5 The 5-level scale, and why disjointness is the whole ballgame

`{-1, -0.5, 0, +0.5, +1}`, separated on `(structure, final answer)`:

| level | meaning | rule |
|---|---|---|
| `+1.0` | plain true | structure ok, final ok, every step exact |
| `+0.5` | mostly true, something wrong | structure ok, final ok, a working step off |
| `0.0` | 50/50 | a step is omitted — unjudgeable either way |
| `-0.5` | probably false, something right | structure ok, **final answer wrong** |
| `-1.0` | plain false | structure wrong **and** final wrong |

An earlier version made `-0.5` by corrupting an *intermediate* step while
leaving structure and final intact — indistinguishable from `+0.5` by
construction. The model responded correctly and sent **74% of true `-0.5` to
`+1`**, i.e. "probably false" read as "plain true". Making the classes disjoint:
**0.7097 -> 0.8428** exact, MAE **0.347 -> 0.145**.

Label provenance: ground truth is the **injected defect type**, decided at
generation time, and is **never re-derived by the checker**. If it were, labels
and features would both be functions of the checker and the network would only
re-encode the checker's output — success that validates nothing. The checker
supplies measurements; the defect supplies truth.

Two further defects of mine, both silent rather than crashing:

- `_bump` computed `x + int(x*scale)`, a **no-op whenever `|x*scale| < 1`**, so
  answers below ~3 were never corrupted and whole `-0.5`/`-1` samples were
  relabelled `+1`.
- Per-step feature rows carried **no step index**, so the net could not tell
  which step it was judging and collapsed to `+1` — **99% of true `-0.5` steps
  became `+1`**.

With both fixed and a 5% floor on defect magnitude: **overall 0.9370 exact,
MAE 0.046**.

### 18.6 Trained from scratch, and the two failed improvements

`examples/train_verifier5.c`: 65 -> 96 -> 5, tanh, softmax CE, SGDM with L2
into the gradient, He init via Box-Muller (the PRM trainer multiplied a
*uniform* variate by sigma, drawing std of `sigma/sqrt(3)`).

```
gradient check: 20 probes on W1, worst scale-relative err = 8.891e-08  OK
overall    n=1574  exact=0.9377  within1=0.9752  macroF1=0.6385  MAE=0.0562  kappa=0.9218
per-step   n=6764  exact=0.9808  within1=0.9871  macroF1=0.6499  MAE=0.0203  kappa=0.9708
```

Model selection uses a validation split carved from **train**; test is touched
once. A validation forward that indexed the shuffled rows as contiguous read
wrong and partly out-of-bounds data, reporting **0.28** while the same model
scored **0.70** on test — every epoch-selection decision was being made on that
noise, which is how a split present in the code but not in the arithmetic
launders test into the loop.

Both attempts to improve the weak algorithms by rescaling features **failed**:
log-transforming the error features scored **0.9320** and clipping at 10 scored
**0.9346**, both below the 0.9377 baseline. The *magnitude* of the error is
what separates `+0.5` from `-1.0`; monotone compression destroys the signal, and
the 1e6 outliers were never the problem. Reverted, with the reasoning left in
`verifier5.py` so the dead ends are not re-derived.

### 18.7 2,000 runtime tests, and what the residual error is

`tools/prm/run_tests.py`: 20 algorithms x 100 parameter sets, cycling all five
levels. **1869/2000 overall (93.45%)**, **8230/8380 per-step (98.21%)**, 6.1s.

| level | accuracy |
|---|---|
| `-1.0` | 0.9725 |
| `-0.5` | 0.9050 |
| `0.0` | 0.9950 |
| `+0.5` | **0.8075** |
| `+1.0` | 0.9925 |

Dominant error: **`+0.5` predicted `+1` in 71 cases**, then `-0.5 -> +1` in 31.
Both certify something wrong as true. Weakest algorithms `euclid_gcd` 0.62 and
`mod_pow_fast` 0.80 — variable-length derivations dilute a single-step defect so
`frac_exact` barely moves, which is an aggregation problem (worst step, not
count), not a scaling one.

### 18.8 What this stack does not establish

- **No real data.** Every label is an injected defect. No human-written
  derivation has ever been scored. This validates the machinery and the scale,
  not real-world accuracy.
- **`+0.5` is a judgement call.** "Slightly off" is a continuum; the 5% floor
  is a choice, not ground truth.
- **No algebraic equivalence.** A candidate reaching the right answer by a
  different route scores badly — the checker compares against *this*
  algorithm's step list.
- **20 algorithms**, integer-friendly coefficients, no radicals, no matrices
  above 2x2, no systems beyond 2 unknowns.
- **Edge cases are out of scope** for the sweep by explicit request, and are
  not trained for.

### 18.9 `.lancius` export, and efficacy/stability across nine configurations

**Export.** The v2 format serialises node structure for every node, but node
*values* only for `LANCIUS_OP_INPUT` nodes that have `runtime_data` bound
(`lancius_serialize.c` sets `has_weights` on exactly that condition); a
`LANCIUS_OP_CONST` node carries only a scalar `attr_val`. A frozen inference
model therefore has to hold its weights as bound INPUT nodes and its features
as an unbound INPUT, which is what `export_lancius()` builds. The stable C API
cannot express this graph at all — `lancius_graph_handle` is a wrapper struct
rather than a `lancius_graph*`, and its builders expose only
input/matmul/relu with no tanh and no bias add. `lancius_graph_save()` is what
`lancius_graph_save_stable()` delegates to internally, so calling it directly is
the same code path without the ABI detour.

The **batch dimension is fixed in the file**: node shapes are written and
restored, so patching `Xn->shape[0]` after load does not propagate downstream.
The row count is declared at export time (`--lancius-rows`, default 256) and the
round-trip is verified at that size. An earlier attempt patched the shape after
loading and silently compared against a graph still locked to one row.

**Round-trip, 9/9 exact.** Every configuration reloads and reproduces the
in-memory logits to **≤ 1.55e-15**. Getting there cost two wrong answers in the
*check*, neither of them a defect in the export: matching the output node on
`shape[1] == NCLASS` first matched `W2`, which is `[H, NCLASS]`, giving a
max |diff| of 4.04; then "first computed match" matched the intermediate matmul,
giving 0.217. Selecting the **sink** — a computed `[n, NCLASS]` node that no
other node consumes — gives 1.3e-15. A save path that is never read back is an
untested claim, and a reader that picks the wrong node reports a failure that
does not exist.

**Nine configurations, 3 widths x 3 seeds** (`tools/prm/sweep_configs.py`):

| H | test exact (mean ± sd) | range | MAE | kappa |
|---|---|---|---|---|
| 64 | 0.9070 ± 0.0388 | [0.8793, 0.9619] | 0.0664 | 0.8815 |
| **96** | **0.9290 ± 0.0170** | [0.9053, 0.9441] | **0.0613** | **0.9107** |
| 128 | 0.9183 ± 0.0280 | [0.8793, 0.9435] | 0.0649 | 0.8973 |

All nine: exact **0.9181 ± 0.0306**, kappa 0.8965 ± 0.0394. **H=96 is the
configuration to ship** — highest mean *and* lowest variance; H=64 is both the
worst mean and the least stable.

Two honest corrections follow from this, and the first is the reason the 3x3 was
worth running over the cheaper 1+3 design:

1. **The width effect in the first pass was a seed artifact.** Varying width at
   one seed and seed at one width independently made H=64 look best (0.9619) and
   would have shipped a 37 KB model on that basis. The 3x3 shows H=64's mean is
   the *worst* of the three and its spread the *widest*; 0.9619 was the lucky
   end of its range. Separating factors one at a time cannot detect an
   interaction, and a configuration chosen from one draw is not a result.
2. **The headline number reported earlier in this stack was a favourable draw.**
   §18.6 quotes 0.9377 at H=96 seed 20261007. Across seeds, H=96 is
   **0.9290 ± 0.0170**, so that figure sits about +0.65 sd above its own
   configuration's mean. The defensible statement is the mean with its spread,
   not the single best draw. Chance is 0.200, so the conclusion is unchanged,
   but the precision claimed for it was not earned.

## 19. V9 bottom-up audit, layer 1: the train-lib against primary sources

§17 verified `adamw_step`, `clip_global_norm`, `lr_cosine` and `he_init` against
their published sources. It did **not** verify the other four public train-lib
entry points: `sgd_step`, `sgdm_step`, `clip_grad_norm`, `lr_warmup_cosine`. Four
of seven checked is not a bottom-up audit. This layer closes that gap.

### 19.1 Results — 169/169 against PyTorch and HuggingFace

`tools/audit/trainlib_oracle.py` + `tools/audit/trainlib_dump.c`, wired into
`oracle_gate.sh`. Nothing on the oracle side links or imports Lancius.

| Quantity | Primary source | Verdict |
|---|---|---|
| `lancius_sgd_step` | `torch.optim.SGD` (lr, no momentum) | **exact**, ~1e-16 over 8 steps x 9 configs x 6 coordinates |
| `lancius_sgdm_step` | `torch.optim.SGD` (momentum) | **exact** in the stable regime; PyTorch's first-step buffer init (`buf = grad`) is identical to a zero buffer under `m = mom*m + g` |
| `lancius_clip_grad_norm` | `torch.nn.utils.clip_grad_norm_` | same scheme, **one characterised divergence** (§19.2) |
| `lancius_lr_warmup_cosine` | `transformers.get_cosine_with_min_lr_schedule_with_warmup` | **exact** on `[0, total]`; **one characterised divergence** (§19.2) |

### 19.2 Two divergences, stated rather than hidden

**`clip_grad_norm` epsilon convention.** PyTorch computes
`clip_coef = max_norm / (total_norm + 1e-6)` and multiplies when
`clip_coef < 1`, so its post-clip norm is **strictly below** `max_norm` by
about `1e-6/norm` — measured `0.99999939338657817` where `max_norm = 1`.
Lancius uses the exact ratio `max_norm / norm` and lands **exactly** on
`max_norm` (`0.99999999999999989`, i.e. 1 to 1e-16). The oracle asserts the
exact property `post == min(pre, max_norm)` and separately bounds the torch
difference by 1e-6. Lancius is the more accurate of the two; this is a
convention difference, not a defect.

**`lr_warmup_cosine` past `total`.** Lancius clamps to `lr_min` once
`step >= total`. HuggingFace's cosine keeps evaluating past `progress = 1`, and
because `cos` has period 2 the learning rate **climbs back up** — at
`step = 22`, `warmup = 5`, `total = 20`, HF yields `4.3e-05` where Lancius
returns `lr_min = 1e-05`. HF's callers stop stepping at `num_training_steps` so
this is never observed in practice. Lancius's clamp is the defensible choice;
recorded because the two genuinely disagree and the assertion pins the
Lancius behaviour rather than assuming agreement.

**SGD at `lr >= 1` with momentum is chaotic.** The update is a linear recurrence
whose error grows by roughly `1/(1-momentum)` per step; at `lr = 1`,
`momentum = 0.99` a single ulp of ordering difference is amplified ~100x per
step. Bit-agreement is not a meaningful requirement in that regime and the
oracle characterises it instead of asserting it. The stable regime agrees to
~1e-16, which is the claim that carries information.

### 19.3 A hole in the external oracle itself

Wiring this layer in exposed a defect in the harness rather than the library:
`oracle_gate.sh` built its probe programs in an unguarded `for` loop with no
failure aggregation, so a probe that failed to compile printed `cc1: fatal
error`, was skipped, and **the gate still exited 0**. A green external oracle
that had verified nothing is precisely the failure this gate exists to prevent,
and it survived every prior run because no probe had ever failed to build. The
loop now fails loudly on any compile or link error, verified by deliberately
breaking `trainlib_dump.c` and confirming the gate exits 1, then restoring it
and confirming 0.

### 19.4 Four more harness defects, same species as §18.4

Every failure found while building this layer was in the **oracle or the dump**,
not in Lancius:

1. `clip_grad_norm_` operates on `.grad`, not on the parameter. Passing the
   parameter clipped nothing and returned nonsense — five large fake failures.
2. `lancius_clip_grad_norm` clips **in place**, so the dump reported pre-clip
   norms next to post-clip gradients; the oracle then handed torch already
   clipped inputs and compared the wrong quantity — four more fake failures.
3. The dump wrote all six weight elements without their index while the oracle
   compared every row against element 5 — the largest "failures", at exactly the
   configs whose weights diverge fastest.
4. The clip assertion demanded `post == max_norm` for cases where nothing was
   clipped; the exact property is `post == min(pre, max_norm)`.

None of these were visible locally, and three of the four would have been
reported as Lancius defects had the oracle not been fixed first. This is now
the sixth consecutive occurrence of the pattern in this document: **the
instrument fails silently and looks like a finding.** The discipline that
follows from it is not "trust the check" but "when a check fires, establish which
side is wrong before believing either."
