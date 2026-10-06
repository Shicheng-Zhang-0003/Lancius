# Lancius Changelog

## 3463-LDFD integration — dataset acquisition stops being manual

Lancius can now acquire its own datasets. `manage_datasets.py download`
routes through the Live Data Feeding Framework (3463-LDFD, submodule) when
its shared library is present, and falls back to the hardened urllib path
when it is not, so the command works either way. `make check` gains an
`ldfd-test` stage that reports an honest SKIP rather than a silent pass.

Nine defects in 3463 were fixed first, because ingesting data through a
component that silently corrupts it is worse than not ingesting at all:

- `assoc.c`: NULL row callback was a guaranteed SIGSEGV on the first data
  row (ASan: `SEGV on unknown address 0x0`); both `_new()` constructors now
  refuse NULL. Headerless feeds emitted **zero** rows — the documented
  positional fallback was unreachable because rows were only forwarded
  after a header was seen; preamble lines are now distinguished from data
  rows by their first byte. A row over `ASSOC_MAX_LINE` (1MB) was silently
  truncated and emitted, producing a plausible-but-wrong number; it is now
  rejected whole. Rows over `ASSOC_MAX_COLS` are rejected instead of
  truncated. RFC4180 `""` now collapses to one quote inside a quoted field
  (was kept as two). GBIF: a value containing the literal text `"results"`
  hijacked the key scan and failed the page; a malformed tail returned -1
  with `kept`/`skipped` still zero even though callbacks had fired.
  `ASSOC_MAX_COLS` raised 32 -> 64.
- `fetch.c`: the async loop set `still_running = 0` after the **first**
  completed transfer, so every sibling source was abandoned mid-flight and
  its easy handle leaked (`curl_multi_cleanup` refuses to run with handles
  attached). Now owned by curl, with an exhaustive detach-on-exit sweep.
  `interval_sec` was parsed, stored, printed by both examples — and read by
  nothing; there is now a real monotonic-clock scheduler that re-polls.
- `parser_csv.c`/`parser_json.c`: a re-used parser kept `header_done` set
  and the JSON buffer non-empty, so every re-poll after the first re-emitted
  the CSV header row as a data record (a 1s-interval source produced a bogus
  leading record every poll) and JSON documents concatenated. New optional
  `snap_parser.reset` clears per-response state while keeping the caller's
  callback — freeing and re-initing the context instead dropped the
  callback, which would have made a re-polled source go silent.
- `context.c`: `curl_global_init`/`cleanup` ran once per context, so two
  live contexts meant two inits and one cleanup, tearing down libcurl global
  state under the survivor. Now one `pthread_once` init, never cleaned up.
  Parser-init failure was ignored, leaving a pipeline with a NULL context.
- `fetch.c`: `CURLOPT_FOLLOWLOCATION` replayed `CURLOPT_HTTPHEADER`, so an
  `auth_header` was re-sent to whatever host a redirect named. Added
  `PROTOCOLS`/`REDIR_PROTOCOLS` (http/https only), `UNRESTRICTED_AUTH=0`,
  and explicit `SSL_VERIFYPEER`/`VERIFYHOST`.
- `snapshot.h` included `<curl/curl.h>` unconditionally, so the buffer and
  the parsers required libcurl headers to compile at all. The multi handle
  is now `void*`; only `fetch.c` sees curl.

New capability:

- `src/output.c`: the output stage existed only as seven undefined externs.
  Real sinks now: `snap_file_output` (atomic append, `fflush` failure
  surfaced), `snap_callback_output`, and a CSV-rows-to-JSONL sink that is
  the integration seam into `data_text/`. The 4 unimplemented transforms
  and 3 unimplemented outputs were deleted from the header rather than left
  to fail at link time.
- `src/decompress.c`: streaming gzip (zlib) and a tar reader, because the
  corpus is MNIST `.gz` and CIFAR-10 `.tar.gz`. gzip accepts 1-byte chunks
  and rejects truncated streams instead of returning a partial dataset; tar
  verifies header checksums, honours GNU `L` and pax `path=` long names,
  refuses `..` members, and contains absolute paths under the destination.
- `snap_fetch_to_buffer`: one-shot raw fetch with a size cap and HTTP >= 400
  reported as an error, so a saved error page is never mistaken for data.
- `ldfd_bridge.py` + `audit_ldfd_bridge.py`: ctypes seam. It deliberately
  exposes only buffer-in/buffer-out entry points, so nothing is called back
  into Python from C. The audit compiles against the C headers to assert
  struct layout and enum values, and exits 77 (skip) when the library is
  absent.
- `examples/poll_once.c`: fetch once, inflate, optionally extract — the
  shape the dataset layer uses. `fire_monitor`/`multi_pipeline` now use the
  scheduler instead of hand-rolled `sleep()` loops.

A second pass, found while hardening the debug/stress path:

- `fetch.c`: `ctx_untrack_task` NULLed the completed task's slot but never
  shrank `n_tasks`, so the async loop's own exit test (`n_tasks == 0`) could
  never fire. A context whose sources were all one-shot finished its work and
  then spun on a 250 ms timer until stopped. Now compacts on untrack.
  Reproduced with a dedicated probe (`n_tasks` 1 -> 0 after the fix) and
  pinned by `test_fetch.c::test_oneshot_loop_drains`.
- `buffer.c`: growth arithmetic could wrap. `buf->len + len` overflowing
  skipped the grow entirely and then `memcpy`'d past the end, and
  `new_cap *= 2` wrapping to 0 spun forever. Now computed in the remaining
  space with an explicit 1 TiB ceiling.
- `output.c`: `strtod` maps overflow to +/-HUGE_VAL, and `%g` printed that
  as `inf` / `nan` — not valid JSON, so one bad numeric field would poison
  every downstream consumer. Non-finite and out-of-range values are now
  emitted as strings, and finite numbers use the shortest representation that
  round-trips (so `-113` stays `-113` instead of becoming
  `-113.00000000000000`, which a bare `%.17g` would have produced).
- `decompress.c`: tar `((size + 511) / 512) * 512` wrapped for a size near
  SIZE_MAX, desynchronising every later offset; a member whose declared size
  runs past the image is now rejected up front. gzip refused a chunk larger
  than zlib's 32-bit `avail_in` rather than truncating the cast and feeding
  inflate a prefix.
- `manage_datasets.py` + `ldfd_bridge.py`: fetch failures can now be
  reported before anything is written, and the landed file is renamed into
  place only after a complete payload.

Test evidence: `make -C 3463-LDFD test` — 4 dependency-free suites, 100
checks, 0 failures (assoc scenarios, assoc regressions, core
buffer/CSV/output, decompression), ASan/UBSan clean via `make sanitize`.
The assoc regression suite was verified to FAIL 15/20 against the pre-fix
`assoc.c`. `make test-fetch` needs libcurl headers and is an honest SKIP
without them.

**Real-libcurl verification (curl 8.5.0).** With libcurl present the fetch
layer was exercised for the first time, and it found a bug the curl stub had
been hiding: the async loop exited as soon as `curl_multi_perform` reported
nothing in flight, but a periodic context is idle for the whole gap between
polls. So `interval_sec` fetched exactly once and the loop was gone before the
first deadline -- the feature the whole integration exists for was inert in
production, while a stub that always had work made it look fine. The exit test
now also asks whether any deadline is still ahead (`sched_has_pending`).
Measured after the fix at 1s interval: 6 polls in 6s; at 3s: 2 polls in 6s.
`tests/test_fetch.c` now passes 22/22 against real curl, including a new
one-shot-termination test so the fix cannot be made by simply never exiting,
and the interval checks were confirmed to FAIL when the fix is reverted.

Also against real curl: `fetch_to_buffer` returns byte-exact payloads, an
HTTP 404 is reported as an error rather than silently landing the error page
as if it were data, and the size cap refuses instead of truncating.

Other fixes found by having the headers available:

- `fetch.c` used `CURLOPT_PROTOCOLS` / `CURLOPT_REDIR_PROTOCOLS`, deprecated
  since curl 7.85. Now uses the `*_STR` variants when the headers provide
  them, keeping the LONG forms for older curl.
- `make test-fetch` linked `-lcjson` through the global `LDFLAGS` and so
  required libcjson-dev even though the suite never calls cJSON; it now links
  curl only and reports a cJSON requirement precisely if the parser vtables
  `context.c` resolves cannot be found.
- `make sanitize` no longer depends on `all`, so it works on a machine that
  can run the tests but lacks the headers needed for the examples.

**zlib was missing from the link line.** `src/decompress.c` needs zlib for
gzip, but the makefile never put `-lz` in `LDFLAGS`. The build compiled every
object cleanly and then failed at the very last link step with
`undefined reference to 'inflateEnd'` / `DSO missing from command line`. That
message reads like a broken zlib install; it was actually a missing flag, and
`libz.so.1` was present the whole time. The makefile now queries zlib through
pkg-config alongside curl and cJSON, with `-lz` as the fallback, and queries
each module separately rather than in one `pkg-config --libs a b c` call --
the single-call form fails wholesale if any one module is unknown, which would
have silently dropped the multiarch include paths for the others too.

**Two more bugs, found by running the examples rather than the tests.**
Neither the unit suites nor `make check` build the examples, so both lived
undetected until the binaries were actually run:

- `snap_tar_extract` returned `SNAP_OK` with `files == 0` for any payload
  shorter than one 512-byte block. `poll_once --extract-tar` on a plain CSV
  printed "extracted 0 file(s)" and wrote nothing at all -- the download was
  discarded with exit code 0. A tar is always a whole number of blocks, so an
  unaligned payload is now rejected with `SNAP_ERR_PARSE`; a genuine empty
  archive (two zero blocks) is still accepted. Three new checks cover this,
  and they were confirmed to fail against the previous behaviour.
- `poll_once` never created its destination directory. Pointed at a path that
  did not exist it failed with "cannot write /tmp/x/f.csv", which reads like a
  permissions problem rather than a missing directory. It now creates parents
  like the tar path always did, and when `--extract-tar` is given a
  non-archive it writes the payload instead of throwing it away.

With cJSON installed, `make` now builds all three examples and
`libsnapshot.so` (60 KB) with no warnings, `test-fetch` runs for real rather
than reporting SKIP, and `audit_ldfd_bridge.py` reaches its live tier: 13/13.

Final state of every suite on this machine:

| suite | result |
|---|---|
| assoc regression | 20/20 |
| assoc | 50/50 |
| core | 35/35 |
| decompress | 35/35 |
| fetch (real curl) | 22/22 |
| bridge audit (real .so) | 13/13 |
| ASan/UBSan | all clean |
| stress full | 23/23 release, 13/13 debug |
| parent `make check` | pass |

A note on why the curl stub was not good enough: it completed every handle on
every pass, so the scheduler always had work to do. Real curl correctly
reports zero running between transfers, which is precisely the state the exit
test mishandled. Stub-based verification of concurrency logic is close to
worthless; only a real library exercises the idle path.

`run_stress.sh` gained a debug mode: it rebuilds the stress binaries into
`build-debug/` with sanitizers so the release build and `liblancius.a` are
never clobbered (unlike `make test_asan`, which rebuilds those same names in
place). It defaults to `-O0` because, measured with gcc+ASan here, both a
heap-buffer-overflow and a leak are caught at `-O0` and both go UNDETECTED at
`-O1` — the optimiser elides or vectorises the operations the sanitizer
instruments, so a green `-O1` run is confident and blind. UBSan is built with
`-fno-sanitize-recover=all` and run with `halt_on_error=1`, because plain
UBSan only prints and continues and would turn undefined behaviour into a
silent pass.


## R3 training-wrap (in progress) — actual framework that safely trains

v12R2 proves Lancius can learn; this batch closes the remaining
fail-loud holes so MLP/CNN/norm graphs train end-to-end with proofs:

- R2-7: `LANCIUS_OP_SUM_AXIS_ND` (id 42) end-to-end — builder, checked
  row-major executor (ahead of the vision router), v2 persistence (axis in
  axes[0]), forward clone, VJP via broadcast-back, general N-dim partial
  reduction in `accum_grad` plus the BROADCAST VJP (2D/4D proven paths
  kept). 3D `[2,1,4]+[2,3,4]` trains (was fail-loud) with exact grads;
  `audit_sum_axis_nd` 21/21 in `make check` (values 1..4-D, 2D
  equivalence, bad-axis reject, roundtrip, 3D exact, finite-diff 4.1e-12).
- R2-8: batched transpose (id 43, forward + persistable) unlocks
  `MATMUL_BATCHED` backward (`dA=dY@Bt, dB=At@dY`, finite-diff ~1e-10);
  dedicated `LAYERNORM_BWD[_GAMMA,_BETA]` (44..46), `RMSNORM_BWD[_GAMMA]`
  (47..48), `GELU_BWD` (49) kernels + pre-router executors + builders +
  VJPs incl. gamma/beta grads (finite-diff ~1e-9..1e-11); fault gate now
  asserts bmm success + GQA still fail-loud; `audit_train_bwd` 24/24.
  Attention/GQA/SwiGLU/RoPE backward stays fail-loud by staged scope.
- R2-9: `lancius_clip_global_norm` multi-tensor clip (+ audit proofs);
  `audit_train_converge` 12/12: 2-4-1 tanh XOR via graph+autodiff+SGD
  solves 4/4 in 28ms, same-seed rerun agrees 1e-12, v2 checkpoint at
  half-time resumes to the same loss 1e-9. Optimizer moments stay
  caller-owned (stateless lib contract); weights checkpoint via v2.
- No format break (ids appended; old files load identically), no
  stable-ABI break (additive builders + widened success surface).


## despot audit V8 (2026-10-03) — external-truth re-audit

Same audit as V7, re-proven against live external sources (not repo lore):
NumPy/PyTorch oracles, ONNX spec, C11/POSIX/OpenMP/CMake/Python docs,
plus independent re-execution of every AI-output claim. 8 confirmed
defects, all fixed and re-proven by `make clean && make -j && make check`
+ `check-sanitizers` + `audit_pytorch_parity.py` green (`3.42e-07`).

- Math: `conv2d_bwd` builder + executor verify grad `[N,C_out,eH,eW]`
  incl. grad-C vs weight-C_out (was silent wrong-channel + OOB; V7 had
  closed bwd_w/maxpool_bwd, this hole remained).
- Programming: stable tensor handles are magic-tagged wrappers borrowed
  from their graph (were raw `node*`, any forged pointer derefed);
  wrong-type/stale wrappers return `INVALID_HANDLE`; graph destroy
  invalidates + frees all wrappers. ONNX Reshape honors `allowzero`
  (explicit-0 raises; 0+-1 mix raises). `export_lancius_onnx.py`
  layout guard is `raise ValueError` (was `assert`, stripped under -O).
  Makefile sets `.DEFAULT_GOAL := all` with depfiles trailed last (bare
  `make` built only `arena.o` after any clean build — incremental builds
  were silently stale for all later edits).
- Converter truth (parity was RED): Reshape shape tensors no longer
  emitted as 1D INPUTs (loader rejects 1D by design); Reshape carries
  data-input only; Relu/Add/Flatten/MatMul ndim follows data rank (2D
  RELU was emitted 4D). Fresh LeNet converts 21 nodes (was 22 with
  helper), loads OK, parity `3.42e-07` bit-for-bit the documented value.
  `parity_runner` + `lancius info` print `(err=N: string)` on reject.
- AI outputs: `train_micromodel` is feat-agnostic (`feat=xn/tn`, any
  1..64; 16-dim distill bins load `32 rows x 16`, loss falls; rejections
  say shape-rejected vs not-found honestly); synthetic stays 8-dim with
  `0.291094->0.000010` endpoints. `eval_verifier` De Morgan is a real
  check (`max(-a,-b)` independently + distinct-input case).
- Proven: `make check` green from clean tree; `check-sanitizers` green;
  parity `3.42e-07`; micromodel synthetic + 16-dim both fall;
  `allowzero=1` + `0+-1` both raise; no `assert` in exporter/bridge/
  converter guards.


## despot audit V7 (2026-10-03) — math, programming, operational truth

Hostile formula-by-formula, line-by-line audit of the entire system.
17 confirmed defects, all fixed and re-proven by `make check` green from a
clean tree (`make clean && make -j && make check`), `-Werror` clean,
finite-difference grad check unchanged, plus targeted execution proofs in
`temp/proofs/`. No format break; no stable-ABI break beyond additive
`INVALID_HANDLE` strictness (wrong-type/stale handles now fail loud).

- Math: per-channel INT8 execution refused loud (`UNSUPPORTED_OP`, dequantize
  first) — was silent smax mis-scale (200x error, err 0); `conv2d_bwd_w` +
  `maxpool2d_bwd` builders + executors verify grad N/C/H_out/W_out — was
  silent partial/wrong gradients (63 vs 1053) + OOB; `test_grad_check`
  copies analytic grads off the scratch arena before reset — was
  use-after-reset by allocation luck.
- Programming: conv overflow guards set `OVERFLOW` (were silent success);
  layernorm/rmsnorm/attention/GQA/conv-bwd OpenMP worker errors propagate
  via shared flags (were `_Thread_local`-lost); v1/v2 save paths set
  `NULL_PTR/IO/OOM/OVERFLOW/LIMIT/INVALID_DTYPE` (were bare -1) and
  `save_stable` maps internal causes (no blind IO); KV-cache API reports
  through the error channel; VM compile early-NULLs + all ~30 execute
  rejects set codes (OOM vs corrupt tape distinguishable); `pool_wait`
  sets `NULL_PTR/INTERNAL/LIMIT`; RoPE `2*head_dim` wrap guarded;
  stable handles carry magic tags (`INVALID_HANDLE` on mismatch) with
  documented lifetimes; `vectors_to_lancius.py` raises `ValueError`
  (no `assert`); `fsync` failures unlink + report `IO` (no false success).
- Operational: CMake globs `src/train/*.c` (was missing train lib);
  `make install` banner `v12R2`; `.gitignore` covers `*.onnx.data` + `temp/`;
  version identity reconciled to `v12R2/V1.2RC2` (README/STATUS/MANIFEST/
  ARCHITECTURE/CONTRIBUTING fixed; `V1.2-RC2` hyphen unified to `V1.2RC2`);
  STATUS validation batch renamed v12R2 + R2 theme + `info/run/eval` +
  manual-only pointers + `quickstart/eval` verbs; `v12R2_SCOPE` V4/V5
  trimmed to essence+pointer (single-owner); KNOWN_LIMITATIONS documents
  per-channel refuse, magic handles, VM/pool/KV error channels, grad-shape
  guards, RoPE guard.
- Proven: `make check` green; `temp/logs/` + `temp/proofs/` hold rebuild,
  gate, CMake-configure, and per-channel/grad-shape/ABI-manual proofs.


## v12R2 (in progress) — mute mathematician <100M, generation scrapped

- Language generation scrapped: `examples/generate_text.c`/`run_llm.c`
  deleted, `makefile` targets + `clean` purged, CLI `generate` fails loud
  (`Use lancius eval`), doctor/TUI switched to `train_micromodel`;
  transformer kernels stay as exact math primitives with known-answer audits.
- Target retargeted: micromodels below 100M params that score math
  (olympiad/millennium), never speak; R2-2 problem encoder `char-v1`
  decision (BPE rejected); R2-6 no streaming generation; docs (`README`,
  `KNOWN_LIMITATIONS`, `STATUS`, `v12R2_SCOPE`) updated.

## v12R2 (in progress) — learn to train micromodels (R2-1..R2-6)

- R2-1: `src/train/lancius_train.c` + `lancius_train.h` (SGD/SGDM/AdamW-decoupled/clip/cosine/warmup-cosine), proven by `audit_train_lib` (SGD/SGDM/AdamW exact, clip scale, schedules, NULL-safety, loss monotonicity).
- R2-2: `text_tokenizer.py` problem encoder `char-v1` (BPE rejected for v12R2: scoring needs bytes, not speaking) + versioned sidecar + SHA256 manifests, proven by `audit_text_pipeline.py` (ASCII/astral roundtrip, sidecar, manifest + tamper-fail).
- R2-3: `train_micromodel` bridge (vendored `.X.bin/.T.bin` or deterministic synthetic, 8->16->1 tanh/MSE, SGD 200 iters, loss `0.291094->0.000010`); `vectors_to_lancius.py` validates rows (no fake graph).
- R2-4: `eval_verifier` harness (tanh/MSE anchors, residual `v(r)=(d-r)/(d+r)` at `0,d/3,d,3d,inf`, weakest-link min, ⊥=-2 absorbing, Gödel idempotent, thresholds); models-side only, no IR opcodes.
- R2-5: `lancius_sandbox.h/.c` (graph caps via liveness peak + node/step bounds + attr finite; weights NaN/Inf abstain; replay bit-identical), proven by `audit_sandbox`.
- R2-6: CLI `eval` verb (vendored e2e), `audit_abi.py` (version + headers/sources), `audit_binding_smoke.py` (headers-only), FP32 parity via existing `audit_fp32_path` 19/19; umbrella headers export train+sandbox; CMake milestone `v12R2`.
- Gates: `make check` extended (train-lib, sandbox, micromodel, eval_verifier, text/abi/binding, `lancius eval`); `check-long` + `check-sanitizers` green.

## despot audit V6 (2026-10-01) — math, runtime, persistence, ops

Hostile formula-by-formula, line-by-line audit of the entire system.
45 defects confirmed by execution and fixed, re-proven by `make check`,
`make check-long`, `make check-sanitizers`, finite-difference grad check
(`8.6e-10`, `5.8e-8`), despot probe, and new `probe_v6`
(`broadcast dim0/1/2/3 4D`, `CE_BWD`, VM rank, layernorm builder).

- Math: BROADCAST 4D backward rewritten exact via permute+reshape+sum_axis
  (was dim2/3 RESHAPE_MISMATCH + dim0/1 INVALID_RANK on 4D; dim2/3 flatten
  was mathematically wrong, summing post dims too); `CE_BWD` ctor requires
  2D + scalar grad (was silent g[0]-only); `CE_BWD` exec requires xe==ye==R*C
  + ge==1 (was OOB); `conv_bwd_w` stride/pad guards (was missing);
  LAYERNORM/RMSNORM exec require input elems == output elems (was OOB).
- Runtime: VM ndim!=2 rejected (was 0D/1D compiled to Rx0); compiler
  validates inputs before reg_map (was NULL+OOB); VM tape OOB guarded;
  out_reg + program invariants validated; RELU/SOFTMAX input==output dims,
  SUM out==1x1 enforced; scheduler validates inputs before deref on
  ATTENTION/ADD/MATMUL/MUL/SUB/TRANSPOSE/SUM/BROADCAST/GQA (was NULL-deref);
  all silent returns set NULL_PTR/GRAPH_INVALID; parallel skips set errors;
  threadpool create sets OOM/INTERNAL, submit sets LIMIT/OOM/GRAPH_INVALID,
  grow is malloc+linearize+free (was realloc-UAF + leak + silent drop);
  arena grow checked + fit re-check (was wrap to 16MB + OOB); static pool
  32B-aligned base required on plan path, bump path aligns start;
  CLI uses posix_memalign 32B.
- Persistence: v2 save mkstemp+fsync+rename (was predictable .tmp);
  per-channel save refused (was silent drop); ftello/off_t + empty graph
  savable (was long truncation + reject); loader sets errors everywhere,
  ndim==0 rejected, INPUT 2/3/4-D only, CONST 1..4-D round-trips, ROPE
  persistable, INT8 scale >0 finite validated; v1 save tmp+rename + FP32
  branch (was truncate + drop); v1 load errors + INT8 scale + FP32 branch;
  quantizer never clears sticky error, per-tensor drops stale per-channel,
  dequant checks per-channel scale + frees stale FP64; fusion requires shape
  equality; conv_bwd/maxpool_bwd/gqa builders validated; vision BWD
  validated + NULL_PTR; runtime queries set errors; stable API scratch
  allocate-first.
- Ops: `manage_datasets.py` chunked capped MNIST/CIFAR (was unbounded
  read); `onnx_to_lancius.py` 2GB pre-stat (was unbounded load);
  `distill_prm800k` realloc tmp, fopen/malloc/fwrite/fseek checks;
  `run_trained_batch`/`parity_runner` unified cleanup + checked elems +
  short-read/write fail; `train_cifar10` tar-slip member validation;
  `make clean` removes `.d` files.

## hardening batch V5 (2026-09-30) — threadpool, IR honesty, example hardening

Comprehensive bug-fix campaign across threadpool, IR, compiler, runtime,
examples, and Python tooling. 12 defects fixed and re-proven by
`make check`, `check-sanitizers`, and despot truth probes.

- Threadpool: queue growth use-before-initialization fixed (was reading from
  uninitialized `nq` buffer during realloc; now reads from `pool->queue`).
- Security: command injection in `distill_prm800k` fixed (replaced `system()`
  with `mkdir()`).
- IR: silent NULL returns fixed — `lancius_matmul_batched`,
  `lancius_cross_entropy`, `lancius_reshape`, `lancius_flatten`,
  `lancius_gqa`, `lancius_permute` now set error codes on validation failure.
- Optimizer: error clearing on success fixed (no longer masks prior errors).
- Quantizer: zero-scale check added in `lancius_dequantize_graph` (validates
  scale > 0 and finite before dequantization).
- Scheduler: cross-entropy consistency fixed (forward and backward now use
  the same R/C source; output shape validated).
- Memory planner: free block splitting now maintains 32-byte alignment.
- Stable API: duplicate `#include` removed.
- Serialization: `fprintf`/`printf` removed from library code (v1 and v2).
- Vision ops: `fprintf` replaced with `lancius_set_error`.
- Examples: `parity_runner` and `run_trained_batch` hardened (unchecked
  allocations, ignored I/O returns, NULL derefs fixed).
- Python: `onnx_to_lancius.py` shape filtering fixed (interior 1s preserved).
- Autodiff: NOP comment clarified (no null pointer dereference).

## hardening batch V4 (2026-09-28) — race conditions, memory safety, quantization, portability

Comprehensive bug-fix campaign across autodiff, kernels, serialization, CLI,
build system, and Python tooling. 26 defects fixed and re-proven by
`make check`, `check-sanitizers`, and pytorch parity.

- Autodiff: BROADCAST backward now correctly reduces over broadcast dimensions
  (was passing grad_out unchanged); NULL checks on `fwd_n->inputs` throughout;
  OOB reads on shape/axes arrays fixed (now pads to 4D); off-by-one in node
  capacity check fixed.
- Kernels: race condition in `kernel_conv2d_bwd_in` fixed (thread-local
  accumulators); race condition in MaxPool2D backward fixed (thread-local
  accumulators).
- Memory: memory leaks in IR node allocation fixed; `abort()` removed from
  library code (replaced with `lancius_set_error` returns); all `fprintf`/`printf`
  in library code replaced with `lancius_set_error`.
- Stable API: dangling `wrapper->sched` fixed; `set_owner` now updates
  `int8_owner`.
- Serialization: portability fixed (`uint64_t` sizing, byte swapping);
  CRC32 table init race fixed (`call_once`); NOP IDs no longer mapped to NULL;
  double-read for CRC eliminated (now computed during parsing).
- Threadpool: `lancius_pool_wait` now accepts a timeout parameter.
- Bytecode VM: overflow checks added.
- Quantization: per-channel quantization support added; dequantization support
  added.
- CLI: command injection fixed (now uses `fork+execvp` instead of `system`).
- Python scripts: security fixes, stale version references removed, error
  handling improved.
- Build system: version consistency enforced, `-Werror` added, Threads
  dependency fixed; `train_cifar10` now links `-lpthread`; dependency tracking
  improved; `.gitignore` updated with missing entries; `lancius.pc.in` version
  fixed.
- Code quality: magic numbers replaced with named constants throughout.

## despot audit V3 (2026-09-28) — every live bug found and implemented

Four forensic sweeps, ~70 code-backed defects, all fixed and re-proven
(`make check`, `check-sanitizers`, despot probe, pytorch parity 3.42e-07):

- Kernels: matmul/conv index-overflow guards, bwd_in stride guards,
  layernorm/rmsnorm degenerate is NUMERICAL (was silent beta/zeros), RoPE
  int-wrap guard, KV-cache hidden/malloc guards, NULL paths set errors.
- Scheduler: CE fwd R/C + shape guards, FP32/INT8 matmul K-match (was OOB),
  FP64 M*K/K*N overflow, elementwise input-size checks, ROPE qk-size +
  offset checks, softmax zero-guard + Inf, permute shape correspondence +
  checked indices, static-plan skips failed assignments, peak/required set
  errors (bounded executor cannot under-alloc), abort-free hot paths kept.
- Vision: inputs/input_count guards on every branch (was NULL+0 deref),
  conv H_out verification vs n->shape, pool index guards, INT8 scale NaN/Inf
  rejected, silent returns set errors.
- IR/autodiff: alloc_node rt-OOM returns NULL, matmul sets errors and stays
  2D (executor is 2D), f32 ownership split (was: leak + free of external),
  broadcast_4d true compat check, bwd_w validation, fused-clone rt attach +
  no aliasing + id-cap guards, loss-node membership/range checks, seed NULL
  aborts, inputs[2] guarded, INPUT clone keeps int8/f32/dtype/scale, NULL
  grad aborts (was neutral-skip), unhandled forward op aborts, permute axes
  validated, builders set SHAPE_MISMATCH (was silent NULL).
- Planner/threadpool/VM/optimizer/quantize: zero-size plan aborts (was
  offset-0 overlap), pool queue grows (was racy inline run) + shutdown
  reject + OOM error, VM materializes CONST (was garbage regs) + inputs
  checks, fusion validates stolen inputs, quantizer syncs rt scale + frees
  stale int8 + skips non-finite max.
- Persistence: v1 checked writes + tmp-less partials unlinked + NULL guards
  + no abort() + ndim 1/3 loads (was 2D coercion) + view-source validation +
  double-free removed (both weight paths + fail path); v2 tmp+rename saves,
  streamed CRC (no 800MB malloc, no long truncation, no seek-bypass),
  duplicate-NOP seen-list, invalid dtype fails, u64 narrowing kept.
- Interop/python: converter true input ranks (was: padded count always 4),
  Reshape rank from resolved dims, packed-field range checks, Gemm byte
  context, write errors + crc0->1; ONNX exporter 2GB cap + header checks +
  checksum==0 refuses (--allow-legacy opts in); pytorch exporter None/cap/
  arity/perm/stride checks + multi-input trace dummies; datasets no-shell
  git clone + cwd-jail cleanup + tar/zip-slip guards + download try/except.
- Operator/examples: CLI fork+exec for user paths (was: 4 shell-injection
  sites) + truncation fails + fread short-read fails + strtol topk/show +
  TUI drain + TUI show passed; trainers validate files/magic/counts/sizes,
  check OOM/graph/schedule, guard grad/loss NULLs, fix cifar eval counting
  (+evaluated denominator) + save check, fix verifier arena use-after-reset
  (heap copies) + worst-step max metric + per-iter g2 destroy + full cleanup
  on fatal paths; run_edge caps + cleanup + rc=1; generate_text cleanup +
  rc propagation; run_llm 3D builders + overflow + output checks.
- Proven: `make check` green (73/73, 265/265, despot probe, verifier with
  corrected worst-step 1.34→0.0013), `check-sanitizers` clean, pytorch
  parity exact.

## operator TUI perfection (2026-09-28) — ordinary users guided, offline honest

- New verbs: `status` (operational snapshot), `models [--check]` (list plus
  validate local models), `demo` (one-command install proof), `help [verb]`
  (plus per-verb `--help` everywhere; `run --help` no longer unknown-arg).
- `doctor` tells the truth: core MISSING fails, training data absent is an
  optional-data warning (was blanket READY), plus network (curl, 5s cap) and
  disk-free sections.
- `train` pre-checks data before work (suggests the exact pull command),
  `--dry` shows binary plus data plus time hint; verifier seconds, mnist
  minutes, cifar10 hours with an explicit are-you-sure in the TUI.
- `run` validates mode/fill/topk/show with suggestions, checks model and
  `--input` files first, explains `--input` size mismatches in bytes.
- `info` flags training-only (`_BWD`) and reserved (`EMBEDDING`, `KV_*`)
  ops, prints human dtype names, hints at `lancius models` on missing files.
- `convert`/`export` pre-check inputs, scripts, and python deps (no raw
  tracebacks); `datasets pull` validates names and refuses cleanly offline.
- TUI rebuilt around direct calls (no `./lancius` re-exec for internal
  verbs, error codes preserved), with header status, plain-language menu
  plus time guide, `help` entry, per-prompt defaults and back (empty),
  validation loops, cifar10 confirm, pause-after-command on ttys, last-status
  line, and `NO_COLOR`/`TERM=dumb`/non-tty safe output.

## despot truth V2 (2026-09-28) — every remaining lie closed, re-proven

- Autodiff truth: `broadcast_to_shape` (any 1..4-D scalar lift); `SUM` grad
  exact for any ndim (was `[1,1]`-for-3D); `SUM_AXIS0/1` + `RESHAPE` VJPs
  added (were silent drops); transformer forward cloned so `fwd_to_full`
  stays complete; `_BWD` in forward returns NULL; N-dim partial broadcast
  reduction fails loud (no `SUM_AXIS_ND` yet) with sticky-error abort of the
  whole training graph; `accum_grad` returns `1/0` with all `NULL` paths
  checked; `add/sub/mul` set `SHAPE_MISMATCH` (was silent NULL).
- Executors: permute stride products checked; batched-matmul batch offsets
  checked (`M·K`, `K·N`, `M·N` + `batch·elems`); liveness/static/parallel
  paths use `_checked` bytes/elements (no abort); `SUM`/`SUM_AXIS`/`TRANSPOSE`
  shape-validated; INT8 Add row-bias-only with FP64 fallback; INT8 matmul
  `scale_a=1.0` for all-zero (consistent with quantizer skip).
- Kernels: attention/GQA `NaN→NUMERICAL` (was silent zeros); KV-cache
  `max/sum NaN→NUMERICAL` (`-inf`/zero stay zeros); GELU documented as
  tanh-approx; `LANCIUS_NORM_EPS=1e-5` pinned.
- Vision: INT8 zero-scale reports `NUMERICAL` (was `INVALID_SHAPE`);
  `MAXPOOL_BWD` output bytes checked.
- Persistence: `checksum==0` doc fixed to rejected-by-default; `u64→size_t`
  narrowing checked; `BROADCAST` loads any 1..4-D.
- ONNX: Conv `dilations/group/auto_pad` rejected; MaxPool
  `pads/dilations/ceil_mode/auto_pad` rejected; Reshape dead code removed.
- Training honesty: CIFAR raw-loss abort (`NaN/>1000/<0` returns 1);
  MNIST raw-loss abort + accuracy gate; both trainers exit 1 at ≤ chance;
  `test_ffi_error` exits 1 on unexpected success/non-NULL handle.
- Docs: new `docs/DESPOT_TRUTH_V2.md` (full math/programming/operational
  audit with formulas and proofs); `docs/ARCHITECTURE.md`,
  `KNOWN_LIMITATIONS.md`, `MANIFEST.md`, `STATUS.md` updated; temp execution
  under `/tmp/opencode/lancius-despot-logs`.
- Proven: `make check` green; `test_grad_check` (`8.6e-10`, `5.8e-8`);
  `probe_v2` (SUM-3D `BROADCAST ndim3`, RESHAPE, SUM_AXIS, `broadcast_to_shape`,
  N-D partial fail-loud, attention NaN→NUMERICAL) all truth holds.

## v12R2 (unreleased) — learn to verify

First work toward the R2 mandate (see `docs/v12R2_SCOPE.md`): the framework
gains only generic trainable primitives so models can learn the
extended-boolean verifier scheme. Truth semantics stay models-side.

- Core: `TANH`/`TANH_BWD` (bounded activation saturating the fluid scale)
  and `MSE`/`MSE_BWD` (regression against step labels in [-1,1]) across
  IR builders, scheduler executors, autodiff VJPs, and v2 loader
  forward cases (ids appended at the end; v11S+ files load identically).
  `_BWD` nodes stay non-serializable, like all training artifacts.
- Gates: 6 new known-answer checks (tanh(0/±1), MSE == 1/3); new
  `train_verifier_head` example trains a tanh-headed MLP by MSE with
  weakest-link credit through the argmin step — loss falls, analytic
  gradients match finite differences to ~1e-11, scores stay in [-1,1],
  all in `make check`.
- Pipeline: `distill_prm800k.py` converts PRM800k step rows to fixed-dim
  numeric vectors + {-1,0,+1} targets with manifests and sha256;
  honestly reports the vendored subset is single-class (+1 only).
- Python-to-C: stdlib-only `distill_prm800k.py` retired in favor of
  `examples/distill_prm800k.c` (self-contained JSON + sha256, no
  dependencies) — byte-identical `.bin` output on all 14,564 vendored
  steps, ASan/UBSan clean including hostile inputs, `--selftest` proofs
  in `make check`. Third-party-bound scripts (torch/onnx/ort/network)
  stay in Python: porting those buys nothing but difficulty.

## v12R1 / V1.2RC1 — 2026-09-21 — hardening plus numerical correctness

First development milestone of the v12 cycle (R1 phase), built on the v11S
stable baseline. Three batches, no format / stable-ABI break beyond additive
error codes:

Hardening batch:

- Scheduler: broadcast column/identity fix, RMSNorm 3D hidden fix,
  parallel FP32 dispatch + FP32 reset hygiene, cycle returns NULL +
  GRAPH_CYCLE, matmul overflow guards, XEnt-bwd numerical guard,
  ROPE even-dim check, permute/batched validation, pool offset bounds
  (`plan->max_id`), parallel/CONST OOM errors.
- Bytecode VM fails loud on unsupported ops; VM input/dim checks.
- `onnx_to_lancius.py`: strict (unknown op / unmapped input / bad perm /
  unresolvable Reshape raise), real CRC32 in header.
- Arena/IR: create/alloc/track/realloc guards; builders validate shapes
  (attention, layernorm/rmsnorm gamma, swiglu, broadcast); fixed
  `bind_external_int8`, `set_owner`, `release_owned` FP32 leak.
- Kernels: null/zero/overflow guards everywhere; attention/GQA
  `-INFINITY` causal sentinel; GELU clamp removed; RoPE odd-dim refuse;
  KV-cache zero-len guards; MaxPool `-INFINITY` + NaN propagate;
  INT8 zero-scale is now a loud error; quantizer clamp + skip-if-INT8.
- Planner/threadpool: all allocs checked, id bounds, overflow-safe
  offsets, threadpool create/teardown hardening.
- Persistence: v2 save CRC fail-closed + `w+b` read-back fix (was silent
  crc=0), v1 loader ndim/duplicate-id/unknown-op hardening, v2 header
  reserved-field + duplicate-NOP rejection, stable API error-map
  completion + FP64-only `read_output` honesty + checked counts.
- Audits updated to `-INFINITY` / unclamped GELU references.
- Removed 15 stale `*.bak*` / `*backup*` files.

Numerical-correctness batch (hostile audit of every numeric path; each
confirmed defect fixed and re-proven by independent execution):

- Scheduler/IR/VM: N-dim broadcast `ADD`/`SUB`/`MUL` (IR emits
  `max`-per-dim output shape; scheduler + VM execute strided broadcast;
  was flat-loop wrong + OOB); softmax zero-sum guard in scheduler and VM.
- Persistence: v2 CRC required by default (`LANCIUS_ALLOW_LEGACY_UNVERIFIED=1`
  opts into legacy unverified loads); O(1) duplicate-ID detection;
  sparse-ID bounds; streaming weight skip; sticky-error clearing on success.
- Runtime: attention cache/heads/dim validation, cache-less long-context
  decode rejected, GQA Q/K/V shape checks, RMSNorm gamma/divisibility
  checks, `_checked` counters on execution/planning/copy paths,
  `FLATTEN`/`RESHAPE` element-equality verification.
- Kernels: INT8 `int64` accumulators (conv + mixed-precision matmul), real
  overflow guards (dead `&& 0` removed), OOM error reporting for
  thread-local and Flash/GQA scratch buffers.
- Core: arena failures carry error codes; `dtype_size(invalid) = 0` with
  `bytes_checked` rejection; `checked_product_shape(NULL)` fails.
- Stable API: additive error codes `GRAPH_CYCLE`/`OVERFLOW`/`NUMERICAL`/
  `INVALID_HANDLE` replace lossy collapsing.
- ONNX: Reshape `0`-copy vs `-1`-infer, Gemm `transA`/`alpha`/`beta`
  rejection + transpose clone-on-write, symmetric-only Conv pads/strides
  and square-only MaxPool, symbolic-dim rejection, static-batch export.
- Training: CIFAR-10 `[-1,1]` normalization matching PyTorch reference,
  He init on MNIST, identity-based parameter binding, per-batch grad
  zeroing, identity-tracked eval logits, raw-vs-clamped loss logging.
- Audits: `audit_modern_llm`, `audit_flash_attention`,
  `audit_threadpool_parity`, `audit_ffi`, `audit_pytorch_parity.py` now
  propagate failures via exit codes.

Despot truth batch (every remaining lie found and implemented):

- Planner records freed offsets: diamond-graph reuse no longer overlaps
  live tensors (was silent corruption on second reuse).
- Broadcast upgraded to trailing-rank (NumPy) semantics in validator, IR
  builders, and scheduler, with per-dim compat guards and bounds checks;
  incompatible shapes rejected, never read out of bounds.
- Conv index math in `int64` (was `int` truncation); KV-cache attention OOM
  reports instead of emitting zeros; cross-entropy forward degenerate
  denominator is `NUMERICAL`, matching backward (was `1e30` sentinel).
- Autodiff: scalar-reduction OOB fixed, 4D `CONST` cloning preserved,
  incompatible accumulation fails loud instead of dropping gradients.
- Scheduler fail-closed on wave-cap overflow; static executor uses
  `_checked` sizing with FP32 binding and NOP skipping; sanitizer gate
  restores a clean build afterwards.
- Audits: `audit_nan_injection`, `audit_memory_pool`, `test_grad_check`,
  `test_path_bg`, `test_torture` (real cycle test), `fuzz_lancius`
  (safe-rejects separated), `audit_internals` (softmax normalization),
  `audit_modern_llm` (GQA values) all propagate failures; new
  `audit_despot_probe` pins broadcast/planner/CE truth in `make check`.
- ONNX: Gemm always clones on transpose (never mutates the shared
  initializer — was order-dependent double-transpose); Reshape preserves
  the batch dim (was forced to 1); `audit_trained_reality.py` exits
  nonzero below 95/100.
- Stable API oversized-input hole proven closed by `audit_fault_injection`
  (11/11).

## v11S / V1.1 — Stable Release

### Changed
-   Model format v2 frozen with CRC32 body integrity verification
-   EXTERNAL_WEIGHTS flag reserved and rejected by loader
-   Stable C API expanded: model load/save, tensor introspection
-   `lancius_read_output` rejects undersized buffers (no silent truncation)
-   Scratch arena auto-sized from liveness analysis
-   Version identity consistent across all documents
-   Full validation suite green (make check, check-long, sanitizers)

### Security
-   CRC32 integrity check on model body (bytes 48..EOF)
-   Reserved header flags rejected at load time

## v11A3 / V1.1-AlphaRC3 — Freeze & Hardening

### Changed
-   Feature freeze: no new operators, runtime subsystems, or training features
-   Loader and model-format validation hardening
-   Sanitizer and fuzz validation expansion
-   Regression-test expansion
-   Documentation cleanup and version-identity consistency
-   Release-candidate preparation for v11S

## v11A2 / V1.1-AlphaRC2 — Transformer Runtime Usability

### Added
-   Dedicated KV-cache runtime object
-   Cache-aware attention execution
-   Explicit prefill and generation execution flows
-   First-class 3D transformer tensor construction (`lancius_input_3d`)
-   Transformer known-answer audit (LayerNorm, RMSNorm, GELU, SwiGLU, RoPE,
    attention, KV-cache parity, prefill/generation parity, GQA)
-   FP32 execution foundation (matmul kernel, scheduler dispatch,
    serialization, path audit)
-   Stronger v2 loader validation

### Improved
-   Reduced demo-style graph mutation in transformer examples
-   Scheduler buffer lifecycle and repeated-execution hygiene

## v11A1 / V1.1-AlphaRC1 — Foundation & Runtime Honesty

### Added

-   Development milestone contract
-   Hardened runtime validation
-   Expanded testing infrastructure
-   Stable API boundary
-   Improved documentation

### Improved

-   Memory safety checks
-   Graph validation
-   Numerical stability
-   Runtime reliability

### Removed

-   Experimental GGUF export path from stable distribution

### Deferred To Future Releases

-   GPU backends
-   Dynamic execution
-   Expanded training support
-   Additional deployment targets
