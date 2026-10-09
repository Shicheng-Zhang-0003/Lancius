# How Lancius Works

The canonical guide to how a model is **defined, processed, executed, trained,
and proved correct** in this framework. `docs/ARCHITECTURE.md` gives the shape
of the system; this document gives the mechanics, with the actual API names,
files, and numbers so that a reader can follow a model from bytes on disk to
numbers coming out, and can add to the system without guessing.

Everything here is checked against the tree. Where a claim is a **contract** it
is enforced by a gate; where it is a **limitation** it is listed as one.

---

## 1. What this framework is

Lancius is a C machine-learning compiler and runtime for **bare-metal inference**:
static graph execution, explicit memory planning, and low-level control over
numeric types and execution order. It is not a tensor library and not a Python
framework. There is no autograd tape, no reference counting, no allocator
hiding in the background.

Roughly 8,350 lines of implementation, 1,182 of headers, 11,382 of examples and
audits, 2,754 of external-oracle and mutation infrastructure.

Design commitments, in priority order:

1. **A wrong answer that reports success is worse than a loud failure.** Every
   operation that cannot do what was asked refuses to build, and the refusal
   propagates through a thread-local error channel that `lancius_get_error()`
   reports.
2. **Self-consistency is not correctness.** A wrong constant, or a flipped
   convolution kernel with a matching flipped backward pass, satisfies every
   finite-difference check. So the load-bearing checks are *external*: they
   recompute results with NumPy, PyTorch, and hand-derived closed form.
3. **Every claim is a gate a machine runs.** `make check` and its siblings run in
   CI on every push; nothing here depends on a human remembering to ask.

---

## 2. The processing pipeline, concretely

Eight stages. Each names the call that performs it and the file that implements
it.

```
  .lancius file / C builder calls
            |
            v
   [1] PARSE OR BUILD          lancius_graph_load()            src/core/lancius_serialize.c
            |                   (or lancius_input/_matmul/...  src/ir/lancius_ir.c
            |                    building nodes directly)
            v
   [2] IR / VALIDATE          lancius_validate_graph()        src/core/lancius_validate.c
            |                   shape, rank, dtype, power-of-two
            v
   [3] SCHEDULE                lancius_ir_schedule(g)          src/runtime/lancius_scheduler.c
            |                   -> lancius_wave[] + memory plan
            v
   [4] MEMORY PLAN             lancius_memory_plan_*()        src/runtime/lancius_memory_planner.c
            |                   liveness -> offsets -> arena layout
            v
   [5] EXECUTE                 lancius_schedule_execute()      src/runtime/lancius_scheduler.c
            |                   (…_parallel for the thread pool)
            |                   (…_static for a flat buffer)
            v
   [6] KERNELS                 kernel_*()                     src/math/lancius_kernels.c
            |                   the actual arithmetic
            v
   [7] RESULT                  node->runtime_data              read by the caller
            |
            v
   [8] (training only)         lancius_ir_autodiff()          src/math/lancius_autodiff.c
                                -> lancius_training_graph, then backprop through
                                   lancius_*_bwd ops
```

### [1] Parse or build

Two ways in, and they produce the same thing.

**From a file.** `lancius_graph_load(path)` reads the v2 format (magic
`0x32434E41`, explicit version `2`, fixed-width little-endian fields, CRC32
required by default; `LANCIUS_ALLOW_LEGACY_UNVERIFIED=1` opts into v1 legacy).
Implemented in `src/core/lancius_serialize.c` and
`src/core/lancius_serialize_v2.c`.

**From code.** The builders in `include/lancius/lancius_ir.h` construct nodes
directly:

```c
lancius_graph*  g = lancius_graph_create();
lancius_node*   X  = lancius_input(g, rows, feats);
lancius_node*   W1 = lancius_input(g, feats, hidden);
lancius_node*   b1 = lancius_input(g, 1, hidden);
lancius_node*   z1 = lancius_add(g, lancius_matmul(g, X, W1), b1);
lancius_node*   z2 = lancius_matmul(g, lancius_tanh(g, z1), W2);
lancius_graph_destroy(g);
```

Weights reach a node through `lancius_node_bind_external(node, ptr)` (the caller
keeps ownership) or `lancius_node_bind_owned_heap(node, ptr)` (the graph takes
ownership and frees it). **Getting this wrong is the classic leak**: assigning a
`calloc` straight to `node->runtime_data` means `lancius_graph_destroy`
correctly declines to free memory it does not own, and the buffer escapes every
run.

**A v2 file stores node values only for `LANCIUS_OP_INPUT` nodes with
`runtime_data` bound.** A `LANCIUS_OP_CONST` node carries a single scalar
`attr_val` and nothing else. So a frozen model must hold its weights as bound
INPUT nodes and its features as an unbound INPUT. The batch dimension is
**fixed in the file** — node shapes are written and restored, so it cannot be
changed after load.

### [2] Validate

`lancius_validate_graph()` checks rank, shape, dtype, and power-of-two
alignment requirements *before* any execution. This is the layer that turns
"silently wrong" into "refused".

Two structural rules worth knowing because they are easy to get backwards:

- **`broadcast_to_shape` follows the NumPy rule** — trailing-rank alignment with
  per-dimension `a == 1 || a == out[i]`, leading missing dimensions treated as 1.
  This must be at least as permissive as the executor, or documented ops become
  unbuildable while a sibling op works.
- **`ADD` on a `[1, H]` bias against a `[R, H]` tensor is a real broadcast**, and
  the backward pass must reduce over axis 0. Getting this wrong yields a gradient
  that is shaped correctly and numerically meaningless.

### [3] Schedule

`lancius_ir_schedule(g)` walks the graph in dependency order and emits a
`lancius_schedule`:

```c
typedef struct {
    lancius_wave*      waves;        /* execution order */
    uint32_t           wave_count;
    struct lancius_memory_plan* plan; /* offsets and lifetimes */
    void*              static_pool;  /* flat buffer, for static execution */
} lancius_schedule;
```

**Waves** are the scheduling unit: nodes whose inputs are all ready can execute
together. This is what makes `lancius_schedule_execute_parallel()` meaningful
and what the thread pool consumes.

### [4] Memory plan

The planner computes per-node lifetimes from the schedule and assigns arena
offsets so that buffers of nodes with disjoint lifetimes can share space. This
is the "memory planning" in the pipeline diagram, and it is a static analysis,
not a runtime allocation. **V9: any planning error returns NULL** (never a
partial plan with a sticky error that poisons a later autodiff abort check).

The bytecode VM has the same fail-closed shape: `lancius_vm_execute_checked`
requires the caller's `out_len` to cover the program output, else `LIMIT`;
a trailing `HALT` no longer falsely fails (V9 fixed a 3-word guard that
rejected `code_len=5` programs).

### [5] Execute

Three execution modes, and they are not interchangeable:

| Call | Buffer | Use |
|---|---|---|
| `lancius_schedule_execute(sched, arena)` | a `lancius_arena*` | general |
| `lancius_schedule_execute_parallel(sched, arena, pool)` | arena + thread pool | multi-core |
| `lancius_schedule_execute_static(sched, flat_buffer)` | one flat `void*` | bare metal |

**The arena** (`src/core/lancius_arena.c`) is a bump-pointer block allocator with
a **32-byte SIMD alignment contract** — required because the build uses
`-mavx2 -mfma`. Sizes are rounded up to a 32-byte footprint; alignment requests
are honoured and capped at 1 MB (an uncapped alignment is an OOM-DoS vector, and
mutation testing proves the cap is load-bearing).

One subtlety that has bitten three separate times in this repository and is
worth stating plainly: **the alignment *argument* only affects
`ALIGN_UP(ptr, alignment)`.** If the arena's *base* already happens to be
32-aligned, then alignment 16 and alignment 32 return the identical address, so
`if (alignment == 0) alignment = 32` can be mutated to `16` and become
completely unobservable. Any test that asserts on returned pointers is therefore
testing the allocator's luck as much as the library's contract.

### [6] Kernels

The arithmetic lives in `src/math/lancius_kernels.c` behind `kernel_*`
functions (`kernel_gelu`, `kernel_layernorm`, `kernel_rmsnorm`, `kernel_swiglu`,
`kernel_conv2d_fwd`, `kernel_attention`, `kernel_cosine`, …). The scheduler
dispatches to these; a kernel never allocates and never reports its own errors —
they surface through the thread-local error channel.

### [7] Result

Node outputs live in `node->runtime_data`. They are valid after
`lancius_schedule_execute` returns and are invalidated by the next
`lancius_arena_reset`.

---

## 3. The operation set

50 opcodes in `include/lancius/lancius_ir.h`. They fall into five families:

| family | ops |
|---|---|
| **structure** | `INPUT`, `CONST`, `RESHAPE`, `PERMUTE`, `FLATTEN`, `TRANSPOSE`, `BROADCAST`, `NOP`, `SUM`, `SUM_AXIS0`, `SUM_AXIS1`, `SUM_AXIS_ND` |
| **arithmetic** | `ADD`, `SUB`, `MUL`, `MATMUL`, `MATMUL_BATCHED`, `TRANSPOSE_BATCHED`, `EMBEDDING` |
| **activations** | `RELU`, `TANH`, `GELU`, `SWIGLU`, `SOFTMAX` |
| **normalisation / attention** | `LAYERNORM`, `RMSNORM`, `ATTENTION`, `GQA`, `ROPE`, `KV_CACHE_READ`, `KV_CACHE_WRITE` |
| **vision** | `CONV2D`, `CONV2D_RELU_FUSED`, `MAXPOOL2D` |
| **loss** | `MSE`, `CROSS_ENTROPY` |
| **backward** | `RELU_BWD`, `TANH_BWD`, `GELU_BWD`, `SOFTMAX_BWD`, `LAYERNORM_BWD`, `LAYERNORM_BWD_GAMMA`, `LAYERNORM_BWD_BETA`, `RMSNORM_BWD`, `RMSNORM_BWD_GAMMA`, `MSE_BWD`, `CROSS_ENTROPY_BWD`, `CONV2D_BWD`, `CONV2D_BWD_W`, `MAXPOOL2D_BWD` |

**conv2d semantics**, because they are the most commonly wrong:

- It is **cross-correlation** — the kernel is **not** flipped. Deep-learning
  `conv2d` (PyTorch, ONNX, Caffe) is cross-correlation; the mathematical
  convolution of LeCun 1998 and MATLAB flips the kernel. Verified against
  `torch.nn.functional.conv2d`.
- Output shape `floor((H + 2*pad - K) / stride) + 1` — floor, not round.
- **Zero padding only.** `reflect`, `replicate`, and `circular` are not
  supported. `stride` and symmetric `pad` are the only parameters; there is no
  dilation and no groups.
- **No bias** in the signature. Bias is a separate `ADD` node.
- Getting the flip wrong is invisible to finite differences, because a flipped
  forward with a matching flipped backward is self-consistent. Only an
  independent engine catches it.

**softmax** normalises along the **last axis of the declared rank**. A softmax
that divides by the wrong denominator still sums to one per row, so "rows sum to
1" is not evidence of correctness.

**Reductions** keep rank: `SUM_AXIS0` on `[R, C]` gives `[1, C]`, not a scalar.

---

## 4. Training

```
build forward graph  ->  lancius_ir_autodiff(g, loss)  ->  training graph  ->  execute  ->  harvest grads
```

`lancius_ir_autodiff()` (`src/math/lancius_autodiff.c`) takes the graph and the
loss node, and returns a `lancius_training_graph` whose `grad_nodes[]` maps each
forward node id to a node holding `d(loss)/d(that node)`.

Three things to know:

- **Fail-loud, not fail-partial.** If a VJP cannot be written, autodiff returns
  `NULL` and sets an error rather than returning an empty gradient graph that a
  caller might treat as "no gradient".
- **Gradient shape is validated, not assumed.** A gradient whose shape does not
  match its parameter is a defect, and there is a standing assertion that it is
  caught.
- **The optimiser is separate.** `include/lancius/lancius_train.h` provides
  `lancius_sgd_step`, `lancius_sgdm_step`, `lancius_adamw_step`,
  `lancius_clip_grad_norm`, `lancius_clip_global_norm`, `lancius_lr_cosine`,
  `lancius_lr_warmup_cosine`. The model-side loop applies harvested gradients;
  the framework does not own the training loop.

A recurring and non-obvious fact when building an objective out of existing
ops: `MSE_BWD` emits `(2/pe) * g * (p - t)`. For a loss of `mean((logits -
onehot)^2)` over `pe` elements that is exactly the softmax cross-entropy
gradient times the constant `2/NCLASS` — so the same graph trains a softmax
classifier without a cross-entropy op, provided the learning rate accounts for
that factor.

---

## 5. The verification discipline

This is the part that makes the rest trustworthy, and it is layered on purpose.

| gate | what it proves | what it cannot prove |
|---|---|---|
| `make check` | every audit's exit code; shape, dtype, and error-channel contracts | that the maths is right |
| `make check-long` | the slower audits | same |
| `make check-sanitizers` | no leaks, no UB, no OOB — over the **library**, not just the audits | correctness |
| `make check-oracle` | results recomputed by NumPy / PyTorch / closed form, sharing no code and no author with the implementation | that the oracle is itself right |
| `make check-ubstrict` | signed overflow and float-cast overflow **abort** rather than warn (`-fno-sanitize-recover=all`) | correctness |
| `make check-mutation` | the gates can **fail**: real defects are injected and the gate must go red | that every defect is covered |

Current standing state: **276 external comparisons, 0 skipped.** Individually:
external oracle 46/46, graph-level 37/37, train-lib 169/169, conv2d vs torch
11/11, reductions+softmax vs torch 13/13. Mutation gate: 18 caught, 0 holes.

### The failure mode this discipline exists to catch

Across nine audit passes, **seven of the "findings" were defects in the checking
harness, not in the library** — a `clip_grad_norm_` reference that clipped the
parameter instead of `.grad`; a dump that recorded pre-clip norms beside
post-clip gradients; a dump omitting element indices; an output-node search
that matched a weight matrix; an error metric dividing by gradients at the 1e-12
level; a bare `import torch`; and an assumption about what a dump filename
meant. Every one of them would have been filed as a library defect had the
harness not been fixed first.

So the operating rule, and it is a rule rather than a slogan:

> **When an oracle fires, establish which side is wrong before believing
> either.** And when a dump's name implies a quantity, confirm what the probe
> actually wrote before trusting it.

---

## 6. Adding an operation

The order matters, because each step is cheap and the reverse is not.

1. **Add the opcode** to `lancius_ir.h` with the next free value. Never renumber;
   opcode values are persisted in model files.
2. **Write the builder** with the same validation discipline as its siblings —
   refuse to build rather than build something that will be wrong later.
3. **Implement the forward kernel**, with no allocation and no direct error
   setting beyond the documented channels.
4. **Add the backward op and its VJP** to `lancius_autodiff.c`. If the VJP
   cannot be written, make autodiff fail loud — do not register the op as
   differentiable and return an empty gradient.
5. **Add it to the scheduler dispatch** and to serialization.
6. **Extend the external oracle**, not just an internal audit. The oracle must
   not import Lancius, and where a convention could be flipped (kernel
   direction, reduction axis, normalisation denominator) it must assert the
   **negative** case too, so that a wrong implementation cannot pass on
   symmetric data.
7. **Add a mutation.** Inject a real defect into the new code and require a gate
   to catch it. A new op with no mutation is an untested op.

---

## 7. Where the honest limits are

Not everything is supported, and the boundaries are as load-bearing as the
features.

- **conv2d** is zero-padded, single-group, undilated, and has no fused bias.
- **softmax backward** has no external oracle; it is reachable through
  `lancius_softmax_bwd` and exercised indirectly by training, but nothing
  compares it against torch autograd.
- **INT8 quantisation** clamps to `[-128, 127]` where TFLite and TensorRT
  specify `[-127, 127]`. Inert as written because `scale = max_abs/127` makes
  `-128` unreachable, but any future path that can produce it must change the
  clamp first.
- **PyTorch-dependent oracle layers require torch.** `trainlib_oracle`,
  `conv_oracle`, and `math_oracle` use torch *as* the independent engine and have
  no non-circular fallback — a NumPy reimplementation of SGD checked against a
  NumPy reimplementation of SGD would agree with itself. The other oracle layers
  do degrade to NumPy plus closed form, and they say so loudly when they skip.
- **`data_vec/` and `temp/`** are generated and gitignored. Trained models in
  `data_vec/` are therefore not in the tree; the `V1.2-RC2` release carries them
  as artifacts.

For the PRM800K verifier stack built on this framework, and specifically what
it does **not** establish, see `docs/DESPOT_TRUTH_V2.md` §18.