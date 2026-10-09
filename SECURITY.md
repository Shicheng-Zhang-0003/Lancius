# Security Policy

## Reporting Issues

Security issues should be reported privately before public disclosure.

## Scope

Security concerns include:

-   memory corruption
-   unsafe deserialization
-   arbitrary execution risks
-   dependency vulnerabilities

## Stable Branch

The v11S branch receives fixes for critical security and correctness
issues.

Security fix enumeration lives in `CHANGELOG.md` (single record).
Essence only below.

## Security Improvements (Hardening Batch V4)

V4 (2026-09-28) essence: CLI fork+execvp, IR leaks, abort→errors,
autodiff OOB, conv/maxpool races, serialization portability, CRC call_once,
dangling sched, Python/build. See `CHANGELOG.md`.

## Security Improvements (Hardening Batch V5)

V5 (2026-09-30) essence: pool use-before-init, distill mkdir, IR NULLs,
optimizer mask, dequant scale, CE R/C, 32B split, silent lib, examples,
converter. See `CHANGELOG.md`.

## Security Improvements (Despot Audit V6)

V6 (2026-10-01) essence: VM tape/out_reg/shape, scheduler NULL-deref,
pool UAF/leak/drop, arena wrap, 32B align, mkstemp/fsync, per-channel refuse,
ndim==0/CONST/ROPE/scale, quantizer/optimizer/vision, capped fetches,
distill I/O, tar-slip member validation. See `CHANGELOG.md` and
`docs/DESPOT_TRUTH_V2.md` §12.

## Security Improvements (Despot Audit V9)

V9 (2026-10-09): v2-only saves (no silent downgrade losing per-channel scales),
v1 trunc-leak + LIMIT, handle secret cookies + owner (magic-only forgery
closed), pool_submit NULL_PTR, VM checked execution + trailing-HALT fix,
planner fail-closed, Python TOCTOU (open+fstat+capped) + 8GB inflate caps,
CLI topk 1..100 + 100k display bound. See `CHANGELOG.md` and
`docs/DESPOT_TRUTH_V2.md` §20.


## Trust model as enforced (V7)

- **Model files are untrusted input.** CRC32 is integrity, not trust: it
  detects corruption and accidental edits, not a hostile writer. The loader
  validates ranks, shapes, element counts and per-node budgets before allocating,
  and `LANCIUS_MAX_TENSOR_BYTES` bounds the total.
- **The arena will not be talked into an OOM.** Alignment must be a power of two
  and at most 1 MiB, element counts are checked with overflow-guarded
  multiplication, and every absurd request returns `OVERFLOW` rather than
  wrapping into a multi-gigabyte grow. `H4` of the V7 audit pins the cap from
  both sides (`1<<60` refused, `1<<20` still honoured).
- **Extraction is contained.** 3463-LDFD's tar reader rejects `../` members and
  absolute-path members, and both the C and Python paths verify that nothing was
  written outside the destination.
- **No shell, ever.** User-supplied paths reach external programs through
  `fork` + `execvp`; `os.system` is not used. `LANCIUS_DATA_DIR` cannot be used
  to inject a command or to point a destructive operation at `/`.
- **Library code never aborts.** A malformed graph, a corrupt model or a
  degenerate tensor produces an error code, not a signal — so a caller can
  always recover and no remote input can terminate the host process through the
  runtime.
