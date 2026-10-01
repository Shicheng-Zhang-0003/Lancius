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
