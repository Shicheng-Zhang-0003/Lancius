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

## Security Improvements (Hardening Batch V4)

The following security-relevant fixes were applied in hardening batch V4
(2026-09-28):

- **CLI command injection eliminated**: User-supplied paths are now executed
  via `fork+execvp` instead of `system()`, preventing shell injection.
- **Memory safety**: IR node allocation leaks fixed; `abort()` removed from
  library code (replaced with `lancius_set_error` returns); OOB reads on
  shape/axes arrays in autodiff fixed (pads to 4D).
- **Race conditions**: `kernel_conv2d_bwd_in` and MaxPool2D backward now use
  thread-local accumulators, eliminating data races.
- **Serialization portability**: `uint64_t` sizing and byte swapping fixed;
  CRC32 table init race fixed (`call_once`); NOP IDs no longer mapped to NULL.
- **Stable API**: Dangling `wrapper->sched` pointer fixed.
- **Python tooling**: Security fixes including stale version references removed
  and error handling improved.
- **Build system**: `-Werror` enforced; version consistency checked.
