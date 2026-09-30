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

## Security Improvements (Hardening Batch V5)

The following security-relevant fixes were applied in hardening batch V5
(2026-09-30):

- **Threadpool queue growth**: Use-before-initialization bug fixed (was
  reading from uninitialized buffer during realloc).
- **Command injection in distill_prm800k**: `system()` replaced with `mkdir()`.
- **IR silent NULL returns**: 6 locations now set error codes on validation
  failure instead of returning NULL silently.
- **Optimizer error clearing**: No longer masks prior errors on success.
- **Quantizer zero-scale check**: Dequantization validates scale > 0.
- **Scheduler cross-entropy consistency**: Forward and backward use same
  R/C source.
- **Memory planner alignment**: Free block splitting maintains 32-byte
  alignment.
- **Stable API**: Duplicate `#include` removed.
- **Serializer/Vision**: `fprintf`/`printf` removed from library code.
- **Examples**: `parity_runner` and `run_trained_batch` hardened (unchecked
  allocations, ignored I/O returns, NULL derefs fixed).
- **Python**: `onnx_to_lancius.py` shape filtering fixed (interior 1s
  preserved).
- **Autodiff**: NOP comment clarified (no null pointer dereference).
