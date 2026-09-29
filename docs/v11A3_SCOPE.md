# Lancius v11A3 Scope

> **Historical document.** v11A3 is a completed development milestone.
> The current stable release is `v11S` (V1.1); the current development
> milestone is `v12R1` (V1.2RC1). This file is preserved for historical
> reference only.

> v11A3 is a development milestone.
> It is not a stable release.

## Theme

Freeze, harden, and hunt bugs.

## Included in v11A3

- Feature freeze
- Loader hardening
- Model-format validation hardening
- Sanitizer and fuzz validation
- Regression-test expansion
- Documentation cleanup
- Release-candidate preparation for v11S

## Not included in v11A3

- New operators
- New runtime subsystems
- GPU acceleration
- Dynamic shapes
- Full production training support
- New ONNX operator expansion unless required for correctness

## Exit criteria

v11A3 is complete when:

- `make check` is green
- `make check-long` is green
- sanitizer validation is green
- malformed model loading is safely rejected
- version identity is consistent
- release notes for v11S can be drafted from the frozen state
