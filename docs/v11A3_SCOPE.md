# Lancius v11A3 Scope

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
