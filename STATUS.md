# Lancius Current Status

Current internal milestone: **v12R1**
Previous internal milestone: **v11S**
Public equivalent: **TBD**
Release line: **12 R1 development**

## Phase

v12R1 is the first development milestone of the v12 cycle
(progression `S → R1 → R2 → R3 → S`), built on the v11S stable baseline.

The v11A3 hardening gate is complete:
- `make check` green
- `make check-long` green
- sanitizer validation green
- malformed model loading safely rejected
- version identity consistent
- model format v2 frozen with CRC32 integrity
- stable C API covers core inference workflow

## Feature freeze

v11S inherits the v11A3 feature freeze:
- no new operators
- no new runtime subsystems
- no new training features
- no new model-format changes

Only critical bug fixes are accepted post-release.

## Hardening batch — 2026-09-12

Bottom-up pass applied (see CHANGELOG): build clean under
`-Wall -Wextra -Werror`; `audit_regression_13c` 49/49,
`audit_known_answer` 67/67, `audit_transformer_known_answer` 265/265,
`audit_fp32_path` 19/19 green. Stale backup files removed.
