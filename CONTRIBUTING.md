# Contributing to Lancius

## Development Philosophy

Lancius values:

-   correctness
-   reproducibility
-   clear architecture
-   documented decisions

## Before Contributing

Please read:

-   README.md
-   docs/ARCHITECTURE.md
-   KNOWN_LIMITATIONS.md
-   docs/DESPOT_TRUTH_V2.md (how this project proves correctness)

## Code Style

Lancius follows a consistent C code style across all source files:

- **Indentation:** 4 spaces (no tabs)
- **Braces:** opening brace on same line for control structures; opening brace on new line for function definitions
- **Naming:** `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for macros and constants, `lancius_` prefix for public API functions
- **Line length:** 100 columns maximum
- **Comments:** `//` for single-line, `/* */` for multi-line; no trailing whitespace
- **Pointers:** asterisk attached to the name (`int *ptr`, not `int* ptr`)
- **Error handling:** all public API functions return error codes; no `abort()` or `exit()` in library code
- **Memory:** every allocation must have a matching free; no leaks on any path (including error paths)

See existing source files (`src/*.c`, `include/*.h`) for examples.

## Pull Requests

Changes should include:

-   explanation of purpose
-   tests where applicable
-   documentation updates

## Validation Requirements

All contributions must pass:

-   `make check` — full test suite (unit tests, audits, known-answer tests)
-   `make check-sanitizers` — ASan + UBSan clean
-   `make check-long` — extended tests (soak fuzz, fuzz)

New primitives must carry:

-   known-answer tests (exact expected values)
-   finite-difference gradient checks (for differentiable ops)
-   exit nonzero on divergence (no false-green)

## Documentation Requirements

Each document owns one thing (single-owner rule):

-   `README.md` — project overview and build instructions
-   `STATUS.md` — current milestone status
-   `KNOWN_LIMITATIONS.md` — explicit boundaries
-   `CHANGELOG.md` — per-batch fix history
-   `docs/ARCHITECTURE.md` — subsystem contracts
-   `docs/DESPOT_TRUTH_V2.md` — mathematical audit
-   `docs/v12R2_SCOPE.md` — next milestone scope
-   `SECURITY.md` — security policy and improvements
-   `CONTRIBUTING.md` — this file

When making changes, update the relevant document(s) to reflect the new
behavior. Do not create new documentation files unless absolutely necessary.
