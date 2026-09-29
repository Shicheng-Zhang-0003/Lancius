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
