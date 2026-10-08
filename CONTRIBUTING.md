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

The style is encoded in `.clang-format` and was extracted from the existing
source rather than invented, so adding the file reformats nothing. Read it as
the authority; this section is the prose version.

-   **Indentation:** 4 spaces, never tabs
-   **Line length:** no limit. Bounds checks and guard clauses stay on one line
    because splitting `if (x > SIZE_MAX / y) { ... }` across lines makes it
    harder to audit, which is the only reason it is long
-   **Pointer style:** `(double*)p`, `lancius_node* n` — asterisk attached to the
    type, cast parentheses attached to the type
-   **Braces:** attached for control structures, on the same line as the closing
    `)` for definitions
-   **Comments:** `/* ... */` throughout, including single-line notes. Long
    comments state *why*, and name the defect they prevent rather than
    restating the code
-   **Naming:** `snake_case` functions and variables, `UPPER_SNAKE_CASE` macros
    and constants, `lancius_` prefix on every public symbol
-   **Error handling:** every public API returns an error code. No `abort()` and
    no `exit()` in library code, ever — not in a "cold" path, not in an audit
    helper. `lancius_set_error` writes thread-local state, so an error raised
    inside an OpenMP worker is invisible to the caller and must be propagated
    through an explicit shared flag
-   **Memory:** every allocation has a matching free on **every** path, including
    error paths. When a function has many failure branches, funnel them through
    one teardown helper rather than repeating the free list — 45 copies of the
    same free list is how `tg->grad_nodes` leaked 45 times

### The rules that are not stylistic

-   **Bind ownership, do not assign.** `node->runtime_data = malloc(...)`
    creates memory the graph does not own and will not free.
    `lancius_node_bind_owned_heap(n, buf)` is the correct form;
    `lancius_node_bind_external(n, buf)` when the caller keeps it
-   **Every branch that computes must `return`.** `execute_node_math` is one
    long `else if` chain followed by a router keyed on `op >= LANCIUS_OP_CONV2D`.
    A branch that falls through is re-dispatched to the router, which rejects
    it — so the op computes the right answer and then reports failure
-   **A successful op must leave `lancius_get_error()` at `OK`.** Sticky errors
    are the contract that makes a later failure honest; a success that leaves
    one behind poisons everything downstream in the same thread
-   **Fail loud, never plausible.** Out of scope means `NULL` plus an error
    code, not zeros, not a plausible number, not a best-effort approximation

## Proving a Change Is Real

Passing gates are necessary and not sufficient — a gate nobody checked for its
own ability to fail is decoration. Before claiming a fix or a new primitive:

```bash
make check           # standing gate
make check-oracle    # recompute it in NumPy / PyTorch / closed form
make check-sanitizers # ASan + UBSan + LSan over the instrumented library
make check-mutation  # does the gate go red if I break this on purpose?
```

New numeric primitives must carry a known-answer test **and** an external
oracle. New invariants must be mutation-tested: inject the inverse of the
invariant, and require the gate to notice. That is how the seven gate holes
recorded in `docs/DESPOT_TRUTH_V2.md` §16 were found and closed.

## Pull Requests

Changes should include:

-   explanation of purpose
-   tests where applicable
-   documentation updates

## Validation Requirements

All contributions must pass:

-   `make check` — full test suite (unit tests, audits, known-answer tests)
-   `make check-sanitizers` — ASan + UBSan + LeakSanitizer clean, library
    instrumented, every audit run
-   `make check-ubstrict` — UBSan with `-fno-sanitize-recover=all`
-   `make check-oracle` — NumPy / PyTorch / closed-form agreement
-   `make check-long` — extended tests (soak fuzz, fuzz)

`make check-mutation` and `.github/workflows/gate.yml` run on every push.

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
-   `docs/v12R2_SCOPE.md` — current milestone scope
-   `SECURITY.md` — security policy and improvements
-   `CONTRIBUTING.md` — this file

When making changes, update the relevant document(s) to reflect the new
behavior. Do not create new documentation files unless absolutely necessary.
