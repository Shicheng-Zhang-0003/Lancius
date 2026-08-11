# Lancius Current Status

Current internal milestone: **v11S**
Previous internal milestone: **v11A3**
Public equivalent: **V1.1**
Release line: **1.1 stable**

## Phase

v11S is the stable release candidate.

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
