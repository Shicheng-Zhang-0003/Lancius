# Lancius v10S

## The Lancius Project

Lancius v10S is the first stable release of the Lancius 1.0 development
cycle.

Lancius is a lightweight machine learning execution framework focused on
bare-metal inference, graph execution, runtime control, and efficient
deployment.

The purpose of v10S is not to represent the final evolution of Lancius.
It is a stable foundation: a verified, documented, and reproducible
release from which future v11 development can proceed.

## Release Philosophy

Lancius follows this development cycle:

-   v10A1: Initial 1.0 architecture development
-   v10A2: Architecture expansion and validation
-   v10A3: Release candidate hardening and adversarial testing
-   v10S: Stable 1.0 release

Future releases will continue the cycle:

-   v11A1
-   v11A2
-   v11A3
-   v11S

## Features

v10S provides:

-   Graph-based execution
-   Stable C API
-   Runtime execution engine
-   Memory planning infrastructure
-   CPU inference backend
-   Transformer-oriented inference primitives
-   Model serialization support
-   Testing and validation tooling

## Supported Focus

Lancius v10S is designed around:

-   deterministic inference
-   low-level runtime control
-   lightweight deployment
-   extensible architecture

## Current Limitations

See:

-   KNOWN_LIMITATIONS.md
-   ARCHITECTURE.md
-   CHANGELOG.md

## Building

Typical build:

``` bash
git clone <repository>
cd lancius
make
```

## License

See LICENSE.
