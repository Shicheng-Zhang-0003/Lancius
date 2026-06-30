# BASIS V8: A Bare-Metal Machine Learning Systems Framework

> A zero-dependency machine learning framework, compiler, and inference
> runtime written in C for understanding, training, and deploying neural
> networks from first principles.

    ┌────────────┐   ┌─────────────┐   ┌──────────────┐
    │ Raw Data   │→→│ Graph / IR   │→→│ Symbolic AD   │
    └────────────┘   └─────────────┘   └──────────────┘
             │                 │                 │
             ▼                 ▼                 ▼
       O(1) Arena      Wave Scheduler      Optimizers
             │                 │
             └──────────┬──────┘
                        ▼
                 Bytecode (.v8b)
                        ▼
                 Standalone Runtime

## Overview

BASIS is an experimental machine learning systems framework built in ISO
C. Rather than relying on existing ML frameworks, BASIS implements its
own tensor operations, computational graphs, symbolic automatic
differentiation, schedulers, serialization format, and standalone
inference runtime.

The project is primarily an educational and engineering exercise:
understanding how modern ML systems work by building them from the
ground up.

## Philosophy

**Infrastructure exists only to serve the model.**

Every subsystem exists because it enables real models to be trained,
validated, serialized, and deployed.

BASIS is **not** intended to replace PyTorch or TensorFlow. Instead, it
explores how a compact, understandable, portable ML framework can be
built from first principles.

## Current Features (V8)

-   N-dimensional tensor operations
-   Static graph / IR execution
-   Symbolic automatic differentiation
-   O(1) scratch arena allocator
-   Wave-based scheduler
-   OpenMP parallel execution
-   SIMD-friendly kernels
-   Custom `.v8b` serialization
-   Standalone inference runtime
-   End-to-end training and deployment pipeline

## Verified Results

  Capability            Status     Result
  --------------------- ---------- ----------------------
  MNIST                 Verified   94.52% test accuracy
  CIFAR-10 (LeNet)      Verified   42.51% test accuracy
  Model Serialization   Verified   `.v8b` save/load
  Edge Runtime          Verified   Standalone inference

## Architecture

Key architectural components include:

-   Static graph intermediate representation (IR)
-   Symbolic vector-Jacobian backpropagation
-   O(1) bump-pointer scratch arena
-   Wave scheduler for dependency-aware execution
-   Thread-local gradient accumulation
-   Bytecode serialization and portable inference runtime

The edge runtime is intentionally separated from training. It loads a
serialized `.v8b` model, reconstructs the execution schedule, and
performs inference without requiring training components.

## Reliability

Current verification work includes:

-   Finite-difference gradient checking
-   Dynamic graph fuzz testing
-   End-to-end training verification
-   Model serialization/deserialization testing
-   Standalone runtime verification

The long-term objective is to increase automated testing,
reproducibility, and mathematical verification before expanding the
framework's feature set.

## Development Process

BASIS was developed using a transparent AI-assisted workflow.

Responsibilities were intentionally divided:

-   **Qwen** acted as the primary implementation ("blue team"),
    generating most of the initial implementation.
-   **The project author** directed the architecture, selected features,
    integrated subsystems, reviewed implementations, and determined the
    roadmap.
-   **ChatGPT, Gemini, GLM, and Kimi** served as the "red team",
    providing design reviews, architectural critiques, implementation
    feedback, stress-testing ideas, and identifying weaknesses before
    features were accepted.

Rather than hiding AI assistance, the project documents it openly. The
long-term goal is to understand, verify, maintain, and extend every
subsystem regardless of how the first implementation was produced.

## Roadmap

Near-term goals include:

-   Expanded automated testing
-   Improved reproducibility
-   Additional benchmarks
-   Transformer primitives (Embedding, LayerNorm, GELU, Attention)
-   Character-level language model training
-   Continued runtime and compiler improvements

Future milestones will be introduced as the project evolves.

## Building

``` bash
make
```

## License

GNU General Public License v3.0 (GPLv3)

## Final Note

BASIS is still an evolving project. The focus today is not on claiming
to be production-ready, but on building a machine learning framework
whose mathematics, implementation, and systems architecture can be
understood, tested, challenged, and improved over time.
