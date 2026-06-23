#!/bin/bash

# ==============================================================================
#  README Generator for llm_maths V4
#  Generates a modern, beautifully formatted Markdown file.
# ==============================================================================

echo "Generating modern README.md..."

cat << 'EOF' > README.md
# 🧠 llm_maths: A Bare-Metal Autograd & Transformer Engine in C

**llm_maths** is a from-scratch, pure-C implementation of a reverse-mode automatic differentiation engine, tensor mathematics, and Transformer architecture components.

Think of it as a miniature, bare-metal PyTorch or JAX, written entirely in C without any external machine learning dependencies. It bridges the gap between abstract symbolic calculus, numeric execution graphs, and information geometry.

---

## 🤝 A Note on Human-AI Collaboration

> **This project is not an "anti-AI" purist exercise.**
> While the foundational architecture, mathematical derivations, and core design philosophy were deeply human-driven, **AI pair-programming was leveraged extensively in the latter stages (V3 $\to$ V4)** of this project.
>
> As the codebase scaled to include symbolic compilers and Transformer attention mechanisms, AI was utilized to dramatically raise efficiency, execute complex multi-file refactors, and identify deep architectural bottlenecks (such as replacing recursive graph traversals with iterative stacks to prevent C-stack overflows, and decoupling optimizer states from the autograd graph). This project represents a modern synthesis: human mathematical intuition guided by AI-assisted systems engineering.

---

## 🏗️ Architecture: The 5 Stages

The library is organized into 5 progressive stages. Understanding these stages is the key to understanding how the framework operates under the hood.

| Stage | Domain | Core Concept |
| :--- | :--- | :--- |
| **Stage 1** | **Atomic** | **The Autograd Tape.** Every number is a `value` node. Operations create new nodes, store pointers to their "parents", and attach a C-function pointer for the chain rule. |
| **Stage 2** | **Linear** | **Tensors as Graphs.** A `tensor` is a 2D grid of `value*` pointers. Matrix multiplication isn't a hardcoded kernel; it's built by chaining scalar operations, meaning gradients flow perfectly through every element. |
| **Stage 3** | **Sequence** | **Transformer Mechanics.** Implements Scaled Dot-Product Attention, Softmax, and RoPE (Rotary Positional Embeddings) natively on the autograd graph. |
| **Stage 4** | **Learning** | **Optimization.** Features Layer Normalization and a highly optimized Adam optimizer. |
| **Stage 5** | **Unified** | **Symbolic Math & Geometry.** A symbolic algebra engine that can differentiate ASTs, compile them into numeric C-graphs, and calculate the Fisher Information Metric. |

---

## ⚡ V4 Architectural Upgrades

If you are comparing this to earlier iterations, V4 includes critical production-ready systems engineering fixes:

1. **Iterative Backpropagation (Stack-Safe):** Replaced recursive topological sorting with an explicit-stack iterative Depth-First Search (DFS). This prevents C-stack overflows on very deep networks (like long unrolled RNNs or deep ResNets).
2. **Decoupled Optimizer State:** In deep learning, optimizer states (like Adam's moving averages) do *not* need gradients. V4 stores Adam moments as raw `double*` arrays, completely decoupled from the autograd graph. This prevents massive memory bloat and keeps training loops lightning fast.
3. **Sanitized Codebase:** Eradicated hidden syntax artifacts ("ghost spaces" in strings and keywords) that caused silent failures in string-matching operations within the symbolic compiler.
4. **Expanded Calculus:** Added native `subtraction` and `tanh` operations, which are mandatory for modern LLM gating mechanisms and LSTMs.

---

## 🚀 Quick Start

### Prerequisites
You just need a standard C compiler (GCC or Clang) and `make`.

### Build and Run
```bash
# 1. Compile the library and test suite
make

# 2. Run the test suite to verify the math
./test_scalar           # Basic calculus & chain rule
./test_tensor           # Matrix math & autograd
./test_attention        # Transformer Q, K, V mechanics
./test_symbolic         # Algebraic differentiation (ASTs)
./test_compiler         # Symbolic-to-Numeric bridging
./test_frontier         # Training loop (Adam optimizer convergence)
./test_grand_unified    # The final boss: Math + Code + Geometry
