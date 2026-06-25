# 🧠 BASIS v5: A Bare-Metal Mathematical Computing, Deep Learning & Inference Engine in C

**BASIS** (Bare-metal Autograd & Symbolic Inference System) is a from-scratch, pure-C implementation of a reverse-mode automatic differentiation engine, tensor mathematics, symbolic calculus, a JIT machine-code compiler, and a Physics-Informed Neural Network (PINN) solver.

Think of it as a miniature, bare-metal synthesis of **PyTorch**, **JAX**, **SymPy**, and **DeepXDE**, written entirely in C without any external machine learning dependencies. It bridges the gap between abstract symbolic calculus, dynamic numeric execution graphs, information geometry, and native hardware inference.

---

## 🤝 A Note on Human-AI Collaboration

> **This project is not an "anti-AI" purist exercise.**
> While the foundational architecture, mathematical derivations, and core design philosophy were deeply human-driven, **AI pair-programming was leveraged extensively in the latter stages (V4 → V5)** of this project.
>
> As the codebase scaled to include JIT compilers, information geometry, dynamic property-based fuzzing, and PDE solvers, AI was utilized to execute complex multi-file refactors, enforce strict memory ownership models, and identify deep architectural bottlenecks (such as replacing recursive graph teardowns with iterative stacks to prevent C-stack overflows on 10,000-deep graphs). This project represents a modern synthesis: human mathematical intuition guided by AI-assisted systems engineering.

---

## 🏗️ The 10 Pillars of BASIS v5

The framework has evolved through rigorous stabilization and expansion phases. Here is what powers the engine under the hood:

| Pillar | Domain | Core Concept |
| :--- | :--- | :--- |
| **1. Autograd Tape** | *Atomic* | Iterative, stack-safe reverse-mode autodiff. Every number is a `basis_value` node with reference counting and strict memory ownership. |
| **2. Tensor Algebra** | *Linear* | 2D grids of autograd nodes. Features **Fused MatMul** (Expression DAG compression) to eliminate intermediate node bloat. |
| **3. Strided Views** | *Memory* | **Zero-copy slicing and transposing** using stride/offset math. Transposing a matrix requires zero memory allocations. |
| **4. Broadcasting** | *Math* | NumPy/PyTorch-style broadcasting via the **Stride=0 trick**. Adding a bias vector to a batch matrix routes gradients perfectly without copying data. |
| **5. Sequence Mechanics** | *Transformers* | Native Scaled Dot-Product Attention, RoPE, and Layer Normalization built directly on the autograd graph. |
| **6. Numerical Stability** | *Robustness* | **Log-Sum-Exp trick** for `basis_tensor_log_softmax`, preventing `NaN` overflows on extreme logits. |
| **7. Optimization** | *Learning* | Decoupled Adam optimizer (raw `double` arrays) and **Natural Gradient Descent** utilizing the Fisher Information Metric. |
| **8. Symbolic Calculus** | *Unified* | A full AST algebraic engine capable of exact symbolic differentiation, constant folding, and Common Subexpression Elimination (CSE). |
| **9. Higher-Order Math** | *Research* | Exact **Hessian-Vector Products (HVP)** computed via the symbolic engine without instantiating the massive Hessian matrix. |
| **10. JIT Compiler** | *Inference* | Translates symbolic ASTs into raw C code, invokes `gcc` at runtime, and loads the AVX-optimized machine code via `dlopen` for zero-overhead inference. |

---

## ⚡ V5 Architectural Upgrades (The Stabilization & Frontier Mandate)

Transitioning from a mathematical prototype to a production-grade framework required strict systems engineering and scientific computing expansion:

1. **Unified Error Handling:** A global `basis_error_t` state with quiet modes for testing. No function fails silently; shape mismatches and domain errors are explicitly caught.
2. **Iterative Graph Teardown:** `basis_value_free` and topological sorting are 100% iterative. The engine can safely dismantle a **10,000-node deep computation graph** without triggering a C-stack overflow.
3. **Collision-Free API & Deployment:** The entire public API is strictly prefixed with `basis_` and accessible via a single `#include <basis.h>` umbrella header. Compiles to `libbasis.so` / `libbasis.a` for system-wide installation.
4. **Binary Serialization:** Native `basis_tensor_save_binary` and `basis_tensor_load_binary` to bridge the gap between heavy training environments and lightweight inference deployment.
5. **Dynamic Property-Based Fuzzing:** The test suite uses `<time.h>` to generate random tensor shapes, extreme logits, and arbitrary ASTs on every run, proving mathematical invariants across the continuous domain.
6. **Physics-Informed Neural Networks (PINNs):** Native support for solving non-linear Partial Differential Equations (like Burgers' Equation and the Heat Equation) using finite-difference stencils routed through the autograd tape.

---

## 🚀 Quick Start

### Prerequisites
You just need a standard C compiler (GCC or Clang), `make`, and a POSIX environment (for the JIT `dlopen`/`mkstemp` features).

### Build and Install
```bash
# 1. Compile the shared/static libraries and the dynamic test suite
make clean && make

# 2. Run the dynamic fuzzing suite (Generates random matrices, ASTs, and weights)
./test_v5_suite

# 3. Install globally to your OS (Linux FHS)
sudo make install
sudo ldconfig
