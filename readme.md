# 🧠 BASIS v6: A Bare-Metal, Compile-to-Native AI & Scientific Computing Engine in C

**BASIS** (Bare-metal Autograd & Symbolic Inference System) is a from-scratch, pure-C implementation of a reverse-mode automatic differentiation engine, tensor mathematics, symbolic calculus, a JIT/AOT machine-code compiler, and a Physics-Informed Neural Network (PINN) solver.

Think of it as a miniature, bare-metal synthesis of **PyTorch**, **JAX**, **SymPy**, and **DeepXDE**, written entirely in C without any external machine learning dependencies. BASIS v6 introduces the **Ghost-Weight Engine**: a static graph IR and AOT lowering compiler that transforms neural networks into standalone, monolithic C source files. Zero Python. Zero dependencies. Absolute architectural clarity.

---

## 🤝 A Note on Human-AI Collaboration

> **This project is not an "anti-AI" purist exercise.**
> While the foundational architecture, mathematical derivations, and core design philosophy were deeply human-driven, **AI pair-programming was leveraged extensively in the latter stages (V4 → V6)** of this project.
>
> As the codebase scaled to include JIT/AOT compilers, information geometry, dynamic property-based fuzzing, and PDE solvers, AI was utilized to execute complex multi-file refactors, enforce strict memory ownership models, and identify deep architectural bottlenecks (such as replacing recursive graph teardowns with iterative stacks to prevent C-stack overflows on 10,000-deep graphs). This project represents a modern synthesis: human mathematical intuition guided by AI-assisted systems engineering.

---

## 🏗️ The 12 Pillars of BASIS v6

The framework has evolved through rigorous stabilization, scientific expansion, and compiler infrastructure phases:

| Pillar | Domain | Core Concept |
| :--- | :--- | :--- |
| **1. Eager Autograd Tape** | *Atomic* | Iterative, stack-safe reverse-mode autodiff. Every number is a `basis_value` node with reference counting and strict memory ownership. |
| **2. Tensor Algebra** | *Linear* | 2D grids of autograd nodes. Features **Fused MatMul** (Expression DAG compression) to eliminate intermediate node bloat. |
| **3. Strided Views** | *Memory* | **Zero-copy slicing and transposing** using stride/offset math. Transposing a matrix requires zero memory allocations. |
| **4. Broadcasting** | *Math* | NumPy/PyTorch-style broadcasting via the **Stride=0 trick**. Routes gradients perfectly without copying data. |
| **5. Sequence Mechanics** | *Transformers* | Native Scaled Dot-Product Attention, RoPE, and Layer Normalization built directly on the autograd graph. |
| **6. Numerical Stability** | *Robustness* | **Log-Sum-Exp trick** for `basis_tensor_log_softmax`, preventing `NaN` overflows on extreme logits. |
| **7. Optimization** | *Learning* | Decoupled Adam optimizer (raw `double` arrays) and **Natural Gradient Descent** utilizing the Fisher Information Metric. |
| **8. Symbolic Calculus** | *Unified* | A full AST algebraic engine capable of exact symbolic differentiation, constant folding, and Common Subexpression Elimination (CSE). |
| **9. Higher-Order Math** | *Research* | Exact **Hessian-Vector Products (HVP)** computed via the symbolic engine without instantiating the massive Hessian matrix. |
| **10. Static Graph IR (v6)** | *Compiler* | Architecture tracing without memory allocation. Defines networks as lightweight metadata DAGs ready for compilation. |
| **11. AOT Lowering Compiler (v6)** | *Compiler* | Lowers IR into monolithic C forward/backward passes. Bypasses the scalar tape entirely for zero-overhead training & inference. |
| **12. Ghost-Weight Bundler (v6)** | *Deployment* | Embeds trained weights directly into C source as `const` arrays. Cross-compilable to edge devices, WASM, or iOS with zero file I/O. |

---

## ⚡ V6 Architectural Upgrades: The Ghost-Weight Engine

BASIS v6 executes a fundamental paradigm shift: **separating Graph Definition from Graph Execution.**

1. **Static Graph Tracing:** Define neural networks using `basis_graph` and `basis_node`. The engine records shapes, operations, and dependencies into a lightweight IR. **Zero weight memory is allocated. Zero autograd nodes are created.**
2. **AOT Forward & Backward Compilation:** The lowering compiler consumes the IR and emits a standalone `ghost_model.c` file containing cache-friendly C loops for both inference and gradient computation. Training shifts from traversing a runtime pointer graph to executing compiled native machine code.
3. **Zero-I/O Deployment:** The bundler reads trained `.basis` weight files and bakes them directly into the generated C code as constant arrays. The resulting executable requires no filesystem, no dynamic loading, and no external libraries.
4. **Hardware-Ready Abstraction:** The clean, loop-based C emission is architected for seamless backend swapping. Future flags (`-DBASIS_BACKEND=CUDA`) will route IR nodes to GPU kernels instead of CPU loops.

---

## 🚀 Quick Start

### Prerequisites
A standard C compiler (GCC or Clang), `make`, and a POSIX environment (for JIT `dlopen`/`mkstemp` features).

### Build and Install
```bash
# 1. Compile the shared/static libraries and the dynamic test suite
make clean && make

# 2. Run the dynamic fuzzing suite (Generates random matrices, ASTs, and weights)
./test_v5_suite

# 3. Install globally to your OS (Linux FHS)
sudo make install
sudo ldconfig
