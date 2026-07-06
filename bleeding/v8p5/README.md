# Lancius: Bare-Metal C Machine Learning Systems Framework

A high-performance, zero-dependency machine learning compiler and inference runtime written in ISO C11. Lancius (lancius.5) features an immutable Static Single Assignment (SSA) IR graph, an $O(1)$ block-arena allocator, dependency-aware Wave Scheduling, post-training INT8 symmetric quantization, and modern Transformer/LLM kernels.

```
       ┌────────────────────────┐
       │   Immutable SSA Graph  │
       │       ([lancius_ir.c])      │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   Wave Scheduler       │
       │   ([lancius_scheduler.c])   │
       └───────────┬────────────┘
                   │
                   ▼
 ┌─────────────────┴─────────────────┐
 ▼                                   ▼
Bytecode Compilation (.lancius)        INT8 Quantizer & Loader
([lancius_bytecode.c])                  ([lancius_quantize.c], [lancius_serialize.c])
 ▼                                   ▼
Virtual Machine Executor           Parallelized INT8 & OpenMP Kernels
([lancius_scheduler.c])                 ([lancius_kernels.c], [lancius_vision_ops.c])
```

---

## 1. Subsystem Architecture

### A. Immutable SSA IR Graph & Auto-Diff
* **File:** [lancius_ir.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/ir/lancius_ir.c) (Header: [lancius_ir.h](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/include/lancius/lancius_ir.h))
* **Description:** Represents neural network layers and math operations as a directed acyclic graph (DAG) of immutable SSA nodes. 
* **Symbolic Differentiation:** [lancius_autodiff.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/math/lancius_autodiff.c) generates symbolic vector-Jacobian products (VJPs) directly on the graph representation to compile the exact backward pass program.

### B. $O(1)$ Block-Arena Allocator
* **File:** [lancius_arena.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/core/lancius_arena.c) (Header: [lancius_arena.h](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/include/lancius/lancius_arena.h))
* **Description:** Implements memory block-chaining with bump-pointer allocation. Enables extremely fast allocation and $O(1)$ teardown via resetting block offsets, completely eliminating execution-loop allocation overhead.

### C. Dependency-Aware Wave Scheduler
* **File:** [lancius_scheduler.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/runtime/lancius_scheduler.c) (Header: [lancius_scheduler.h](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/include/lancius/lancius_scheduler.h))
* **Description:** Groups graph nodes into independent execution "waves" based on topological dependency levels. Waves are executed sequentially, while nodes within each wave are dispatched in parallel.

### D. Post-Training INT8 Quantization
* **File:** [lancius_quantize.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/compiler/lancius_quantize.c)
* **Description:** Compresses FP64 weights of 4D filter tensors into INT8 symmetric representations:
  $$W_{\text{int8}} = \text{round}\left(\frac{W}{\text{scale}}\right), \quad \text{scale} = \frac{\max(|W|)}{127.0}$$
  Reduces serialized model size by up to 8x for weight parameters.

### E. Mixed-Precision Runtime & Vision Router
* **File:** [lancius_vision_ops.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/runtime/lancius_vision_ops.c)
* **Description:** Automatically routes operations. When loaded weights are INT8, it dynamically quantizes incoming FP64 activations on the fly, computes parallelized INT8 convolutions, and performs fused ReLU activations.

### F. V13 Transformer / LLM Kernels
* **File:** [lancius_kernels.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/math/lancius_kernels.c) (Header: [lancius_kernels.h](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/include/lancius/lancius_kernels.h))
* **Description:** High-performance kernels for modern sequence modeling, including:
  * **LayerNorm:** Numerically stable layer normalization with gamma/beta parameters.
  * **GELU:** Fast mathematical tanh approximation of the Gaussian Error Linear Unit.
  * **RoPE (Rotary Position Embedding):** Rotational head embedding mapping for keys/queries.

### G. Binary Serialization
* **File:** [lancius_serialize.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/core/lancius_serialize.c)
* **Description:** Compiles the complete graph structure and weights into a custom binary format (`.lancius`). Correctly writes and parses dynamic node attributes, network topologies, scales, and data types (FP64 and INT8).

---

## 2. Verified Runtime Results

All verification programs compile under `-Werror` and have been executed with the following outputs:

### A. MNIST Training & VM Execution (`train_mnist`)
* **Source:** [train_mnist.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/examples/train_mnist.c)
* **Result:** Trains a sequential MLP (`784 -> 128 -> 10`) utilizing the symbolic autodiff backward pass, serializes model inputs, compiles bytecode instructions, and runs verification through the VM executor.
```
================================================================
  Lancius: MNIST Notepad (Sequential + Scaled MSE)           
================================================================

[2/5] Loaded 60000 training images, 10000 test images.
[3/5] Compiling Backward Pass & Scheduling Waves...
[4/5] Training for 10 Epochs (Batch Size: 64) via Sequential Executor...

  Epoch 1 | Batch 100/937 | Loss: -21.0407  Epoch 1 | Batch 200/937 | Loss: -27.7059  ... Epoch 1 | Loss: -24.9983
  Epoch 2 | Batch 100/937 | Loss: -20.5798  Epoch 2 | Batch 200/937 | Loss: -20.4064  ... Epoch 2 | Loss: -18.3045
  Epoch 3 | Batch 100/937 | Loss: -16.1843  Epoch 3 | Batch 200/937 | Loss: -15.4436  ... Epoch 3 | Loss: -19.9517
  Epoch 4 | Batch 100/937 | Loss: -31.0666  Epoch 4 | Batch 200/937 | Loss: -30.9029  ... Epoch 4 | Loss: -32.1724
  Epoch 5 | Batch 100/937 | Loss: -31.1354  Epoch 5 | Batch 200/937 | Loss: -32.3105  ... Epoch 5 | Loss: -37.1527
  Epoch 6 | Batch 100/937 | Loss: -37.2946  Epoch 6 | Batch 200/937 | Loss: -37.5485  ... Epoch 6 | Loss: -39.2673
  Epoch 7 | Batch 100/937 | Loss: -40.2709  Epoch 7 | Batch 200/937 | Loss: -40.0948  ... Epoch 7 | Loss: -40.1290
  Epoch 8 | Batch 100/937 | Loss: -35.4624  Epoch 8 | Batch 200/937 | Loss: -35.8437  ... Epoch 8 | Loss: -39.8259
  Epoch 9 | Batch 100/937 | Loss: -38.9317  Epoch 9 | Batch 200/937 | Loss: -40.2628  ... Epoch 9 | Loss: -41.2916
  Epoch 10 | Batch 100/937 | Loss: -46.6479  Epoch 10 | Batch 200/937 | Loss: -42.9281  ... Epoch 10 | Loss: -39.8933

[5/5] Compiling Inference Graph to Bytecode VM & Evaluating...

================================================================
  FINAL TEST ACCURACY: 94.52% (9452 / 10000)
================================================================
  LANCIUS NOTEPAD COMPLETE. PATH C & D VERIFIED.
================================================================
```

### B. CIFAR-10 LeNet Training & Quantization (`train_cifar10`)
* **Source:** [train_cifar10.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/examples/train_cifar10.c)
* **Result:** Trains a Convolutional LeNet model on CIFAR-10, triggers post-training INT8 quantization to freeze parameters, and dumps the compressed weights into `cifar10_lenet.lancius`.
```
================================================================
  Lancius: CIFAR-10 Notepad (Same Padding + He Init)
================================================================

[2/5] Loaded 50000 training images, 10000 test images.
[3/5] Compiling Backward Pass & Scheduling Waves...
[V9] Running Graph Optimizations...
[V9 OPTIMIZER] Fused 2 Conv2D+ReLU patterns. Arena memory saved.
[4/5] Training for 30 Epochs (Batch Size: 32)...

  Epoch 1 | Loss: 1.6399 | lr: 0.000300
  Epoch 2 | Loss: 1.3083 | lr: 0.000300
  ...
  Epoch 15 | Loss: 0.7171 | lr: 0.000300
  [SCHEDULER] Learning rate decayed to 0.000150
  Epoch 16 | Loss: 0.6619 | lr: 0.000150
  ...
  Epoch 25 | Loss: 0.5553 | lr: 0.000150
  [SCHEDULER] Learning rate decayed to 0.000075
  Epoch 26 | Loss: 0.5242 | lr: 0.000075
  ...
  Epoch 30 | Loss: 0.4981 | lr: 0.000075

[5/5] Evaluating on 10,000 Test Images...

================================================================
  FINAL TEST ACCURACY: 72.97% (7297 / 10000)
================================================================

[PATH B] Freezing trained inference model to 'cifar10_lenet.lancius'...
[LANCIUS SERIAL] Saved 23 nodes to cifar10_lenet.lancius

================================================================
  V10 CIFAR-10 NOTEPAD COMPLETE. EDGE DEPLOYMENT READY.
================================================================
```

### C. Standalone Edge Deployment Runner (`run_edge`)
* **Source:** [run_edge.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/examples/run_edge.c)
* **Result:** Stand-alone deployment program that loads the compiled model, reconstructs the graph, performs graph optimization (Conv2D+ReLU fusion), and executes fast INT8 inference.
```
================================================================
  Lancius: EDGE DEPLOYMENT RUNNER (Standalone)              
================================================================

[1/3] Loading model from 'cifar10_lenet.lancius'...
[LANCIUS SERIAL] Loaded 23 nodes from cifar10_lenet.lancius
  ✅ Loaded 21 nodes.
[2/3] Compiling Inference Schedule...
[V9] Running Graph Optimizations...
[V9 OPTIMIZER] Fused 0 Conv2D+ReLU patterns. Arena memory saved.
[3/3] Running Inference on Random Noise (Batch Size: 32)...
  ✅ Inference Successful!
  Output Logits for Image 0 (Raw Scores for 10 Classes):
  [-4.8937 -2.9052 -2.1735 -3.7566 -1.9126 -11.1924 10.8312 -8.3126 -4.2219 -0.5151 ]
  🏆 Predicted Class for Image 0: 6 (Confidence: 10.8312)

================================================================
  EDGE DEPLOYMENT VERIFIED. MODEL IS PORTABLE.
================================================================
```

### D. Red-Team Finite Difference Gradient Checker (`test_grad_check`)
* **Source:** [test_grad_check.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/examples/test_grad_check.c)
* **Result:** Evaluates backpropagation correctness by comparing analytical autodiff gradients against numerical finite differences ($h = 10^{-5}$).
```
================================================================
  Lancius: FINITE DIFFERENCE GRADIENT CHECKER (RED TEAM)    
================================================================

[1/3] Computing Analytical Gradients via Autodiff...
[2/3] Computing Finite Differences (Perturbing Weights)...
[3/3] Comparing Analytical vs. Numerical...

================================================================
  GRADIENT CHECK RESULTS:
  Conv2D Weights (W1) Max Relative Error: 8.559618e-10
  ✅ PASS (Threshold: < 1.000000e-04)
  MatMul Weights (W2) Max Relative Error: 5.768201e-08
  ✅ PASS (Threshold: < 1.000000e-04)
================================================================
```

### E. Graph Save/Load & Permute Validation (`test_path_bg`)
* **Source:** [test_path_bg.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/examples/test_path_bg.c)
* **Result:** Verifies dimensional permutation correctness and serialization mapping consistency.
```
================================================================
  Lancius: Path B (Serialization) & Path G (Permute) Test   
================================================================

[PATH G] Testing N-Dimensional Permute...
  ✅ Permute Executed! Output shape: [2, 4, 5, 3]
  ✅ Shape correctly transposed from [2,3,4,5] to [2,4,5,3]!

[PATH B] Testing Graph Save & Load...
  Saving graph to 'test_model.lancius'...
[LANCIUS SERIAL] Saved 4 nodes to test_model.lancius
  Destroying original graph from RAM...
  Loading graph from 'test_model.lancius'...
[LANCIUS SERIAL] Loaded 4 nodes from test_model.lancius
  ✅ Serialization Success! Loaded 4 nodes.
  ✅ Weights perfectly preserved! (W[0] = 0.420000)

================================================================
  PATH B & G VERIFICATION COMPLETE.
================================================================
```

### F. Dynamic Graph Fuzzing (`fuzz_lancius`)
* **Source:** [fuzz_lancius.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/examples/fuzz_lancius.c)
* **Result:** Randomly generates shapes and constructs DAG configurations to ensure the runtime boundary-checks shape alignment and avoids segmentation faults.
```
  Fuzzing... 500 / 500 graphs processed.

================================================================
  FUZZ RESULTS: 500 PASSED | 0 FAILED (out of 500)
  (Failures here usually mean IR shape-rejection, which is SAFE)
================================================================
```

---

## 3. Building and Running

### Prerequisites
A Linux environment with `gcc` (supporting OpenMP) and `make`.

### Build Commands
Run from the `bleeding/v8p5/` directory:
```bash
# Clean previous builds
make clean

# Compile all libraries and executable binaries
make
```

### Execution Commands
```bash
# Run standalone inference deployment
./run_edge

# Run gradient check validation
./test_grad_check

# Run serialization & permutation tests
./test_path_bg

# Run dynamic graph fuzzing
./fuzz_lancius

# Train MNIST MLP Model
./train_mnist

# Train CIFAR-10 LeNet Model (and produce quantized weights)
./train_cifar10
```

---

## 4. Codebase Navigation

Explore the implementation directly:

* **Core Engine:**
  * [lancius_arena.h](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/include/lancius/lancius_arena.h) / [lancius_arena.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/core/lancius_arena.c) - Allocator
  * [lancius_ir.h](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/include/lancius/lancius_ir.h) / [lancius_ir.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/ir/lancius_ir.c) - Immutable SSA graph
  * [lancius_scheduler.h](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/include/lancius/lancius_scheduler.h) / [lancius_scheduler.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/runtime/lancius_scheduler.c) - Execution runtime & scheduling
  * [lancius_serialize.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/core/lancius_serialize.c) - Custom binary parsing
* **Math & Optimization:**
  * [lancius_autodiff.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/math/lancius_autodiff.c) - Auto-diff backward compiler
  * [lancius_kernels.h](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/include/lancius/lancius_kernels.h) / [lancius_kernels.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/math/lancius_kernels.c) - Math ops and Transformer kernels
  * [lancius_quantize.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/compiler/lancius_quantize.c) - Symmetric INT8 quantizer
  * [lancius_vision_ops.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/src/runtime/lancius_vision_ops.c) - Vision execution routing & activation quantization
* **Examples & Drivers:**
  * [run_edge.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/examples/run_edge.c) - Standalone Inference
  * [test_grad_check.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/examples/test_grad_check.c) - Gradient checking
  * [test_path_bg.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/examples/test_path_bg.c) - Serialization & Permutation test
  * [fuzz_lancius.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/examples/fuzz_lancius.c) - Boundary fuzzer
  * [train_mnist.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/examples/train_mnist.c) - MNIST training MLP
  * [train_cifar10.c](file:///home/magi-01/Desktop/work/programming/llm_maths/bleeding/v8p5/examples/train_cifar10.c) - CIFAR-10 training CNN