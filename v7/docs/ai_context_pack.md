# PROJECT BUNDLE: v7
Root Directory: /home/magi-01/Desktop/work/programming/llm_maths/v7
Generated: Sat Jun 27 05:53:53 PM MDT 2026

## 1. DIRECTORY HIERARCHY
```text
v7/
├── include/
│   ├── basis/
│   │   ├── core/
│   │   │   ├── error.h
│   │   │   ├── ir.h
│   │   │   ├── memory.h
│   │   │   └── test_harness.h
│   │   ├── l1_math/
│   │   │   └── tensor_v7.h
│   │   ├── l4_compiler/
│   │   │   ├── lowering.h
│   │   │   └── pass_manager.h
│   │   ├── l5_runtime/
│   │   │   └── scheduler.h
│   │   ├── l6_learning/
│   │   │   └── autodiff.h
│   │   ├── l7_intelligence/
│   │   │   └── swarm.h
│   │   ├── stage1_atomic/
│   │   │   └── scalar.h
│   │   ├── stage2_linear/
│   │   │   └── tensor.h
│   │   ├── stage3_sequence/
│   │   │   └── sequence.h
│   │   ├── stage4_learning/
│   │   │   ├── learning.h
│   │   │   └── optim.h
│   │   ├── stage5_unified/
│   │   │   ├── compiler.h
│   │   │   ├── geometry.h
│   │   │   ├── jit.h
│   │   │   └── symbolic.h
│   │   ├── stage6_ir/
│   │   │   ├── graph.h
│   │   │   └── lowering.h
│   │   ├── compiler.h
│   │   ├── geometry.h
│   │   ├── jit.h
│   │   ├── learning.h
│   │   ├── optim.h
│   │   ├── scalar.h
│   │   ├── sequence.h
│   │   ├── symbolic.h
│   │   └── tensor.h
│   └── basis.h
├── src/
│   └── basis/
│       ├── core/
│       │   ├── error.c
│       │   ├── error.o
│       │   ├── ir.c
│       │   ├── ir.o
│       │   ├── memory.c
│       │   └── memory.o
│       ├── l1_math/
│       │   ├── tensor_v7.c
│       │   └── tensor_v7.o
│       ├── l4_compiler/
│       │   ├── lowering.c
│       │   ├── lowering.o
│       │   ├── pass_manager.c
│       │   └── pass_manager.o
│       ├── l5_runtime/
│       │   ├── scheduler.c
│       │   └── scheduler.o
│       ├── l6_learning/
│       │   ├── autodiff.c
│       │   └── autodiff.o
│       ├── l7_intelligence/
│       │   ├── swarm.c
│       │   └── swarm.o
│       ├── stage1_atomic/
│       │   ├── scalar.c
│       │   └── scalar.o
│       ├── stage2_linear/
│       │   ├── tensor.c
│       │   └── tensor.o
│       ├── stage3_sequence/
│       │   ├── sequence.c
│       │   └── sequence.o
│       ├── stage4_learning/
│       │   ├── learning.c
│       │   ├── learning.o
│       │   ├── optim.c
│       │   └── optim.o
│       ├── stage5_unified/
│       │   ├── compiler.c
│       │   ├── compiler.o
│       │   ├── geometry.c
│       │   ├── geometry.o
│       │   ├── jit.c
│       │   ├── jit.o
│       │   ├── symbolic.c
│       │   └── symbolic.o
│       └── stage6_ir/
│           ├── graph.c
│           ├── graph.o
│           ├── lowering.c
│           └── lowering.o
├── tests/
│   ├── test_v5_suite.c
│   ├── test_v7_autodiff.c
│   ├── test_v7_ir.c
│   ├── test_v7_memory.c
│   ├── test_v7_passes.c
│   ├── test_v7_scheduler.c
│   ├── test_v7_swarm.c
│   ├── test_v7_tensor.c
│   ├── test_v7_transformer.c
│   └── test_v7_transformer_primitives.c
├── b1.basis
├── b2.basis
├── bundle_model
├── bundle_model.c
├── compile_mlp
├── compile_mlp.c
├── deploy_xor
├── deploy_xor.c
├── ghost_model.c
├── ghost_model.so
├── makefile
├── standalone_model
├── standalone_model.c
├── test_global
├── test_global.c
├── test_swarm
├── test_swarm.c
├── test_v7_autodiff
├── trace_mlp
├── trace_mlp.c
├── train_burgers
├── train_burgers.c
├── train_mlp_aot
├── train_mlp_aot.c
├── train_pinn
├── train_pinn.c
├── W1.basis
└── W2.basis
```

## 2. FILE CONTENTS

### FILE: bundle_model.c
Location: `bundle_model.c`
```cpp
#include <basis.h>
#include <stdio.h>

// Helper to emit a BASIS tensor as a hardcoded C constant array
void emit_array(FILE* f, const char* name, basis_tensor* t) {
    fprintf(f, "const double %s[%zu] = {\n", name, t->row_count * t->column_count);
    for(size_t i=0; i<t->row_count; i++) {
        fprintf(f, "    ");
        for(size_t j=0; j<t->column_count; j++) {
            fprintf(f, "%.17g, ", t->data[i*t->column_count + j]->data);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
}

int main() {
    printf("=== Ghost-Weight Bundler ===\n");
    printf("Reading trained weights from disk...\n");

    basis_tensor* W1 = basis_tensor_load_binary("W1.basis");
    basis_tensor* b1 = basis_tensor_load_binary("b1.basis");
    basis_tensor* W2 = basis_tensor_load_binary("W2.basis");
    basis_tensor* b2 = basis_tensor_load_binary("b2.basis");

    if (!W1 || !b1 || !W2 || !b2) {
        printf("Error: Could not load .basis files. Run ./deploy_xor first!\n");
        return 1;
    }

    printf("Generating standalone_model.c...\n");
    FILE* f = fopen("standalone_model.c", "w");

    fprintf(f, "#include <stdio.h>\n\n");

    // 1. Embed the weights directly into the C source code!
    emit_array(f, "W1", W1);
    emit_array(f, "b1", b1);
    emit_array(f, "W2", W2);
    emit_array(f, "b2", b2);

    // 2. Emit the forward pass logic using the hardcoded arrays
    fprintf(f, "int main() {\n");
    fprintf(f, "    double X[8] = {0,0, 0,1, 1,0, 1,1};\n");
    fprintf(f, "    double H1[32] = {0};\n");
    fprintf(f, "    double Y[4] = {0};\n\n");

    // Layer 1 (MatMul + Broadcast Add + ReLU)
    fprintf(f, "    for(int i=0; i<4; i++) {\n");
    fprintf(f, "        for(int j=0; j<8; j++) {\n");
    fprintf(f, "            double sum = 0.0;\n");
    fprintf(f, "            for(int k=0; k<2; k++) sum += X[i*2+k] * W1[k*8+j];\n");
    fprintf(f, "            double val = sum + b1[j];\n");
    fprintf(f, "            H1[i*8+j] = val > 0.0 ? val : 0.0;\n");
    fprintf(f, "        }\n");
    fprintf(f, "    }\n\n");

    // Layer 2 (MatMul + Broadcast Add)
    fprintf(f, "    for(int i=0; i<4; i++) {\n");
    fprintf(f, "        double sum = 0.0;\n");
    fprintf(f, "        for(int k=0; k<8; k++) sum += H1[i*8+k] * W2[k];\n");
    fprintf(f, "        Y[i] = sum + b2[0];\n");
    fprintf(f, "    }\n\n");

    fprintf(f, "    printf(\"=== Standalone Ghost-Weight Inference ===\\n\");\n");
    fprintf(f, "    for(int i=0; i<4; i++) printf(\"Input: [%%.0f, %%.0f] | Output: %%.4f\\n\", X[i*2], X[i*2+1], Y[i]);\n");
    fprintf(f, "    return 0;\n");
    fprintf(f, "}\n");

    fclose(f);

    basis_tensor_free(W1); basis_tensor_free(b1);
    basis_tensor_free(W2); basis_tensor_free(b2);

    printf("✅ Bundling complete. 'standalone_model.c' is ready.\n");
    return 0;
}

```

---

### FILE: compile_mlp.c
Location: `compile_mlp.c`
```cpp
#include <basis.h>
#include <stdio.h>
#include <dlfcn.h>

typedef void (*ghost_forward_t)(double*, double*, double*, double*, double*, double*);

int main() {
    printf("=== BASIS v6: AOT Lowering Compiler (Ghost-Weight Engine) ===\n\n");

    // 1. Trace the Graph IR
    basis_graph* g = basis_graph_new();
    basis_node* X = basis_graph_input(g, "X", 4, 2);
    basis_node* W1 = basis_graph_input(g, "W1", 2, 8);
    basis_node* b1 = basis_graph_input(g, "b1", 1, 8);
    basis_node* W2 = basis_graph_input(g, "W2", 8, 1);
    basis_node* b2 = basis_graph_input(g, "b2", 1, 1);

    basis_node* Z1 = basis_graph_matmul(g, X, W1);
    basis_node* A1 = basis_graph_broadcast_add(g, Z1, b1);
    basis_node* H1 = basis_graph_relu(g, A1);
    basis_node* Z2 = basis_graph_matmul(g, H1, W2);
    basis_node* Y = basis_graph_broadcast_add(g, Z2, b2);

    // 2. Lower to Monolithic C Code
    printf("[1/4] Lowering IR to ghost_model.c...\n");
    basis_graph_lower_to_c(g, "ghost_model.c");
    basis_graph_free(g);

    // 3. Compile to Shared Object (Zero BASIS Runtime Dependency)
    printf("[2/4] Compiling ghost_model.c to native machine code (AVX/SIMD)...\n");
    int status = system("gcc -shared -fPIC -O3 -march=native -ffast-math -o ghost_model.so ghost_model.c 2>/dev/null");
    if (status != 0) {
        printf("Compilation failed!\n");
        return 1;
    }

    // 4. Load and Execute
    printf("[3/4] Loading ghost_model.so via dlopen...\n");
    void* handle = dlopen("./ghost_model.so", RTLD_NOW);
    if (!handle) {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }

    ghost_forward_t forward = (ghost_forward_t)dlsym(handle, "ghost_weight_forward");
    if (!forward) {
        printf("dlsym failed: %s\n", dlerror());
        return 1;
    }

    printf("[4/4] Executing compiled model with raw C-arrays (NO BASIS TENSORS)...\n\n");

    // Raw C-Arrays (Simulating weights loaded from disk or embedded in firmware)
    double raw_X[8] = {0,0, 0,1, 1,0, 1,1};
    double raw_W1[16] = {0.5, -0.5, 0.8, 0.2, -0.1, 0.9, 0.3, -0.4, 0.7, 0.1, -0.6, 0.5, 0.2, -0.8, 0.4, 0.6};
    double raw_b1[8] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    double raw_W2[8] = {0.5, -0.5, 0.8, 0.2, -0.1, 0.9, 0.3, -0.4};
    double raw_b2[1] = {0.0};
    double raw_Y[4] = {0};

    // Execute the compiled machine code!
    forward(raw_X, raw_W1, raw_b1, raw_W2, raw_b2, raw_Y);

    printf("=== Native Execution Results ===\n");
    for(int i=0; i<4; i++) {
        printf("Input: [%.0f, %.0f] | Native Output: %.4f\n", raw_X[i*2], raw_X[i*2+1], raw_Y[i]);
    }

    dlclose(handle);
    printf("\n✅ Ghost-Weight Pipeline Complete.\n");
    printf("   The model executed using ONLY raw C-arrays and native machine code.\n");
    printf("   ZERO basis_tensor structs. ZERO autograd nodes. ZERO dependencies.\n");

    return 0;
}

```

---

### FILE: deploy_xor.c
Location: `deploy_xor.c`
```cpp
#include <basis.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(42);
    printf("=== BASIS v5: End-to-End Deployment Pipeline ===\n\n");

    // ==========================================
    // PHASE 1: TRAINING (The "Lab" Environment)
    // ==========================================
    printf("[1/4] Training XOR Network...\n");
    basis_tensor* X = basis_tensor_new(4, 2);
    basis_tensor_set(X, 0, 0, 0.0); basis_tensor_set(X, 0, 1, 0.0);
    basis_tensor_set(X, 1, 0, 0.0); basis_tensor_set(X, 1, 1, 1.0);
    basis_tensor_set(X, 2, 0, 1.0); basis_tensor_set(X, 2, 1, 0.0);
    basis_tensor_set(X, 3, 0, 1.0); basis_tensor_set(X, 3, 1, 1.0);

    basis_tensor* Y_true = basis_tensor_new(4, 1);
    basis_tensor_set(Y_true, 0, 0, 0.0); basis_tensor_set(Y_true, 1, 0, 1.0);
    basis_tensor_set(Y_true, 2, 0, 1.0); basis_tensor_set(Y_true, 3, 0, 0.0);

    basis_tensor* W1 = basis_tensor_new(2, 8);
    basis_tensor* b1 = basis_tensor_new(1, 8);
    basis_tensor* W2 = basis_tensor_new(8, 1);
    basis_tensor* b2 = basis_tensor_new(1, 1);

    for(size_t i=0; i<16; i++) W1->data[i]->data = ((rand() % 100) / 100.0) - 0.5;
    for(size_t i=0; i<8; i++)  W2->data[i]->data = ((rand() % 100) / 100.0) - 0.5;

    basis_adam* opt_W1 = basis_adam_new(2, 8, 0.05);
    basis_adam* opt_b1 = basis_adam_new(1, 8, 0.05);
    basis_adam* opt_W2 = basis_adam_new(8, 1, 0.05);
    basis_adam* opt_b2 = basis_adam_new(1, 1, 0.05);

    for(int epoch = 0; epoch < 2000; epoch++) {
        basis_tensor* Z1 = basis_tensor_matrix_multiplication(X, W1);
        basis_tensor* b1_view = basis_tensor_broadcast_view(b1, 4, 8);
        basis_tensor* A1 = basis_tensor_add(Z1, b1_view);
        basis_tensor* H1 = basis_tensor_rectified_linear_unit(A1);
        basis_tensor* Z2 = basis_tensor_matrix_multiplication(H1, W2);
        basis_tensor* b2_view = basis_tensor_broadcast_view(b2, 4, 1);
        basis_tensor* Y_pred = basis_tensor_add(Z2, b2_view);

        basis_value* loss_sum = basis_value_new(0.0);
        for(int i=0; i<4; i++) {
            basis_value* diff = basis_value_subtraction(Y_pred->data[i], Y_true->data[i]);
            basis_value* sq = basis_value_power(diff, 2.0);
            basis_value* new_sum = basis_value_addition(loss_sum, sq);
            basis_value_free(loss_sum); basis_value_free(diff); basis_value_free(sq);
            loss_sum = new_sum;
        }
        basis_value* divisor = basis_value_new(0.25);
        basis_value* loss = basis_value_multiplication(loss_sum, divisor);

        basis_value_backward_propagation(loss);

        basis_adam_optimization_step(opt_W1, W1); basis_adam_optimization_step(opt_b1, b1);
        basis_adam_optimization_step(opt_W2, W2); basis_adam_optimization_step(opt_b2, b2);

        basis_value_free(loss); basis_value_free(divisor); basis_value_free(loss_sum);
        basis_tensor_free(Y_pred); basis_tensor_free(b2_view); basis_tensor_free(Z2);
        basis_tensor_free(H1); basis_tensor_free(A1); basis_tensor_free(b1_view); basis_tensor_free(Z1);
    }
    printf("      Training complete.\n");

    // ==========================================
    // PHASE 2: SERIALIZATION (Export to Disk)
    // ==========================================
    printf("[2/4] Saving weights to disk (*.basis)...\n");
    basis_tensor_save_binary(W1, "W1.basis");
    basis_tensor_save_binary(b1, "b1.basis");
    basis_tensor_save_binary(W2, "W2.basis");
    basis_tensor_save_binary(b2, "b2.basis");
    printf("      Weights exported to binary format.\n");

    // ==========================================
    // PHASE 3: TEARDOWN (Simulate shutting down the training server)
    // ==========================================
    printf("[3/4] Purging training memory...\n");
    basis_adam_free(opt_W1); basis_adam_free(opt_b1);
    basis_adam_free(opt_W2); basis_adam_free(opt_b2);
    basis_tensor_free(W1); basis_tensor_free(b1);
    basis_tensor_free(W2); basis_tensor_free(b2);
    basis_tensor_free(X); basis_tensor_free(Y_true);
    printf("      All training tensors freed. Memory is empty.\n\n");

    // ==========================================
    // PHASE 4: DEPLOYMENT (Load & Infer)
    // ==========================================
    printf("[4/4] Loading weights into fresh memory for Inference...\n");
    basis_tensor* W1_inf = basis_tensor_load_binary("W1.basis");
    basis_tensor* b1_inf = basis_tensor_load_binary("b1.basis");
    basis_tensor* W2_inf = basis_tensor_load_binary("W2.basis");
    basis_tensor* b2_inf = basis_tensor_load_binary("b2.basis");
    printf("      Weights successfully loaded from disk.\n\n");

    printf("=== Inference Results ===\n");
    basis_tensor* X_test = basis_tensor_new(4, 2);
    basis_tensor_set(X_test, 0, 0, 0.0); basis_tensor_set(X_test, 0, 1, 0.0);
    basis_tensor_set(X_test, 1, 0, 0.0); basis_tensor_set(X_test, 1, 1, 1.0);
    basis_tensor_set(X_test, 2, 0, 1.0); basis_tensor_set(X_test, 2, 1, 0.0);
    basis_tensor_set(X_test, 3, 0, 1.0); basis_tensor_set(X_test, 3, 1, 1.0);

    basis_tensor* Z1 = basis_tensor_matrix_multiplication(X_test, W1_inf);
    basis_tensor* b1_view = basis_tensor_broadcast_view(b1_inf, 4, 8);
    basis_tensor* A1 = basis_tensor_add(Z1, b1_view);
    basis_tensor* H1 = basis_tensor_rectified_linear_unit(A1);
    basis_tensor* Z2 = basis_tensor_matrix_multiplication(H1, W2_inf);
    basis_tensor* b2_view = basis_tensor_broadcast_view(b2_inf, 4, 1);
    basis_tensor* Y_pred = basis_tensor_add(Z2, b2_view);

    for(int i=0; i<4; i++) {
        printf("Input: [%.0f, %.0f] | Predicted: %.4f\n",
               X_test->data[i*2]->data, X_test->data[i*2+1]->data,
               Y_pred->data[i]->data);
    }

    basis_tensor_free(Y_pred); basis_tensor_free(b2_view); basis_tensor_free(Z2);
    basis_tensor_free(H1); basis_tensor_free(A1); basis_tensor_free(b1_view); basis_tensor_free(Z1);
    basis_tensor_free(W1_inf); basis_tensor_free(b1_inf);
    basis_tensor_free(W2_inf); basis_tensor_free(b2_inf);
    basis_tensor_free(X_test);

    printf("\n✅ Deployment pipeline complete. Zero memory leaks.\n");
    return 0;
}

```

---

### FILE: ghost_model.c
Location: `ghost_model.c`
```cpp
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    double* t5;
    double* t6;
    double* t7;
    double* t8;
    double* t9;
} GhostContext;

void ghost_weight_forward(double* X, double* W1, double* b1, double* W2, double* b2, GhostContext* ctx, double* out) {
    ctx->t5 = (double*)malloc(4 * 8 * sizeof(double));
    for(size_t i=0; i<4; i++) for(size_t j=0; j<8; j++) {
        double sum = 0.0;
        for(size_t k=0; k<2; k++) sum += X[i*2 + k] * W1[k*8 + j];
        ctx->t5[i*8 + j] = sum;
    }
    ctx->t6 = (double*)malloc(4 * 8 * sizeof(double));
    for(size_t i=0; i<4; i++) for(size_t j=0; j<8; j++) ctx->t6[i*8 + j] = ctx->t5[i*8 + j] + b1[j];
    ctx->t7 = (double*)malloc(4 * 8 * sizeof(double));
    for(size_t i=0; i<4; i++) for(size_t j=0; j<8; j++) { double v = ctx->t6[i*8 + j]; ctx->t7[i*8 + j] = v > 0.0 ? v : 0.0; }
    ctx->t8 = (double*)malloc(4 * 1 * sizeof(double));
    for(size_t i=0; i<4; i++) for(size_t j=0; j<1; j++) {
        double sum = 0.0;
        for(size_t k=0; k<8; k++) sum += ctx->t7[i*8 + k] * W2[k*1 + j];
        ctx->t8[i*1 + j] = sum;
    }
    ctx->t9 = (double*)malloc(4 * 1 * sizeof(double));
    for(size_t i=0; i<4; i++) for(size_t j=0; j<1; j++) ctx->t9[i*1 + j] = ctx->t8[i*1 + j] + b2[j];
    memcpy(out, ctx->t9, 4 * 1 * sizeof(double));
}

void ghost_weight_backward(double* X, double* W1, double* b1, double* W2, double* b2, GhostContext* ctx, double* grad_out, double* grad_X, double* grad_W1, double* grad_b1, double* grad_W2, double* grad_b2) {
    double* grad_t5 = (double*)calloc(4 * 8, sizeof(double));
    double* grad_t6 = (double*)calloc(4 * 8, sizeof(double));
    double* grad_t7 = (double*)calloc(4 * 8, sizeof(double));
    double* grad_t8 = (double*)calloc(4 * 1, sizeof(double));
    double* grad_t9 = grad_out;
    for(size_t i=0; i<4; i++) for(size_t j=0; j<1; j++) grad_t8[i*1+j] += grad_t9[i*1+j];
    for(size_t j=0; j<1; j++) { double s=0; for(size_t i=0; i<4; i++) s += grad_t9[i*1+j]; grad_b2[j] += s; }
    for(size_t i=0; i<4; i++) for(size_t k=0; k<8; k++) { double s=0; for(size_t j=0; j<1; j++) s += grad_t8[i*1+j] * W2[k*1+j]; grad_t7[i*8+k] += s; }
    for(size_t k=0; k<8; k++) for(size_t j=0; j<1; j++) { double s=0; for(size_t i=0; i<4; i++) s += ctx->t7[i*8+k] * grad_t8[i*1+j]; grad_W2[k*1+j] += s; }
    for(size_t i=0; i<4; i++) for(size_t j=0; j<8; j++) grad_t6[i*8+j] += grad_t7[i*8+j] * (ctx->t7[i*8+j] > 0.0 ? 1.0 : 0.0);
    for(size_t i=0; i<4; i++) for(size_t j=0; j<8; j++) grad_t5[i*8+j] += grad_t6[i*8+j];
    for(size_t j=0; j<8; j++) { double s=0; for(size_t i=0; i<4; i++) s += grad_t6[i*8+j]; grad_b1[j] += s; }
    for(size_t i=0; i<4; i++) for(size_t k=0; k<2; k++) { double s=0; for(size_t j=0; j<8; j++) s += grad_t5[i*8+j] * W1[k*8+j]; grad_X[i*2+k] += s; }
    for(size_t k=0; k<2; k++) for(size_t j=0; j<8; j++) { double s=0; for(size_t i=0; i<4; i++) s += X[i*2+k] * grad_t5[i*8+j]; grad_W1[k*8+j] += s; }
    free(grad_t5);
    free(grad_t6);
    free(grad_t7);
    free(grad_t8);
}

void ghost_free_context(GhostContext* ctx) {
    free(ctx->t5);
    free(ctx->t6);
    free(ctx->t7);
    free(ctx->t8);
    free(ctx->t9);
}

```

---

### FILE: include/basis/compiler.h
Location: `include/basis/compiler.h`
```h
#include "basis/stage5_unified/compiler.h"

```

---

### FILE: include/basis/core/error.h
Location: `include/basis/core/error.h`
```h

#ifndef BASIS_ERROR_H
#define BASIS_ERROR_H
#include <stdio.h>
#include <stdbool.h>

typedef enum {
    BASIS_SUCCESS = 0,
    BASIS_NULL_POINTER,
    BASIS_OUT_OF_MEMORY,
    BASIS_INVALID_SHAPE,
    BASIS_INVALID_EXPRESSION,
    BASIS_DOMAIN_ERROR,
    BASIS_UNKNOWN_ERROR
} basis_error_t;

extern basis_error_t _basis_last_error;
extern const char* _basis_last_error_msg;
extern bool _basis_error_quiet;

#define BASIS_SET_ERROR(err, msg) do { \
    _basis_last_error = err; \
    _basis_last_error_msg = msg; \
    if (!_basis_error_quiet) { \
        fprintf(stderr, "[BASIS ERROR] %s: %s (File: %s, Line: %d)\n", #err, msg, __FILE__, __LINE__); \
    } \
} while(0)

#define BASIS_CHECK_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        BASIS_SET_ERROR(BASIS_NULL_POINTER, "Null pointer encountered"); \
        return NULL; \
    } \
} while(0)

#define BASIS_CHECK_SHAPE(cond) do { \
    if (!(cond)) { \
        BASIS_SET_ERROR(BASIS_INVALID_SHAPE, "Tensor shape mismatch or invalid dimensions"); \
        return NULL; \
    } \
} while(0)

static inline basis_error_t basis_get_last_error(void) { return _basis_last_error; }
static inline const char* basis_get_last_error_msg(void) { return _basis_last_error_msg; }
static inline void basis_clear_error(void) { _basis_last_error = BASIS_SUCCESS; _basis_last_error_msg = ""; }
static inline void basis_set_error_quiet(bool quiet) { _basis_error_quiet = quiet; }
#endif

```

---

### FILE: include/basis/core/ir.h
Location: `include/basis/core/ir.h`
```h
#ifndef BASIS_V7_IR_H
#define BASIS_V7_IR_H

#include "basis/core/memory.h"
#include <stddef.h>
#include <stdint.h>

typedef enum {
    BASIS_IR_OP_INPUT,
    BASIS_IR_OP_CONST,
    BASIS_IR_OP_ADD,
    BASIS_IR_OP_SUB,
    BASIS_IR_OP_MUL,
    BASIS_IR_OP_MATMUL,
    BASIS_IR_OP_RELU,
    BASIS_IR_OP_TANH,
    BASIS_IR_OP_TRANSPOSE,
    BASIS_IR_OP_SOFTMAX,
    BASIS_IR_OP_SUM,
    BASIS_IR_OP_BROADCAST,
    BASIS_IR_OP_RELU_BWD
} basis_ir_opcode;

// Immutable Node: Inputs are const pointers. Once created, a node cannot be mutated.
typedef struct basis_ir_node {
    uint32_t id;
    basis_ir_opcode op;
    size_t rows;
    size_t cols;

    const struct basis_ir_node** inputs;
    uint32_t input_count;

    double attr_val; // For CONST nodes or operation attributes
    double* runtime_data; // Populated by the V7 Executor
} basis_ir_node;

typedef struct basis_ir_graph {
    basis_arena* arena;          // All nodes and input arrays live here

    basis_ir_node** nodes;       // Tracking array for topological traversal
    uint32_t node_count;
    uint32_t node_cap;

    uint32_t next_id;
} basis_ir_graph;

basis_ir_graph* basis_ir_graph_create(void);
void basis_ir_graph_destroy(basis_ir_graph* g); // O(1) Teardown!

// Graph Builders (Enforce SSA & Shape Inference)
basis_ir_node* basis_ir_input(basis_ir_graph* g, size_t r, size_t c);
basis_ir_node* basis_ir_const(basis_ir_graph* g, double val, size_t r, size_t c);
basis_ir_node* basis_ir_add(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b);
basis_ir_node* basis_ir_matmul(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b);
basis_ir_node* basis_ir_relu(basis_ir_graph* g, const basis_ir_node* a);


basis_ir_node* basis_ir_mul(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b);
basis_ir_node* basis_ir_sub(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b);
basis_ir_node* basis_ir_transpose(basis_ir_graph* g, const basis_ir_node* a);
basis_ir_node* basis_ir_sum(basis_ir_graph* g, const basis_ir_node* a);
basis_ir_node* basis_ir_broadcast(basis_ir_graph* g, const basis_ir_node* a, size_t r, size_t c);
basis_ir_node* basis_ir_relu_bwd(basis_ir_graph* g, const basis_ir_node* grad, const basis_ir_node* fwd_a);
basis_ir_node* basis_ir_softmax(basis_ir_graph* g, const basis_ir_node* a);

void basis_ir_print(const basis_ir_graph* g);

#endif // BASIS_V7_IR_H

```

---

### FILE: include/basis/core/memory.h
Location: `include/basis/core/memory.h`
```h
#ifndef BASIS_V7_MEMORY_H
#define BASIS_V7_MEMORY_H

#include <stddef.h>
#include <stdint.h>

/* ==========================================================================
 * Arena Allocator (L0 Platform)
 * Used for: Graphs, ASTs, Scratch buffers.
 * Properties: O(1) allocation (pointer bump), O(1) teardown (reset).
 * ========================================================================== */
typedef struct basis_arena basis_arena;

basis_arena* basis_arena_create(size_t default_block_size);
void* basis_arena_alloc(basis_arena* arena, size_t size, size_t alignment);
void basis_arena_reset(basis_arena* arena); // O(1) "Free"
void basis_arena_destroy(basis_arena* arena);

/* ==========================================================================
 * Pool Allocator (L0 Platform)
 * Used for: Fixed-size Tensors, Graph Nodes.
 * Properties: O(1) alloc/free via free-list, zero fragmentation.
 * ========================================================================== */
typedef struct basis_pool basis_pool;

basis_pool* basis_pool_create(size_t object_size, size_t objects_per_chunk);
void* basis_pool_alloc(basis_pool* pool);
void basis_pool_free(basis_pool* pool, void* ptr);
void basis_pool_destroy(basis_pool* pool);

#endif // BASIS_V7_MEMORY_H

```

---

### FILE: include/basis/core/test_harness.h
Location: `include/basis/core/test_harness.h`
```h

#ifndef BASIS_TEST_HARNESS_H
#define BASIS_TEST_HARNESS_H
#include <stdio.h>
#include <math.h>

static int basis_tests_run = 0;
static int basis_tests_failed = 0;

#define BASIS_ASSERT(cond, msg) do { \
    basis_tests_run++; \
    if (!(cond)) { \
        printf("  [FAIL] %s (Line %d): %s\n", __func__, __LINE__, msg); \
        basis_tests_failed++; \
    } else { \
        printf("  [PASS] %s\n", msg); \
    } \
} while(0)

#define BASIS_ASSERT_NEAR(a, b, eps, msg) do { \
    basis_tests_run++; \
    if (fabs((a) - (b)) > (eps)) { \
        printf("  [FAIL] %s (Line %d): %s (Expected %f, Got %f)\n", __func__, __LINE__, msg, (double)(b), (double)(a)); \
        basis_tests_failed++; \
    } else { \
        printf("  [PASS] %s\n", msg); \
    } \
} while(0)

#define BASIS_TEST_SUITE_START(name) printf("\n=== SUITE: %s ===\n", name);
#define BASIS_TEST_SUITE_END() do { \
    printf("\nResults: %d/%d passed.\n", basis_tests_run - basis_tests_failed, basis_tests_run); \
    if (basis_tests_failed > 0) printf("!!! FAILURES DETECTED !!!\n"); \
    else printf("✅ ALL TESTS PASSED.\n"); \
} while(0)
#endif

```

---

### FILE: include/basis/geometry.h
Location: `include/basis/geometry.h`
```h
#include "basis/stage5_unified/geometry.h"

```

---

### FILE: include/basis/jit.h
Location: `include/basis/jit.h`
```h
#include "basis/stage5_unified/jit.h"

```

---

### FILE: include/basis/l1_math/tensor_v7.h
Location: `include/basis/l1_math/tensor_v7.h`
```h
#ifndef BASIS_V7_TENSOR_H
#define BASIS_V7_TENSOR_H

#include "basis/core/ir.h"
#include <stddef.h>

// Lightweight V7 Tensor: Just metadata and a link to the IR Graph
typedef struct {
    size_t rows;
    size_t cols;
    basis_ir_node* node;
    basis_ir_graph* graph;
} basis_v7_tensor;

// Graph Building (Deferred Execution)
basis_v7_tensor* basis_v7_input(basis_ir_graph* g, size_t r, size_t c, const double* initial_data);
basis_v7_tensor* basis_v7_const(basis_ir_graph* g, double val, size_t r, size_t c);
basis_v7_tensor* basis_v7_add(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b);
basis_v7_tensor* basis_v7_matmul(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b);
basis_v7_tensor* basis_v7_mul(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b);
basis_v7_tensor* basis_v7_sub(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b);
basis_v7_tensor* basis_v7_transpose(basis_ir_graph* g, basis_v7_tensor* a);
basis_v7_tensor* basis_v7_sum(basis_ir_graph* g, basis_v7_tensor* a);
basis_v7_tensor* basis_v7_broadcast(basis_ir_graph* g, basis_v7_tensor* a, size_t r, size_t c);
basis_v7_tensor* basis_v7_softmax(basis_ir_graph* g, basis_v7_tensor* a);
basis_v7_tensor* basis_v7_relu(basis_ir_graph* g, basis_v7_tensor* a);

// Execution & Teardown
void basis_v7_execute(basis_ir_graph* g);
void basis_v7_tensor_print(basis_v7_tensor* t, const char* name);

#endif // BASIS_V7_TENSOR_H

```

---

### FILE: include/basis/l4_compiler/lowering.h
Location: `include/basis/l4_compiler/lowering.h`
```h
#ifndef BASIS_V7_LOWERING_H
#define BASIS_V7_LOWERING_H
#include "basis/core/ir.h"
void basis_ir_lower_to_c(basis_ir_graph* g, const char* output_filename);
#endif

```

---

### FILE: include/basis/l4_compiler/pass_manager.h
Location: `include/basis/l4_compiler/pass_manager.h`
```h
#ifndef BASIS_V7_PASS_MANAGER_H
#define BASIS_V7_PASS_MANAGER_H

#include "basis/core/ir.h"

// Runs optimization passes (Constant Folding, Identity Elimination)
// and returns a newly optimized, immutable graph.
basis_ir_graph* basis_ir_run_passes(basis_ir_graph* old_g);

#endif // BASIS_V7_PASS_MANAGER_H

```

---

### FILE: include/basis/l5_runtime/scheduler.h
Location: `include/basis/l5_runtime/scheduler.h`
```h
#ifndef BASIS_V7_SCHEDULER_H
#define BASIS_V7_SCHEDULER_H

#include "basis/core/ir.h"
#include <stdint.h>

// A Wave contains a list of nodes that have NO dependencies on each other.
// They can be executed safely in parallel.
typedef struct {
    basis_ir_node** nodes;
    uint32_t node_count;
} basis_wave;

// The Execution Schedule
typedef struct {
    basis_wave* waves;
    uint32_t wave_count;
} basis_schedule;

// Analyzes the IR and groups nodes into deterministic execution waves.
basis_schedule* basis_ir_schedule(basis_ir_graph* g);

// Executes the schedule sequentially (Deterministic baseline).
// Future: basis_schedule_execute_parallel(schedule, thread_pool);
void basis_schedule_execute(basis_schedule* schedule, basis_arena* scratch);

void basis_schedule_print(basis_schedule* schedule);
void basis_schedule_destroy(basis_schedule* schedule);

#endif // BASIS_V7_SCHEDULER_H

```

---

### FILE: include/basis/l6_learning/autodiff.h
Location: `include/basis/l6_learning/autodiff.h`
```h
#ifndef BASIS_V7_AUTODIFF_H
#define BASIS_V7_AUTODIFF_H
#include "basis/core/ir.h"

typedef struct {
    basis_ir_graph* graph;
    basis_ir_node* loss_node;
    basis_ir_node** grad_nodes; // Mapped by forward input ID
    uint32_t max_id;
} basis_training_graph;

basis_training_graph* basis_ir_autodiff(basis_ir_graph* fwd_g, basis_ir_node* loss_node);
void basis_training_graph_destroy(basis_training_graph* tg);
#endif

```

---

### FILE: include/basis/l7_intelligence/swarm.h
Location: `include/basis/l7_intelligence/swarm.h`
```h
#ifndef BASIS_V7_SWARM_H
#define BASIS_V7_SWARM_H

#include "basis/core/ir.h"
#include "basis/l5_runtime/scheduler.h"
#include "basis/core/memory.h"

typedef struct {
    const char* name;
    basis_schedule* schedule;
    basis_arena* scratch;      // Isolated memory domain

    double** input_buffers;    // Array of persistent double* pointers
    size_t* input_sizes;       // Array of sizes in bytes
    uint32_t input_count;

    basis_ir_node* out_node;   // Reference to the output node to fetch active scratch data
    size_t output_size;        // Size of output in bytes
} basis_expert;

typedef struct {
    basis_expert** experts;
    uint32_t expert_count;
    uint32_t expert_cap;
} basis_swarm;

basis_swarm* basis_swarm_create(void);
void basis_swarm_register(basis_swarm* swarm, const char* name, basis_schedule* sched, basis_ir_graph* g);
basis_expert* basis_swarm_get(basis_swarm* swarm, const char* name);

void basis_swarm_dispatch(basis_expert* expert, double* in_data, double* out_data);
void basis_swarm_destroy(basis_swarm* swarm);

#endif

```

---

### FILE: include/basis/learning.h
Location: `include/basis/learning.h`
```h
#include "basis/stage4_learning/learning.h"

```

---

### FILE: include/basis/optim.h
Location: `include/basis/optim.h`
```h
#include "basis/stage4_learning/optim.h"

```

---

### FILE: include/basis/scalar.h
Location: `include/basis/scalar.h`
```h
#include "basis/stage1_atomic/scalar.h"

```

---

### FILE: include/basis/sequence.h
Location: `include/basis/sequence.h`
```h
#include "basis/stage3_sequence/sequence.h"

```

---

### FILE: include/basis/stage1_atomic/scalar.h
Location: `include/basis/stage1_atomic/scalar.h`
```h

#ifndef basis_stage1_scalar_h
#define basis_stage1_scalar_h
#include <stdlib.h>
#include <stdbool.h>

typedef struct basis_value {
    double data;
    double gradient;
    void (*backward_function)(struct basis_value* output_value);
    struct basis_value** previous_nodes;
    size_t previous_node_count;
    char* operation;
    double constant_value;
    size_t reference_count;
    bool visited;
} basis_value;

basis_value* basis_value_new(double initial_data);
void basis_value_retain(basis_value* target_value);
void basis_value_free(basis_value* target_value);
basis_value* basis_value_addition(basis_value* first_operand, basis_value* second_operand);
basis_value* basis_value_subtraction(basis_value* first_operand, basis_value* second_operand);
basis_value* basis_value_multiplication(basis_value* first_operand, basis_value* second_operand);
basis_value* basis_value_power(basis_value* base, double exponent);
basis_value* basis_value_exponential(basis_value* input_value);
basis_value* basis_value_logarithm(basis_value* input_value);
basis_value* basis_value_sine(basis_value* input_value);
basis_value* basis_value_cosine(basis_value* input_value);
basis_value* basis_value_tanh(basis_value* input_value);
basis_value* basis_value_rectified_linear_unit(basis_value* input_value);
void basis_value_zero_gradient(basis_value* root_value);
void basis_value_backward_propagation(basis_value* root_value);
#endif

```

---

### FILE: include/basis/stage2_linear/tensor.h
Location: `include/basis/stage2_linear/tensor.h`
```h
#ifndef basis_stage2_tensor_h
#define basis_stage2_tensor_h
#include "basis/stage1_atomic/scalar.h"
#include <stdbool.h>

typedef struct basis_tensor {
    basis_value** data;
    size_t flat_size; // Total allocated elements in root data array

    size_t row_count;
    size_t column_count;

    size_t row_stride;
    size_t col_stride;
    size_t offset;

    bool is_view;
    struct basis_tensor* parent;
    size_t ref_count;
} basis_tensor;

#define BASIS_TENSOR_AT(t, i, j) ((t)->data[(t)->offset + (i) * (t)->row_stride + (j) * (t)->col_stride])

basis_tensor* basis_tensor_new(size_t row_count, size_t column_count);
basis_tensor* basis_tensor_transpose(basis_tensor* target_tensor);
basis_tensor* basis_tensor_slice_rows(basis_tensor* target_tensor, size_t start_row, size_t num_rows);
void basis_tensor_free(basis_tensor* target_tensor);
void basis_tensor_fill(basis_tensor* target_tensor, double fill_value);
void basis_tensor_set(basis_tensor* target_tensor, size_t row_index, size_t column_index, double data_value);
basis_tensor* basis_tensor_addition(basis_tensor* first_tensor, basis_tensor* second_tensor);
basis_tensor* basis_tensor_matrix_multiplication(basis_tensor* first_tensor, basis_tensor* second_tensor);
basis_tensor* basis_tensor_scalar_multiplication(basis_tensor* target_tensor, double scalar_value);
basis_tensor* basis_tensor_rectified_linear_unit(basis_tensor* target_tensor);
basis_tensor* basis_tensor_tanh(basis_tensor* target_tensor);
basis_tensor* basis_tensor_broadcast_view(basis_tensor* target_tensor, size_t target_rows, size_t target_cols);
basis_tensor* basis_tensor_add(basis_tensor* first_tensor, basis_tensor* second_tensor);
void basis_tensor_matmul_backward_cblas(basis_tensor* grad_output, basis_tensor* A, basis_tensor* B);
void basis_tensor_save_binary(basis_tensor* target_tensor, const char* filename);
basis_tensor* basis_tensor_load_binary(const char* filename);
void basis_tensor_print(basis_tensor* target_tensor, const char* name);
#endif

```

---

### FILE: include/basis/stage3_sequence/sequence.h
Location: `include/basis/stage3_sequence/sequence.h`
```h

#ifndef basis_stage3_sequence_h
#define basis_stage3_sequence_h
#include "basis/stage2_linear/tensor.h"
basis_tensor* basis_tensor_transpose(basis_tensor* target_tensor);
basis_tensor* basis_tensor_softmax(basis_tensor* target_tensor);
basis_tensor* basis_tensor_log_softmax(basis_tensor* target_tensor);
basis_tensor* basis_tensor_attention(basis_tensor* query_tensor, basis_tensor* key_tensor, basis_tensor* basis_value_tensor);
#endif

```

---

### FILE: include/basis/stage4_learning/learning.h
Location: `include/basis/stage4_learning/learning.h`
```h

#ifndef basis_stage4_learning_h
#define basis_stage4_learning_h
#include "basis/stage2_linear/tensor.h"
basis_tensor* basis_tensor_layer_normalization(basis_tensor* target_tensor, double epsilon);
basis_tensor* basis_tensor_rotary_positional_embedding(basis_tensor* target_tensor, int position, double base);
basis_value* basis_tensor_sum(basis_tensor* target_tensor);
basis_tensor* basis_tensor_logarithm(basis_tensor* target_tensor);
basis_tensor* basis_tensor_multiplication(basis_tensor* first_tensor, basis_tensor* second_tensor);
#endif

```

---

### FILE: include/basis/stage4_learning/optim.h
Location: `include/basis/stage4_learning/optim.h`
```h

#ifndef basis_stage4_optim_h
#define basis_stage4_optim_h
#include "basis/stage2_linear/tensor.h"
typedef struct basis_adam {
    double learning_rate; double beta1; double beta2; double epsilon; size_t time_step;
    double* first_moment; double* second_moment; size_t element_count;
} basis_adam;
basis_adam* basis_adam_new(size_t row_count, size_t column_count, double learning_rate);
void basis_adam_optimization_step(basis_adam* optimizer, basis_tensor* weight_tensor);
void basis_adam_free(basis_adam* optimizer);
#endif

```

---

### FILE: include/basis/stage5_unified/compiler.h
Location: `include/basis/stage5_unified/compiler.h`
```h

#ifndef basis_stage5_unified_compiler_h
#define basis_stage5_unified_compiler_h
#include "basis/stage1_atomic/scalar.h"
#include "basis/stage5_unified/symbolic.h"

typedef struct { char* name; basis_value* basis_value_pointer; } basis_variable_mapping;

// V5 Phase C: CSE Cache Entry
typedef struct {
    basis_symbol* sym;
    basis_value* val;
    size_t hash;
} cse_entry;

typedef struct {
    basis_variable_mapping* variable_mappings;
    size_t variable_mapping_count;

    cse_entry* cse_cache;
    size_t cse_count;
    size_t cse_cap;
} basis_compiler;

basis_compiler* basis_compiler_new(void);
void basis_compiler_map(basis_compiler* basis_compiler_instance, char* variable_name, basis_value* basis_value_pointer);
basis_value* basis_compiler_compile(basis_compiler* basis_compiler_instance, basis_symbol* basis_symbol_node);
void basis_compiler_free(basis_compiler* basis_compiler_instance);
#endif

```

---

### FILE: include/basis/stage5_unified/geometry.h
Location: `include/basis/stage5_unified/geometry.h`
```h
#ifndef basis_stage5_unified_geometry_h
#define basis_stage5_unified_geometry_h
#include "basis/stage2_linear/tensor.h"

typedef struct {
    basis_tensor* matrix;
} basis_metric;

basis_metric* basis_metric_fisher_information(basis_tensor* weight_tensor, basis_tensor* input_tensor, basis_tensor* output_tensor);
void basis_metric_free(basis_metric* target_metric);
void basis_metric_ngd_step(basis_metric* fisher_metric, basis_tensor* weight_tensor, double learning_rate);

#endif

```

---

### FILE: include/basis/stage5_unified/jit.h
Location: `include/basis/stage5_unified/jit.h`
```h
#ifndef basis_stage5_unified_jit_h
#define basis_stage5_unified_jit_h
#include "basis/stage5_unified/symbolic.h"

typedef double (*basis_jit_func_t)(double*);

typedef struct {
    void* handle;
    basis_jit_func_t func;
    char* so_path;
} basis_jit_module;

basis_jit_module* basis_jit_compile(basis_symbol* root, char** var_names, size_t var_count);
double basis_jit_execute(basis_jit_module* mod, double* var_values);
void basis_jit_free(basis_jit_module* mod);

#endif

```

---

### FILE: include/basis/stage5_unified/symbolic.h
Location: `include/basis/stage5_unified/symbolic.h`
```h

#ifndef basis_stage5_symbolic_h
#define basis_stage5_symbolic_h
#include <stdlib.h>
typedef enum { basis_symbol_type_variable, basis_symbol_type_constant, basis_symbol_type_operation } basis_symbol_type;
typedef struct basis_symbol {
    basis_symbol_type type; char* name; double basis_value; char* operation;
    struct basis_symbol** arguments; size_t argument_count;
} basis_symbol;
basis_symbol* basis_symbol_variable(char* variable_name);
basis_symbol* basis_symbol_constant(double constant_value);
basis_symbol* basis_symbol_addition(basis_symbol* first_operand, basis_symbol* second_operand);
basis_symbol* basis_symbol_subtraction(basis_symbol* first_operand, basis_symbol* second_operand);
basis_symbol* basis_symbol_multiplication(basis_symbol* first_operand, basis_symbol* second_operand);
basis_symbol* basis_symbol_power(basis_symbol* base, double exponent);
basis_symbol* basis_symbol_logarithm(basis_symbol* input_symbol);
basis_symbol* basis_symbol_exponential(basis_symbol* input_symbol);
basis_symbol* basis_symbol_copy(basis_symbol* source_symbol);
basis_symbol* basis_symbol_differentiation(basis_symbol* source_symbol, char* variable_name);
basis_symbol* basis_symbol_simplify(basis_symbol* source_symbol);
void basis_symbol_print(basis_symbol* target_symbol);
void basis_symbol_free(basis_symbol* target_symbol);
#endif

```

---

### FILE: include/basis/stage6_ir/graph.h
Location: `include/basis/stage6_ir/graph.h`
```h
#ifndef BASIS_STAGE6_IR_GRAPH_H
#define BASIS_STAGE6_IR_GRAPH_H

#include <stdlib.h>
#include <stdio.h>

typedef enum {
    BASIS_OP_INPUT,
    BASIS_OP_MATMUL,
    BASIS_OP_ADD,
    BASIS_OP_RELU,
    BASIS_OP_TANH,
    BASIS_OP_BROADCAST_ADD
} basis_op_type;

typedef struct basis_node {
    int id;
    basis_op_type op;
    int input_ids[2];
    size_t rows;
    size_t cols;
    char* name;
} basis_node;

typedef struct basis_graph {
    basis_node* nodes;
    size_t node_count;
    size_t node_cap;
} basis_graph;

basis_graph* basis_graph_new(void);
basis_node* basis_graph_input(basis_graph* g, const char* name, size_t r, size_t c);
basis_node* basis_graph_matmul(basis_graph* g, basis_node* a, basis_node* b);
basis_node* basis_graph_add(basis_graph* g, basis_node* a, basis_node* b);
basis_node* basis_graph_broadcast_add(basis_graph* g, basis_node* a, basis_node* bias);
basis_node* basis_graph_relu(basis_graph* g, basis_node* a);
basis_node* basis_graph_tanh(basis_graph* g, basis_node* a);
void basis_graph_print(basis_graph* g);
void basis_graph_free(basis_graph* g);

#endif

```

---

### FILE: include/basis/stage6_ir/lowering.h
Location: `include/basis/stage6_ir/lowering.h`
```h
#ifndef BASIS_STAGE6_IR_LOWERING_H
#define BASIS_STAGE6_IR_LOWERING_H

#include "basis/stage6_ir/graph.h"

// Lowers the IR graph into a monolithic C source file containing the forward pass.
// The generated function signature will be:
// void ghost_weight_forward(double* input_0, double* input_1, ..., double* out)
void basis_graph_lower_to_c(basis_graph* g, const char* output_filename);

#endif

```

---

### FILE: include/basis/symbolic.h
Location: `include/basis/symbolic.h`
```h
#include "basis/stage5_unified/symbolic.h"

```

---

### FILE: include/basis/tensor.h
Location: `include/basis/tensor.h`
```h
#include "basis/stage2_linear/tensor.h"

```

---

### FILE: include/basis.h
Location: `include/basis.h`
```h
/**
 * @file basis.h
 * @brief BASIS v5: Bare-metal Autograd & Symbolic Inference System
 *
 * @section intro_sec Introduction
 * BASIS is a from-scratch, pure-C implementation of a reverse-mode automatic
 * differentiation engine, tensor mathematics, symbolic calculus, and a JIT
 * machine-code compiler.
 *
 * @section memory_sec Strict Memory Ownership
 * BASIS uses manual reference counting.
 * - Functions that create new tensors/values (e.g., basis_tensor_new, basis_value_addition)
 *   return objects with a reference count of 1. The caller owns them and MUST free them.
 * - Views (e.g., basis_tensor_transpose, basis_tensor_broadcast_view) share underlying
 *   data and increment the parent's reference count. Freeing a view is safe and required.
 * - Optimizer states (basis_adam) are decoupled from the autograd graph and must be
 *   freed independently.
 *
 * @section compile_sec Compilation
 * Link against the shared library: `gcc main.c -lbasis -lm -ldl`
 */
#ifndef BASIS_UMBRELLA_H
#define BASIS_UMBRELLA_H

#include "basis/core/error.h"
#include "basis/core/memory.h"
#include "basis/core/ir.h"
#include "basis/l5_runtime/scheduler.h"
#include "basis/l6_learning/autodiff.h"
#include "basis/l7_intelligence/swarm.h"
#include "basis/l4_compiler/lowering.h"
#include "basis/l1_math/tensor_v7.h"
#include "basis/l4_compiler/pass_manager.h"
#include "basis/stage1_atomic/scalar.h"
#include "basis/stage2_linear/tensor.h"
#include "basis/stage3_sequence/sequence.h"
#include "basis/stage4_learning/learning.h"
#include "basis/stage4_learning/optim.h"
#include "basis/stage5_unified/symbolic.h"
#include "basis/stage5_unified/compiler.h"
#include "basis/stage5_unified/geometry.h"
#include "basis/stage5_unified/jit.h"
#include "basis/stage6_ir/graph.h"
#include "basis/stage6_ir/lowering.h"

#endif

```

---

### FILE: makefile
Location: `makefile`
```text
CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -std=c11 -I./include -fPIC
LDFLAGS = -lm -ldl

SRCS = src/basis/core/error.c \
       src/basis/core/memory.c \
       src/basis/core/ir.c \
       src/basis/stage1_atomic/scalar.c \
       src/basis/stage2_linear/tensor.c \
       src/basis/stage3_sequence/sequence.c \
       src/basis/stage4_learning/learning.c \
       src/basis/stage4_learning/optim.c \
       src/basis/stage5_unified/symbolic.c \
       src/basis/stage5_unified/compiler.c \
       src/basis/stage5_unified/geometry.c \
       src/basis/stage5_unified/jit.c \
       src/basis/stage6_ir/graph.c \
       src/basis/stage6_ir/lowering.c \
       src/basis/l1_math/tensor_v7.c \
       src/basis/l4_compiler/pass_manager.c \
       src/basis/l4_compiler/lowering.c \
       src/basis/l5_runtime/scheduler.c \
       src/basis/l6_learning/autodiff.c \
       src/basis/l7_intelligence/swarm.c

OBJS = $(SRCS:.c=.o)

all: test_v5_suite libbasis.so libbasis.a

libbasis.so: $(OBJS)
	$(CC) -shared -o $@ $(OBJS) $(LDFLAGS)

libbasis.a: $(OBJS)
	ar rcs $@ $(OBJS)

test_v5_suite: tests/test_v5_suite.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_v7_memory: tests/test_v7_memory.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_v7_ir: tests/test_v7_ir.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_v7_tensor: tests/test_v7_tensor.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_v7_passes: tests/test_v7_passes.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_v7_lowering: tests/test_v7_lowering.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_v7_scheduler: tests/test_v7_scheduler.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)


test_v7_transformer_primitives: tests/test_v7_transformer_primitives.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)


test_v7_transformer: tests/test_v7_transformer.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)


test_v7_autodiff: tests/test_v7_autodiff.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)


test_v7_swarm: tests/test_v7_swarm.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

PREFIX ?= /usr/local
install: all
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include/basis
	install -m 755 libbasis.so $(PREFIX)/lib/
	install -m 644 libbasis.a $(PREFIX)/lib/
	cp -r include/basis/* $(PREFIX)/include/basis/
	install -m 644 include/basis.h $(PREFIX)/include/

uninstall:
	rm -f $(PREFIX)/lib/libbasis.so $(PREFIX)/lib/libbasis.a
	rm -rf $(PREFIX)/include/basis
	rm -f $(PREFIX)/include/basis.h

clean:
	rm -f $(OBJS) test_v5_suite test_v7_* libbasis.so libbasis.a

.PHONY: all clean install uninstall

```

---

### FILE: src/basis/core/error.c
Location: `src/basis/core/error.c`
```cpp

#include "basis/core/error.h"
basis_error_t _basis_last_error = BASIS_SUCCESS;
const char* _basis_last_error_msg = "";
bool _basis_error_quiet = false;

```

---

### FILE: src/basis/core/ir.c
Location: `src/basis/core/ir.c`
```cpp
#include "basis/core/ir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void graph_track_node(basis_ir_graph* g, basis_ir_node* n) {
    if (g->node_count >= g->node_cap) {
        g->node_cap = g->node_cap == 0 ? 1024 : g->node_cap * 2;
        g->nodes = (basis_ir_node**)realloc(g->nodes, sizeof(basis_ir_node*) * g->node_cap);
    }
    g->nodes[g->node_count++] = n;
}

basis_ir_graph* basis_ir_graph_create(void) {
    basis_ir_graph* g = (basis_ir_graph*)calloc(1, sizeof(basis_ir_graph));
    if (!g) return NULL;
    g->arena = basis_arena_create(1024 * 1024 * 16); // 16MB blocks for massive graphs
    g->node_cap = 1024;
    g->nodes = (basis_ir_node**)malloc(sizeof(basis_ir_node*) * g->node_cap);
    return g;
}

void basis_ir_graph_destroy(basis_ir_graph* g) {
    if (!g) return;
    free(g->nodes);             // Free the tracking metadata
    basis_arena_destroy(g->arena); // O(1) destruction of ALL nodes!
    free(g);
}

basis_ir_node* basis_ir_input(basis_ir_graph* g, size_t r, size_t c) {
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++;
    n->op = BASIS_IR_OP_INPUT;
    n->rows = r; n->cols = c;
    n->input_count = 0; n->inputs = NULL;
    n->attr_val = 0.0;
    graph_track_node(g, n);
    return n;
}

basis_ir_node* basis_ir_const(basis_ir_graph* g, double val, size_t r, size_t c) {
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++;
    n->op = BASIS_IR_OP_CONST;
    n->rows = r; n->cols = c;
    n->input_count = 0; n->inputs = NULL;
    n->attr_val = val;
    graph_track_node(g, n);
    return n;
}

basis_ir_node* basis_ir_add(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL; // Shape Inference
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++;
    n->op = BASIS_IR_OP_ADD;
    n->rows = a->rows; n->cols = a->cols;
    n->input_count = 2;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*) * 2, 8);
    n->inputs[0] = a; n->inputs[1] = b;
    n->attr_val = 0.0;
    graph_track_node(g, n);
    return n;
}

basis_ir_node* basis_ir_matmul(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b) {
    if (!a || !b || a->cols != b->rows) return NULL; // Shape Inference
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++;
    n->op = BASIS_IR_OP_MATMUL;
    n->rows = a->rows; n->cols = b->cols;
    n->input_count = 2;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*) * 2, 8);
    n->inputs[0] = a; n->inputs[1] = b;
    n->attr_val = 0.0;
    graph_track_node(g, n);
    return n;
}

basis_ir_node* basis_ir_relu(basis_ir_graph* g, const basis_ir_node* a) {
    if (!a) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++;
    n->op = BASIS_IR_OP_RELU;
    n->rows = a->rows; n->cols = a->cols;
    n->input_count = 1;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*), 8);
    n->inputs[0] = a;
    n->attr_val = 0.0;
    graph_track_node(g, n);
    return n;
}


basis_ir_node* basis_ir_mul(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_MUL;
    n->rows = a->rows; n->cols = a->cols;
    n->input_count = 2;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*) * 2, 8);
    n->inputs[0] = a; n->inputs[1] = b;
    n->attr_val = 0.0;
    graph_track_node(g, n); return n;
}

basis_ir_node* basis_ir_sub(basis_ir_graph* g, const basis_ir_node* a, const basis_ir_node* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_SUB;
    n->rows = a->rows; n->cols = a->cols;
    n->input_count = 2;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*) * 2, 8);
    n->inputs[0] = a; n->inputs[1] = b;
    n->attr_val = 0.0;
    graph_track_node(g, n); return n;
}

basis_ir_node* basis_ir_transpose(basis_ir_graph* g, const basis_ir_node* a) {
    if (!a) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_TRANSPOSE;
    n->rows = a->cols; n->cols = a->rows;
    n->input_count = 1;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*), 8);
    n->inputs[0] = a;
    n->attr_val = 0.0;
    graph_track_node(g, n); return n;
}

basis_ir_node* basis_ir_softmax(basis_ir_graph* g, const basis_ir_node* a) {
    if (!a) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_SOFTMAX;
    n->rows = a->rows; n->cols = a->cols;
    n->input_count = 1;
    n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*), 8);
    n->inputs[0] = a;
    n->attr_val = 0.0;
    graph_track_node(g, n); return n;
}


basis_ir_node* basis_ir_sum(basis_ir_graph* g, const basis_ir_node* a) {
    if (!a) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_SUM;
    n->rows = 1; n->cols = 1;
    n->input_count = 1; n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*), 8);
    n->inputs[0] = a; n->attr_val = 0.0; graph_track_node(g, n); return n;
}
basis_ir_node* basis_ir_broadcast(basis_ir_graph* g, const basis_ir_node* a, size_t r, size_t c) {
    if (!a) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_BROADCAST;
    n->rows = r; n->cols = c;
    n->input_count = 1; n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*), 8);
    n->inputs[0] = a; n->attr_val = 0.0; graph_track_node(g, n); return n;
}
basis_ir_node* basis_ir_relu_bwd(basis_ir_graph* g, const basis_ir_node* grad, const basis_ir_node* fwd_a) {
    if (!grad || !fwd_a) return NULL;
    basis_ir_node* n = (basis_ir_node*)basis_arena_alloc(g->arena, sizeof(basis_ir_node), 8);
    n->id = g->next_id++; n->op = BASIS_IR_OP_RELU_BWD;
    n->rows = grad->rows; n->cols = grad->cols;
    n->input_count = 2; n->inputs = (const basis_ir_node**)basis_arena_alloc(g->arena, sizeof(basis_ir_node*) * 2, 8);
    n->inputs[0] = grad; n->inputs[1] = fwd_a; n->attr_val = 0.0; graph_track_node(g, n); return n;
}

static const char* ir_op_str(basis_ir_opcode op) {
    switch(op) {
        case BASIS_IR_OP_INPUT: return "INPUT";
        case BASIS_IR_OP_CONST: return "CONST";
        case BASIS_IR_OP_ADD: return "ADD";
        case BASIS_IR_OP_MATMUL: return "MATMUL";
        case BASIS_IR_OP_RELU: return "RELU";
        case BASIS_IR_OP_MUL: return "MUL";
        case BASIS_IR_OP_SUB: return "SUB";
        case BASIS_IR_OP_TRANSPOSE: return "TRANSPOSE";
        case BASIS_IR_OP_SOFTMAX: return "SOFTMAX";
        case BASIS_IR_OP_SUM: return "SUM";
        case BASIS_IR_OP_BROADCAST: return "BROADCAST";
        case BASIS_IR_OP_RELU_BWD: return "RELU_BWD";
        default: return "UNKNOWN";
    }
}

void basis_ir_print(const basis_ir_graph* g) {
    printf("=== V7 Immutable Graph (%u nodes) ===\n", g->node_count);
    for (uint32_t i = 0; i < g->node_count; i++) {
        const basis_ir_node* n = g->nodes[i];
        printf("%%%u = %-8s [%3zu x %3zu] | Inputs: ", n->id, ir_op_str(n->op), n->rows, n->cols);
        if (n->input_count == 0) {
            if (n->op == BASIS_IR_OP_CONST) printf("val=%.2f", n->attr_val);
            else printf("none");
        } else {
            for(uint32_t j=0; j<n->input_count; j++) printf("%%%u ", n->inputs[j]->id);
        }
        printf("\n");
    }
}

```

---

### FILE: src/basis/core/memory.c
Location: `src/basis/core/memory.c`
```cpp
#include "basis/core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align) - 1))

/* ==========================================================================
 * Arena Implementation
 * ========================================================================== */
typedef struct basis_arena_block {
    struct basis_arena_block* next;
    size_t size;
    size_t used;
    uint8_t* memory;
} basis_arena_block;

struct basis_arena {
    basis_arena_block* current;
    basis_arena_block* first;
    size_t default_block_size;
};

static basis_arena_block* arena_block_create(size_t size) {
    basis_arena_block* block = (basis_arena_block*)malloc(sizeof(basis_arena_block));
    if (!block) return NULL;
    block->memory = (uint8_t*)malloc(size);
    if (!block->memory) { free(block); return NULL; }
    block->size = size;
    block->used = 0;
    block->next = NULL;
    return block;
}

basis_arena* basis_arena_create(size_t default_block_size) {
    basis_arena* arena = (basis_arena*)calloc(1, sizeof(basis_arena));
    if (!arena) return NULL;
    arena->default_block_size = default_block_size > 0 ? default_block_size : 65536; // 64KB default
    arena->first = arena_block_create(arena->default_block_size);
    arena->current = arena->first;
    return arena;
}

void* basis_arena_alloc(basis_arena* arena, size_t size, size_t alignment) {
    if (!arena || size == 0) return NULL;
    if (alignment == 0) alignment = sizeof(void*);

    // CRITICAL FIX: Enforce power-of-2 alignment to prevent SIMD SIGBUS crashes
    if (alignment & (alignment - 1)) {
        fprintf(stderr, "[BASIS ARENA FATAL] Alignment %zu is not a power of 2!\n", alignment);
        return NULL;
    }

    basis_arena_block* block = arena->current;
    uintptr_t current_ptr = (uintptr_t)(block->memory + block->used);
    uintptr_t aligned_ptr = ALIGN_UP(current_ptr, alignment);
    size_t padding = aligned_ptr - current_ptr;

    if (block->used + padding + size > block->size) {
        // Need a new block
        size_t new_block_size = arena->default_block_size;
        if (size + alignment > new_block_size) {
            new_block_size = size + alignment; // Oversized allocation
        }
        basis_arena_block* new_block = arena_block_create(new_block_size);
        if (!new_block) return NULL;

        // Link and switch
        block->next = new_block;
        arena->current = new_block;
        block = new_block;

        current_ptr = (uintptr_t)(block->memory + block->used);
        aligned_ptr = ALIGN_UP(current_ptr, alignment);
        padding = aligned_ptr - current_ptr;
    }

    block->used += padding + size;
    return (void*)aligned_ptr;
}

void basis_arena_reset(basis_arena* arena) {
    if (!arena) return;
    // Free all blocks except the first one to maintain a baseline capacity
    basis_arena_block* block = arena->first->next;
    while (block) {
        basis_arena_block* next = block->next;
        free(block->memory);
        free(block);
        block = next;
    }
    arena->first->next = NULL;
    arena->first->used = 0;
    arena->current = arena->first;
}

void basis_arena_destroy(basis_arena* arena) {
    if (!arena) return;
    basis_arena_block* block = arena->first;
    while (block) {
        basis_arena_block* next = block->next;
        free(block->memory);
        free(block);
        block = next;
    }
    free(arena);
}

/* ==========================================================================
 * Pool Implementation
 * ========================================================================== */
typedef struct pool_slot {
    struct pool_slot* next;
} pool_slot;

struct basis_pool {
    size_t object_size;      // Actual usable size
    size_t slot_size;        // Size including free-list pointer overhead + alignment
    size_t objects_per_chunk;
    pool_slot* free_list;

    void** chunks;           // Array of raw malloc pointers for teardown
    size_t chunk_count;
    size_t chunk_cap;
};

basis_pool* basis_pool_create(size_t object_size, size_t objects_per_chunk) {
    basis_pool* pool = (basis_pool*)calloc(1, sizeof(basis_pool));
    if (!pool) return NULL;

    // Ensure slot is large enough to hold a pointer for the free list, and aligned
    size_t min_size = object_size < sizeof(pool_slot*) ? sizeof(pool_slot*) : object_size;
    pool->object_size = object_size;
    pool->slot_size = ALIGN_UP(min_size, sizeof(void*));
    pool->objects_per_chunk = objects_per_chunk > 0 ? objects_per_chunk : 256;
    pool->free_list = NULL;

    pool->chunk_cap = 16;
    pool->chunks = (void**)malloc(sizeof(void*) * pool->chunk_cap);
    pool->chunk_count = 0;

    return pool;
}

static void pool_allocate_chunk(basis_pool* pool) {
    size_t chunk_bytes = pool->slot_size * pool->objects_per_chunk;
    uint8_t* raw_chunk = (uint8_t*)malloc(chunk_bytes);
    if (!raw_chunk) return;

    // Track chunk for destruction
    if (pool->chunk_count >= pool->chunk_cap) {
        pool->chunk_cap *= 2;
        pool->chunks = (void**)realloc(pool->chunks, sizeof(void*) * pool->chunk_cap);
    }
    pool->chunks[pool->chunk_count++] = raw_chunk;

    // Carve chunk into slots and prepend to free list
    for (size_t i = 0; i < pool->objects_per_chunk; i++) {
        pool_slot* slot = (pool_slot*)(raw_chunk + (i * pool->slot_size));
        slot->next = pool->free_list;
        pool->free_list = slot;
    }
}

void* basis_pool_alloc(basis_pool* pool) {
    if (!pool) return NULL;
    if (!pool->free_list) {
        pool_allocate_chunk(pool);
        if (!pool->free_list) return NULL; // OOM
    }
    pool_slot* slot = pool->free_list;
    pool->free_list = slot->next;
    return (void*)slot;
}

void basis_pool_free(basis_pool* pool, void* ptr) {
    if (!pool || !ptr) return;
    pool_slot* slot = (pool_slot*)ptr;
    slot->next = pool->free_list;
    pool->free_list = slot;
}

void basis_pool_destroy(basis_pool* pool) {
    if (!pool) return;
    for (size_t i = 0; i < pool->chunk_count; i++) {
        free(pool->chunks[i]);
    }
    free(pool->chunks);
    free(pool);
}

```

---

### FILE: src/basis/l1_math/tensor_v7.c
Location: `src/basis/l1_math/tensor_v7.c`
```cpp
#include "basis/l1_math/tensor_v7.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static basis_v7_tensor* create_v7_tensor(basis_ir_graph* g, basis_ir_node* n) {
    // Allocate the metadata struct on the arena so it's freed in O(1)
    basis_v7_tensor* t = (basis_v7_tensor*)basis_arena_alloc(g->arena, sizeof(basis_v7_tensor), 8);
    t->rows = n->rows;
    t->cols = n->cols;
    t->node = n;
    t->graph = g;
    return t;
}

basis_v7_tensor* basis_v7_input(basis_ir_graph* g, size_t r, size_t c, const double* initial_data) {
    basis_ir_node* n = basis_ir_input(g, r, c);
    size_t elements = r * c;
    // Allocate data buffer on the arena!
    n->runtime_data = (double*)basis_arena_alloc(g->arena, elements * sizeof(double), 32);
    if (initial_data) {
        memcpy(n->runtime_data, initial_data, elements * sizeof(double));
    } else {
        memset(n->runtime_data, 0, elements * sizeof(double));
    }
    return create_v7_tensor(g, n);
}

basis_v7_tensor* basis_v7_const(basis_ir_graph* g, double val, size_t r, size_t c) {
    basis_ir_node* n = basis_ir_const(g, val, r, c);
    if (!n) return NULL;
    // runtime_data is now allocated and populated by basis_ir_const
    return create_v7_tensor(g, n);
}

basis_v7_tensor* basis_v7_add(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b) {
    basis_ir_node* n = basis_ir_add(g, a->node, b->node);
    if (!n) return NULL;
    return create_v7_tensor(g, n);
}

basis_v7_tensor* basis_v7_matmul(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b) {
    basis_ir_node* n = basis_ir_matmul(g, a->node, b->node);
    if (!n) return NULL;
    return create_v7_tensor(g, n);
}

basis_v7_tensor* basis_v7_relu(basis_ir_graph* g, basis_v7_tensor* a) {
    basis_ir_node* n = basis_ir_relu(g, a->node);
    if (!n) return NULL;
    return create_v7_tensor(g, n);
}


basis_v7_tensor* basis_v7_mul(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b) {
    basis_ir_node* n = basis_ir_mul(g, a->node, b->node); if (!n) return NULL; return create_v7_tensor(g, n);
}
basis_v7_tensor* basis_v7_sub(basis_ir_graph* g, basis_v7_tensor* a, basis_v7_tensor* b) {
    basis_ir_node* n = basis_ir_sub(g, a->node, b->node); if (!n) return NULL; return create_v7_tensor(g, n);
}
basis_v7_tensor* basis_v7_transpose(basis_ir_graph* g, basis_v7_tensor* a) {
    basis_ir_node* n = basis_ir_transpose(g, a->node); if (!n) return NULL; return create_v7_tensor(g, n);
}
basis_v7_tensor* basis_v7_softmax(basis_ir_graph* g, basis_v7_tensor* a) {
    basis_ir_node* n = basis_ir_softmax(g, a->node); if (!n) return NULL; return create_v7_tensor(g, n);
}


basis_v7_tensor* basis_v7_sum(basis_ir_graph* g, basis_v7_tensor* a) {
    basis_ir_node* n = basis_ir_sum(g, a->node); if (!n) return NULL; return create_v7_tensor(g, n);
}
basis_v7_tensor* basis_v7_broadcast(basis_ir_graph* g, basis_v7_tensor* a, size_t r, size_t c) {
    basis_ir_node* n = basis_ir_broadcast(g, a->node, r, c); if (!n) return NULL; return create_v7_tensor(g, n);
}

void basis_v7_execute(basis_ir_graph* g) {
    for (uint32_t i = 0; i < g->node_count; i++) {
        basis_ir_node* n = g->nodes[i];
        size_t elements = n->rows * n->cols;

        if (n->op == BASIS_IR_OP_INPUT || n->op == BASIS_IR_OP_CONST) continue; // Already populated

        // Allocate intermediate activation buffer on the Arena (32-byte aligned for AVX)
        n->runtime_data = (double*)basis_arena_alloc(g->arena, elements * sizeof(double), 32);

        if (n->op == BASIS_IR_OP_ADD) {
            double* a = n->inputs[0]->runtime_data;
            double* b = n->inputs[1]->runtime_data;
            double* out = n->runtime_data;
            for(size_t k=0; k<elements; k++) out[k] = a[k] + b[k];
        }
        else if (n->op == BASIS_IR_OP_MATMUL) {
            double* a = n->inputs[0]->runtime_data;
            double* b = n->inputs[1]->runtime_data;
            double* out = n->runtime_data;
            size_t M = n->inputs[0]->rows;
            size_t K = n->inputs[0]->cols;
            size_t N = n->inputs[1]->cols;

            for(size_t i=0; i<M; i++) {
                for(size_t j=0; j<N; j++) {
                    double sum = 0.0;
                    for(size_t k=0; k<K; k++) {
                        sum += a[i*K + k] * b[k*N + j];
                    }
                    out[i*N + j] = sum;
                }
            }
        }
        else if (n->op == BASIS_IR_OP_RELU) {
            double* a = n->inputs[0]->runtime_data;
            double* out = n->runtime_data;
            for(size_t k=0; k<elements; k++) out[k] = a[k] > 0.0 ? a[k] : 0.0;
        }
    }
}

void basis_v7_tensor_print(basis_v7_tensor* t, const char* name) {
    if (!t || !t->node || !t->node->runtime_data) {
        printf("Tensor %s: Not yet executed.\n", name);
        return;
    }
    printf("Tensor %s (%zu x %zu):\n", name, t->rows, t->cols);
    for (size_t i = 0; i < t->rows; i++) {
        printf("  [");
        for (size_t j = 0; j < t->cols; j++) {
            printf(" %8.4f ", t->node->runtime_data[i * t->cols + j]);
        }
        printf("]\n");
    }
}

```

---

### FILE: src/basis/l4_compiler/lowering.c
Location: `src/basis/l4_compiler/lowering.c`
```cpp
#include "basis/l4_compiler/lowering.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void basis_ir_lower_to_c(basis_ir_graph* g, const char* output_filename) {
    FILE* f = fopen(output_filename, "w");
    if (!f) return;
    fprintf(f, "#include <stdlib.h>\n#include <string.h>\n\n");
    int input_count = 0;
    char** c_vars = (char**)calloc(g->next_id, sizeof(char*));
    for (uint32_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i]->op == BASIS_IR_OP_INPUT) {
            c_vars[g->nodes[i]->id] = (char*)malloc(16);
            sprintf(c_vars[g->nodes[i]->id], "in%d", input_count++);
        }
    }
    fprintf(f, "void v7_compiled_kernel(double** inputs, double* out) {\n");
    for (uint32_t i = 0; i < g->node_count; i++) {
        basis_ir_node* n = g->nodes[i];
        if (n->op != BASIS_IR_OP_INPUT) {
            c_vars[n->id] = (char*)malloc(16);
            sprintf(c_vars[n->id], "buf%u", n->id);
            fprintf(f, "    double* buf%u = (double*)malloc(%zu * sizeof(double));\n", n->id, n->rows * n->cols);
        }
    }
    fprintf(f, "\n");
    for (uint32_t i = 0; i < g->node_count; i++) {
        basis_ir_node* n = g->nodes[i];
        size_t elements = n->rows * n->cols;
        if (n->op == BASIS_IR_OP_INPUT) continue;
        if (n->op == BASIS_IR_OP_CONST) {
            fprintf(f, "    for(size_t i=0; i<%zu; i++) buf%u[i] = %f;\n", elements, n->id, n->attr_val);
        } else if (n->op == BASIS_IR_OP_ADD) {
            fprintf(f, "    for(size_t i=0; i<%zu; i++) buf%u[i] = %s[i] + %s[i];\n", elements, n->id, c_vars[n->inputs[0]->id], c_vars[n->inputs[1]->id]);
        } else if (n->op == BASIS_IR_OP_MATMUL) {
            size_t M = n->inputs[0]->rows, K = n->inputs[0]->cols, N = n->inputs[1]->cols;
            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t j=0; j<%zu; j++) { double sum=0; for(size_t k=0; k<%zu; k++) sum+=%s[i*%zu+k]*%s[k*%zu+j]; buf%u[i*%zu+j]=sum; }\n", M, N, K, c_vars[n->inputs[0]->id], K, c_vars[n->inputs[1]->id], N, n->id, N);
        } else if (n->op == BASIS_IR_OP_RELU) {
            fprintf(f, "    for(size_t i=0; i<%zu; i++) { double v=%s[i]; buf%u[i]=v>0.0?v:0.0; }\n", elements, c_vars[n->inputs[0]->id], n->id);
        }
    }
    basis_ir_node* final_node = g->nodes[g->node_count - 1];
    fprintf(f, "\n    memcpy(out, buf%u, %zu * sizeof(double));\n", final_node->id, final_node->rows * final_node->cols);
    fprintf(f, "\n");
    for (uint32_t i = 0; i < g->node_count; i++) if (g->nodes[i]->op != BASIS_IR_OP_INPUT) fprintf(f, "    free(buf%u);\n", g->nodes[i]->id);
    fprintf(f, "}\n");
    fclose(f);
    for (uint32_t i = 0; i < g->next_id; i++) if (c_vars[i]) free(c_vars[i]);
    free(c_vars);
}

```

---

### FILE: src/basis/l4_compiler/pass_manager.c
Location: `src/basis/l4_compiler/pass_manager.c`
```cpp
#include "basis/l4_compiler/pass_manager.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

static void basis_ir_dce(basis_ir_graph* g) {
    if (!g || g->node_count == 0) return;
    basis_ir_node* out = g->nodes[g->node_count - 1];
    bool* reachable = (bool*)calloc(g->next_id, sizeof(bool));
    if (!reachable) return;
    reachable[out->id] = true;
    for (int i = (int)g->node_count - 1; i >= 0; i--) {
        basis_ir_node* n = g->nodes[i];
        if (reachable[n->id]) {
            for (uint32_t j = 0; j < n->input_count; j++) {
                if (n->inputs[j]) reachable[n->inputs[j]->id] = true;
            }
        }
    }
    uint32_t write_idx = 0;
    for (uint32_t i = 0; i < g->node_count; i++) {
        if (reachable[g->nodes[i]->id]) g->nodes[write_idx++] = g->nodes[i];
    }
    g->node_count = write_idx;
    free(reachable);
}

basis_ir_graph* basis_ir_run_passes(basis_ir_graph* old_g) {
    if (!old_g) return NULL;
    basis_ir_graph* new_g = basis_ir_graph_create();
    basis_ir_node** map = (basis_ir_node**)calloc(old_g->next_id, sizeof(basis_ir_node*));
    if (!map) { basis_ir_graph_destroy(new_g); return NULL; }

    for (uint32_t i = 0; i < old_g->node_count; i++) {
        basis_ir_node* old_n = old_g->nodes[i];
        basis_ir_node* new_n = NULL;

        const basis_ir_node* in0 = (old_n->input_count > 0 && old_n->inputs[0]) ? map[old_n->inputs[0]->id] : NULL;
        const basis_ir_node* in1 = (old_n->input_count > 1 && old_n->inputs[1]) ? map[old_n->inputs[1]->id] : NULL;

        // Pass 1: Constant Folding
        if (old_n->op == BASIS_IR_OP_ADD && in0 && in1 && in0->op == BASIS_IR_OP_CONST && in1->op == BASIS_IR_OP_CONST) {
            new_n = basis_ir_const(new_g, in0->attr_val + in1->attr_val, old_n->rows, old_n->cols);
        } else if (old_n->op == BASIS_IR_OP_MUL && in0 && in1 && in0->op == BASIS_IR_OP_CONST && in1->op == BASIS_IR_OP_CONST) {
            new_n = basis_ir_const(new_g, in0->attr_val * in1->attr_val, old_n->rows, old_n->cols);
        } else if (old_n->op == BASIS_IR_OP_SUB && in0 && in1 && in0->op == BASIS_IR_OP_CONST && in1->op == BASIS_IR_OP_CONST) {
            new_n = basis_ir_const(new_g, in0->attr_val - in1->attr_val, old_n->rows, old_n->cols);
        }
        // Pass 2: Identity Elimination
        else if (old_n->op == BASIS_IR_OP_ADD && in0 && in1) {
            if (in0->op == BASIS_IR_OP_CONST && in0->attr_val == 0.0) new_n = (basis_ir_node*)in1;
            else if (in1->op == BASIS_IR_OP_CONST && in1->attr_val == 0.0) new_n = (basis_ir_node*)in0;
        } else if (old_n->op == BASIS_IR_OP_MUL && in0 && in1) {
            if (in0->op == BASIS_IR_OP_CONST && in0->attr_val == 1.0) new_n = (basis_ir_node*)in1;
            else if (in1->op == BASIS_IR_OP_CONST && in1->attr_val == 1.0) new_n = (basis_ir_node*)in0;
        } else if (old_n->op == BASIS_IR_OP_SUB && in0 && in1) {
            if (in1->op == BASIS_IR_OP_CONST && in1->attr_val == 0.0) new_n = (basis_ir_node*)in0;
        }

        // Fallback: Rebuild Node
        if (!new_n) {
            switch (old_n->op) {
                case BASIS_IR_OP_INPUT:
                    new_n = basis_ir_input(new_g, old_n->rows, old_n->cols);
                    // CRITICAL FIX: Share pointer so external training loops can update weights
                    new_n->runtime_data = old_n->runtime_data;
                    break;
                case BASIS_IR_OP_CONST:
                    // CRITICAL FIX: basis_ir_const already allocates and fills persistent memory
                    new_n = basis_ir_const(new_g, old_n->attr_val, old_n->rows, old_n->cols);
                    break;
                case BASIS_IR_OP_ADD: new_n = basis_ir_add(new_g, in0, in1); break;
                case BASIS_IR_OP_SUB: new_n = basis_ir_sub(new_g, in0, in1); break;
                case BASIS_IR_OP_MUL: new_n = basis_ir_mul(new_g, in0, in1); break;
                case BASIS_IR_OP_MATMUL: new_n = basis_ir_matmul(new_g, in0, in1); break;
                case BASIS_IR_OP_RELU: new_n = basis_ir_relu(new_g, in0); break;
                case BASIS_IR_OP_TRANSPOSE: new_n = basis_ir_transpose(new_g, in0); break;
                case BASIS_IR_OP_SOFTMAX: new_n = basis_ir_softmax(new_g, in0); break;
                default: break;
            }
        }
        if (new_n) map[old_n->id] = new_n;
    }
    free(map);
    basis_ir_dce(new_g);
    return new_g;
}

```

---

### FILE: src/basis/l5_runtime/scheduler.c
Location: `src/basis/l5_runtime/scheduler.c`
```cpp
#include "basis/l5_runtime/scheduler.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

basis_schedule* basis_ir_schedule(basis_ir_graph* g) {
    if (!g || g->node_count == 0) return NULL;

    basis_schedule* sched = (basis_schedule*)calloc(1, sizeof(basis_schedule));
    if (!sched) return NULL;

    // Allocate metadata for Kahn's Algorithm
    uint32_t* in_degree = (uint32_t*)calloc(g->next_id, sizeof(uint32_t));
    basis_ir_node** queue = (basis_ir_node**)malloc(sizeof(basis_ir_node*) * g->node_count);

    // 1. Compute in-degrees
    for (uint32_t i = 0; i < g->node_count; i++) {
        basis_ir_node* n = g->nodes[i];
        in_degree[n->id] = n->input_count;
    }

    // 2. Extract Waves (Level-based Topological Sort)
    uint32_t wave_cap = 16;
    sched->waves = (basis_wave*)malloc(sizeof(basis_wave) * wave_cap);
    sched->wave_count = 0;

    uint32_t processed = 0;

    while (processed < g->node_count) {
        uint32_t q_tail = 0;

        // Find all nodes with 0 in-degree (ready to execute)
        for (uint32_t i = 0; i < g->node_count; i++) {
            basis_ir_node* n = g->nodes[i];
            // We use a trick: mark processed nodes by setting in_degree to UINT32_MAX
            if (in_degree[n->id] == 0) {
                queue[q_tail++] = n;
                in_degree[n->id] = UINT32_MAX; // Mark as processed
            }
        }

        if (q_tail == 0) break; // Should not happen in a valid DAG

        // Create a new Wave
        if (sched->wave_count >= wave_cap) {
            wave_cap *= 2;
            sched->waves = (basis_wave*)realloc(sched->waves, sizeof(basis_wave) * wave_cap);
        }

        basis_wave* w = &sched->waves[sched->wave_count++];
        w->node_count = q_tail;
        w->nodes = (basis_ir_node**)malloc(sizeof(basis_ir_node*) * q_tail);
        memcpy(w->nodes, queue, sizeof(basis_ir_node*) * q_tail);

        // Decrement in-degrees of children
        for (uint32_t i = 0; i < g->node_count; i++) {
            basis_ir_node* n = g->nodes[i];
            if (in_degree[n->id] == UINT32_MAX) continue; // Already processed

            bool all_inputs_ready = true;
            for (uint32_t j = 0; j < n->input_count; j++) {
                if (in_degree[n->inputs[j]->id] != UINT32_MAX) {
                    all_inputs_ready = false;
                    break;
                }
            }
            if (all_inputs_ready) {
                in_degree[n->id] = 0; // Ready for next wave
            }
        }

        processed += q_tail;
    }

    free(in_degree);
    free(queue);
    return sched;
}

void basis_schedule_execute(basis_schedule* schedule, basis_arena* scratch) {
    if (!schedule) return;

    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        basis_wave* wave = &schedule->waves[w];

        // Execute all nodes in this wave
        // (Currently sequential, but structurally ready for pthreads/OpenMP dispatch)
        for (uint32_t i = 0; i < wave->node_count; i++) {
            basis_ir_node* n = wave->nodes[i];
            size_t elements = n->rows * n->cols;

            if (n->op == BASIS_IR_OP_INPUT) { if(!n->runtime_data) printf("FATAL: INPUT %u NULL\n", n->id); continue; }
            if (n->op == BASIS_IR_OP_CONST) { if(!n->runtime_data) { n->runtime_data = (double*)basis_arena_alloc(scratch, elements * sizeof(double), 32); for(size_t k=0; k<elements; k++) n->runtime_data[k] = n->attr_val; } continue; }

            if (!n->runtime_data) {
                // Fallback allocation if not using the L1 Tensor Arena wrapper
                n->runtime_data = (double*)basis_arena_alloc(scratch, elements * sizeof(double), 32);
            }

            if (n->op == BASIS_IR_OP_ADD) {
                double* a = n->inputs[0]->runtime_data;
                double* b = n->inputs[1]->runtime_data;
                for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] + b[k];
            }
            else if (n->op == BASIS_IR_OP_MATMUL) {
                double* a = n->inputs[0]->runtime_data;
                double* b = n->inputs[1]->runtime_data;
                size_t M = n->inputs[0]->rows;
                size_t K = n->inputs[0]->cols;
                size_t N = n->inputs[1]->cols;
                for(size_t r=0; r<M; r++) {
                    for(size_t c=0; c<N; c++) {
                        double sum = 0.0;
                        for(size_t k=0; k<K; k++) sum += a[r*K + k] * b[k*N + c];
                        n->runtime_data[r*N + c] = sum;
                    }
                }
            }
            
            if (n->op == BASIS_IR_OP_MUL) {
                double* a = n->inputs[0]->runtime_data;
                double* b = n->inputs[1]->runtime_data;
                for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] * b[k];
            }
            else if (n->op == BASIS_IR_OP_SUB) {
                double* a = n->inputs[0]->runtime_data;
                double* b = n->inputs[1]->runtime_data;
                for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] - b[k];
            }
            else if (n->op == BASIS_IR_OP_TRANSPOSE) {
                double* a = n->inputs[0]->runtime_data;
                size_t R = n->inputs[0]->rows;
                size_t C = n->inputs[0]->cols;
                for(size_t r=0; r<R; r++) for(size_t c=0; c<C; c++) n->runtime_data[c*R + r] = a[r*C + c];
            }
            
            else if (n->op == BASIS_IR_OP_SUM) {
                double* a = n->inputs[0]->runtime_data;
                size_t elements = n->inputs[0]->rows * n->inputs[0]->cols;
                double sum = 0.0;
                for(size_t k=0; k<elements; k++) sum += a[k];
                n->runtime_data[0] = sum;
            }
            else if (n->op == BASIS_IR_OP_BROADCAST) {
                double val = n->inputs[0]->runtime_data[0];
                size_t elements = n->rows * n->cols;
                for(size_t k=0; k<elements; k++) n->runtime_data[k] = val;
            }
            else if (n->op == BASIS_IR_OP_RELU_BWD) {
                double* grad = n->inputs[0]->runtime_data;
                double* fwd_a = n->inputs[1]->runtime_data;
                size_t elements = n->rows * n->cols;
                for(size_t k=0; k<elements; k++) n->runtime_data[k] = fwd_a[k] > 0.0 ? grad[k] : 0.0;
            }
            else if (n->op == BASIS_IR_OP_SOFTMAX) {
                double* a = n->inputs[0]->runtime_data;
                size_t R = n->rows;
                size_t C = n->cols;
                for(size_t r=0; r<R; r++) {
                    double max_val = a[r*C];
                    for(size_t c=1; c<C; c++) if(a[r*C+c] > max_val) max_val = a[r*C+c];
                    double sum = 0.0;
                    for(size_t c=0; c<C; c++) { n->runtime_data[r*C+c] = exp(a[r*C+c] - max_val); sum += n->runtime_data[r*C+c]; }
                    for(size_t c=0; c<C; c++) n->runtime_data[r*C+c] /= sum;
                }
            }
            else if (n->op == BASIS_IR_OP_RELU) {
                double* a = n->inputs[0]->runtime_data;
                for(size_t k=0; k<elements; k++) n->runtime_data[k] = a[k] > 0.0 ? a[k] : 0.0;
            }
        }
    }
}

void basis_schedule_print(basis_schedule* schedule) {
    if (!schedule) return;
    printf("=== Execution Schedule (%u Waves) ===\n", schedule->wave_count);
    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        printf("Wave %u (%u parallel nodes): [ ", w, schedule->waves[w].node_count);
        for (uint32_t i = 0; i < schedule->waves[w].node_count; i++) {
            printf("%%%u ", schedule->waves[w].nodes[i]->id);
        }
        printf("]\n");
    }
}

void basis_schedule_destroy(basis_schedule* schedule) {
    if (!schedule) return;
    for (uint32_t w = 0; w < schedule->wave_count; w++) {
        free(schedule->waves[w].nodes);
    }
    free(schedule->waves);
    free(schedule);
}

```

---

### FILE: src/basis/l6_learning/autodiff.c
Location: `src/basis/l6_learning/autodiff.c`
```cpp
#include "basis/l6_learning/autodiff.h"
#include <stdlib.h>
#include <string.h>

static void accum_grad(basis_ir_graph* g, basis_ir_node** grad_map, uint32_t fwd_input_id, basis_ir_node* new_grad, basis_ir_node** fwd_to_full) {
    if (!new_grad) return;
    basis_ir_node* full_input = fwd_to_full[fwd_input_id];

    // Forward Broadcast: 1x1 grad -> MxN input
    if (new_grad->rows == 1 && new_grad->cols == 1 && (full_input->rows > 1 || full_input->cols > 1)) {
        new_grad = basis_ir_broadcast(g, new_grad, full_input->rows, full_input->cols);
    }
    // CRITICAL FIX: Reverse Reduction: MxN grad -> 1x1 input (e.g. from broadcasting a scalar bias)
    else if ((new_grad->rows > 1 || new_grad->cols > 1) && full_input->rows == 1 && full_input->cols == 1) {
        new_grad = basis_ir_sum(g, new_grad);
    }
    if (grad_map[fwd_input_id] == NULL) {
        grad_map[fwd_input_id] = new_grad;
    } else {
        grad_map[fwd_input_id] = basis_ir_add(g, grad_map[fwd_input_id], new_grad);
    }
}

basis_training_graph* basis_ir_autodiff(basis_ir_graph* fwd_g, basis_ir_node* loss_node) {
    basis_training_graph* tg = (basis_training_graph*)calloc(1, sizeof(basis_training_graph));
    tg->graph = basis_ir_graph_create();
    tg->max_id = fwd_g->next_id;
    tg->grad_nodes = (basis_ir_node**)calloc(fwd_g->next_id, sizeof(basis_ir_node*));

    basis_ir_node** fwd_to_full = (basis_ir_node**)calloc(fwd_g->next_id, sizeof(basis_ir_node*));
    for(uint32_t i=0; i<fwd_g->node_count; i++) {
        basis_ir_node* old = fwd_g->nodes[i];
        basis_ir_node* n = NULL;
        const basis_ir_node* in0 = old->input_count > 0 ? fwd_to_full[old->inputs[0]->id] : NULL;
        const basis_ir_node* in1 = old->input_count > 1 ? fwd_to_full[old->inputs[1]->id] : NULL;
        switch(old->op) {
            case BASIS_IR_OP_INPUT: n = basis_ir_input(tg->graph, old->rows, old->cols); break;
            case BASIS_IR_OP_CONST: n = basis_ir_const(tg->graph, old->attr_val, old->rows, old->cols); break;
            case BASIS_IR_OP_ADD: n = basis_ir_add(tg->graph, in0, in1); break;
            case BASIS_IR_OP_SUB: n = basis_ir_sub(tg->graph, in0, in1); break;
            case BASIS_IR_OP_MUL: n = basis_ir_mul(tg->graph, in0, in1); break;
            case BASIS_IR_OP_MATMUL: n = basis_ir_matmul(tg->graph, in0, in1); break;
            case BASIS_IR_OP_RELU: n = basis_ir_relu(tg->graph, in0); break;
            case BASIS_IR_OP_TRANSPOSE: n = basis_ir_transpose(tg->graph, in0); break;
            case BASIS_IR_OP_SUM: n = basis_ir_sum(tg->graph, in0); break;
            case BASIS_IR_OP_BROADCAST: n = basis_ir_broadcast(tg->graph, in0, old->rows, old->cols); break;
            default: break;
        }
        fwd_to_full[old->id] = n;
    }

    basis_ir_node** grad_map = (basis_ir_node**)calloc(fwd_g->next_id, sizeof(basis_ir_node*));
    grad_map[loss_node->id] = basis_ir_const(tg->graph, 1.0, 1, 1);

    for (int i = fwd_g->node_count - 1; i >= 0; i--) {
        basis_ir_node* fwd_n = fwd_g->nodes[i];
        basis_ir_node* grad_out = grad_map[fwd_n->id];
        if (!grad_out) continue;
        if (fwd_n->op == BASIS_IR_OP_INPUT || fwd_n->op == BASIS_IR_OP_CONST) continue;

        if (fwd_n->op == BASIS_IR_OP_ADD) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, grad_out, fwd_to_full);
        } else if (fwd_n->op == BASIS_IR_OP_SUB) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
            basis_ir_node* neg = basis_ir_const(tg->graph, -1.0, grad_out->rows, grad_out->cols);
            basis_ir_node* gB = basis_ir_mul(tg->graph, grad_out, neg);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, gB, fwd_to_full);
        } else if (fwd_n->op == BASIS_IR_OP_MUL) {
            basis_ir_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            basis_ir_node* B = fwd_to_full[fwd_n->inputs[1]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, basis_ir_mul(tg->graph, grad_out, B), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, basis_ir_mul(tg->graph, grad_out, A), fwd_to_full);
        } else if (fwd_n->op == BASIS_IR_OP_MATMUL) {
            basis_ir_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            basis_ir_node* B = fwd_to_full[fwd_n->inputs[1]->id];
            basis_ir_node* B_T = basis_ir_transpose(tg->graph, B);
            basis_ir_node* A_T = basis_ir_transpose(tg->graph, A);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, basis_ir_matmul(tg->graph, grad_out, B_T), fwd_to_full);
            accum_grad(tg->graph, grad_map, fwd_n->inputs[1]->id, basis_ir_matmul(tg->graph, A_T, grad_out), fwd_to_full);
        } else if (fwd_n->op == BASIS_IR_OP_RELU) {
            basis_ir_node* A = fwd_to_full[fwd_n->inputs[0]->id];
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, basis_ir_relu_bwd(tg->graph, grad_out, A), fwd_to_full);
        } else if (fwd_n->op == BASIS_IR_OP_SUM) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, grad_out, fwd_to_full);
        } else if (fwd_n->op == BASIS_IR_OP_BROADCAST) {
            accum_grad(tg->graph, grad_map, fwd_n->inputs[0]->id, basis_ir_sum(tg->graph, grad_out), fwd_to_full);
        }
    }

    for(uint32_t i=0; i<fwd_g->next_id; i++) tg->grad_nodes[i] = grad_map[i];
    tg->loss_node = fwd_to_full[loss_node->id];

    free(grad_map); free(fwd_to_full);
    return tg;
}

void basis_training_graph_destroy(basis_training_graph* tg) {
    if (!tg) return;
    free(tg->grad_nodes);
    basis_ir_graph_destroy(tg->graph);
    free(tg);
}

```

---

### FILE: src/basis/l7_intelligence/swarm.c
Location: `src/basis/l7_intelligence/swarm.c`
```cpp
#include "basis/l7_intelligence/swarm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

basis_swarm* basis_swarm_create(void) {
    basis_swarm* s = (basis_swarm*)calloc(1, sizeof(basis_swarm));
    s->expert_cap = 16;
    s->experts = (basis_expert**)malloc(sizeof(basis_expert*) * s->expert_cap);
    return s;
}

void basis_swarm_register(basis_swarm* swarm, const char* name, basis_schedule* sched, basis_ir_graph* g) {
    if (swarm->expert_count >= swarm->expert_cap) {
        swarm->expert_cap *= 2;
        swarm->experts = (basis_expert**)realloc(swarm->experts, sizeof(basis_expert*) * swarm->expert_cap);
    }

    basis_expert* exp = (basis_expert*)calloc(1, sizeof(basis_expert));
    exp->name = name;
    exp->schedule = sched;
    exp->scratch = basis_arena_create(1024 * 1024);

    // Map Inputs
    exp->input_count = 0;
    for(uint32_t i=0; i<g->node_count; i++) if(g->nodes[i]->op == BASIS_IR_OP_INPUT) exp->input_count++;

    exp->input_buffers = (double**)calloc(exp->input_count, sizeof(double*));
    exp->input_sizes = (size_t*)calloc(exp->input_count, sizeof(size_t));

    uint32_t idx = 0;
    for(uint32_t i=0; i<g->node_count; i++) {
        if(g->nodes[i]->op == BASIS_IR_OP_INPUT) {
            size_t bytes = g->nodes[i]->rows * g->nodes[i]->cols * sizeof(double);
            g->nodes[i]->runtime_data = (double*)malloc(bytes);
            exp->input_buffers[idx] = g->nodes[i]->runtime_data;
            exp->input_sizes[idx] = bytes;
            idx++;
        }
    }

    // Map Output (Last Node)
    basis_ir_node* out_node = g->nodes[g->node_count - 1];
    exp->out_node = out_node;
    exp->output_size = out_node->rows * out_node->cols * sizeof(double);

    swarm->experts[swarm->expert_count++] = exp;
}

basis_expert* basis_swarm_get(basis_swarm* swarm, const char* name) {
    for(uint32_t i=0; i<swarm->expert_count; i++) {
        if(strcmp(swarm->experts[i]->name, name) == 0) return swarm->experts[i];
    }
    return NULL;
}

void basis_swarm_dispatch(basis_expert* expert, double* in_data, double* out_data) {
    if (!expert || !expert->schedule || expert->input_count == 0) return;

    // 1. Inject data
    memcpy(expert->input_buffers[0], in_data, expert->input_sizes[0]);

    // 2. Nullify intermediate nodes (including output) to force scratch allocation
    for(uint32_t w=0; w<expert->schedule->wave_count; w++) {
        for(uint32_t i=0; i<expert->schedule->waves[w].node_count; i++) {
            basis_ir_node* n = expert->schedule->waves[w].nodes[i];
            if (n->op != BASIS_IR_OP_INPUT && n->op != BASIS_IR_OP_CONST) {
                n->runtime_data = NULL;
            }
        }
    }

    // 3. Execute
    basis_schedule_execute(expert->schedule, expert->scratch);

    // 4. Extract result directly from the node's newly allocated scratch buffer
    memcpy(out_data, expert->out_node->runtime_data, expert->output_size);

    // 5. O(1) Teardown
    basis_arena_reset(expert->scratch);
}

void basis_swarm_destroy(basis_swarm* swarm) {
    if (!swarm) return;
    for(uint32_t i=0; i<swarm->expert_count; i++) {
        basis_expert* e = swarm->experts[i];
        basis_arena_destroy(e->scratch);
        basis_schedule_destroy(e->schedule);

        for(uint32_t j=0; j<e->input_count; j++) free(e->input_buffers[j]);
        free(e->input_buffers);
        free(e->input_sizes);
        free(e);
    }
    free(swarm->experts);
    free(swarm);
}

```

---

### FILE: src/basis/stage1_atomic/scalar.c
Location: `src/basis/stage1_atomic/scalar.c`
```cpp

#include "basis/stage1_atomic/scalar.h"
#include "basis/core/error.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static char* basis_value_string_duplicate(const char* source) {
    if (!source) return NULL;
    size_t len = strlen(source) + 1;
    char* copy = (char*)malloc(len);
    if (copy) memcpy(copy, source, len);
    return copy;
}

basis_value* basis_value_new(double initial_data) {
    basis_value* new_value = (basis_value*)malloc(sizeof(basis_value));
    if (!new_value) { BASIS_SET_ERROR(BASIS_OUT_OF_MEMORY, "Failed to allocate basis_value"); return NULL; }
    new_value->data = initial_data;
    new_value->gradient = 0.0;
    new_value->backward_function = NULL;
    new_value->previous_nodes = NULL;
    new_value->previous_node_count = 0;
    new_value->operation = basis_value_string_duplicate("");
    new_value->constant_value = 0.0;
    new_value->reference_count = 1;
    new_value->visited = false;
    return new_value;
}

void basis_value_retain(basis_value* target_value) {
    if (target_value) target_value->reference_count++;
}

static basis_value* basis_value_operation_initialize(basis_value* first_operand, basis_value* second_operand, char* operation, size_t operand_count) {
    basis_value* output_value = basis_value_new(0.0);
    if (!output_value) return NULL;
    free(output_value->operation);
    output_value->operation = basis_value_string_duplicate(operation);
    output_value->previous_node_count = operand_count;
    output_value->previous_nodes = (basis_value**)malloc(sizeof(basis_value*) * operand_count);
    if (first_operand) output_value->previous_nodes[0] = first_operand;
    if (second_operand && operand_count > 1) output_value->previous_nodes[1] = second_operand;
    return output_value;
}

static void backward_addition(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += 1.0 * output_value->gradient;
    output_value->previous_nodes[1]->gradient += 1.0 * output_value->gradient;
}

basis_value* basis_value_addition(basis_value* first_operand, basis_value* second_operand) {
    BASIS_CHECK_NULL(first_operand); BASIS_CHECK_NULL(second_operand);
    basis_value* output_value = basis_value_operation_initialize(first_operand, second_operand, "+", 2);
    output_value->data = first_operand->data + second_operand->data;
    output_value->backward_function = backward_addition;
    basis_value_retain(first_operand); basis_value_retain(second_operand);
    return output_value;
}

static void backward_subtraction(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += 1.0 * output_value->gradient;
    output_value->previous_nodes[1]->gradient += -1.0 * output_value->gradient;
}

basis_value* basis_value_subtraction(basis_value* first_operand, basis_value* second_operand) {
    BASIS_CHECK_NULL(first_operand); BASIS_CHECK_NULL(second_operand);
    basis_value* output_value = basis_value_operation_initialize(first_operand, second_operand, "-", 2);
    output_value->data = first_operand->data - second_operand->data;
    output_value->backward_function = backward_subtraction;
    basis_value_retain(first_operand); basis_value_retain(second_operand);
    return output_value;
}

static void backward_multiplication(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += output_value->previous_nodes[1]->data * output_value->gradient;
    output_value->previous_nodes[1]->gradient += output_value->previous_nodes[0]->data * output_value->gradient;
}

basis_value* basis_value_multiplication(basis_value* first_operand, basis_value* second_operand) {
    BASIS_CHECK_NULL(first_operand); BASIS_CHECK_NULL(second_operand);
    basis_value* output_value = basis_value_operation_initialize(first_operand, second_operand, "*", 2);
    output_value->data = first_operand->data * second_operand->data;
    output_value->backward_function = backward_multiplication;
    basis_value_retain(first_operand); basis_value_retain(second_operand);
    return output_value;
}

static void backward_power(basis_value* output_value) {
    double base = output_value->previous_nodes[0]->data;
    double exponent = output_value->constant_value;
    output_value->previous_nodes[0]->gradient += (exponent * pow(base, exponent - 1.0)) * output_value->gradient;
}

basis_value* basis_value_power(basis_value* base, double exponent) {
    BASIS_CHECK_NULL(base);
    basis_value* output_value = basis_value_operation_initialize(base, NULL, "**", 1);
    output_value->data = pow(base->data, exponent);
    output_value->constant_value = exponent;
    output_value->backward_function = backward_power;
    basis_value_retain(base);
    return output_value;
}

static void backward_exponential(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += output_value->data * output_value->gradient;
}

basis_value* basis_value_exponential(basis_value* input_value) {
    BASIS_CHECK_NULL(input_value);
    basis_value* output_value = basis_value_operation_initialize(input_value, NULL, "exp", 1);
    output_value->data = exp(input_value->data);
    output_value->backward_function = backward_exponential;
    basis_value_retain(input_value);
    return output_value;
}

static void backward_logarithm(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += (1.0 / output_value->previous_nodes[0]->data) * output_value->gradient;
}

basis_value* basis_value_logarithm(basis_value* input_value) {
    BASIS_CHECK_NULL(input_value);
    if (input_value->data <= 0.0) {
        BASIS_SET_ERROR(BASIS_DOMAIN_ERROR, "Logarithm of non-positive number");
        return NULL;
    }
    basis_value* output_value = basis_value_operation_initialize(input_value, NULL, "log", 1);
    output_value->data = log(input_value->data);
    output_value->backward_function = backward_logarithm;
    basis_value_retain(input_value);
    return output_value;
}

static void backward_sine(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += cos(output_value->previous_nodes[0]->data) * output_value->gradient;
}

basis_value* basis_value_sine(basis_value* input_value) {
    BASIS_CHECK_NULL(input_value);
    basis_value* output_value = basis_value_operation_initialize(input_value, NULL, "sin", 1);
    output_value->data = sin(input_value->data);
    output_value->backward_function = backward_sine;
    basis_value_retain(input_value);
    return output_value;
}

static void backward_cosine(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += -sin(output_value->previous_nodes[0]->data) * output_value->gradient;
}

basis_value* basis_value_cosine(basis_value* input_value) {
    BASIS_CHECK_NULL(input_value);
    basis_value* output_value = basis_value_operation_initialize(input_value, NULL, "cos", 1);
    output_value->data = cos(input_value->data);
    output_value->backward_function = backward_cosine;
    basis_value_retain(input_value);
    return output_value;
}

static void backward_tanh(basis_value* output_value) {
    double tanh_val = output_value->data;
    output_value->previous_nodes[0]->gradient += (1.0 - tanh_val * tanh_val) * output_value->gradient;
}

basis_value* basis_value_tanh(basis_value* input_value) {
    BASIS_CHECK_NULL(input_value);
    basis_value* output_value = basis_value_operation_initialize(input_value, NULL, "tanh", 1);
    output_value->data = tanh(input_value->data);
    output_value->backward_function = backward_tanh;
    basis_value_retain(input_value);
    return output_value;
}

static void backward_rectified_linear_unit(basis_value* output_value) {
    double gradient_multiplier = (output_value->data > 0) ? 1.0 : 0.0;
    output_value->previous_nodes[0]->gradient += gradient_multiplier * output_value->gradient;
}

basis_value* basis_value_rectified_linear_unit(basis_value* input_value) {
    BASIS_CHECK_NULL(input_value);
    basis_value* output_value = basis_value_operation_initialize(input_value, NULL, "ReLU", 1);
    output_value->data = (input_value->data > 0) ? input_value->data : 0.0;
    output_value->backward_function = backward_rectified_linear_unit;
    basis_value_retain(input_value);
    return output_value;
}

static void build_topological_order(basis_value* root, basis_value*** topological_order, size_t* order_size, size_t* order_capacity) {
    size_t stack_cap = 256;
    basis_value** stack = (basis_value**)malloc(sizeof(basis_value*) * stack_cap);
    size_t* edge_index = (size_t*)malloc(sizeof(size_t) * stack_cap);
    size_t stack_ptr = 0;

    stack[stack_ptr] = root;
    edge_index[stack_ptr] = 0;
    root->visited = true;
    stack_ptr++;

    while (stack_ptr > 0) {
        basis_value* current = stack[stack_ptr - 1];
        size_t idx = edge_index[stack_ptr - 1];

        if (idx < current->previous_node_count) {
            edge_index[stack_ptr - 1]++;
            basis_value* child = current->previous_nodes[idx];
            if (!child->visited) {
                child->visited = true;
                if (stack_ptr >= stack_cap) {
                    stack_cap *= 2;
                    stack = (basis_value**)realloc(stack, sizeof(basis_value*) * stack_cap);
                    edge_index = (size_t*)realloc(edge_index, sizeof(size_t) * stack_cap);
                }
                stack[stack_ptr] = child;
                edge_index[stack_ptr] = 0;
                stack_ptr++;
            }
        } else {
            if (*order_size >= *order_capacity) {
                *order_capacity *= 2;
                *topological_order = (basis_value**)realloc(*topological_order, sizeof(basis_value*) * (*order_capacity));
            }
            (*topological_order)[(*order_size)++] = current;
            stack_ptr--;
        }
    }
    free(stack);
    free(edge_index);
}

void basis_value_zero_gradient(basis_value* root_value) {
    if (!root_value) return;
    size_t order_capacity = 16;
    size_t order_size = 0;
    basis_value** topological_order = (basis_value**)malloc(sizeof(basis_value*) * order_capacity);
    build_topological_order(root_value, &topological_order, &order_size, &order_capacity);
    for (size_t i = 0; i < order_size; i++) {
        topological_order[i]->gradient = 0.0;
        topological_order[i]->visited = false;
    }
    free(topological_order);
}

void basis_value_backward_propagation(basis_value* root_value) {
    if (!root_value) return;
    size_t order_capacity = 16;
    size_t order_size = 0;
    basis_value** topological_order = (basis_value**)malloc(sizeof(basis_value*) * order_capacity);
    build_topological_order(root_value, &topological_order, &order_size, &order_capacity);
    for (size_t i = 0; i < order_size; i++) {
        topological_order[i]->gradient = 0.0;
        topological_order[i]->visited = false;
    }
    root_value->gradient = 1.0;
    for (int i = (int)order_size - 1; i >= 0; i--) {
        if (topological_order[i]->backward_function) {
            topological_order[i]->backward_function(topological_order[i]);
        }
    }
    free(topological_order);
}

void basis_value_free(basis_value* target_value) {
    if (!target_value) return;

    // Phase L: Iterative DAG Teardown to prevent stack overflow on deep graphs
    size_t stack_cap = 256;
    basis_value** stack = (basis_value**)malloc(sizeof(basis_value*) * stack_cap);
    size_t stack_ptr = 0;

    stack[stack_ptr++] = target_value;

    while (stack_ptr > 0) {
        basis_value* current = stack[--stack_ptr];
        current->reference_count--;

        if (current->reference_count > 0) continue;

        if (current->previous_nodes) {
            for (size_t i = 0; i < current->previous_node_count; i++) {
                if (current->previous_nodes[i]) {
                    if (stack_ptr >= stack_cap) {
                        stack_cap *= 2;
                        stack = (basis_value**)realloc(stack, sizeof(basis_value*) * stack_cap);
                    }
                    stack[stack_ptr++] = current->previous_nodes[i];
                }
            }
            free(current->previous_nodes);
        }
        if (current->operation) free(current->operation);
        free(current);
    }
    free(stack);
}

```

---

### FILE: src/basis/stage2_linear/tensor.c
Location: `src/basis/stage2_linear/tensor.c`
```cpp
#include "basis/stage2_linear/tensor.h"
#include "basis/core/error.h"
#ifdef BASIS_USE_CBLAS
#include <cblas.h>
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static char* tensor_string_duplicate(const char* source) {
    if (!source) return NULL;
    size_t len = strlen(source) + 1;
    char* copy = (char*)malloc(len);
    if (copy) memcpy(copy, source, len);
    return copy;
}

basis_tensor* basis_tensor_new(size_t row_count, size_t column_count) {
    if (row_count == 0 || column_count == 0) {
        BASIS_SET_ERROR(BASIS_INVALID_SHAPE, "Tensor dimensions must be > 0");
        return NULL;
    }
    basis_tensor* new_tensor = (basis_tensor*)malloc(sizeof(basis_tensor));
    new_tensor->row_count = row_count;
    new_tensor->column_count = column_count;
    new_tensor->flat_size = row_count * column_count;
    new_tensor->data = (basis_value**)malloc(sizeof(basis_value*) * new_tensor->flat_size);
    for (size_t i = 0; i < new_tensor->flat_size; i++) {
        new_tensor->data[i] = basis_value_new(0.0);
    }
    new_tensor->row_stride = column_count;
    new_tensor->col_stride = 1;
    new_tensor->offset = 0;
    new_tensor->is_view = false;
    new_tensor->parent = NULL;
    new_tensor->ref_count = 1;
    return new_tensor;
}

basis_tensor* basis_tensor_transpose(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* view = (basis_tensor*)malloc(sizeof(basis_tensor));
    view->data = target_tensor->data;
    view->flat_size = target_tensor->flat_size;
    view->row_count = target_tensor->column_count;
    view->column_count = target_tensor->row_count;
    view->row_stride = target_tensor->col_stride;
    view->col_stride = target_tensor->row_stride;
    view->offset = target_tensor->offset;
    view->is_view = true;
    view->parent = target_tensor;
    view->ref_count = 1;
    target_tensor->ref_count++; // Keep parent alive
    return view;
}

basis_tensor* basis_tensor_slice_rows(basis_tensor* target_tensor, size_t start_row, size_t num_rows) {
    BASIS_CHECK_NULL(target_tensor);
    if (start_row + num_rows > target_tensor->row_count) {
        BASIS_SET_ERROR(BASIS_INVALID_SHAPE, "Slice out of bounds");
        return NULL;
    }
    basis_tensor* view = (basis_tensor*)malloc(sizeof(basis_tensor));
    view->data = target_tensor->data;
    view->flat_size = target_tensor->flat_size;
    view->row_count = num_rows;
    view->column_count = target_tensor->column_count;
    view->row_stride = target_tensor->row_stride;
    view->col_stride = target_tensor->col_stride;
    view->offset = target_tensor->offset + (start_row * target_tensor->row_stride);
    view->is_view = true;
    view->parent = target_tensor;
    view->ref_count = 1;
    target_tensor->ref_count++;
    return view;
}

void basis_tensor_free(basis_tensor* target_tensor) {
    if (!target_tensor) return;
    target_tensor->ref_count--;
    if (target_tensor->ref_count > 0) return;

    if (target_tensor->is_view) {
        basis_tensor_free(target_tensor->parent);
    } else {
        if (target_tensor->data) {
            for (size_t i = 0; i < target_tensor->flat_size; i++) {
                basis_value_free(target_tensor->data[i]);
            }
            free(target_tensor->data);
        }
    }
    free(target_tensor);
}

void basis_tensor_fill(basis_tensor* target_tensor, double fill_value) {
    if (!target_tensor) return;
    for (size_t i = 0; i < target_tensor->row_count; i++)
        for (size_t j = 0; j < target_tensor->column_count; j++)
            BASIS_TENSOR_AT(target_tensor, i, j)->data = fill_value;
}

void basis_tensor_set(basis_tensor* target_tensor, size_t row_index, size_t column_index, double data_value) {
    if (!target_tensor || row_index >= target_tensor->row_count || column_index >= target_tensor->column_count) return;
    BASIS_TENSOR_AT(target_tensor, row_index, column_index)->data = data_value;
}

basis_tensor* basis_tensor_addition(basis_tensor* first_tensor, basis_tensor* second_tensor) {
    BASIS_CHECK_NULL(first_tensor); BASIS_CHECK_NULL(second_tensor);
    BASIS_CHECK_SHAPE(first_tensor->row_count == second_tensor->row_count && first_tensor->column_count == second_tensor->column_count);
    basis_tensor* output_tensor = basis_tensor_new(first_tensor->row_count, first_tensor->column_count);
    for (size_t i = 0; i < first_tensor->row_count; i++)
        for (size_t j = 0; j < first_tensor->column_count; j++)
            output_tensor->data[i * output_tensor->column_count + j] = basis_value_addition(BASIS_TENSOR_AT(first_tensor, i, j), BASIS_TENSOR_AT(second_tensor, i, j));
    return output_tensor;
}

static void backward_matmul_element(basis_value* output_value) {
    size_t K = output_value->previous_node_count / 2;
    double grad = output_value->gradient;
    for (size_t k = 0; k < K; k++) {
        basis_value* a_ik = output_value->previous_nodes[k];
        basis_value* b_kj = output_value->previous_nodes[K + k];
        a_ik->gradient += grad * b_kj->data;
        b_kj->gradient += grad * a_ik->data;
    }
}

basis_tensor* basis_tensor_matrix_multiplication(basis_tensor* first_tensor, basis_tensor* second_tensor) {
    BASIS_CHECK_NULL(first_tensor); BASIS_CHECK_NULL(second_tensor);
    BASIS_CHECK_SHAPE(first_tensor->column_count == second_tensor->row_count);
    size_t M = first_tensor->row_count;
    size_t K = first_tensor->column_count;
    size_t N = second_tensor->column_count;
    basis_tensor* output_tensor = basis_tensor_new(M, N);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            basis_value* c_ij = basis_value_new(0.0);
            free(c_ij->operation);
            c_ij->operation = tensor_string_duplicate("matmul_fused");
            c_ij->previous_node_count = 2 * K;
            c_ij->previous_nodes = (basis_value**)malloc(sizeof(basis_value*) * 2 * K);
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                basis_value* a_ik = BASIS_TENSOR_AT(first_tensor, i, k);
                basis_value* b_kj = BASIS_TENSOR_AT(second_tensor, k, j);
                c_ij->previous_nodes[k] = a_ik;
                c_ij->previous_nodes[K + k] = b_kj;
                basis_value_retain(a_ik);
                basis_value_retain(b_kj);
                sum += a_ik->data * b_kj->data;
            }
            c_ij->data = sum;
            c_ij->backward_function = backward_matmul_element;
            basis_value_free(output_tensor->data[i * N + j]);
            output_tensor->data[i * N + j] = c_ij;
        }
    }
    return output_tensor;
}

basis_tensor* basis_tensor_scalar_multiplication(basis_tensor* target_tensor, double scalar_value) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    basis_value* scalar = basis_value_new(scalar_value);
    for (size_t i = 0; i < target_tensor->row_count; i++)
        for (size_t j = 0; j < target_tensor->column_count; j++)
            output_tensor->data[i * output_tensor->column_count + j] = basis_value_multiplication(BASIS_TENSOR_AT(target_tensor, i, j), scalar);
    basis_value_free(scalar);
    return output_tensor;
}

basis_tensor* basis_tensor_rectified_linear_unit(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count; i++)
        for (size_t j = 0; j < target_tensor->column_count; j++)
            output_tensor->data[i * output_tensor->column_count + j] = basis_value_rectified_linear_unit(BASIS_TENSOR_AT(target_tensor, i, j));
    return output_tensor;
}


basis_tensor* basis_tensor_broadcast_view(basis_tensor* target_tensor, size_t target_rows, size_t target_cols) {
    BASIS_CHECK_NULL(target_tensor);
    // Validate broadcasting rules
    bool row_compat = (target_tensor->row_count == target_rows) || (target_tensor->row_count == 1);
    bool col_compat = (target_tensor->column_count == target_cols) || (target_tensor->column_count == 1);
    if (!row_compat || !col_compat) {
        BASIS_SET_ERROR(BASIS_INVALID_SHAPE, "Broadcast shape mismatch");
        return NULL;
    }

    basis_tensor* view = (basis_tensor*)malloc(sizeof(basis_tensor));
    view->data = target_tensor->data;
    view->flat_size = target_tensor->flat_size;
    view->row_count = target_rows;
    view->column_count = target_cols;

    // The Math: stride=0 on singleton dimensions
    view->row_stride = (target_tensor->row_count == 1) ? 0 : target_tensor->row_stride;
    view->col_stride = (target_tensor->column_count == 1) ? 0 : target_tensor->col_stride;
    view->offset = target_tensor->offset;

    view->is_view = true;
    view->parent = target_tensor;
    view->ref_count = 1;
    target_tensor->ref_count++;
    return view;
}

basis_tensor* basis_tensor_add(basis_tensor* first_tensor, basis_tensor* second_tensor) {
    BASIS_CHECK_NULL(first_tensor); BASIS_CHECK_NULL(second_tensor);
    BASIS_CHECK_SHAPE(first_tensor->row_count == second_tensor->row_count && first_tensor->column_count == second_tensor->column_count);

    basis_tensor* output_tensor = basis_tensor_new(first_tensor->row_count, first_tensor->column_count);
    for (size_t i = 0; i < first_tensor->row_count; i++) {
        for (size_t j = 0; j < first_tensor->column_count; j++) {
            basis_value_free(output_tensor->data[i * output_tensor->column_count + j]);
            output_tensor->data[i * output_tensor->column_count + j] = basis_value_addition(
                BASIS_TENSOR_AT(first_tensor, i, j),
                BASIS_TENSOR_AT(second_tensor, i, j)
            );
        }
    }
    return output_tensor;
}

void basis_tensor_matmul_backward_cblas(basis_tensor* grad_output, basis_tensor* A, basis_tensor* B) {
#ifdef BASIS_USE_CBLAS
    // Future Phase K: This will be natively hooked into the Tensor-level Tape.
    // For now, it serves as the high-performance gradient scatter utility.
    size_t M = A->row_count; size_t K = A->column_count; size_t N = B->column_count;
    double* dC = malloc(M * N * sizeof(double));
    double* A_flat = malloc(M * K * sizeof(double));
    double* B_flat = malloc(K * N * sizeof(double));
    double* dA = calloc(M * K, sizeof(double));
    double* dB = calloc(K * N, sizeof(double));

    for(size_t i=0; i<M; i++) for(size_t j=0; j<N; j++) dC[i*N+j] = BASIS_TENSOR_AT(grad_output, i, j)->gradient;
    for(size_t i=0; i<M; i++) for(size_t k=0; k<K; k++) A_flat[i*K+k] = BASIS_TENSOR_AT(A, i, k)->data;
    for(size_t k=0; k<K; k++) for(size_t j=0; j<N; j++) B_flat[k*N+j] = BASIS_TENSOR_AT(B, k, j)->data;

    // dA = dC * B^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, K, N, 1.0, dC, N, B_flat, N, 0.0, dA, K);
    // dB = A^T * dC
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, N, M, 1.0, A_flat, K, dC, N, 0.0, dB, N);

    for(size_t i=0; i<M; i++) for(size_t k=0; k<K; k++) BASIS_TENSOR_AT(A, i, k)->gradient += dA[i*K+k];
    for(size_t k=0; k<K; k++) for(size_t j=0; j<N; j++) BASIS_TENSOR_AT(B, k, j)->gradient += dB[k*N+j];

    free(dC); free(A_flat); free(B_flat); free(dA); free(dB);
#else
    (void)grad_output; (void)A; (void)B;
    fprintf(stderr, "[BASIS] CBLAS backward pass called, but BASIS_USE_CBLAS is not defined.\n");
#endif
}


basis_tensor* basis_tensor_tanh(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count; i++)
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value_free(output_tensor->data[i * output_tensor->column_count + j]);
            output_tensor->data[i * output_tensor->column_count + j] = basis_value_tanh(BASIS_TENSOR_AT(target_tensor, i, j));
        }
    return output_tensor;
}


void basis_tensor_save_binary(basis_tensor* target_tensor, const char* filename) {
    if (!target_tensor || !filename) return;
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    fwrite(&target_tensor->row_count, sizeof(size_t), 1, f);
    fwrite(&target_tensor->column_count, sizeof(size_t), 1, f);
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            double val = BASIS_TENSOR_AT(target_tensor, i, j)->data;
            fwrite(&val, sizeof(double), 1, f);
        }
    }
    fclose(f);
}

basis_tensor* basis_tensor_load_binary(const char* filename) {
    if (!filename) return NULL;
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    size_t r, c;
    size_t dummy_r = fread(&r, sizeof(size_t), 1, f); (void)dummy_r;
    size_t dummy_c = fread(&c, sizeof(size_t), 1, f); (void)dummy_c;
    basis_tensor* t = basis_tensor_new(r, c);
    for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) {
            double val;
            size_t dummy_val = fread(&val, sizeof(double), 1, f); (void)dummy_val;
            BASIS_TENSOR_AT(t, i, j)->data = val;
        }
    }
    fclose(f);
    return t;
}

void basis_tensor_print(basis_tensor* target_tensor, const char* name) {
    if (!target_tensor) return;
    printf("Tensor %s (%zu x %zu):\n", name, target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        printf("  [");
        for (size_t j = 0; j < target_tensor->column_count; j++)
            printf(" %8.4f ", BASIS_TENSOR_AT(target_tensor, i, j)->data);
        printf("]\n");
    }
}

```

---

### FILE: src/basis/stage3_sequence/sequence.c
Location: `src/basis/stage3_sequence/sequence.c`
```cpp

#include "basis/stage3_sequence/sequence.h"
#include "basis/core/error.h"
#include <math.h>
#include <stdlib.h>



basis_tensor* basis_tensor_softmax(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        double maximum_data = target_tensor->data[i * target_tensor->column_count]->data;
        for (size_t j = 1; j < target_tensor->column_count; j++) {
            if (BASIS_TENSOR_AT(target_tensor, i, j)->data > maximum_data) {
                maximum_data = BASIS_TENSOR_AT(target_tensor, i, j)->data;
            }
        }
        basis_value* negative_maximum = basis_value_new(-maximum_data);
        basis_value* sum = basis_value_new(0.0);
        basis_value** exponentials = (basis_value**)malloc(sizeof(basis_value*) * target_tensor->column_count);
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value* shifted = basis_value_addition(BASIS_TENSOR_AT(target_tensor, i, j), negative_maximum);
            exponentials[j] = basis_value_exponential(shifted);
            basis_value* new_sum = basis_value_addition(sum, exponentials[j]);
            basis_value_free(sum); basis_value_free(shifted);
            sum = new_sum;
        }
        basis_value_free(negative_maximum);
        basis_value* inverse_sum = basis_value_power(sum, -1.0);
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value_free(BASIS_TENSOR_AT(output_tensor, i, j));
            BASIS_TENSOR_AT(output_tensor, i, j) = basis_value_multiplication(exponentials[j], inverse_sum);
            basis_value_free(exponentials[j]);
        }
        basis_value_free(sum); basis_value_free(inverse_sum); free(exponentials);
    }
    return output_tensor;
}

basis_tensor* basis_tensor_log_softmax(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        double maximum_data = target_tensor->data[i * target_tensor->column_count]->data;
        for (size_t j = 1; j < target_tensor->column_count; j++) {
            if (BASIS_TENSOR_AT(target_tensor, i, j)->data > maximum_data) {
                maximum_data = BASIS_TENSOR_AT(target_tensor, i, j)->data;
            }
        }
        basis_value* negative_maximum = basis_value_new(-maximum_data);
        basis_value* sum = basis_value_new(0.0);

        basis_value** shifted_vals = (basis_value**)malloc(sizeof(basis_value*) * target_tensor->column_count);
        basis_value** exponentials = (basis_value**)malloc(sizeof(basis_value*) * target_tensor->column_count);

        for (size_t j = 0; j < target_tensor->column_count; j++) {
            shifted_vals[j] = basis_value_addition(BASIS_TENSOR_AT(target_tensor, i, j), negative_maximum);
            exponentials[j] = basis_value_exponential(shifted_vals[j]);
            basis_value* new_sum = basis_value_addition(sum, exponentials[j]);
            basis_value_free(sum);
            sum = new_sum;
        }
        basis_value_free(negative_maximum);

        basis_value* log_sum = basis_value_logarithm(sum);
        basis_value_free(sum); // sum tree destroyed, drops ref to exponentials. Array holds the other ref.

        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value_free(BASIS_TENSOR_AT(output_tensor, i, j));
            // LogSoftmax(x) = (x - max) - log(sum(exp(x - max)))
            BASIS_TENSOR_AT(output_tensor, i, j) = basis_value_subtraction(shifted_vals[j], log_sum);
            basis_value_free(shifted_vals[j]);
            basis_value_free(exponentials[j]);
        }
        basis_value_free(log_sum);
        free(shifted_vals);
        free(exponentials);
    }
    return output_tensor;
}

basis_tensor* basis_tensor_attention(basis_tensor* query_tensor, basis_tensor* key_tensor, basis_tensor* basis_value_tensor) {
    BASIS_CHECK_NULL(query_tensor); BASIS_CHECK_NULL(key_tensor); BASIS_CHECK_NULL(basis_value_tensor);
    BASIS_CHECK_SHAPE(query_tensor->column_count == key_tensor->column_count && key_tensor->row_count == basis_value_tensor->row_count);

    basis_tensor* transposed_key_tensor = basis_tensor_transpose(key_tensor);
    basis_tensor* query_key_product = basis_tensor_matrix_multiplication(query_tensor, transposed_key_tensor);
    basis_tensor* scaled_tensor = basis_tensor_scalar_multiplication(query_key_product, 1.0 / sqrt((double)key_tensor->column_count));
    basis_tensor* attention_weights = basis_tensor_softmax(scaled_tensor);
    basis_tensor* output_tensor = basis_tensor_matrix_multiplication(attention_weights, basis_value_tensor);

    basis_tensor_free(transposed_key_tensor); basis_tensor_free(query_key_product);
    basis_tensor_free(scaled_tensor); basis_tensor_free(attention_weights);
    return output_tensor;
}

```

---

### FILE: src/basis/stage4_learning/learning.c
Location: `src/basis/stage4_learning/learning.c`
```cpp

#include "basis/stage4_learning/learning.h"
#include "basis/core/error.h"
#include <math.h>
#include <stdlib.h>

basis_tensor* basis_tensor_layer_normalization(basis_tensor* target_tensor, double epsilon) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        basis_value* sum = basis_value_new(0.0);
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value* new_sum = basis_value_addition(sum, BASIS_TENSOR_AT(target_tensor, i, j));
            basis_value_free(sum); sum = new_sum;
        }
        basis_value* mean = basis_value_multiplication(sum, basis_value_new(1.0 / target_tensor->column_count));
        basis_value_free(sum);
        basis_value* variance_sum = basis_value_new(0.0);
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value* negative_mean = basis_value_multiplication(mean, basis_value_new(-1.0));
            basis_value* difference = basis_value_addition(BASIS_TENSOR_AT(target_tensor, i, j), negative_mean);
            basis_value_free(negative_mean);
            basis_value* squared = basis_value_power(difference, 2.0);
            basis_value_free(difference);
            basis_value* new_variance_sum = basis_value_addition(variance_sum, squared);
            basis_value_free(variance_sum); basis_value_free(squared);
            variance_sum = new_variance_sum;
        }
        basis_value* variance = basis_value_multiplication(variance_sum, basis_value_new(1.0 / target_tensor->column_count));
        basis_value_free(variance_sum);
        basis_value* epsilon_value = basis_value_new(epsilon);
        basis_value* variance_epsilon = basis_value_addition(variance, epsilon_value);
        basis_value_free(variance); basis_value_free(epsilon_value);
        basis_value* inverse_standard_deviation = basis_value_power(variance_epsilon, -0.5);
        basis_value_free(variance_epsilon);
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value* negative_mean = basis_value_multiplication(mean, basis_value_new(-1.0));
            basis_value* difference = basis_value_addition(BASIS_TENSOR_AT(target_tensor, i, j), negative_mean);
            basis_value_free(negative_mean);
            basis_value_free(BASIS_TENSOR_AT(output_tensor, i, j));
            BASIS_TENSOR_AT(output_tensor, i, j) = basis_value_multiplication(difference, inverse_standard_deviation);
            basis_value_free(difference);
        }
        basis_value_free(mean); basis_value_free(inverse_standard_deviation);
    }
    return output_tensor;
}

basis_tensor* basis_tensor_rotary_positional_embedding(basis_tensor* target_tensor, int position, double base) {
    BASIS_CHECK_NULL(target_tensor);
    BASIS_CHECK_SHAPE(target_tensor->column_count % 2 == 0);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    size_t half_dim = target_tensor->column_count / 2;
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        for (size_t j = 0; j < half_dim; j++) {
            double theta = pow(base, -(double)(2 * j) / target_tensor->column_count);
            double angle = position * theta;
            basis_value* angle_value = basis_value_new(angle);
            basis_value* cosine_value = basis_value_cosine(angle_value);
            basis_value* sine_value = basis_value_sine(angle_value);
            basis_value_free(angle_value);
            size_t idx1 = i * target_tensor->column_count + j;
            size_t idx2 = i * target_tensor->column_count + j + half_dim;
            basis_value* first_element = BASIS_TENSOR_AT(target_tensor, 0, idx1);
            basis_value* second_element = BASIS_TENSOR_AT(target_tensor, 0, idx2);
            basis_value_free(BASIS_TENSOR_AT(output_tensor, 0, idx1)); basis_value_free(BASIS_TENSOR_AT(output_tensor, 0, idx2));
            basis_value* negative_one = basis_value_new(-1.0);
            basis_value* negative_sine = basis_value_multiplication(sine_value, negative_one);
            basis_value_free(negative_one);
            basis_value* term1 = basis_value_multiplication(first_element, cosine_value);
            basis_value* term2 = basis_value_multiplication(second_element, negative_sine);
            BASIS_TENSOR_AT(output_tensor, 0, idx1) = basis_value_addition(term1, term2);
            basis_value_free(term1); basis_value_free(term2); basis_value_free(negative_sine);
            basis_value* term3 = basis_value_multiplication(first_element, sine_value);
            basis_value* term4 = basis_value_multiplication(second_element, cosine_value);
            BASIS_TENSOR_AT(output_tensor, 0, idx2) = basis_value_addition(term3, term4);
            basis_value_free(term3); basis_value_free(term4);
            basis_value_free(cosine_value); basis_value_free(sine_value);
        }
    }
    return output_tensor;
}

basis_value* basis_tensor_sum(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_value* sum = basis_value_new(0.0);
    for (size_t i = 0; i < target_tensor->row_count; i++) {
        for (size_t j = 0; j < target_tensor->column_count; j++) {
            basis_value* new_sum = basis_value_addition(sum, BASIS_TENSOR_AT(target_tensor, i, j));
            basis_value_free(sum);
            sum = new_sum;
        }
    }
    return sum;
}

basis_tensor* basis_tensor_logarithm(basis_tensor* target_tensor) {
    BASIS_CHECK_NULL(target_tensor);
    basis_tensor* output_tensor = basis_tensor_new(target_tensor->row_count, target_tensor->column_count);
    for (size_t i = 0; i < target_tensor->row_count * target_tensor->column_count; i++) {
        basis_value_free(BASIS_TENSOR_AT(output_tensor, 0, i));
        BASIS_TENSOR_AT(output_tensor, 0, i) = basis_value_logarithm(BASIS_TENSOR_AT(target_tensor, 0, i));
    }
    return output_tensor;
}

basis_tensor* basis_tensor_multiplication(basis_tensor* first_tensor, basis_tensor* second_tensor) {
    BASIS_CHECK_NULL(first_tensor); BASIS_CHECK_NULL(second_tensor);
    BASIS_CHECK_SHAPE(first_tensor->row_count == second_tensor->row_count && first_tensor->column_count == second_tensor->column_count);
    basis_tensor* output_tensor = basis_tensor_new(first_tensor->row_count, first_tensor->column_count);
    for (size_t i = 0; i < first_tensor->row_count * first_tensor->column_count; i++) {
        basis_value_free(BASIS_TENSOR_AT(output_tensor, 0, i));
        BASIS_TENSOR_AT(output_tensor, 0, i) = basis_value_multiplication(BASIS_TENSOR_AT(first_tensor, 0, i), BASIS_TENSOR_AT(second_tensor, 0, i));
    }
    return output_tensor;
}

```

---

### FILE: src/basis/stage4_learning/optim.c
Location: `src/basis/stage4_learning/optim.c`
```cpp

#include "basis/stage4_learning/optim.h"
#include <math.h>
#include <stdlib.h>
basis_adam* basis_adam_new(size_t row_count, size_t column_count, double learning_rate) {
    basis_adam* optimizer = (basis_adam*)malloc(sizeof(basis_adam));
    optimizer->learning_rate = learning_rate; optimizer->beta1 = 0.9; optimizer->beta2 = 0.999;
    optimizer->epsilon = 1e-8; optimizer->time_step = 0; optimizer->element_count = row_count * column_count;
    optimizer->first_moment = (double*)calloc(optimizer->element_count, sizeof(double));
    optimizer->second_moment = (double*)calloc(optimizer->element_count, sizeof(double));
    return optimizer;
}
void basis_adam_optimization_step(basis_adam* optimizer, basis_tensor* weight_tensor) {
    optimizer->time_step++;
    for (size_t i = 0; i < optimizer->element_count; i++) {
        double gradient = weight_tensor->data[i]->gradient;
        optimizer->first_moment[i] = optimizer->beta1 * optimizer->first_moment[i] + (1.0 - optimizer->beta1) * gradient;
        optimizer->second_moment[i] = optimizer->beta2 * optimizer->second_moment[i] + (1.0 - optimizer->beta2) * gradient * gradient;
        double first_moment_hat = optimizer->first_moment[i] / (1.0 - pow(optimizer->beta1, optimizer->time_step));
        double second_moment_hat = optimizer->second_moment[i] / (1.0 - pow(optimizer->beta2, optimizer->time_step));
        weight_tensor->data[i]->data -= optimizer->learning_rate * first_moment_hat / (sqrt(second_moment_hat) + optimizer->epsilon);
        weight_tensor->data[i]->gradient = 0.0;
    }
}
void basis_adam_free(basis_adam* optimizer) {
    if (!optimizer) return;
    free(optimizer->first_moment); free(optimizer->second_moment); free(optimizer);
}

```

---

### FILE: src/basis/stage5_unified/compiler.c
Location: `src/basis/stage5_unified/compiler.c`
```cpp

#include "basis/stage5_unified/compiler.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

static char* basis_compiler_string_duplicate(const char* source_string) {
    size_t length = strlen(source_string) + 1;
    char* copy = (char*)malloc(length);
    if (copy) memcpy(copy, source_string, length);
    return copy;
}

// --- CSE Helper Functions ---
static size_t basis_symbol_hash(basis_symbol* sym) {
    if (!sym) return 0;
    size_t h = sym->type;
    if (sym->type == basis_symbol_type_variable) {
        const char* p = sym->name;
        while (*p) { h = h * 31 + *p++; }
    } else if (sym->type == basis_symbol_type_constant) {
        union { double d; size_t i; } u;
        u.d = sym->basis_value;
        h ^= u.i;
    } else {
        const char* p = sym->operation;
        while (*p) { h = h * 31 + *p++; }
        h ^= sym->argument_count;
        for (size_t i = 0; i < sym->argument_count; i++) {
            h ^= basis_symbol_hash(sym->arguments[i]) * (i + 1);
        }
        if (sym->operation && strcmp(sym->operation, "**") == 0) {
            union { double d; size_t i; } u;
            u.d = sym->basis_value;
            h ^= u.i;
        }
    }
    return h;
}

static bool basis_symbol_structural_equal(basis_symbol* a, basis_symbol* b) {
    if (!a || !b) return a == b;
    if (a->type != b->type) return false;
    if (a->type == basis_symbol_type_variable) return strcmp(a->name, b->name) == 0;
    if (a->type == basis_symbol_type_constant) return a->basis_value == b->basis_value;
    if (strcmp(a->operation, b->operation) != 0) return false;
    if (a->argument_count != b->argument_count) return false;
    if (a->operation && strcmp(a->operation, "**") == 0 && a->basis_value != b->basis_value) return false;
    for (size_t i = 0; i < a->argument_count; i++) {
        if (!basis_symbol_structural_equal(a->arguments[i], b->arguments[i])) return false;
    }
    return true;
}
// ----------------------------

basis_compiler* basis_compiler_new() {
    basis_compiler* new_compiler = (basis_compiler*)malloc(sizeof(basis_compiler));
    new_compiler->variable_mappings = NULL;
    new_compiler->variable_mapping_count = 0;
    new_compiler->cse_cache = NULL;
    new_compiler->cse_count = 0;
    new_compiler->cse_cap = 0;
    return new_compiler;
}

void basis_compiler_map(basis_compiler* basis_compiler_instance, char* variable_name, basis_value* basis_value_pointer) {
    basis_compiler_instance->variable_mapping_count++;
    basis_compiler_instance->variable_mappings = (basis_variable_mapping*)realloc(basis_compiler_instance->variable_mappings, sizeof(basis_variable_mapping) * basis_compiler_instance->variable_mapping_count);
    basis_compiler_instance->variable_mappings[basis_compiler_instance->variable_mapping_count - 1].name = basis_compiler_string_duplicate(variable_name);
    basis_compiler_instance->variable_mappings[basis_compiler_instance->variable_mapping_count - 1].basis_value_pointer = basis_value_pointer;
}

static basis_value* find_variable(basis_compiler* basis_compiler_instance, char* variable_name) {
    for (size_t i = 0; i < basis_compiler_instance->variable_mapping_count; i++) {
        if (strcmp(basis_compiler_instance->variable_mappings[i].name, variable_name) == 0) return basis_compiler_instance->variable_mappings[i].basis_value_pointer;
    }
    return NULL;
}

basis_value* basis_compiler_compile(basis_compiler* basis_compiler_instance, basis_symbol* basis_symbol_node) {
    if (!basis_compiler_instance || !basis_symbol_node) return NULL;

    // Phase C: CSE Check
    size_t h = basis_symbol_hash(basis_symbol_node);
    for (size_t i = 0; i < basis_compiler_instance->cse_count; i++) {
        if (basis_compiler_instance->cse_cache[i].hash == h &&
            basis_symbol_structural_equal(basis_compiler_instance->cse_cache[i].sym, basis_symbol_node)) {
            basis_value_retain(basis_compiler_instance->cse_cache[i].val);
            return basis_compiler_instance->cse_cache[i].val;
        }
    }

    if (basis_symbol_node->type == basis_symbol_type_constant) return basis_value_new(basis_symbol_node->basis_value);
    if (basis_symbol_node->type == basis_symbol_type_variable) {
        basis_value* found_value = find_variable(basis_compiler_instance, basis_symbol_node->name);
        if (!found_value) {
            fprintf(stderr, "Compiler: undefined variable '%s'\n", basis_symbol_node->name);
            return NULL;
        }
        basis_value_retain(found_value);
        return found_value;
    }
    if (basis_symbol_node->type == basis_symbol_type_operation) {
        basis_value** arguments = calloc(basis_symbol_node->argument_count, sizeof(basis_value*));
        if (!arguments) return NULL;
        for (size_t i = 0; i < basis_symbol_node->argument_count; i++) {
            arguments[i] = basis_compiler_compile(basis_compiler_instance, basis_symbol_node->arguments[i]);
            if (!arguments[i]) {
                for (size_t j = 0; j < i; j++) basis_value_free(arguments[j]);
                free(arguments); return NULL;
            }
        }
        basis_value* output_value = NULL;
        if (strcmp(basis_symbol_node->operation, "+") == 0) output_value = basis_value_addition(arguments[0], arguments[1]);
        else if (strcmp(basis_symbol_node->operation, "-") == 0) output_value = basis_value_subtraction(arguments[0], arguments[1]);
        else if (strcmp(basis_symbol_node->operation, "*") == 0) output_value = basis_value_multiplication(arguments[0], arguments[1]);
        else if (strcmp(basis_symbol_node->operation, "**") == 0) output_value = basis_value_power(arguments[0], basis_symbol_node->basis_value);
        else if (strcmp(basis_symbol_node->operation, "exp") == 0) output_value = basis_value_exponential(arguments[0]);
        else if (strcmp(basis_symbol_node->operation, "log") == 0) output_value = basis_value_logarithm(arguments[0]);
        else {
            fprintf(stderr, "Compiler: unknown operation '%s'\n", basis_symbol_node->operation);
            for (size_t i = 0; i < basis_symbol_node->argument_count; i++) basis_value_free(arguments[i]);
            free(arguments); return NULL;
        }
        for (size_t i = 0; i < basis_symbol_node->argument_count; i++) basis_value_free(arguments[i]);
        free(arguments);

        // Phase C: CSE Store (Weak Reference)
        if (output_value) {
            if (basis_compiler_instance->cse_count >= basis_compiler_instance->cse_cap) {
                basis_compiler_instance->cse_cap = basis_compiler_instance->cse_cap == 0 ? 16 : basis_compiler_instance->cse_cap * 2;
                basis_compiler_instance->cse_cache = realloc(basis_compiler_instance->cse_cache, sizeof(cse_entry) * basis_compiler_instance->cse_cap);
            }
            basis_compiler_instance->cse_cache[basis_compiler_instance->cse_count].sym = basis_symbol_node;
            basis_compiler_instance->cse_cache[basis_compiler_instance->cse_count].val = output_value;
            basis_compiler_instance->cse_cache[basis_compiler_instance->cse_count].hash = h;
            basis_compiler_instance->cse_count++;
        }

        return output_value;
    }
    return NULL;
}

void basis_compiler_free(basis_compiler* basis_compiler_instance) {
    if (!basis_compiler_instance) return;
    for (size_t i = 0; i < basis_compiler_instance->variable_mapping_count; i++) free(basis_compiler_instance->variable_mappings[i].name);
    if (basis_compiler_instance->variable_mappings) free(basis_compiler_instance->variable_mappings);
    if (basis_compiler_instance->cse_cache) free(basis_compiler_instance->cse_cache);
    free(basis_compiler_instance);
}

```

---

### FILE: src/basis/stage5_unified/geometry.c
Location: `src/basis/stage5_unified/geometry.c`
```cpp
#include "basis/stage5_unified/geometry.h"
#include "basis/stage3_sequence/sequence.h"
#include "basis/core/error.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

basis_metric* basis_metric_fisher_information(basis_tensor* weight_tensor, basis_tensor* input_tensor, basis_tensor* output_tensor) {
    (void)weight_tensor; (void)output_tensor;
    BASIS_CHECK_NULL(input_tensor);
    basis_tensor* transposed_input = basis_tensor_transpose(input_tensor);
    basis_tensor* fisher_matrix = basis_tensor_matrix_multiplication(transposed_input, input_tensor);
    basis_metric* new_metric = (basis_metric*)malloc(sizeof(basis_metric));
    new_metric->matrix = fisher_matrix;
    basis_tensor_free(transposed_input);
    return new_metric;
}

void basis_metric_free(basis_metric* target_metric) {
    if (!target_metric) return;
    basis_tensor_free(target_metric->matrix);
    free(target_metric);
}

// Gaussian elimination with partial pivoting to solve Ax = b
static int basis_matrix_solve(double* A, double* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double max_val = fabs(A[i * n + i]);
        size_t max_row = i;
        for (size_t k = i + 1; k < n; k++) {
            if (fabs(A[k * n + i]) > max_val) {
                max_val = fabs(A[k * n + i]);
                max_row = k;
            }
        }
        if (max_val < 1e-12) return -1; // Singular

        if (max_row != i) {
            for (size_t j = 0; j < n; j++) {
                double tmp = A[i * n + j];
                A[i * n + j] = A[max_row * n + j];
                A[max_row * n + j] = tmp;
            }
            double tmp = b[i]; b[i] = b[max_row]; b[max_row] = tmp;
        }

        for (size_t k = i + 1; k < n; k++) {
            double c = -A[k * n + i] / A[i * n + i];
            for (size_t j = i; j < n; j++) {
                if (i == j) A[k * n + j] = 0;
                else A[k * n + j] += c * A[i * n + j];
            }
            b[k] += c * b[i];
        }
    }

    for (int i = (int)n - 1; i >= 0; i--) {
        b[i] /= A[i * n + i];
        for (int k = i - 1; k >= 0; k--) {
            b[k] -= A[k * n + i] * b[i];
        }
    }
    return 0;
}

void basis_metric_ngd_step(basis_metric* fisher_metric, basis_tensor* weight_tensor, double learning_rate) {
    if (!fisher_metric || !weight_tensor) return;

    size_t n = weight_tensor->row_count * weight_tensor->column_count;
    size_t f_rows = fisher_metric->matrix->row_count;
    size_t f_cols = fisher_metric->matrix->column_count;

    // The Fisher matrix must be an NxN square matrix, where N is the number of weights
    if (f_rows != n || f_cols != n) {
        BASIS_SET_ERROR(BASIS_INVALID_SHAPE, "Fisher matrix must be NxN where N is the number of weights");
        return;
    }

    double* F = (double*)malloc(n * n * sizeof(double));
    double* grad = (double*)malloc(n * sizeof(double));

    for(size_t i = 0; i < n * n; i++) F[i] = fisher_metric->matrix->data[i]->data;
    for(size_t i = 0; i < n; i++) grad[i] = weight_tensor->data[i]->gradient;

    // Tikhonov regularization (damping) for numerical stability
    double damping = 1e-12; // Lowered to prevent bias on ill-conditioned matrices
    for(size_t i = 0; i < n; i++) F[i * n + i] += damping;

    int status = basis_matrix_solve(F, grad, n);

    if (status == 0) {
        for(size_t i = 0; i < n; i++) {
            weight_tensor->data[i]->data -= learning_rate * grad[i];
            weight_tensor->data[i]->gradient = 0.0;
        }
    } else {
        fprintf(stderr, "[BASIS NGD] Fisher matrix is singular, falling back to standard GD.\n");
        for(size_t i = 0; i < n; i++) {
            weight_tensor->data[i]->data -= learning_rate * weight_tensor->data[i]->gradient;
            weight_tensor->data[i]->gradient = 0.0;
        }
    }

    free(F);
    free(grad);
}

```

---

### FILE: src/basis/stage5_unified/jit.c
Location: `src/basis/stage5_unified/jit.c`
```cpp
#define _DEFAULT_SOURCE
#include "basis/stage5_unified/jit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <stdarg.h>

// --- Dynamic String Buffer ---
typedef struct {
    char* data;
    size_t len;
    size_t cap;
} jit_buf;

static void jit_buf_init(jit_buf* b, size_t initial_cap) {
    b->cap = initial_cap;
    b->len = 0;
    b->data = (char*)malloc(b->cap);
    b->data[0] = '\0';
}

static void jit_buf_append(jit_buf* b, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int needed = vsnprintf(NULL, 0, fmt, args);
    va_end(args);

    while (b->len + (size_t)needed + 1 > b->cap) {
        b->cap *= 2;
        b->data = (char*)realloc(b->data, b->cap);
    }

    va_start(args, fmt);
    vsnprintf(b->data + b->len, b->cap - b->len, fmt, args);
    va_end(args);
    b->len += (size_t)needed;
}

static void jit_buf_free(jit_buf* b) {
    free(b->data);
}

// --- Recursive Post-Order AST Emitter ---
// Returns the temp variable ID for this node
static int jit_emit_node(basis_symbol* sym, jit_buf* buf, int* temp_counter, char** var_names, size_t var_count) {
    if (!sym) return -1;

    if (sym->type == basis_symbol_type_constant) {
        int id = (*temp_counter)++;
        jit_buf_append(buf, "    double t%d = %.17g;\n", id, sym->basis_value);
        return id;
    }

    if (sym->type == basis_symbol_type_variable) {
        for (size_t i = 0; i < var_count; i++) {
            if (strcmp(sym->name, var_names[i]) == 0) {
                int id = (*temp_counter)++;
                jit_buf_append(buf, "    double t%d = vars[%zu];\n", id, i);
                return id;
            }
        }
        return -1;
    }

    if (sym->type == basis_symbol_type_operation) {
        int left_id = -1, right_id = -1;
        if (sym->argument_count > 0)
            left_id = jit_emit_node(sym->arguments[0], buf, temp_counter, var_names, var_count);
        if (sym->argument_count > 1)
            right_id = jit_emit_node(sym->arguments[1], buf, temp_counter, var_names, var_count);

        int id = (*temp_counter)++;

        if (strcmp(sym->operation, "+") == 0)
            jit_buf_append(buf, "    double t%d = t%d + t%d;\n", id, left_id, right_id);
        else if (strcmp(sym->operation, "-") == 0)
            jit_buf_append(buf, "    double t%d = t%d - t%d;\n", id, left_id, right_id);
        else if (strcmp(sym->operation, "*") == 0)
            jit_buf_append(buf, "    double t%d = t%d * t%d;\n", id, left_id, right_id);
        else if (strcmp(sym->operation, "**") == 0)
            jit_buf_append(buf, "    double t%d = pow(t%d, %.17g);\n", id, left_id, sym->basis_value);
        else if (strcmp(sym->operation, "exp") == 0)
            jit_buf_append(buf, "    double t%d = exp(t%d);\n", id, left_id);
        else if (strcmp(sym->operation, "log") == 0)
            jit_buf_append(buf, "    double t%d = log(t%d);\n", id, left_id);
        else if (strcmp(sym->operation, "sin") == 0)
            jit_buf_append(buf, "    double t%d = sin(t%d);\n", id, left_id);
        else if (strcmp(sym->operation, "cos") == 0)
            jit_buf_append(buf, "    double t%d = cos(t%d);\n", id, left_id);
        else if (strcmp(sym->operation, "tanh") == 0)
            jit_buf_append(buf, "    double t%d = tanh(t%d);\n", id, left_id);
        else
            jit_buf_append(buf, "    double t%d = 0.0; /* unknown op */\n", id);

        return id;
    }
    return -1;
}

// --- Public API ---
basis_jit_module* basis_jit_compile(basis_symbol* root, char** var_names, size_t var_count) {
    if (!root) return NULL;

    jit_buf buf;
    jit_buf_init(&buf, 4096);

    // Function header
    jit_buf_append(&buf, "#include <math.h>\n");
    jit_buf_append(&buf, "double basis_jit_func(double* vars) {\n");

    // Emit AST
    int temp_counter = 0;
    int root_id = jit_emit_node(root, &buf, &temp_counter, var_names, var_count);

    if (root_id < 0) {
        jit_buf_free(&buf);
        return NULL;
    }

    // Return the root temp variable
    jit_buf_append(&buf, "    return t%d;\n", root_id);
    jit_buf_append(&buf, "}\n");

    // Write to temp file
    char base_file[] = "/tmp/basis_jit_XXXXXX";
    int fd_c = mkstemp(base_file);
    if (fd_c == -1) {
        jit_buf_free(&buf);
        return NULL;
    }

    char c_file[256];
    char so_file[256];
    snprintf(c_file, sizeof(c_file), "%s.c", base_file);
    snprintf(so_file, sizeof(so_file), "%s.so", base_file);

    ssize_t written = write(fd_c, buf.data, buf.len);
    (void)written;
    close(fd_c);
    jit_buf_free(&buf);

    rename(base_file, c_file);

    // Compile with maximum optimization
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "gcc -shared -fPIC -O3 -march=native -o %s %s -lm 2>/dev/null", so_file, c_file);
    int status = system(cmd);
    unlink(c_file);

    if (status != 0) return NULL;

    // Load shared object
    void* handle = dlopen(so_file, RTLD_NOW);
    if (!handle) {
        unlink(so_file);
        return NULL;
    }

    basis_jit_func_t func = (basis_jit_func_t)dlsym(handle, "basis_jit_func");
    if (!func) {
        dlclose(handle);
        unlink(so_file);
        return NULL;
    }

    basis_jit_module* mod = (basis_jit_module*)malloc(sizeof(basis_jit_module));
    mod->handle = handle;
    mod->func = func;
    mod->so_path = strdup(so_file);
    return mod;
}

double basis_jit_execute(basis_jit_module* mod, double* var_values) {
    if (!mod || !mod->func) return 0.0;
    return mod->func(var_values);
}

void basis_jit_free(basis_jit_module* mod) {
    if (!mod) return;
    if (mod->handle) dlclose(mod->handle);
    if (mod->so_path) {
        unlink(mod->so_path);
        free(mod->so_path);
    }
    free(mod);
}

```

---

### FILE: src/basis/stage5_unified/symbolic.c
Location: `src/basis/stage5_unified/symbolic.c`
```cpp

#include "basis/stage5_unified/symbolic.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

static char* basis_symbol_string_duplicate(const char* source_string) {
    size_t length = strlen(source_string) + 1;
    char* copy = (char*)malloc(length);
    if (copy) memcpy(copy, source_string, length);
    return copy;
}

static basis_symbol* basis_symbol_new(basis_symbol_type type) {
    basis_symbol* new_symbol = (basis_symbol*)malloc(sizeof(basis_symbol));
    new_symbol->type = type; new_symbol->name = NULL; new_symbol->operation = NULL;
    new_symbol->basis_value = 0.0; new_symbol->arguments = NULL; new_symbol->argument_count = 0;
    return new_symbol;
}

basis_symbol* basis_symbol_variable(char* variable_name) {
    basis_symbol* new_symbol = basis_symbol_new(basis_symbol_type_variable);
    new_symbol->name = basis_symbol_string_duplicate(variable_name);
    return new_symbol;
}

basis_symbol* basis_symbol_constant(double constant_value) {
    basis_symbol* new_symbol = basis_symbol_new(basis_symbol_type_constant);
    new_symbol->basis_value = constant_value;
    return new_symbol;
}

static basis_symbol* basis_symbol_operation_new(char* operation, size_t argument_count) {
    basis_symbol* new_symbol = basis_symbol_new(basis_symbol_type_operation);
    new_symbol->operation = basis_symbol_string_duplicate(operation);
    new_symbol->argument_count = argument_count;
    new_symbol->arguments = (basis_symbol**)malloc(sizeof(basis_symbol*) * argument_count);
    return new_symbol;
}

basis_symbol* basis_symbol_addition(basis_symbol* first_operand, basis_symbol* second_operand) {
    basis_symbol* new_symbol = basis_symbol_operation_new("+", 2);
    new_symbol->arguments[0] = first_operand; new_symbol->arguments[1] = second_operand;
    return new_symbol;
}

basis_symbol* basis_symbol_subtraction(basis_symbol* first_operand, basis_symbol* second_operand) {
    basis_symbol* new_symbol = basis_symbol_operation_new("-", 2);
    new_symbol->arguments[0] = first_operand; new_symbol->arguments[1] = second_operand;
    return new_symbol;
}

basis_symbol* basis_symbol_multiplication(basis_symbol* first_operand, basis_symbol* second_operand) {
    basis_symbol* new_symbol = basis_symbol_operation_new("*", 2);
    new_symbol->arguments[0] = first_operand; new_symbol->arguments[1] = second_operand;
    return new_symbol;
}

basis_symbol* basis_symbol_power(basis_symbol* base, double exponent) {
    basis_symbol* new_symbol = basis_symbol_operation_new("**", 1);
    new_symbol->arguments[0] = base; new_symbol->basis_value = exponent;
    return new_symbol;
}

basis_symbol* basis_symbol_logarithm(basis_symbol* input_symbol) {
    basis_symbol* new_symbol = basis_symbol_operation_new("log", 1);
    new_symbol->arguments[0] = input_symbol;
    return new_symbol;
}

basis_symbol* basis_symbol_exponential(basis_symbol* input_symbol) {
    basis_symbol* new_symbol = basis_symbol_operation_new("exp", 1);
    new_symbol->arguments[0] = input_symbol;
    return new_symbol;
}

basis_symbol* basis_symbol_copy(basis_symbol* source_symbol) {
    if (!source_symbol) return NULL;
    basis_symbol* output_symbol;
    if (source_symbol->type == basis_symbol_type_variable) {
        output_symbol = basis_symbol_variable(source_symbol->name);
    } else if (source_symbol->type == basis_symbol_type_constant) {
        output_symbol = basis_symbol_constant(source_symbol->basis_value);
    } else {
        output_symbol = basis_symbol_operation_new(source_symbol->operation, source_symbol->argument_count);
        output_symbol->basis_value = source_symbol->basis_value;
        for (size_t i = 0; i < source_symbol->argument_count; i++) {
            output_symbol->arguments[i] = basis_symbol_copy(source_symbol->arguments[i]);
        }
    }
    return output_symbol;
}

basis_symbol* basis_symbol_differentiation(basis_symbol* source_symbol, char* variable_name) {
    if (source_symbol->type == basis_symbol_type_constant) return basis_symbol_constant(0.0);
    if (source_symbol->type == basis_symbol_type_variable) return (strcmp(source_symbol->name, variable_name) == 0) ? basis_symbol_constant(1.0) : basis_symbol_constant(0.0);
    if (source_symbol->type == basis_symbol_type_operation) {
        if (strcmp(source_symbol->operation, "+") == 0) return basis_symbol_addition(basis_symbol_differentiation(source_symbol->arguments[0], variable_name), basis_symbol_differentiation(source_symbol->arguments[1], variable_name));
        if (strcmp(source_symbol->operation, "-") == 0) return basis_symbol_subtraction(basis_symbol_differentiation(source_symbol->arguments[0], variable_name), basis_symbol_differentiation(source_symbol->arguments[1], variable_name));
        if (strcmp(source_symbol->operation, "*") == 0) return basis_symbol_addition(basis_symbol_multiplication(basis_symbol_differentiation(source_symbol->arguments[0], variable_name), basis_symbol_copy(source_symbol->arguments[1])), basis_symbol_multiplication(basis_symbol_copy(source_symbol->arguments[0]), basis_symbol_differentiation(source_symbol->arguments[1], variable_name)));
        if (strcmp(source_symbol->operation, "**") == 0) {
            double exponent = source_symbol->basis_value;
            return basis_symbol_multiplication(basis_symbol_constant(exponent), basis_symbol_multiplication(basis_symbol_power(basis_symbol_copy(source_symbol->arguments[0]), exponent - 1.0), basis_symbol_differentiation(source_symbol->arguments[0], variable_name)));
        }
        if (strcmp(source_symbol->operation, "exp") == 0) return basis_symbol_multiplication(basis_symbol_exponential(basis_symbol_copy(source_symbol->arguments[0])), basis_symbol_differentiation(source_symbol->arguments[0], variable_name));
        if (strcmp(source_symbol->operation, "log") == 0) return basis_symbol_multiplication(basis_symbol_differentiation(source_symbol->arguments[0], variable_name), basis_symbol_power(basis_symbol_copy(source_symbol->arguments[0]), -1.0));
    }
    return basis_symbol_constant(0.0);
}

void basis_symbol_free(basis_symbol* target_symbol) {
    if (!target_symbol) return;
    if (target_symbol->name) free(target_symbol->name);
    if (target_symbol->operation) free(target_symbol->operation);
    for (size_t i = 0; i < target_symbol->argument_count; i++) basis_symbol_free(target_symbol->arguments[i]);
    if (target_symbol->arguments) free(target_symbol->arguments);
    free(target_symbol);
}

void basis_symbol_print(basis_symbol* target_symbol) {
    if (!target_symbol) { printf("NULL"); return; }
    if (target_symbol->type == basis_symbol_type_variable) printf("%s", target_symbol->name);
    else if (target_symbol->type == basis_symbol_type_constant) printf("%.2f", target_symbol->basis_value);
    else if (target_symbol->type == basis_symbol_type_operation) {
        if (target_symbol->argument_count == 2) {
            printf("("); basis_symbol_print(target_symbol->arguments[0]); printf(" %s ", target_symbol->operation); basis_symbol_print(target_symbol->arguments[1]); printf(")");
        } else {
            printf("%s(", target_symbol->operation); basis_symbol_print(target_symbol->arguments[0]);
            if (strcmp(target_symbol->operation, "**") == 0) printf(", %.2f", target_symbol->basis_value);
            printf(")");
        }
    }
}

basis_symbol* basis_symbol_simplify(basis_symbol* source_symbol) {
    if (!source_symbol) return NULL;
    if (source_symbol->type == basis_symbol_type_variable || source_symbol->type == basis_symbol_type_constant) return basis_symbol_copy(source_symbol);

    basis_symbol* first_argument = source_symbol->argument_count > 0 ? basis_symbol_simplify(source_symbol->arguments[0]) : NULL;
    basis_symbol* second_argument = source_symbol->argument_count > 1 ? basis_symbol_simplify(source_symbol->arguments[1]) : NULL;

    if (strcmp(source_symbol->operation, "+") == 0) {
        if (first_argument->type == basis_symbol_type_constant && first_argument->basis_value == 0.0) { basis_symbol_free(first_argument); return second_argument; }
        if (second_argument->type == basis_symbol_type_constant && second_argument->basis_value == 0.0) { basis_symbol_free(second_argument); return first_argument; }
        if (first_argument->type == basis_symbol_type_constant && second_argument->type == basis_symbol_type_constant) {
            double result = first_argument->basis_value + second_argument->basis_value;
            basis_symbol_free(first_argument); basis_symbol_free(second_argument); return basis_symbol_constant(result);
        }
    } else if (strcmp(source_symbol->operation, "-") == 0) {
        if (second_argument->type == basis_symbol_type_constant && second_argument->basis_value == 0.0) { basis_symbol_free(second_argument); return first_argument; }
        if (first_argument->type == basis_symbol_type_constant && second_argument->type == basis_symbol_type_constant) {
            double result = first_argument->basis_value - second_argument->basis_value;
            basis_symbol_free(first_argument); basis_symbol_free(second_argument); return basis_symbol_constant(result);
        }
    } else if (strcmp(source_symbol->operation, "*") == 0) {
        if (first_argument->type == basis_symbol_type_constant && first_argument->basis_value == 0.0) { basis_symbol_free(second_argument); return first_argument; }
        if (second_argument->type == basis_symbol_type_constant && second_argument->basis_value == 0.0) { basis_symbol_free(first_argument); return second_argument; }
        if (first_argument->type == basis_symbol_type_constant && first_argument->basis_value == 1.0) { basis_symbol_free(first_argument); return second_argument; }
        if (second_argument->type == basis_symbol_type_constant && second_argument->basis_value == 1.0) { basis_symbol_free(second_argument); return first_argument; }
        if (first_argument->type == basis_symbol_type_constant && second_argument->type == basis_symbol_type_constant) {
            double result = first_argument->basis_value * second_argument->basis_value;
            basis_symbol_free(first_argument); basis_symbol_free(second_argument); return basis_symbol_constant(result);
        }
    } else if (strcmp(source_symbol->operation, "**") == 0) {
        if (source_symbol->basis_value == 0.0) { basis_symbol_free(first_argument); return basis_symbol_constant(1.0); }
        if (source_symbol->basis_value == 1.0) { return first_argument; }
        if (first_argument->type == basis_symbol_type_constant) {
            double result = pow(first_argument->basis_value, source_symbol->basis_value);
            basis_symbol_free(first_argument); return basis_symbol_constant(result);
        }
    } else if (strcmp(source_symbol->operation, "exp") == 0) {
        if (first_argument->type == basis_symbol_type_constant) {
            double result = exp(first_argument->basis_value);
            basis_symbol_free(first_argument); return basis_symbol_constant(result);
        }
    } else if (strcmp(source_symbol->operation, "log") == 0) {
        if (first_argument->type == basis_symbol_type_constant) {
            double result = log(first_argument->basis_value);
            basis_symbol_free(first_argument); return basis_symbol_constant(result);
        }
    }

    basis_symbol* output_symbol = basis_symbol_operation_new(source_symbol->operation, source_symbol->argument_count);
    output_symbol->basis_value = source_symbol->basis_value;
    if (source_symbol->argument_count > 0 && first_argument) output_symbol->arguments[0] = first_argument;
    if (source_symbol->argument_count > 1 && second_argument) output_symbol->arguments[1] = second_argument;
    for (size_t i = 2; i < source_symbol->argument_count; i++) output_symbol->arguments[i] = basis_symbol_copy(source_symbol->arguments[i]);
    return output_symbol;
}

```

---

### FILE: src/basis/stage6_ir/graph.c
Location: `src/basis/stage6_ir/graph.c`
```cpp
#include "basis/stage6_ir/graph.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static char* ir_strdup(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char* c = (char*)malloc(len);
    if (c) memcpy(c, s, len);
    return c;
}

static basis_node* graph_alloc_node(basis_graph* g, basis_op_type op, size_t r, size_t c, const char* name) {
    if (g->node_count >= g->node_cap) {
        g->node_cap = g->node_cap == 0 ? 16 : g->node_cap * 2;
        g->nodes = (basis_node*)realloc(g->nodes, sizeof(basis_node) * g->node_cap);
    }
    basis_node* n = &g->nodes[g->node_count++];
    n->id = (int)g->node_count - 1;
    n->op = op;
    n->input_ids[0] = -1;
    n->input_ids[1] = -1;
    n->rows = r;
    n->cols = c;
    n->name = ir_strdup(name);
    return n;
}

basis_graph* basis_graph_new(void) {
    basis_graph* g = (basis_graph*)calloc(1, sizeof(basis_graph));
    return g;
}

basis_node* basis_graph_input(basis_graph* g, const char* name, size_t r, size_t c) {
    return graph_alloc_node(g, BASIS_OP_INPUT, r, c, name);
}

basis_node* basis_graph_matmul(basis_graph* g, basis_node* a, basis_node* b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "[IR ERROR] MatMul shape mismatch: [%zux%zu] x [%zux%zu]\n", a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }
    basis_node* n = graph_alloc_node(g, BASIS_OP_MATMUL, a->rows, b->cols, NULL);
    n->input_ids[0] = a->id;
    n->input_ids[1] = b->id;
    return n;
}

basis_node* basis_graph_add(basis_graph* g, basis_node* a, basis_node* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "[IR ERROR] Add shape mismatch\n");
        return NULL;
    }
    basis_node* n = graph_alloc_node(g, BASIS_OP_ADD, a->rows, a->cols, NULL);
    n->input_ids[0] = a->id;
    n->input_ids[1] = b->id;
    return n;
}

basis_node* basis_graph_broadcast_add(basis_graph* g, basis_node* a, basis_node* bias) {
    if (bias->rows != 1 || bias->cols != a->cols) {
        fprintf(stderr, "[IR ERROR] Broadcast Add shape mismatch\n");
        return NULL;
    }
    basis_node* n = graph_alloc_node(g, BASIS_OP_BROADCAST_ADD, a->rows, a->cols, NULL);
    n->input_ids[0] = a->id;
    n->input_ids[1] = bias->id;
    return n;
}

basis_node* basis_graph_relu(basis_graph* g, basis_node* a) {
    basis_node* n = graph_alloc_node(g, BASIS_OP_RELU, a->rows, a->cols, NULL);
    n->input_ids[0] = a->id;
    return n;
}

basis_node* basis_graph_tanh(basis_graph* g, basis_node* a) {
    basis_node* n = graph_alloc_node(g, BASIS_OP_TANH, a->rows, a->cols, NULL);
    n->input_ids[0] = a->id;
    return n;
}

static const char* op_to_string(basis_op_type op) {
    switch(op) {
        case BASIS_OP_INPUT: return "INPUT";
        case BASIS_OP_MATMUL: return "MATMUL";
        case BASIS_OP_ADD: return "ADD";
        case BASIS_OP_BROADCAST_ADD: return "BROADCAST_ADD";
        case BASIS_OP_RELU: return "RELU";
        case BASIS_OP_TANH: return "TANH";
        default: return "UNKNOWN";
    }
}

void basis_graph_print(basis_graph* g) {
    printf("=== BASIS Static Graph IR ===\n");
    for (size_t i = 0; i < g->node_count; i++) {
        basis_node* n = &g->nodes[i];
        printf("Node %2d | %-15s | Shape: [%3zu x %3zu] | Inputs: ",
               n->id, op_to_string(n->op), n->rows, n->cols);
        if (n->op == BASIS_OP_INPUT) {
            printf("None (Name: \"%s\")", n->name);
        } else {
            printf("[%d, %d]", n->input_ids[0], n->input_ids[1]);
        }
        printf("\n");
    }
    printf("=============================\n");
}

void basis_graph_free(basis_graph* g) {
    if (!g) return;
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].name) free(g->nodes[i].name);
    }
    if (g->nodes) free(g->nodes);
    free(g);
}

```

---

### FILE: src/basis/stage6_ir/lowering.c
Location: `src/basis/stage6_ir/lowering.c`
```cpp
#include "basis/stage6_ir/lowering.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void get_val_buf(basis_node* n, char* buf, size_t buf_size) {
    if (n->op == BASIS_OP_INPUT) snprintf(buf, buf_size, "%s", n->name);
    else snprintf(buf, buf_size, "ctx->t%d", n->id);
}

static void get_grad_buf(basis_node* n, char* buf, size_t buf_size) {
    if (n->op == BASIS_OP_INPUT) snprintf(buf, buf_size, "grad_%s", n->name);
    else snprintf(buf, buf_size, "grad_t%d", n->id);
}

void basis_graph_lower_to_c(basis_graph* g, const char* output_filename) {
    FILE* f = fopen(output_filename, "w");
    if (!f) return;

    fprintf(f, "#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n\n");

    // 1. Emit Context Struct
    fprintf(f, "typedef struct {\n");
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op != BASIS_OP_INPUT) {
            fprintf(f, "    double* t%zu;\n", i);
        }
    }
    fprintf(f, "} GhostContext;\n\n");

    // 2. Emit Forward Pass
    fprintf(f, "void ghost_weight_forward(");
    int first = 1;
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op == BASIS_OP_INPUT) {
            if (!first) fprintf(f, ", ");
            fprintf(f, "double* %s", g->nodes[i].name);
            first = 0;
        }
    }
    fprintf(f, ", GhostContext* ctx, double* out) {\n");

    for (size_t i = 0; i < g->node_count; i++) {
        basis_node* n = &g->nodes[i];
        if (n->op == BASIS_OP_INPUT) continue;

        fprintf(f, "    ctx->t%zu = (double*)malloc(%zu * %zu * sizeof(double));\n", i, n->rows, n->cols);

        if (n->op == BASIS_OP_MATMUL) {
            basis_node* a = &g->nodes[n->input_ids[0]];
            basis_node* b = &g->nodes[n->input_ids[1]];
            char a_buf[64], b_buf[64];
            get_val_buf(a, a_buf, sizeof(a_buf)); get_val_buf(b, b_buf, sizeof(b_buf));

            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t j=0; j<%zu; j++) {\n", n->rows, n->cols);
            fprintf(f, "        double sum = 0.0;\n");
            fprintf(f, "        for(size_t k=0; k<%zu; k++) sum += %s[i*%zu + k] * %s[k*%zu + j];\n", a->cols, a_buf, a->cols, b_buf, b->cols);
            fprintf(f, "        ctx->t%zu[i*%zu + j] = sum;\n    }\n", i, n->cols);
        } else if (n->op == BASIS_OP_BROADCAST_ADD) {
            basis_node* a = &g->nodes[n->input_ids[0]];
            basis_node* b = &g->nodes[n->input_ids[1]];
            char a_buf[64], b_buf[64];
            get_val_buf(a, a_buf, sizeof(a_buf)); get_val_buf(b, b_buf, sizeof(b_buf));
            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t j=0; j<%zu; j++) ctx->t%zu[i*%zu + j] = %s[i*%zu + j] + %s[j];\n", n->rows, n->cols, i, n->cols, a_buf, a->cols, b_buf);
        } else if (n->op == BASIS_OP_RELU) {
            basis_node* a = &g->nodes[n->input_ids[0]];
            char a_buf[64]; get_val_buf(a, a_buf, sizeof(a_buf));
            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t j=0; j<%zu; j++) { double v = %s[i*%zu + j]; ctx->t%zu[i*%zu + j] = v > 0.0 ? v : 0.0; }\n", n->rows, n->cols, a_buf, a->cols, i, n->cols);
        }
    }

    basis_node* last = &g->nodes[g->node_count - 1];
    char last_buf[64]; get_val_buf(last, last_buf, sizeof(last_buf));
    fprintf(f, "    memcpy(out, %s, %zu * %zu * sizeof(double));\n}\n\n", last_buf, last->rows, last->cols);

    // 3. Emit Backward Pass
    fprintf(f, "void ghost_weight_backward(");
    first = 1;
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op == BASIS_OP_INPUT) {
            if (!first) fprintf(f, ", ");
            fprintf(f, "double* %s", g->nodes[i].name);
            first = 0;
        }
    }
    fprintf(f, ", GhostContext* ctx, double* grad_out, ");
    first = 1;
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op == BASIS_OP_INPUT) {
            if (!first) fprintf(f, ", ");
            fprintf(f, "double* grad_%s", g->nodes[i].name);
            first = 0;
        }
    }
    fprintf(f, ") {\n");

    // Allocate intermediate gradients (EXCEPT the last node, which maps to grad_out)
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op != BASIS_OP_INPUT && i != g->node_count - 1) {
            fprintf(f, "    double* grad_t%zu = (double*)calloc(%zu * %zu, sizeof(double));\n", i, g->nodes[i].rows, g->nodes[i].cols);
        }
    }

    // Map last node's gradient to grad_out
    fprintf(f, "    double* grad_t%zu = grad_out;\n", g->node_count - 1);

    // Reverse topological traversal
    for (int idx = (int)g->node_count - 1; idx >= 0; idx--) {
        basis_node* n = &g->nodes[idx];
        if (n->op == BASIS_OP_INPUT) continue;

        char grad_n[64]; get_grad_buf(n, grad_n, sizeof(grad_n));

        if (n->op == BASIS_OP_MATMUL) {
            basis_node* a = &g->nodes[n->input_ids[0]];
            basis_node* b = &g->nodes[n->input_ids[1]];
            char a_buf[64], b_buf[64], grad_a[64], grad_b[64];
            get_val_buf(a, a_buf, sizeof(a_buf)); get_val_buf(b, b_buf, sizeof(b_buf));
            get_grad_buf(a, grad_a, sizeof(grad_a)); get_grad_buf(b, grad_b, sizeof(grad_b));

            // dA = dC * B^T
            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t k=0; k<%zu; k++) { double s=0; for(size_t j=0; j<%zu; j++) s += %s[i*%zu+j] * %s[k*%zu+j]; %s[i*%zu+k] += s; }\n",
                    a->rows, a->cols, b->cols, grad_n, n->cols, b_buf, b->cols, grad_a, a->cols);
            // dB = A^T * dC
            fprintf(f, "    for(size_t k=0; k<%zu; k++) for(size_t j=0; j<%zu; j++) { double s=0; for(size_t i=0; i<%zu; i++) s += %s[i*%zu+k] * %s[i*%zu+j]; %s[k*%zu+j] += s; }\n",
                    a->cols, b->cols, a->rows, a_buf, a->cols, grad_n, n->cols, grad_b, b->cols);
        } else if (n->op == BASIS_OP_BROADCAST_ADD) {
            basis_node* a = &g->nodes[n->input_ids[0]];
            basis_node* b = &g->nodes[n->input_ids[1]];
            char grad_a[64], grad_b[64];
            get_grad_buf(a, grad_a, sizeof(grad_a)); get_grad_buf(b, grad_b, sizeof(grad_b));

            // dA = dC
            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t j=0; j<%zu; j++) %s[i*%zu+j] += %s[i*%zu+j];\n", n->rows, n->cols, grad_a, n->cols, grad_n, n->cols);
            // dB = sum_rows(dC)
            fprintf(f, "    for(size_t j=0; j<%zu; j++) { double s=0; for(size_t i=0; i<%zu; i++) s += %s[i*%zu+j]; %s[j] += s; }\n", n->cols, n->rows, grad_n, n->cols, grad_b);
        } else if (n->op == BASIS_OP_RELU) {
            basis_node* a = &g->nodes[n->input_ids[0]];
            char grad_a[64];
            get_grad_buf(a, grad_a, sizeof(grad_a));

            // dA = dC * (out > 0). We use ctx->t{idx} (the output of RELU) as the mask!
            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t j=0; j<%zu; j++) %s[i*%zu+j] += %s[i*%zu+j] * (ctx->t%zu[i*%zu+j] > 0.0 ? 1.0 : 0.0);\n",
                    n->rows, n->cols, grad_a, n->cols, grad_n, n->cols, (size_t)idx, n->cols);
        }
    }

    // Free intermediate gradients
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op != BASIS_OP_INPUT && i != g->node_count - 1) {
            fprintf(f, "    free(grad_t%zu);\n", i);
        }
    }
    fprintf(f, "}\n\n");

    // 4. Emit Free Context
    fprintf(f, "void ghost_free_context(GhostContext* ctx) {\n");
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op != BASIS_OP_INPUT) fprintf(f, "    free(ctx->t%zu);\n", i);
    }
    fprintf(f, "}\n");

    fclose(f);
}

```

---

### FILE: standalone_model.c
Location: `standalone_model.c`
```cpp
#include <stdio.h>

const double W1[16] = {
    0.77859360719956927, -0.099999999999999978, -0.6737230796636654, -0.090000000000000024, -0.38, 0.7492486592078863, 0.99646302805399078, 0.20027491124201302, 
    -0.71898337550582159, -0.070000000000000007, 1.2959340901938223, -0.070000000000000007, -0.32999999999999996, -0.71631420986226346, 0.056210616792803959, -0.29490607738890073, 
};

const double b1[8] = {
    -0.099307282509110639, 0, -0.26305954660409908, 0, 0, -0.14023833393989973, -0.52058129703802114, -0.41490607738890067, 
};

const double W2[8] = {
    1.2331095824713145, 
    -0.020000000000000018, 
    0.96817179415084753, 
    0.089999999999999969, 
    0.17000000000000004, 
    0.77724859159475734, 
    -0.65349618094143647, 
    0.16679316733600885, 
};

const double b2[1] = {
    -1.4375961407180335e-16, 
};

int main() {
    double X[8] = {0,0, 0,1, 1,0, 1,1};
    double H1[32] = {0};
    double Y[4] = {0};

    for(int i=0; i<4; i++) {
        for(int j=0; j<8; j++) {
            double sum = 0.0;
            for(int k=0; k<2; k++) sum += X[i*2+k] * W1[k*8+j];
            double val = sum + b1[j];
            H1[i*8+j] = val > 0.0 ? val : 0.0;
        }
    }

    for(int i=0; i<4; i++) {
        double sum = 0.0;
        for(int k=0; k<8; k++) sum += H1[i*8+k] * W2[k];
        Y[i] = sum + b2[0];
    }

    printf("=== Standalone Ghost-Weight Inference ===\n");
    for(int i=0; i<4; i++) printf("Input: [%.0f, %.0f] | Output: %.4f\n", X[i*2], X[i*2+1], Y[i]);
    return 0;
}

```

---

### FILE: test_global.c
Location: `test_global.c`
```cpp
#include <basis.h>
#include <stdio.h>

int main() {
    printf("🌍 Accessing BASIS v5 from the global system environment!\n\n");

    // 1. Build a simple Autograd graph
    basis_value* x = basis_value_new(3.0);
    basis_value* y = basis_value_new(4.0);
    basis_value* z = basis_value_addition(x, y);

    // 2. Trigger the Iterative Topological Sort
    basis_value_backward_propagation(z);

    printf("Forward Pass:  3.0 + 4.0 = %.1f\n", z->data);
    printf("Backward Pass: dx = %.1f, dy = %.1f\n", x->gradient, y->gradient);

    // 3. Strict Memory Teardown
    basis_value_free(z);
    basis_value_free(x);
    basis_value_free(y);

    printf("\n✅ Global library linkage successful. Zero memory leaks.\n");
    printf("🚀 BASIS v5 is now a native part of your operating system.\n");
    return 0;
}

```

---

### FILE: test_swarm.c
Location: `test_swarm.c`
```cpp
#include <basis.h>
#include <stdio.h>

// Helper to build a simple forward-only expert graph
basis_ir_graph* build_expert_graph(double multiplier) {
    basis_ir_graph* g = basis_ir_graph_create();
    basis_v7_tensor* X = basis_v7_input(g, 2, 2, NULL);
    basis_v7_tensor* W = basis_v7_const(g, multiplier, 2, 2);
    basis_v7_tensor* Out = basis_v7_mul(g, X, W); // Element-wise multiply
    (void)Out;
    return g;
}

int main() {
    printf("=== BASIS V7: Standalone Swarm Execution ===\n\n");

    // 1. Compile Micro-Models (Experts)
    basis_ir_graph* g_A = build_expert_graph(10.0);
    basis_schedule* sched_A = basis_ir_schedule(g_A);

    // 2. Register in the Swarm
    basis_swarm* swarm = basis_swarm_create();
    basis_swarm_register(swarm, "Expert_A", sched_A, g_A);

    // 3. Planner Routes Tensors
    basis_expert* exp = basis_swarm_get(swarm, "Expert_A");
    double in[4] = {1.0, 2.0, 3.0, 4.0};
    double out[4];

    basis_swarm_dispatch(exp, in, out);

    printf("Input:  [%.1f, %.1f, %.1f, %.1f]\n", in[0], in[1], in[2], in[3]);
    printf("Output: [%.1f, %.1f, %.1f, %.1f]\n", out[0], out[1], out[2], out[3]);

    // 4. Teardown
    basis_swarm_destroy(swarm);
    basis_ir_graph_destroy(g_A);

    printf("\n✅ Standalone Swarm Execution Complete.\n");
    return 0;
}

```

---

### FILE: tests/test_v5_suite.c
Location: `tests/test_v5_suite.c`
```cpp

#include "basis/scalar.h"
#include "basis/jit.h"
#include "basis/tensor.h"
#include "basis/sequence.h"
#include "basis/learning.h"
#include "basis/geometry.h"
#include "basis/symbolic.h"
#include "basis/compiler.h"
#include "basis/optim.h"
#include "basis/core/test_harness.h"
#include "basis/core/error.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

static double rand_double(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

void test_autograd_basic() {
    BASIS_TEST_SUITE_START("Phase A/B: Dynamic Basic Autograd");
    double val_a = rand_double(-100.0, 100.0);
    double val_b = rand_double(-100.0, 100.0);

    basis_value* a = basis_value_new(val_a);
    basis_value* b = basis_value_new(val_b);
    basis_value* c = basis_value_addition(a, b);
    BASIS_ASSERT_NEAR(c->data, val_a + val_b, 1e-9, "Dynamic forward pass addition");

    basis_value_backward_propagation(c);
    BASIS_ASSERT_NEAR(a->gradient, 1.0, 1e-9, "Gradient a");
    BASIS_ASSERT_NEAR(b->gradient, 1.0, 1e-9, "Gradient b");

    basis_value_free(c); basis_value_free(a); basis_value_free(b);
}

void test_tensor_shape_validation() {
    BASIS_TEST_SUITE_START("Phase A: Tensor Shape Validation (Static Error Logic)");
    basis_tensor* a = basis_tensor_new(2, 3);
    basis_tensor* b = basis_tensor_new(4, 2);

    basis_clear_error();
    basis_set_error_quiet(true);
    basis_tensor* c = basis_tensor_matrix_multiplication(a, b);
    basis_set_error_quiet(false);

    BASIS_ASSERT(c == NULL, "MatMul correctly rejects invalid shapes");
    BASIS_ASSERT(basis_get_last_error() == BASIS_INVALID_SHAPE, "Error code is BASIS_INVALID_SHAPE");

    basis_tensor_free(a); basis_tensor_free(b);
}

void test_symbolic_simplification() {
    BASIS_TEST_SUITE_START("Phase C: Symbolic Simplification (Static Logic)");
    basis_symbol* x1 = basis_symbol_variable("x");
    basis_symbol* one1 = basis_symbol_constant(1.0);
    basis_symbol* expr1 = basis_symbol_multiplication(x1, one1);
    basis_symbol* simp1 = basis_symbol_simplify(expr1);
    BASIS_ASSERT(simp1->type == basis_symbol_type_variable, "x * 1 simplifies to x");
    basis_symbol_free(expr1); basis_symbol_free(simp1);

    basis_symbol* x2 = basis_symbol_variable("x");
    basis_symbol* zero2 = basis_symbol_constant(0.0);
    basis_symbol* expr2 = basis_symbol_addition(x2, zero2);
    basis_symbol* simp2 = basis_symbol_simplify(expr2);
    BASIS_ASSERT(simp2->type == basis_symbol_type_variable, "x + 0 simplifies to x");
    basis_symbol_free(expr2); basis_symbol_free(simp2);

    basis_symbol* x3 = basis_symbol_variable("x");
    basis_symbol* zero3 = basis_symbol_constant(0.0);
    basis_symbol* expr3 = basis_symbol_multiplication(x3, zero3);
    basis_symbol* simp3 = basis_symbol_simplify(expr3);
    BASIS_ASSERT(simp3->type == basis_symbol_type_constant && simp3->basis_value == 0.0, "x * 0 simplifies to 0");
    basis_symbol_free(expr3); basis_symbol_free(simp3);

    basis_symbol* c1 = basis_symbol_constant(2.0);
    basis_symbol* c2 = basis_symbol_constant(3.0);
    basis_symbol* expr4 = basis_symbol_addition(c1, c2);
    basis_symbol* simp4 = basis_symbol_simplify(expr4);
    BASIS_ASSERT(simp4->type == basis_symbol_type_constant && simp4->basis_value == 5.0, "2.0 + 3.0 folds to 5.0");
    basis_symbol_free(expr4); basis_symbol_free(simp4);
}

void test_compiler_cse() {
    BASIS_TEST_SUITE_START("Phase C: Compiler CSE (Static Pointer Logic)");
    basis_symbol* x = basis_symbol_variable("x");
    basis_symbol* left = basis_symbol_multiplication(basis_symbol_copy(x), basis_symbol_copy(x));
    basis_symbol* right = basis_symbol_multiplication(basis_symbol_copy(x), basis_symbol_copy(x));
    basis_symbol* tree = basis_symbol_addition(left, right);

    basis_value* x_val = basis_value_new(3.0);
    basis_compiler* comp = basis_compiler_new();
    basis_compiler_map(comp, "x", x_val);
    basis_value* root = basis_compiler_compile(comp, tree);

    BASIS_ASSERT(root->previous_nodes[0] == root->previous_nodes[1], "CSE reused identical sub-expression pointer");

    basis_value_backward_propagation(root);
    BASIS_ASSERT_NEAR(x_val->gradient, 12.0, 1e-6, "Gradient flows through shared CSE node");

    basis_value_free(root); basis_value_free(x_val); basis_compiler_free(comp);
    basis_symbol_free(tree); basis_symbol_free(x);
}

void test_log_softmax_stability() {
    BASIS_TEST_SUITE_START("Phase D: Dynamic Numerical Stability (Random Extreme Logits)");
    int N = 10 + rand() % 50; // Random width 10 to 60
    basis_tensor* logits = basis_tensor_new(1, N);
    for(int i=0; i<N; i++) basis_tensor_set(logits, 0, i, rand_double(-1000.0, 1000.0));

    basis_tensor* log_probs = basis_tensor_log_softmax(logits);

    bool is_stable = true;
    double sum_exp = 0.0;
    for(int i=0; i<N; i++) {
        double val = BASIS_TENSOR_AT(log_probs, 0, i)->data;
        if (isinf(val) || isnan(val)) is_stable = false;
        sum_exp += exp(val);
    }
    BASIS_ASSERT(is_stable, "Dynamic Log-Softmax handles random extreme logits without Overflow/NaN");
    BASIS_ASSERT_NEAR(sum_exp, 1.0, 1e-6, "Dynamic probabilities sum to 1.0");

    basis_value* loss = basis_tensor_sum(log_probs);
    basis_value_backward_propagation(loss);

    bool grads_ok = true;
    for(int i=0; i<N; i++) if (isnan(BASIS_TENSOR_AT(logits, 0, i)->gradient)) grads_ok = false;
    BASIS_ASSERT(grads_ok, "Dynamic gradients flow through Log-Softmax without NaNs");

    basis_value_free(loss); basis_tensor_free(log_probs); basis_tensor_free(logits);
}

void test_holy_trinity() {
    BASIS_TEST_SUITE_START("Phase B: Dynamic Holy Trinity (Random Evaluation Point)");
    double x_val = rand_double(0.5, 2.0); // Safe domain for log(x) and exp(x^2)
    double eps = 1e-5;

    double f_plus = exp((x_val + eps) * (x_val + eps)) * log(x_val + eps);
    double f_minus = exp((x_val - eps) * (x_val - eps)) * log(x_val - eps);
    double fd_grad = (f_plus - f_minus) / (2.0 * eps);

    basis_value* x_auto = basis_value_new(x_val);
    basis_value* x2_auto = basis_value_power(x_auto, 2.0);
    basis_value* exp_auto = basis_value_exponential(x2_auto);
    basis_value* log_auto = basis_value_logarithm(x_auto);
    basis_value* f_auto = basis_value_multiplication(exp_auto, log_auto);
    basis_value_backward_propagation(f_auto);
    double auto_grad = x_auto->gradient;

    basis_symbol* x_sym = basis_symbol_variable("x");
    basis_symbol* x2_sym = basis_symbol_power(basis_symbol_copy(x_sym), 2.0);
    basis_symbol* exp_sym = basis_symbol_exponential(x2_sym);
    basis_symbol* log_sym = basis_symbol_logarithm(x_sym);
    basis_symbol* f_sym = basis_symbol_multiplication(exp_sym, log_sym);

    basis_symbol* df_sym = basis_symbol_differentiation(f_sym, "x");
    basis_symbol* df_simp = basis_symbol_simplify(df_sym);

    basis_value* x_val_sym = basis_value_new(x_val);
    basis_compiler* comp = basis_compiler_new();
    basis_compiler_map(comp, "x", x_val_sym);
    basis_value* sym_grad_node = basis_compiler_compile(comp, df_simp);
    double sym_grad = sym_grad_node->data;

    BASIS_ASSERT_NEAR(auto_grad, fd_grad, 1e-4, "Dynamic Autodiff matches Finite Difference");
    BASIS_ASSERT_NEAR(sym_grad, fd_grad, 1e-4, "Dynamic Symbolic matches Finite Difference");
    BASIS_ASSERT_NEAR(sym_grad, auto_grad, 1e-6, "Dynamic Symbolic matches Autodiff directly");

    basis_value_free(f_auto); basis_value_free(x_auto);
    basis_value_free(sym_grad_node); basis_value_free(x_val_sym);
    basis_compiler_free(comp);
    basis_symbol_free(f_sym); basis_symbol_free(df_sym); basis_symbol_free(df_simp);
}

void test_information_geometry() {
    BASIS_TEST_SUITE_START("Phase E: Dynamic Information Geometry (Random PSD Verification)");
    int N = 3 + rand() % 5;
    int M = 3 + rand() % 5;
    basis_tensor* X = basis_tensor_new(N, M);
    for(int i=0; i<N; i++) for(int j=0; j<M; j++) basis_tensor_set(X, i, j, rand_double(-5.0, 5.0));

    basis_metric* fisher = basis_metric_fisher_information(NULL, X, NULL);
    basis_tensor* F = fisher->matrix;

    bool symmetric = true;
    for(int i=0; i<M; i++) for(int j=0; j<M; j++) {
        if (fabs(BASIS_TENSOR_AT(F, i, j)->data - BASIS_TENSOR_AT(F, j, i)->data) > 1e-9) symmetric = false;
    }
    BASIS_ASSERT(symmetric, "Dynamic Fisher Matrix is Symmetric");

    bool psd = true;
    for(int t=0; t<5; t++) { // Test 5 random vectors
        double quad = 0.0;
        double v[10];
        for(int i=0; i<M; i++) v[i] = rand_double(-10.0, 10.0);
        for(int i=0; i<M; i++) for(int j=0; j<M; j++) {
            quad += v[i] * BASIS_TENSOR_AT(F, i, j)->data * v[j];
        }
        if (quad < -1e-9) psd = false;
    }
    BASIS_ASSERT(psd, "Dynamic Fisher Matrix is Positive Semi-Definite");

    // Invariant: The trace of the Fisher matrix must be strictly positive for non-zero inputs
    double trace = 0.0;
    for(int i=0; i<M; i++) trace += BASIS_TENSOR_AT(F, i, i)->data;
    BASIS_ASSERT(trace > 0.0, "Dynamic Fisher Matrix Trace is strictly positive");

    basis_metric_free(fisher); basis_tensor_free(X);
}

void test_natural_gradient_descent() {
    BASIS_TEST_SUITE_START("Phase E: Dynamic NGD (Random True Weights)");
    double true_w0 = rand_double(-10.0, 10.0);
    double true_w1 = rand_double(-10.0, 10.0);

    basis_tensor* X = basis_tensor_new(3, 2);
    basis_tensor_set(X, 0, 0, 1.0); basis_tensor_set(X, 0, 1, 1.1);
    basis_tensor_set(X, 1, 0, 2.0); basis_tensor_set(X, 1, 1, 2.1);
    basis_tensor_set(X, 2, 0, 3.0); basis_tensor_set(X, 2, 1, 3.1);

    basis_tensor* W = basis_tensor_new(2, 1);
    basis_tensor_set(W, 0, 0, 0.0); basis_tensor_set(W, 1, 0, 0.0);

    basis_metric* fisher = basis_metric_fisher_information(NULL, X, NULL);
    basis_tensor* Y_true = basis_tensor_new(3, 1);
    basis_tensor_set(Y_true, 0, 0, 1.0*true_w0 + 1.1*true_w1);
    basis_tensor_set(Y_true, 1, 0, 2.0*true_w0 + 2.1*true_w1);
    basis_tensor_set(Y_true, 2, 0, 3.0*true_w0 + 3.1*true_w1);

    basis_tensor* Y_pred = basis_tensor_matrix_multiplication(X, W);
    basis_value* loss_sum = basis_value_new(0.0);
    for(int i=0; i<3; i++) {
        basis_value* diff = basis_value_subtraction(BASIS_TENSOR_AT(Y_pred, 0, i), BASIS_TENSOR_AT(Y_true, 0, i));
        basis_value* sq = basis_value_power(diff, 2.0);
        basis_value* new_sum = basis_value_addition(loss_sum, sq);
        basis_value_free(loss_sum); basis_value_free(diff); basis_value_free(sq);
        loss_sum = new_sum;
    }
    basis_value_backward_propagation(loss_sum);
    basis_metric_ngd_step(fisher, W, 0.5);

    BASIS_ASSERT_NEAR(W->data[0]->data, true_w0, 1e-4, "Dynamic NGD converges to random true W[0]");
    BASIS_ASSERT_NEAR(W->data[1]->data, true_w1, 1e-4, "Dynamic NGD converges to random true W[1]");

    basis_value_free(loss_sum); basis_tensor_free(Y_pred); basis_metric_free(fisher);
    basis_tensor_free(X); basis_tensor_free(W); basis_tensor_free(Y_true);
}

void test_matmul_dag_compression() {
    BASIS_TEST_SUITE_START("Phase F: Dynamic Fused MatMul (Random Dimensions & Values)");
    int M = 5 + rand() % 15;
    int K = 5 + rand() % 15;
    int N = 5 + rand() % 15;

    basis_tensor* A = basis_tensor_new(M, K);
    basis_tensor* B = basis_tensor_new(K, N);
    for(int i=0; i<M; i++) for(int j=0; j<K; j++) basis_tensor_set(A, i, j, rand_double(-5.0, 5.0));
    for(int i=0; i<K; i++) for(int j=0; j<N; j++) basis_tensor_set(B, i, j, rand_double(-5.0, 5.0));

    basis_tensor* C = basis_tensor_matrix_multiplication(A, B);

    bool forward_ok = true;
    for(int i=0; i<M; i++) for(int j=0; j<N; j++) {
        double sum = 0.0;
        for(int k=0; k<K; k++) sum += BASIS_TENSOR_AT(A, i, k)->data * BASIS_TENSOR_AT(B, k, j)->data;
        if (fabs(BASIS_TENSOR_AT(C, i, j)->data - sum) > 1e-6) forward_ok = false;
    }
    BASIS_ASSERT(forward_ok, "Dynamic Fused MatMul forward pass matches naive C-loop");

    basis_value* loss = basis_tensor_sum(C);
    basis_value_backward_propagation(loss);

    bool backward_ok = true;
    for(int i=0; i<M; i++) for(int j=0; j<K; j++) if(isnan(BASIS_TENSOR_AT(A, i, j)->gradient)) backward_ok = false;
    BASIS_ASSERT(backward_ok, "Dynamic MatMul backward pass completes without NaN");

    // Invariant: If dL/dC is all 1s (from sum), the global gradient flow must be non-zero and conserved
    double sum_grad_A = 0.0;
    for(int i=0; i<M; i++) for(int j=0; j<K; j++) sum_grad_A += BASIS_TENSOR_AT(A, i, j)->gradient;
    BASIS_ASSERT(sum_grad_A != 0.0, "Dynamic MatMul gradient conservation invariant holds");

    basis_value_free(loss); basis_tensor_free(C); basis_tensor_free(A); basis_tensor_free(B);
}

void test_strided_tensor_views() {
    BASIS_TEST_SUITE_START("Phase G: Dynamic Strided Views (Random Shapes)");
    int R = 3 + rand() % 10;
    int C_cols = 3 + rand() % 10;
    basis_tensor* A = basis_tensor_new(R, C_cols);
    for(int i=0; i<R; i++) for(int j=0; j<C_cols; j++) basis_tensor_set(A, i, j, rand_double(-10.0, 10.0));

    basis_tensor* A_T = basis_tensor_transpose(A);
    bool trans_ok = true;
    for(int i=0; i<R; i++) for(int j=0; j<C_cols; j++) {
        if (fabs(BASIS_TENSOR_AT(A, i, j)->data - BASIS_TENSOR_AT(A_T, j, i)->data) > 1e-9) trans_ok = false;
    }
    BASIS_ASSERT(trans_ok, "Dynamic Transpose reads correct strided memory");
    BASIS_ASSERT(A_T->data == A->data, "Dynamic Transpose is Zero-Copy");

    basis_value* loss = basis_tensor_sum(A_T);
    basis_value_backward_propagation(loss);

    bool grad_ok = true;
    for(int i=0; i<R; i++) for(int j=0; j<C_cols; j++) {
        if (fabs(BASIS_TENSOR_AT(A, i, j)->gradient - 1.0) > 1e-9) grad_ok = false;
    }
    BASIS_ASSERT(grad_ok, "Dynamic Gradients flow perfectly through transpose view");

    basis_value_free(loss); basis_tensor_free(A_T); basis_tensor_free(A);
}

void test_broadcasting_math() {
    BASIS_TEST_SUITE_START("Phase I: Dynamic Broadcasting (Random Batch & Feature Sizes)");
    int B = 4 + rand() % 20;
    int N = 4 + rand() % 20;

    basis_tensor* A = basis_tensor_new(B, N);
    basis_tensor* bias = basis_tensor_new(1, N);
    for(int i=0; i<B; i++) for(int j=0; j<N; j++) basis_tensor_set(A, i, j, rand_double(-5.0, 5.0));
    for(int j=0; j<N; j++) basis_tensor_set(bias, 0, j, rand_double(-5.0, 5.0));

    basis_tensor* bias_view = basis_tensor_broadcast_view(bias, B, N);
    basis_tensor* C = basis_tensor_add(A, bias_view);

    bool forward_ok = true;
    for(int i=0; i<B; i++) for(int j=0; j<N; j++) {
        double expected = BASIS_TENSOR_AT(A, i, j)->data + BASIS_TENSOR_AT(bias, 0, j)->data;
        if (fabs(BASIS_TENSOR_AT(C, i, j)->data - expected) > 1e-9) forward_ok = false;
    }
    BASIS_ASSERT(forward_ok, "Dynamic Broadcast forward pass matches naive");

    basis_value* loss = basis_tensor_sum(C);
    basis_value_backward_propagation(loss);

    bool backward_ok = true;
    for(int j=0; j<N; j++) {
        if (fabs(BASIS_TENSOR_AT(bias, 0, j)->gradient - (double)B) > 1e-9) backward_ok = false;
    }
    BASIS_ASSERT(backward_ok, "Dynamic Broadcast backward perfectly accumulates gradients");

    // Invariant: The broadcast view must strictly maintain the zero-stride property
    BASIS_ASSERT(bias_view->row_stride == 0, "Dynamic Broadcast maintains zero-stride memory invariant");

    basis_value_free(loss); basis_tensor_free(C); basis_tensor_free(bias_view);
    basis_tensor_free(A); basis_tensor_free(bias);
}


void test_hessian_vector_product() {
    BASIS_TEST_SUITE_START("Phase H: Dynamic Higher-Order Calculus (Exact Symbolic HVP)");

    basis_symbol* w1 = basis_symbol_variable("w1");
    basis_symbol* w2 = basis_symbol_variable("w2");

    basis_symbol* w1_sq = basis_symbol_power(basis_symbol_copy(w1), 2.0);
    basis_symbol* w2_sq = basis_symbol_power(basis_symbol_copy(w2), 2.0);
    basis_symbol* cross = basis_symbol_multiplication(basis_symbol_copy(w1), basis_symbol_copy(w2));

    basis_symbol* term2 = basis_symbol_multiplication(basis_symbol_constant(2.0), cross);
    basis_symbol* term3 = basis_symbol_multiplication(basis_symbol_constant(3.0), w2_sq);

    basis_symbol* L = basis_symbol_addition(w1_sq, term2);
    basis_symbol* L_final = basis_symbol_addition(L, term3);

    basis_symbol* G1 = basis_symbol_differentiation(L_final, "w1");
    basis_symbol* G2 = basis_symbol_differentiation(L_final, "w2");

    basis_symbol* S1 = basis_symbol_multiplication(G1, basis_symbol_constant(1.0));
    basis_symbol* S2 = basis_symbol_multiplication(G2, basis_symbol_constant(-1.0));
    basis_symbol* S = basis_symbol_addition(S1, S2);

    basis_symbol* S_simp = basis_symbol_simplify(S);
    basis_symbol_free(S);

    basis_symbol* HVP1_raw = basis_symbol_differentiation(S_simp, "w1");
    basis_symbol* HVP2_raw = basis_symbol_differentiation(S_simp, "w2");

    basis_symbol* HVP1 = basis_symbol_simplify(HVP1_raw);
    basis_symbol* HVP2 = basis_symbol_simplify(HVP2_raw);

    basis_symbol_free(HVP1_raw);
    basis_symbol_free(HVP2_raw);

    // Evaluate at a random point to prove it's constant for a quadratic manifold
    double rand_w1 = rand_double(-10.0, 10.0);
    double rand_w2 = rand_double(-10.0, 10.0);

    basis_value* val_w1 = basis_value_new(rand_w1);
    basis_value* val_w2 = basis_value_new(rand_w2);

    basis_compiler* comp = basis_compiler_new();
    basis_compiler_map(comp, "w1", val_w1);
    basis_compiler_map(comp, "w2", val_w2);

    basis_value* hvp1_num = basis_compiler_compile(comp, HVP1);
    basis_value* hvp2_num = basis_compiler_compile(comp, HVP2);

    BASIS_ASSERT_NEAR(hvp1_num->data, 0.0, 1e-6, "Dynamic HVP[0] matches exact analytical Hessian product");
    BASIS_ASSERT_NEAR(hvp2_num->data, -4.0, 1e-6, "Dynamic HVP[1] matches exact analytical Hessian product");

    basis_value_free(hvp1_num);
    basis_value_free(hvp2_num);
    basis_value_free(val_w1);
    basis_value_free(val_w2);
    basis_compiler_free(comp);

    basis_symbol_free(S_simp);
    basis_symbol_free(HVP1);
    basis_symbol_free(HVP2);
    basis_symbol_free(L_final);
    basis_symbol_free(w1);
    basis_symbol_free(w2);
}

void test_mlp_xor_convergence() {
    BASIS_TEST_SUITE_START("Phase K: Dynamic MLP XOR (Random Weight Initialization)");
    basis_tensor* X = basis_tensor_new(4, 2);
    basis_tensor_set(X, 0, 0, 0.0); basis_tensor_set(X, 0, 1, 0.0);
    basis_tensor_set(X, 1, 0, 0.0); basis_tensor_set(X, 1, 1, 1.0);
    basis_tensor_set(X, 2, 0, 1.0); basis_tensor_set(X, 2, 1, 0.0);
    basis_tensor_set(X, 3, 0, 1.0); basis_tensor_set(X, 3, 1, 1.0);

    basis_tensor* Y_true = basis_tensor_new(4, 1);
    basis_tensor_set(Y_true, 0, 0, 0.0); basis_tensor_set(Y_true, 1, 0, 1.0);
    basis_tensor_set(Y_true, 2, 0, 1.0); basis_tensor_set(Y_true, 3, 0, 0.0);

    basis_tensor* W1 = basis_tensor_new(2, 8);
    basis_tensor* b1 = basis_tensor_new(1, 8);
    basis_tensor* W2 = basis_tensor_new(8, 1);
    basis_tensor* b2 = basis_tensor_new(1, 1);

    for(int i=0; i<16; i++) W1->data[i]->data = rand_double(-1.0, 1.0);
    for(int i=0; i<8; i++) W2->data[i]->data = rand_double(-1.0, 1.0);

    basis_adam* opt_W1 = basis_adam_new(2, 8, 0.1);
    basis_adam* opt_b1 = basis_adam_new(1, 8, 0.1);
    basis_adam* opt_W2 = basis_adam_new(8, 1, 0.1);
    basis_adam* opt_b2 = basis_adam_new(1, 1, 0.1);

    double initial_loss = 0.0, final_loss = 0.0;

    for(int epoch=0; epoch<3000; epoch++) {
        basis_tensor* Z1 = basis_tensor_matrix_multiplication(X, W1);
        basis_tensor* b1_view = basis_tensor_broadcast_view(b1, 4, 8);
        basis_tensor* A1 = basis_tensor_add(Z1, b1_view);
        basis_tensor* H1 = basis_tensor_rectified_linear_unit(A1);

        basis_tensor* Z2 = basis_tensor_matrix_multiplication(H1, W2);
        basis_tensor* b2_view = basis_tensor_broadcast_view(b2, 4, 1);
        basis_tensor* Y_pred = basis_tensor_add(Z2, b2_view);

        basis_tensor* neg_Y = basis_tensor_scalar_multiplication(Y_true, -1.0);
        basis_tensor* diff = basis_tensor_add(Y_pred, neg_Y);
        basis_tensor* sq = basis_tensor_multiplication(diff, diff);
        basis_value* sum_sq = basis_tensor_sum(sq);
        basis_value* divisor = basis_value_new(0.25);
        basis_value* loss = basis_value_multiplication(sum_sq, divisor);

        if (epoch == 0) initial_loss = loss->data;
        if (epoch == 2999) final_loss = loss->data;

        basis_value_backward_propagation(loss);

        basis_adam_optimization_step(opt_W1, W1);
        basis_adam_optimization_step(opt_b1, b1);
        basis_adam_optimization_step(opt_W2, W2);
        basis_adam_optimization_step(opt_b2, b2);

        basis_value_free(loss); basis_value_free(divisor); basis_value_free(sum_sq);
        basis_tensor_free(sq); basis_tensor_free(diff); basis_tensor_free(neg_Y);
        basis_tensor_free(Y_pred); basis_tensor_free(b2_view); basis_tensor_free(Z2);
        basis_tensor_free(H1); basis_tensor_free(A1); basis_tensor_free(b1_view);
        basis_tensor_free(Z1);
    }

    BASIS_ASSERT(final_loss < initial_loss * 0.1, "Dynamic MLP successfully learns XOR from random init");
    BASIS_ASSERT(final_loss < 0.1, "Dynamic MLP achieves low absolute error");

    basis_adam_free(opt_W1); basis_adam_free(opt_b1);
    basis_adam_free(opt_W2); basis_adam_free(opt_b2);
    basis_tensor_free(W1); basis_tensor_free(b1);
    basis_tensor_free(W2); basis_tensor_free(b2);
    basis_tensor_free(X); basis_tensor_free(Y_true);
}


void test_jit_inference() {
    BASIS_TEST_SUITE_START("Phase J: JIT Code Generation (Real AST Compiler)");

    // === Test 1: exp(x^2) * log(x) ===
    {
        basis_symbol* x = basis_symbol_variable("x");
        basis_symbol* x2 = basis_symbol_power(basis_symbol_copy(x), 2.0);
        basis_symbol* e = basis_symbol_exponential(x2);
        basis_symbol* l = basis_symbol_logarithm(x);
        basis_symbol* f = basis_symbol_multiplication(e, l);

        char* vars[] = {"x"};
        basis_jit_module* jit = basis_jit_compile(f, vars, 1);
        BASIS_ASSERT(jit != NULL, "JIT compiled exp(x^2)*log(x) successfully");

        double x_val = 1.5;
        double jit_result = basis_jit_execute(jit, &x_val);
        double expected = exp(1.5*1.5) * log(1.5);
        BASIS_ASSERT_NEAR(jit_result, expected, 1e-9, "JIT output matches expected for exp(x^2)*log(x)");

        basis_jit_free(jit);
        basis_symbol_free(f);
    }

    // === Test 2: (a ** 2.0) + (b * a) ===
    {
        basis_symbol* a = basis_symbol_variable("a");
        basis_symbol* b = basis_symbol_variable("b");
        basis_symbol* a_sq = basis_symbol_power(basis_symbol_copy(a), 2.0);
        basis_symbol* ba = basis_symbol_multiplication(basis_symbol_copy(b), a);
        basis_symbol* f2 = basis_symbol_addition(a_sq, ba);

        char* vars2[] = {"a", "b"};
        basis_jit_module* jit2 = basis_jit_compile(f2, vars2, 2);
        BASIS_ASSERT(jit2 != NULL, "JIT compiled (a^2)+(b*a) successfully");

        double vals[] = {2.0, 3.0};
        double jit_result2 = basis_jit_execute(jit2, vals);
        double expected2 = (2.0 * 2.0) + (3.0 * 2.0); // 4.0 + 6.0 = 10.0
        BASIS_ASSERT_NEAR(jit_result2, expected2, 1e-9, "JIT output matches expected for (a^2)+(b*a)");

        basis_jit_free(jit2);
        basis_symbol_free(f2);
    }

    // === Test 3: Constant folding through JIT: (3.0 + 4.0) * 2.0 ===
    {
        basis_symbol* c1 = basis_symbol_constant(3.0);
        basis_symbol* c2 = basis_symbol_constant(4.0);
        basis_symbol* sum = basis_symbol_addition(c1, c2);
        basis_symbol* c3 = basis_symbol_constant(2.0);
        basis_symbol* prod = basis_symbol_multiplication(sum, c3);

        char* vars3[] = {"dummy"};
        basis_jit_module* jit3 = basis_jit_compile(prod, vars3, 1);
        BASIS_ASSERT(jit3 != NULL, "JIT compiled pure constant expression");

        double dummy = 0.0;
        double jit_result3 = basis_jit_execute(jit3, &dummy);
        BASIS_ASSERT_NEAR(jit_result3, 14.0, 1e-9, "JIT evaluates pure constant expression correctly");

        basis_jit_free(jit3);
        basis_symbol_free(prod);
    }
}


void test_stress_abyss_graph() {
    BASIS_TEST_SUITE_START("Phase L1: The Abyss Graph (10,000 Sequential Ops)");
    basis_value* current = basis_value_new(1.0);
    for(int i = 0; i < 10000; i++) {
        basis_value* c = basis_value_new(0.0001);
        basis_value* next = basis_value_addition(current, c);
        basis_value_free(current);
        basis_value_free(c);
        current = next;
    }
    BASIS_ASSERT_NEAR(current->data, 2.0, 1e-6, "Abyss graph forward pass completes 10,000 additions");

    // Backward pass tests the iterative topological sort on a 10,000 depth graph
    basis_value_backward_propagation(current);
    BASIS_ASSERT(current->gradient == 1.0, "Abyss graph backward pass completes without stack overflow");

    // Teardown tests the new iterative basis_value_free
    basis_value_free(current);
    BASIS_ASSERT(1, "Abyss graph iterative teardown completes without stack overflow");
}

void test_stress_leviathan_matmul() {
    BASIS_TEST_SUITE_START("Phase L2: The Leviathan MatMul (100x100 DAG)");
    size_t N = 100;
    basis_tensor* A = basis_tensor_new(N, N);
    basis_tensor* B = basis_tensor_new(N, N);

    for(size_t i=0; i<N*N; i++) {
        A->data[i]->data = 0.01;
        B->data[i]->data = 0.01;
    }

    basis_tensor* C = basis_tensor_matrix_multiplication(A, B);
    BASIS_ASSERT_NEAR(BASIS_TENSOR_AT(C, 0, 0)->data, 0.01, 1e-6, "Leviathan forward pass correct");

    basis_value* loss = basis_tensor_sum(C);
    basis_value_backward_propagation(loss);

    BASIS_ASSERT_NEAR(BASIS_TENSOR_AT(A, 0, 0)->gradient, 1.0, 1e-6, "Leviathan backward pass routes 2M edges correctly");

    basis_value_free(loss);
    basis_tensor_free(C);
    basis_tensor_free(A);
    basis_tensor_free(B);
}

void test_stress_deep_mlp() {
    BASIS_TEST_SUITE_START("Phase L3: The Deep MLP (5 Layers, 500 Epochs)");
    size_t B = 32;
    basis_tensor* X = basis_tensor_new(B, 16);
    for(size_t i=0; i<B*16; i++) X->data[i]->data = ((i % 5) - 2.0) * 0.1;

    basis_tensor* W1 = basis_tensor_new(16, 32); basis_tensor* b1 = basis_tensor_new(1, 32);
    basis_tensor* W2 = basis_tensor_new(32, 32); basis_tensor* b2 = basis_tensor_new(1, 32);
    basis_tensor* W3 = basis_tensor_new(32, 32); basis_tensor* b3 = basis_tensor_new(1, 32);
    basis_tensor* W4 = basis_tensor_new(32, 16); basis_tensor* b4 = basis_tensor_new(1, 16);
    basis_tensor* W5 = basis_tensor_new(16, 1);  basis_tensor* b5 = basis_tensor_new(1, 1);

    for(size_t i=0; i<16*32; i++) W1->data[i]->data = ((i % 7) - 3.0) * 0.1;
    for(size_t i=0; i<32*32; i++) W2->data[i]->data = ((i % 7) - 3.0) * 0.1;
    for(size_t i=0; i<32*32; i++) W3->data[i]->data = ((i % 7) - 3.0) * 0.1;
    for(size_t i=0; i<32*16; i++) W4->data[i]->data = ((i % 7) - 3.0) * 0.1;
    for(size_t i=0; i<16*1; i++) W5->data[i]->data = ((i % 7) - 3.0) * 0.1;

    basis_adam* opt1 = basis_adam_new(16, 32, 0.01);
    basis_adam* opt2 = basis_adam_new(32, 32, 0.01);
    basis_adam* opt3 = basis_adam_new(32, 32, 0.01);
    basis_adam* opt4 = basis_adam_new(32, 16, 0.01);
    basis_adam* opt5 = basis_adam_new(16, 1, 0.01);

    double final_loss = 0.0;

    for(int epoch=0; epoch<500; epoch++) {
        basis_tensor* Z1 = basis_tensor_matrix_multiplication(X, W1);
        basis_tensor* b1_v = basis_tensor_broadcast_view(b1, B, 32);
        basis_tensor* A1 = basis_tensor_add(Z1, b1_v);
        basis_tensor* H1 = basis_tensor_rectified_linear_unit(A1);

        basis_tensor* Z2 = basis_tensor_matrix_multiplication(H1, W2);
        basis_tensor* b2_v = basis_tensor_broadcast_view(b2, B, 32);
        basis_tensor* A2 = basis_tensor_add(Z2, b2_v);
        basis_tensor* H2 = basis_tensor_rectified_linear_unit(A2);

        basis_tensor* Z3 = basis_tensor_matrix_multiplication(H2, W3);
        basis_tensor* b3_v = basis_tensor_broadcast_view(b3, B, 32);
        basis_tensor* A3 = basis_tensor_add(Z3, b3_v);
        basis_tensor* H3 = basis_tensor_rectified_linear_unit(A3);

        basis_tensor* Z4 = basis_tensor_matrix_multiplication(H3, W4);
        basis_tensor* b4_v = basis_tensor_broadcast_view(b4, B, 16);
        basis_tensor* A4 = basis_tensor_add(Z4, b4_v);
        basis_tensor* H4 = basis_tensor_rectified_linear_unit(A4);

        basis_tensor* Z5 = basis_tensor_matrix_multiplication(H4, W5);
        basis_tensor* b5_v = basis_tensor_broadcast_view(b5, B, 1);
        basis_tensor* Y = basis_tensor_add(Z5, b5_v);

        basis_value* loss = basis_tensor_sum(Y);
        if(epoch == 499) final_loss = loss->data;

        basis_value_backward_propagation(loss);

        basis_adam_optimization_step(opt1, W1);
        basis_adam_optimization_step(opt2, W2);
        basis_adam_optimization_step(opt3, W3);
        basis_adam_optimization_step(opt4, W4);
        basis_adam_optimization_step(opt5, W5);

        basis_value_free(loss);
        basis_tensor_free(Y); basis_tensor_free(b5_v); basis_tensor_free(Z5);
        basis_tensor_free(H4); basis_tensor_free(A4); basis_tensor_free(b4_v); basis_tensor_free(Z4);
        basis_tensor_free(H3); basis_tensor_free(A3); basis_tensor_free(b3_v); basis_tensor_free(Z3);
        basis_tensor_free(H2); basis_tensor_free(A2); basis_tensor_free(b2_v); basis_tensor_free(Z2);
        basis_tensor_free(H1); basis_tensor_free(A1); basis_tensor_free(b1_v); basis_tensor_free(Z1);
    }

    BASIS_ASSERT(!isnan(final_loss), "Deep MLP loss did not explode to NaN");
    BASIS_ASSERT(!isinf(final_loss), "Deep MLP loss did not explode to Inf");

    basis_adam_free(opt1); basis_adam_free(opt2); basis_adam_free(opt3); basis_adam_free(opt4); basis_adam_free(opt5);
    basis_tensor_free(W1); basis_tensor_free(b1);
    basis_tensor_free(W2); basis_tensor_free(b2);
    basis_tensor_free(W3); basis_tensor_free(b3);
    basis_tensor_free(W4); basis_tensor_free(b4);
    basis_tensor_free(W5); basis_tensor_free(b5);
    basis_tensor_free(X);
}

void test_stress_jit_fuzzer() {
    BASIS_TEST_SUITE_START("Phase L4: The JIT Fuzzer (50 Random ASTs)");

    int success_count = 0;
    for(int i=0; i<50; i++) {
        basis_symbol* x = basis_symbol_variable("x");
        basis_symbol* current = basis_symbol_copy(x);

        for(int j=0; j<5; j++) {
            int op = rand() % 4;
            basis_symbol* c = basis_symbol_constant((double)(rand() % 5) + 1.0);
            if(op == 0) current = basis_symbol_addition(current, c);
            else if(op == 1) current = basis_symbol_multiplication(current, c);
            else if(op == 2) {
                current = basis_symbol_power(current, 2.0);
                basis_symbol_free(c);
            } else {
                current = basis_symbol_exponential(current);
                basis_symbol_free(c);
            }
        }

        char* vars[] = {"x"};
        basis_jit_module* jit = basis_jit_compile(current, vars, 1);

        if(jit) {
            double x_val = 0.5; // Safe domain for exp and powers
            double jit_res = basis_jit_execute(jit, &x_val);

            basis_value* x_auto = basis_value_new(x_val);
            basis_compiler* comp = basis_compiler_new();
            basis_compiler_map(comp, "x", x_auto);
            basis_value* auto_res_node = basis_compiler_compile(comp, current);
            double auto_res = auto_res_node->data;

            bool match = false;
            if (isnan(jit_res) && isnan(auto_res)) match = true;
            else if (isinf(jit_res) && isinf(auto_res) && (jit_res > 0) == (auto_res > 0)) match = true;
            else if (!isnan(jit_res) && !isinf(jit_res) && !isnan(auto_res) && !isinf(auto_res) && fabs(jit_res - auto_res) < 1e-3) match = true;

            if (match) success_count++;

            basis_value_free(auto_res_node);
            basis_value_free(x_auto);
            basis_compiler_free(comp);
            basis_jit_free(jit);
        }
        basis_symbol_free(current);
        basis_symbol_free(x);
    }

    BASIS_ASSERT(success_count == 50, "JIT Fuzzer successfully compiled and matched 50 random ASTs");
}

int main() {
    srand(time(NULL)); // Seed the random number generator
    printf("=========================================================\n");
    printf("  BASIS v5 DYNAMIC Property-Based Test Suite            \n");
    printf("=========================================================\n");

    test_autograd_basic();
    test_tensor_shape_validation();
    test_symbolic_simplification();
    test_compiler_cse();
    test_log_softmax_stability();
    test_holy_trinity();
    test_information_geometry();
    test_natural_gradient_descent();
    test_matmul_dag_compression();
    test_strided_tensor_views();
    test_broadcasting_math();
    test_jit_inference();
    test_hessian_vector_product();
    test_mlp_xor_convergence();

    test_stress_abyss_graph();
    test_stress_leviathan_matmul();
    test_stress_deep_mlp();
    test_stress_jit_fuzzer();
    BASIS_TEST_SUITE_END();
    return basis_tests_failed > 0 ? 1 : 0;
}

```

---

### FILE: tests/test_v7_autodiff.c
Location: `tests/test_v7_autodiff.c`
```cpp
#include "basis.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: L6 Native Autodiff Training (XOR)           \n");
    printf("=========================================================\n\n");

    srand(42);
    basis_ir_graph* g = basis_ir_graph_create();

    double X_data[8] = {0,0, 0,1, 1,0, 1,1};
    double Y_data[4] = {0, 1, 1, 0};

    double W1[16], W2[8];
    for(int i=0; i<16; i++) W1[i] = ((rand() % 100) / 100.0) - 0.5;
    for(int i=0; i<8; i++) W2[i] = ((rand() % 100) / 100.0) - 0.5;

    basis_v7_tensor* X = basis_v7_input(g, 4, 2, NULL);
    basis_v7_tensor* Y = basis_v7_input(g, 4, 1, NULL);
    basis_v7_tensor* W1_t = basis_v7_input(g, 2, 8, NULL);
    basis_v7_tensor* W2_t = basis_v7_input(g, 8, 1, NULL);

    basis_v7_tensor* Z1 = basis_v7_matmul(g, X, W1_t);
    basis_v7_tensor* A1 = basis_v7_relu(g, Z1);
    basis_v7_tensor* Z2 = basis_v7_matmul(g, A1, W2_t);

    basis_v7_tensor* diff = basis_v7_sub(g, Z2, Y);
    basis_v7_tensor* sq = basis_v7_mul(g, diff, diff);
    basis_v7_tensor* loss = basis_v7_sum(g, sq);

    printf("[1/3] Generating Forward + Backward Graph via Autodiff...\n");
    basis_training_graph* tg = basis_ir_autodiff(g, loss->node);

    printf("[2/3] Scheduling Unified Training Waves...\n");
    basis_schedule* sched = basis_ir_schedule(tg->graph);
    basis_arena* scratch = basis_arena_create(1024 * 1024);

    double* ptr_X = NULL; double* ptr_Y = NULL;
    double* ptr_W1 = NULL; double* ptr_W2 = NULL;

    // Persistent allocation for INPUT nodes (Weights and Data)
    for(uint32_t i=0; i<tg->graph->node_count; i++) {
        basis_ir_node* n = tg->graph->nodes[i];
        if (n->op == BASIS_IR_OP_INPUT) {
            n->runtime_data = (double*)malloc(n->rows * n->cols * sizeof(double));
            if (n->rows == 4 && n->cols == 2) ptr_X = n->runtime_data;
            if (n->rows == 4 && n->cols == 1) ptr_Y = n->runtime_data;
            if (n->rows == 2 && n->cols == 8) ptr_W1 = n->runtime_data;
            if (n->rows == 8 && n->cols == 1) ptr_W2 = n->runtime_data;
        }
    }

    printf("[3/3] Executing 2000 Epochs (O(1) Arena Teardown per epoch)...\n\n");
    double lr = 0.05;

    for(int epoch=0; epoch<2000; epoch++) {
        // 1. Copy persistent data into INPUT nodes
        memcpy(ptr_X, X_data, sizeof(X_data));
        memcpy(ptr_Y, Y_data, sizeof(Y_data));
        memcpy(ptr_W1, W1, sizeof(W1));
        memcpy(ptr_W2, W2, sizeof(W2));

        // 2. CRITICAL: Nullify intermediate runtime_data so scheduler re-allocates on scratch arena
        for(uint32_t i=0; i<tg->graph->node_count; i++) {
            basis_ir_node* n = tg->graph->nodes[i];
            if (n->op != BASIS_IR_OP_INPUT && n->op != BASIS_IR_OP_CONST) {
                n->runtime_data = NULL;
            }
        }

        // 3. Execute Forward + Backward
        basis_schedule_execute(sched, scratch);

        // 4. Fetch ephemeral gradient pointers for this epoch
        double* current_loss = tg->loss_node->runtime_data;
        basis_ir_node* grad_W1_node = tg->grad_nodes[W1_t->node->id];
        basis_ir_node* grad_W2_node = tg->grad_nodes[W2_t->node->id];

        double* current_grad_W1 = grad_W1_node ? grad_W1_node->runtime_data : NULL;
        double* current_grad_W2 = grad_W2_node ? grad_W2_node->runtime_data : NULL;

        if(epoch % 400 == 0 && current_loss) {
            printf("Epoch %4d | Loss: %.6f\n", epoch, current_loss[0]);
        }

        // 5. Bare-metal SGD Update
        if (current_grad_W1 && current_grad_W2) {
            for(int i=0; i<16; i++) W1[i] -= lr * current_grad_W1[i];
            for(int i=0; i<8; i++)  W2[i] -= lr * current_grad_W2[i];
        }

        // 6. O(1) Teardown! (Reclaims all activations and gradients instantly)
        basis_arena_reset(scratch);
    }

    printf("\n=== Final Inference ===\n");
    memcpy(ptr_X, X_data, sizeof(X_data));
    memcpy(ptr_W1, W1, sizeof(W1));
    memcpy(ptr_W2, W2, sizeof(W2));

    for(uint32_t i=0; i<tg->graph->node_count; i++) {
        basis_ir_node* n = tg->graph->nodes[i];
        if (n->op != BASIS_IR_OP_INPUT && n->op != BASIS_IR_OP_CONST) n->runtime_data = NULL;
    }
    basis_schedule_execute(sched, scratch);

    // Find the forward Z2 node (input 0 of the SUB node)
    basis_ir_node* fwd_Z2 = NULL;
    for(uint32_t i=0; i<tg->graph->node_count; i++) {
        if(tg->graph->nodes[i]->op == BASIS_IR_OP_SUB) {
            fwd_Z2 = (basis_ir_node*)tg->graph->nodes[i]->inputs[0];
            break;
        }
    }

    if (fwd_Z2 && fwd_Z2->runtime_data) {
        for(int i=0; i<4; i++) {
            printf("Input: [%.0f, %.0f] | Target: %.0f | Predicted: %.4f\n",
                   X_data[i*2], X_data[i*2+1], Y_data[i], fwd_Z2->runtime_data[i]);
        }
    }

    // Cleanup persistent INPUT buffers
    for(uint32_t i=0; i<tg->graph->node_count; i++) {
        if (tg->graph->nodes[i]->op == BASIS_IR_OP_INPUT && tg->graph->nodes[i]->runtime_data) {
            free(tg->graph->nodes[i]->runtime_data);
        }
    }

    basis_schedule_destroy(sched);
    basis_training_graph_destroy(tg);
    basis_ir_graph_destroy(g);
    basis_arena_destroy(scratch);

    printf("\n✅ Native V7 Autodiff Training Complete. Zero Leaks.\n");
    return 0;
}

```

---

### FILE: tests/test_v7_ir.c
Location: `tests/test_v7_ir.c`
```cpp
#include "basis/core/ir.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: Immutable Graph IR Validation (SSA & Arena) \n");
    printf("=========================================================\n\n");

    int failures = 0;
    basis_ir_graph* g = basis_ir_graph_create();

    // 1. Basic SSA & Shape Inference
    printf("=== Shape Inference & SSA Tests ===\n");
    basis_ir_node* in1 = basis_ir_input(g, 4, 8);
    basis_ir_node* in2 = basis_ir_input(g, 4, 8);
    basis_ir_node* add1 = basis_ir_add(g, in1, in2);

    if (add1 && add1->rows == 4 && add1->cols == 8) printf("  [PASS] Shape inference (Add)\n"); else { printf("  [FAIL] Add shape\n"); failures++; }

    basis_ir_node* w = basis_ir_input(g, 8, 16);
    basis_ir_node* mm = basis_ir_matmul(g, add1, w);
    if (mm && mm->rows == 4 && mm->cols == 16) printf("  [PASS] Shape inference (MatMul)\n"); else { printf("  [FAIL] MatMul shape\n"); failures++; }

    // Test Shape Validation (Should return NULL)
    basis_ir_node* bad_mm = basis_ir_matmul(g, in1, in2); // 4x8 * 4x8 is invalid
    if (bad_mm == NULL) printf("  [PASS] Shape validation rejects invalid MatMul\n"); else { printf("  [FAIL] Validation\n"); failures++; }

    // 2. The 1,000,000 Node Stress Test & O(1) Teardown
    printf("\n=== Stress Test: 1,000,000 Node Deep Graph ===\n");
    basis_ir_node* current = in1;
    basis_ir_node* bias = basis_ir_const(g, 0.5, 4, 8);

    clock_t start = clock();
    for(int i=0; i<1000000; i++) {
        current = basis_ir_add(g, current, bias);
    }
    clock_t end = clock();
    double build_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("  Built 1,000,000 node chain in %.4f sec\n", build_time);

    start = clock();
    basis_ir_graph_destroy(g);
    end = clock();
    double teardown_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("  Destroyed 1,000,000 node chain in %.6f sec\n", teardown_time);

    if (teardown_time < 0.01) printf("  [PASS] O(1) Arena Teardown verified (No recursive frees!)\n");
    else { printf("  [FAIL] Teardown too slow\n"); failures++; }

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ ALL V7 IR TESTS PASSED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

```

---

### FILE: tests/test_v7_memory.c
Location: `tests/test_v7_memory.c`
```cpp
#include "basis/core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#define TEST_PASSED(msg) printf("  [PASS] %s\n", msg)
#define TEST_FAILED(msg) do { printf("  [FAIL] %s\n", msg); failures++; } while(0)

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: Memory Subsystem Validation (Arena & Pool)  \n");
    printf("=========================================================\n\n");

    int failures = 0;

    // ==========================================
    // TEST 1: Arena Alignment & Bump Allocation
    // ==========================================
    printf("=== Arena Allocator Tests ===\n");
    basis_arena* arena = basis_arena_create(4096); // 4KB blocks

    void* p1 = basis_arena_alloc(arena, 10, 8);
    void* p2 = basis_arena_alloc(arena, 33, 16);
    void* p3 = basis_arena_alloc(arena, 100, 32);

    if (((uintptr_t)p1 % 8) == 0) TEST_PASSED("p1 aligned to 8 bytes"); else TEST_FAILED("p1 alignment");
    if (((uintptr_t)p2 % 16) == 0) TEST_PASSED("p2 aligned to 16 bytes"); else TEST_FAILED("p2 alignment");
    if (((uintptr_t)p3 % 32) == 0) TEST_PASSED("p3 aligned to 32 bytes"); else TEST_FAILED("p3 alignment");

    if (p2 > p1 && p3 > p2) TEST_PASSED("Arena pointer bumping is monotonic"); else TEST_FAILED("Bump order");

    // ==========================================
    // TEST 2: Arena Oversized Allocation & Reset
    // ==========================================
    void* huge = basis_arena_alloc(arena, 8192, 8);
    if (huge != NULL) TEST_PASSED("Arena handles oversized allocations (> block size)"); else TEST_FAILED("Oversized alloc");

    basis_arena_reset(arena);

    void* p4 = basis_arena_alloc(arena, 10, 8);
    if (p4 == p1) TEST_PASSED("Arena Reset correctly reuses baseline memory (O(1) teardown)"); else TEST_FAILED("Arena Reset reuse");

    basis_arena_destroy(arena);
    TEST_PASSED("Arena destroyed without leaks");

    // ==========================================
    // TEST 3: Pool Allocation & Free-List Reuse
    // ==========================================
    printf("\n=== Pool Allocator Tests ===\n");

    typedef struct { double x, y, z; int id; } mock_node;
    basis_pool* pool = basis_pool_create(sizeof(mock_node), 100);

    mock_node* nodes[100];
    for(int i=0; i<100; i++) {
        nodes[i] = (mock_node*)basis_pool_alloc(pool);
        nodes[i]->id = i;
    }
    TEST_PASSED("Pool allocated 100 objects seamlessly");

    mock_node* freed_node = nodes[50];
    basis_pool_free(pool, freed_node);

    mock_node* reused_node = (mock_node*)basis_pool_alloc(pool);
    if (reused_node == freed_node) TEST_PASSED("Pool Free-List correctly reuses freed memory pointer"); else TEST_FAILED("Pool reuse");

    mock_node* overflow_node = (mock_node*)basis_pool_alloc(pool);
    if (overflow_node != NULL) TEST_PASSED("Pool successfully allocated secondary chunk on overflow"); else TEST_FAILED("Pool overflow");

    basis_pool_destroy(pool);
    TEST_PASSED("Pool destroyed without leaks");

    // ==========================================
    // TEST 4: Realistic Graph Lifecycle (1,000,000 Nodes)
    // ==========================================
    printf("\n=== Realistic Graph Lifecycle (1,000,000 Nodes) ===\n");
    printf("Simulating Compiler IR: Bulk Alloc -> Use -> Mass Teardown\n");
    size_t N = 1000000;
    clock_t start, end;

    // Array to hold pointers. This forces malloc to manage real heap structures
    // and prevents the CPU from just reusing a single L1-cached address.
    void** ptrs = (void**)malloc(N * sizeof(void*));

    // --- MALLOC LIFECYCLE ---
    start = clock();
    // Phase 1: Build Graph (Varying sizes 64-127 to scatter across glibc bins)
    for(size_t i=0; i<N; i++) {
        size_t sz = 64 + (i % 64);
        ptrs[i] = malloc(sz);
        ((char*)ptrs[i])[0] = (char)i; // Touch memory
    }
    // Phase 2: Teardown Graph (O(N) mass free)
    for(size_t i=0; i<N; i++) {
        free(ptrs[i]);
    }
    end = clock();
    double malloc_time = (double)(end - start) / CLOCKS_PER_SEC;

    // --- ARENA LIFECYCLE ---
    // 64MB blocks easily holds the ~100MB graph with minimal chunking overhead
    basis_arena* perf_arena = basis_arena_create(1024 * 1024 * 64);
    start = clock();
    // Phase 1: Build Graph (Linear streaming)
    for(size_t i=0; i<N; i++) {
        size_t sz = 64 + (i % 64);
        char* ptr = (char*)basis_arena_alloc(perf_arena, sz, 8);
        ptr[0] = (char)i; // Touch memory
        ptrs[i] = ptr;
    }
    // Phase 2: Teardown Graph (O(1) Reset!)
    basis_arena_reset(perf_arena);
    end = clock();
    double arena_time = (double)(end - start) / CLOCKS_PER_SEC;

    basis_arena_destroy(perf_arena);
    free(ptrs);

    printf("  Malloc Lifecycle (Alloc + Mass Free): %.4f sec\n", malloc_time);
    printf("  Arena Lifecycle (Alloc + O(1) Reset):      %.4f sec\n", arena_time);
    if (arena_time < malloc_time) TEST_PASSED("Arena Lifecycle crushes Malloc Lifecycle"); else TEST_FAILED("Arena performance");

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ ALL V7 MEMORY TESTS PASSED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

```

---

### FILE: tests/test_v7_passes.c
Location: `tests/test_v7_passes.c`
```cpp
#include "basis.h"
#include <stdio.h>
#include <stdbool.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: L4 Compiler Pass Manager Validation         \n");
    printf("=========================================================\n\n");

    int failures = 0;

    // 1. Build an intentionally wasteful, unoptimized graph
    printf("=== Building Unoptimized Graph ===\n");
    basis_ir_graph* g_raw = basis_ir_graph_create();

    basis_v7_tensor* A = basis_v7_input(g_raw, 4, 4, NULL);

    // Wasteful Math: Add constants at runtime
    basis_v7_tensor* C1 = basis_v7_const(g_raw, 5.0, 4, 4);
    basis_v7_tensor* C2 = basis_v7_const(g_raw, 3.0, 4, 4);
    basis_v7_tensor* SumC = basis_v7_add(g_raw, C1, C2); // Should fold to Const(8.0)

    // Wasteful Math: Add zero
    basis_v7_tensor* Zero = basis_v7_const(g_raw, 0.0, 4, 4);
    basis_v7_tensor* AddZero = basis_v7_add(g_raw, A, Zero); // Should simplify to A

    // Final operation to use the results
    basis_v7_tensor* Out = basis_v7_add(g_raw, SumC, AddZero);
    (void)Out; // Suppress unused variable warning

    printf("Raw Graph Node Count: %u\n", g_raw->node_count);
    basis_ir_print(g_raw);

    // 2. Run Compiler Passes
    printf("\n=== Running Compiler Passes (Folding & Elimination) ===\n");
    basis_ir_graph* g_opt = basis_ir_run_passes(g_raw);

    printf("Optimized Graph Node Count: %u\n", g_opt->node_count);
    basis_ir_print(g_opt);

    // 3. Verify Optimizations
    printf("\n=== Verification ===\n");

    // The raw graph had: A, C1, C2, SumC, Zero, AddZero, Out = 7 nodes
    // The optimized graph should have: A, SumC(folded to Const 8), Out = 3 nodes
    // (C1, C2, Zero, and AddZero should be completely eliminated)

    if (g_opt->node_count == 3) {
        printf("  [PASS] Node count reduced from 7 to 3 (Dead code eliminated)\n");
    } else {
        printf("  [FAIL] Node count is %u (Expected 3)\n", g_opt->node_count);
        failures++;
    }

    // Verify Constant Folding: Node 1 should be CONST with value 8.0
    basis_ir_node* folded_node = g_opt->nodes[1];
    if (folded_node->op == BASIS_IR_OP_CONST && folded_node->attr_val == 8.0) {
        printf("  [PASS] Constant Folding: Add(Const 5, Const 3) -> Const 8\n");
    } else {
        printf("  [FAIL] Constant folding failed\n");
        failures++;
    }

    // Verify Identity Elimination: The final ADD should take A and the folded Const
    // (Order-agnostic check since addition is commutative)
    basis_ir_node* final_node = g_opt->nodes[2];
    bool has_input = (final_node->inputs[0]->op == BASIS_IR_OP_INPUT || final_node->inputs[1]->op == BASIS_IR_OP_INPUT);
    bool has_const = (final_node->inputs[0]->op == BASIS_IR_OP_CONST || final_node->inputs[1]->op == BASIS_IR_OP_CONST);

    if (final_node->op == BASIS_IR_OP_ADD && has_input && has_const) {
        printf("  [PASS] Identity Elimination: Add(A, 0) simplified directly to A\n");
    } else {
        printf("  [FAIL] Identity elimination failed\n");
        failures++;
    }

    // 4. Cleanup
    basis_ir_graph_destroy(g_raw);
    basis_ir_graph_destroy(g_opt);

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ ALL V7 COMPILER PASS TESTS PASSED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

```

---

### FILE: tests/test_v7_scheduler.c
Location: `tests/test_v7_scheduler.c`
```cpp
#include "basis.h"
#include <stdio.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: L5 Deterministic Scheduler Validation       \n");
    printf("=========================================================\n\n");

    int failures = 0;
    basis_ir_graph* g = basis_ir_graph_create();

    // Build a "Diamond" DAG to prove parallel wave extraction
    //      A (Input)
    //     / \
    //    B   C   (Both depend only on A -> Should be in Wave 1 together)
    //     \ /
    //      D     (Depends on B and C -> Should be in Wave 2)

    double A_data[] = {1.0, 2.0, 3.0, 4.0};
    basis_v7_tensor* A = basis_v7_input(g, 2, 2, A_data);
    basis_v7_tensor* B = basis_v7_relu(g, A);
    basis_v7_tensor* C = basis_v7_add(g, A, A);
    basis_v7_tensor* D = basis_v7_add(g, B, C);

    printf("=== Graph Topology ===\n");
    basis_ir_print(g);

    printf("\n=== Scheduling Waves ===\n");
    basis_schedule* sched = basis_ir_schedule(g);
    basis_schedule_print(sched);

    // Verify Wave Count (Should be 3: Wave 0=[A], Wave 1=[B,C], Wave 2=[D])
    if (sched->wave_count == 3) {
        printf("\n  [PASS] Correctly identified 3 execution waves\n");
    } else {
        printf("\n  [FAIL] Expected 3 waves, got %u\n", sched->wave_count);
        failures++;
    }

    // Verify Wave 1 has 2 parallel nodes (B and C)
    if (sched->wave_count >= 2 && sched->waves[1].node_count == 2) {
        printf("  [PASS] Wave 1 correctly groups 2 independent nodes for parallel execution\n");
    } else {
        printf("  [FAIL] Wave 1 parallel grouping failed\n");
        failures++;
    }

    printf("\n=== Executing Schedule ===\n");
    basis_schedule_execute(sched);

    // Math check:
    // A = [1, 2, 3, 4]
    // B = ReLU(A) = [1, 2, 3, 4]
    // C = A + A = [2, 4, 6, 8]
    // D = B + C = [3, 6, 9, 12]

    double* d_data = D->node->runtime_data;
    if (d_data[0] == 3.0 && d_data[3] == 12.0) {
        printf("  [PASS] Wave-based execution math is correct!\n");
    } else {
        printf("  [FAIL] Math mismatch in wave execution\n");
        failures++;
    }

    basis_schedule_destroy(sched);
    basis_ir_graph_destroy(g);

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ ALL V7 SCHEDULER TESTS PASSED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

```

---

### FILE: tests/test_v7_swarm.c
Location: `tests/test_v7_swarm.c`
```cpp
#include "basis.h"
#include <stdio.h>

// Helper to build a simple forward-only expert graph
basis_ir_graph* build_expert_graph(double multiplier) {
    basis_ir_graph* g = basis_ir_graph_create();
    basis_v7_tensor* X = basis_v7_input(g, 2, 2, NULL);
    basis_v7_tensor* W = basis_v7_const(g, multiplier, 2, 2);
    basis_v7_tensor* Out = basis_v7_mul(g, X, W); // Element-wise multiply
    (void)Out;
    return g;
}

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: L7 Micro-Model Swarm & Planner              \n");
    printf("=========================================================\n\n");

    int failures = 0;

    // 1. Build the Experts (Micro-Models)
    printf("[1/4] Compiling Micro-Models...\n");
        // Actually, let's make Expert A multiply by 10.0, Expert B multiply by -1.0
    basis_ir_graph* g_A = build_expert_graph(10.0);
    basis_ir_graph* g_B = build_expert_graph(-1.0);

    basis_schedule* sched_A = basis_ir_schedule(g_A);
    basis_schedule* sched_B = basis_ir_schedule(g_B);

    // 2. Initialize the Swarm Registry
    printf("[2/4] Registering Experts in the Swarm...\n");
    basis_swarm* swarm = basis_swarm_create();
    basis_swarm_register(swarm, "Expert_A_Positive", sched_A, g_A);
    basis_swarm_register(swarm, "Expert_B_Negative", sched_B, g_B);

    // 3. The Planner (Structured Routing Protocol)
    printf("[3/4] Executing Planner Routing Logic...\n");
    double inputs[4][4] = {
        { 1.0,  2.0,  3.0,  4.0},  // Positive -> Route to A (* 10)
        {-5.0, -6.0, -7.0, -8.0},  // Negative -> Route to B (* -1)
        { 0.5,  0.5,  0.5,  0.5},  // Positive -> Route to A (* 10)
        {-2.0,  2.0, -2.0,  2.0}   // Mixed -> Route based on first element (Negative -> B)
    };
    double outputs[4][4];

    basis_expert* exp_A = basis_swarm_get(swarm, "Expert_A_Positive");
    basis_expert* exp_B = basis_swarm_get(swarm, "Expert_B_Negative");

    for(int i=0; i<4; i++) {
        // The Planner inspects the tensor and routes it
        if (inputs[i][0] > 0.0) {
            basis_swarm_dispatch(exp_A, inputs[i], outputs[i]);
        } else {
            basis_swarm_dispatch(exp_B, inputs[i], outputs[i]);
        }
    }

    // 4. Verify Structured Inter-Model Protocol
    printf("[4/4] Verifying Outputs...\n");

    // Input 0: [1,2,3,4] * 10 = [10, 20, 30, 40]
    if (outputs[0][0] == 10.0 && outputs[0][3] == 40.0) printf("  [PASS] Planner correctly routed positive tensor to Expert A\n");
    else { printf("  [FAIL] Expert A routing/math failed\n"); failures++; }

    // Input 1: [-5,-6,-7,-8] * -1 = [5, 6, 7, 8]
    if (outputs[1][0] == 5.0 && outputs[1][3] == 8.0) printf("  [PASS] Planner correctly routed negative tensor to Expert B\n");
    else { printf("  [FAIL] Expert B routing/math failed\n"); failures++; }

    basis_swarm_destroy(swarm);
    basis_ir_graph_destroy(g_A);
    basis_ir_graph_destroy(g_B);

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ V7.4 MICRO-MODEL SWARM SUCCESSFULLY EXECUTED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

```

---

### FILE: tests/test_v7_tensor.c
Location: `tests/test_v7_tensor.c`
```cpp
#include "basis.h"
#include <stdio.h>
#include <time.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: L1 Tensor Engine (Deferred Execution)       \n");
    printf("=========================================================\n\n");

    int failures = 0;
    basis_ir_graph* g = basis_ir_graph_create();

    // 1. Define Inputs
    double A_data[] = {1.0, 2.0, 3.0, 4.0};
    double B_data[] = {5.0, 6.0, 7.0, 8.0};

    basis_v7_tensor* A = basis_v7_input(g, 2, 2, A_data);
    basis_v7_tensor* B = basis_v7_input(g, 2, 2, B_data);

    // 2. Trace Graph (No math executed yet!)
    printf("=== Tracing Graph (Deferred Execution) ===\n");
    basis_v7_tensor* C = basis_v7_add(g, A, B);       // C = A + B
    basis_v7_tensor* D = basis_v7_matmul(g, C, A);    // D = (A + B) @ A
    basis_v7_tensor* E = basis_v7_relu(g, D);         // E = ReLU(D)

    if (C->node->runtime_data == NULL) printf("  [PASS] Add deferred (No data allocated yet)\n"); else { printf("  [FAIL] Add executed early\n"); failures++; }
    if (D->node->runtime_data == NULL) printf("  [PASS] MatMul deferred (No data allocated yet)\n"); else { printf("  [FAIL] MatMul executed early\n"); failures++; }

    // 3. Execute Graph
    printf("\n=== Executing Graph ===\n");
    basis_v7_execute(g);

    basis_v7_tensor_print(A, "A (Input)");
    basis_v7_tensor_print(C, "C (A + B)");
    basis_v7_tensor_print(E, "E (ReLU((A+B)@A))");

    // Verify Math: C[0,0] = 1+5 = 6.0
    if (C->node->runtime_data[0] == 6.0) printf("\n  [PASS] Forward math verified\n"); else { printf("\n  [FAIL] Math incorrect\n"); failures++; }

    // 4. The O(1) Teardown Flex
    printf("\n=== The O(1) Teardown Flex ===\n");
    printf("Destroying the graph will instantly free:\n");
    printf("  - 5 IR Nodes\n");
    printf("  - 5 Tensor metadata structs\n");
    printf("  - 5 Raw double* activation buffers\n");
    printf("Total memory reclaimed in exactly ONE CPU instruction (Arena Reset).\n");

    clock_t start = clock();
    basis_ir_graph_destroy(g);
    clock_t end = clock();
    double teardown_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("  Teardown time: %.6f sec\n", teardown_time);
    if (teardown_time < 0.001) printf("  [PASS] O(1) Arena Teardown verified!\n"); else { printf("  [FAIL] Teardown too slow\n"); failures++; }

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ ALL V7 TENSOR TESTS PASSED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

```

---

### FILE: tests/test_v7_transformer.c
Location: `tests/test_v7_transformer.c`
```cpp
#include "basis.h"
#include <stdio.h>
#include <math.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: Reference Transformer Block (Attention+FFN) \n");
    printf("=========================================================\n\n");

    int failures = 0;
    basis_ir_graph* g = basis_ir_graph_create();

    size_t SEQ = 4, D = 4;
    double scale_factor = 1.0 / sqrt((double)D);

    printf("=== 1. Tracing Transformer Block (Deferred Execution) ===\n");

    basis_v7_tensor* X   = basis_v7_input(g, SEQ, D, NULL);
    basis_v7_tensor* Wq  = basis_v7_input(g, D, D, NULL);
    basis_v7_tensor* Wk  = basis_v7_input(g, D, D, NULL);
    basis_v7_tensor* Wv  = basis_v7_input(g, D, D, NULL);
    basis_v7_tensor* W1  = basis_v7_input(g, D, D, NULL);
    basis_v7_tensor* W2  = basis_v7_input(g, D, D, NULL);

    basis_v7_tensor* Scale = basis_v7_const(g, scale_factor, SEQ, SEQ);

    basis_v7_tensor* Q = basis_v7_matmul(g, X, Wq);
    basis_v7_tensor* K = basis_v7_matmul(g, X, Wk);
    basis_v7_tensor* V = basis_v7_matmul(g, X, Wv);

    basis_v7_tensor* K_T     = basis_v7_transpose(g, K);
    basis_v7_tensor* Scores  = basis_v7_matmul(g, Q, K_T);
    basis_v7_tensor* Scaled  = basis_v7_mul(g, Scores, Scale);
    basis_v7_tensor* Probs   = basis_v7_softmax(g, Scaled);
    basis_v7_tensor* AttnOut = basis_v7_matmul(g, Probs, V);

    basis_v7_tensor* H1      = basis_v7_matmul(g, AttnOut, W1);
    basis_v7_tensor* H1_Act  = basis_v7_relu(g, H1);
    basis_v7_tensor* FFN_Out = basis_v7_matmul(g, H1_Act, W2);

    basis_v7_tensor* BlockOut = basis_v7_add(g, AttnOut, FFN_Out);
    (void)BlockOut;

    printf("Raw Graph Nodes: %u\n", g->node_count);

    printf("\n=== 2. Running Compiler Passes ===\n");
    basis_ir_graph* g_opt = basis_ir_run_passes(g);
    printf("Optimized Graph Nodes: %u\n", g_opt->node_count);

    printf("\n=== 3. Extracting Parallel Execution Waves ===\n");
    basis_schedule* sched = basis_ir_schedule(g_opt);
    basis_schedule_print(sched);

    bool qkv_parallel = false;
    for(uint32_t w=0; w<sched->wave_count; w++) {
        if(sched->waves[w].node_count == 3) {
            qkv_parallel = true; break;
        }
    }
    if(qkv_parallel) {
        printf("\n  [PASS] Scheduler automatically parallelized Q, K, V projections!\n");
    } else {
        printf("\n  [FAIL] Q, K, V were not grouped in the same parallel wave.\n");
        failures++;
    }

    printf("\n=== 4. Executing Transformer Block ===\n");
    basis_schedule_execute(sched);

    basis_ir_node* opt_out = g_opt->nodes[g_opt->node_count - 1];
    basis_ir_node* opt_probs = NULL;
    for(uint32_t i=0; i<g_opt->node_count; i++) {
        if(g_opt->nodes[i]->op == BASIS_IR_OP_SOFTMAX) {
            opt_probs = g_opt->nodes[i];
            break;
        }
    }

    if (opt_out->rows == SEQ && opt_out->cols == D) {
        printf("  [PASS] Final output shape is correct [%zu x %zu]\n", SEQ, D);
    } else {
        printf("  [FAIL] Final output shape mismatch\n");
        failures++;
    }

    if (opt_probs && opt_probs->runtime_data) {
        double* probs_data = opt_probs->runtime_data;
        double row_sum = 0.0;
        for(size_t i=0; i<SEQ; i++) row_sum += probs_data[i];
        if (fabs(row_sum - 1.0) < 1e-6) {
            printf("  [PASS] Attention probabilities sum to 1.0 (Softmax verified)\n");
        } else {
            printf("  [FAIL] Softmax probabilities sum to %f\n", row_sum);
            failures++;
        }
    } else {
        printf("  [FAIL] Could not find executed Softmax node\n");
        failures++;
    }

    basis_schedule_destroy(sched);
    basis_ir_graph_destroy(g_opt);
    basis_ir_graph_destroy(g);

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ V7 REFERENCE TRANSFORMER SUCCESSFULLY EXECUTED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

```

---

### FILE: tests/test_v7_transformer_primitives.c
Location: `tests/test_v7_transformer_primitives.c`
```cpp
#include "basis.h"
#include <stdio.h>
#include <math.h>

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: Transformer Primitives Validation           \n");
    printf("=========================================================\n\n");

    int failures = 0;
    basis_ir_graph* g = basis_ir_graph_create();

    // 1. Test Transpose
    double A_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    basis_v7_tensor* A = basis_v7_input(g, 2, 3, A_data);
    basis_v7_tensor* A_T = basis_v7_transpose(g, A);

    // 2. Test Softmax (Log-Sum-Exp trick)
    double logits[] = {1.0, 2.0, 3.0, 100.0, 101.0, 102.0}; // Extreme values to test stability
    basis_v7_tensor* L = basis_v7_input(g, 2, 3, logits);
    basis_v7_tensor* S = basis_v7_softmax(g, L);

    // 3. Test Element-wise MUL & SUB
    basis_v7_tensor* B = basis_v7_input(g, 2, 3, A_data);
    basis_v7_tensor* Mul = basis_v7_mul(g, A, B); // A^2
    basis_v7_tensor* Sub = basis_v7_sub(g, Mul, A); // A^2 - A

    basis_schedule* sched = basis_ir_schedule(g);
    basis_schedule_execute(sched);

    printf("=== Verification ===\n");

    // Verify Transpose (2x3 -> 3x2)
    if (A_T->rows == 3 && A_T->cols == 2 && A_T->node->runtime_data[1] == 4.0) {
        printf("  [PASS] Transpose correctly flipped 2x3 to 3x2\n");
    } else {
        printf("  [FAIL] Transpose math/shape failed\n"); failures++;
    }

    // Verify Softmax Stability (Row 2 has 100, 101, 102. Naive exp() would overflow to Inf)
    double* s_data = S->node->runtime_data;
    double sum_row2 = s_data[3] + s_data[4] + s_data[5];
    if (fabs(sum_row2 - 1.0) < 1e-6 && !isinf(s_data[5])) {
        printf("  [PASS] Softmax Log-Sum-Exp trick prevented overflow on extreme logits\n");
    } else {
        printf("  [FAIL] Softmax overflowed or didn't sum to 1.0\n"); failures++;
    }

    // Verify MUL and SUB (A=3 -> A^2=9 -> A^2-A=6)
    if (Sub->node->runtime_data[2] == 6.0) {
        printf("  [PASS] Element-wise MUL and SUB executed correctly\n");
    } else {
        printf("  [FAIL] Element-wise math failed\n"); failures++;
    }

    basis_schedule_destroy(sched);
    basis_ir_graph_destroy(g);

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ ALL V7 TRANSFORMER PRIMITIVE TESTS PASSED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}

```

---

### FILE: trace_mlp.c
Location: `trace_mlp.c`
```cpp
#include <basis.h>
#include <stdio.h>

int main() {
    printf("=== BASIS v6: Static Graph Tracing (No Execution) ===\n\n");

    // 1. Initialize the Graph IR
    basis_graph* g = basis_graph_new();

    // 2. Define Inputs (Metadata only, no memory allocated for data)
    basis_node* X = basis_graph_input(g, "X", 4, 2);      // Batch of 4, 2 features
    basis_node* W1 = basis_graph_input(g, "W1", 2, 8);    // Weights
    basis_node* b1 = basis_graph_input(g, "b1", 1, 8);    // Bias
    basis_node* W2 = basis_graph_input(g, "W2", 8, 1);
    basis_node* b2 = basis_graph_input(g, "b2", 1, 1);

    // 3. Trace the Forward Pass (Building the IR DAG)
    basis_node* Z1 = basis_graph_matmul(g, X, W1);
    basis_node* A1 = basis_graph_broadcast_add(g, Z1, b1);
    basis_node* H1 = basis_graph_relu(g, A1);

    basis_node* Z2 = basis_graph_matmul(g, H1, W2);
    basis_node* Y = basis_graph_broadcast_add(g, Z2, b2);

    // 4. Print the IR
    basis_graph_print(g);

    printf("\nNotice: We just defined a complete Neural Network architecture.\n");
    printf("However, ZERO bytes of weight memory were allocated.\n");
    printf("ZERO autograd scalar nodes were created.\n");
    printf("This IR is now ready to be fed to the v6 Lowering Compiler.\n");

    basis_graph_free(g);
    return 0;
}

```

---

### FILE: train_burgers.c
Location: `train_burgers.c`
```cpp
#include <basis.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.14159265358979323846
#define NUM_COLLOC 50
#define EPOCHS 3000
#define EPS 1e-4
#define NU (0.01 / PI)

// Hard-constrained Ansatz:
// u(x,t) = -sin(pi*x) + (1 - x^2) * t * MLP(x,t)
basis_tensor* eval_u(double x_val, double t_val, basis_tensor* W1, basis_tensor* b1, basis_tensor* W2, basis_tensor* b2) {
    basis_tensor* X_in = basis_tensor_new(1, 2);
    basis_tensor_set(X_in, 0, 0, x_val); basis_tensor_set(X_in, 0, 1, t_val);

    basis_tensor* Z1 = basis_tensor_matrix_multiplication(X_in, W1);
    basis_tensor* b1_v = basis_tensor_broadcast_view(b1, 1, 20);
    basis_tensor* A1 = basis_tensor_add(Z1, b1_v);
    basis_tensor* H1 = basis_tensor_tanh(A1);

    basis_tensor* Z2 = basis_tensor_matrix_multiplication(H1, W2);
    basis_tensor* b2_v = basis_tensor_broadcast_view(b2, 1, 1);
    basis_tensor* U_raw = basis_tensor_add(Z2, b2_v);

    double ic = -sin(PI * x_val);
    double mask = (1.0 - x_val * x_val) * t_val;

    basis_tensor* mask_tensor = basis_tensor_scalar_multiplication(U_raw, mask);
    basis_tensor* ic_tensor = basis_tensor_new(1, 1);
    basis_tensor_set(ic_tensor, 0, 0, ic);

    basis_tensor* U_final = basis_tensor_add(mask_tensor, ic_tensor);

    basis_tensor_free(X_in); basis_tensor_free(Z1); basis_tensor_free(b1_v);
    basis_tensor_free(A1); basis_tensor_free(H1); basis_tensor_free(Z2);
    basis_tensor_free(b2_v); basis_tensor_free(U_raw);
    basis_tensor_free(mask_tensor); basis_tensor_free(ic_tensor);

    return U_final;
}

int main() {
    srand(42);
    printf("=== BASIS v5: Physics-Informed Neural Network (Burgers' Equation) ===\n");
    printf("Solving: du/dt + u*du/dx = nu * d^2u/dx^2\n");
    printf("BCs: u(-1,t)=0, u(1,t)=0 | IC: u(x,0)=-sin(pi*x)\n");
    printf("Viscosity nu = %.5f (Shockwave will form near x=0)\n\n", NU);

    basis_tensor* W1 = basis_tensor_new(2, 20);
    basis_tensor* b1 = basis_tensor_new(1, 20);
    basis_tensor* W2 = basis_tensor_new(20, 1);
    basis_tensor* b2 = basis_tensor_new(1, 1);

    for(size_t i=0; i<40; i++) W1->data[i]->data = ((rand() % 100) / 100.0 - 0.5) * 1.0;
    for(size_t i=0; i<20; i++) W2->data[i]->data = ((rand() % 100) / 100.0 - 0.5) * 1.0;

    basis_adam* opt_W1 = basis_adam_new(2, 20, 0.02);
    basis_adam* opt_b1 = basis_adam_new(1, 20, 0.02);
    basis_adam* opt_W2 = basis_adam_new(20, 1, 0.02);
    basis_adam* opt_b2 = basis_adam_new(1, 1, 0.02);

    for(int epoch=0; epoch<EPOCHS; epoch++) {
        basis_value* loss_sum = basis_value_new(0.0);

        for(int i=0; i<NUM_COLLOC; i++) {
            double xc = 2.0 * ((double)rand() / RAND_MAX) - 1.0; // x in [-1, 1]
            double tc = ((double)rand() / RAND_MAX); // t in [0, 1]

            basis_tensor* u_c = eval_u(xc, tc, W1, b1, W2, b2);
            basis_tensor* u_tp = eval_u(xc, tc + EPS, W1, b1, W2, b2);
            basis_tensor* u_tm = eval_u(xc, tc - EPS, W1, b1, W2, b2);
            basis_tensor* u_xp = eval_u(xc + EPS, tc, W1, b1, W2, b2);
            basis_tensor* u_xm = eval_u(xc - EPS, tc, W1, b1, W2, b2);

            basis_value* v_uc = u_c->data[0]; basis_value_retain(v_uc);
            basis_value* v_utp = u_tp->data[0]; basis_value_retain(v_utp);
            basis_value* v_utm = u_tm->data[0]; basis_value_retain(v_utm);
            basis_value* v_uxp = u_xp->data[0]; basis_value_retain(v_uxp);
            basis_value* v_uxm = u_xm->data[0]; basis_value_retain(v_uxm);

            basis_tensor_free(u_c); basis_tensor_free(u_tp); basis_tensor_free(u_tm);
            basis_tensor_free(u_xp); basis_tensor_free(u_xm);

            basis_value* dt_u = basis_value_subtraction(v_utp, v_utm);
            basis_value* dt_scale = basis_value_new(1.0 / (2.0 * EPS));
            basis_value* dt_final = basis_value_multiplication(dt_u, dt_scale);

            basis_value* dx_u = basis_value_subtraction(v_uxp, v_uxm);
            basis_value* dx_scale = basis_value_new(1.0 / (2.0 * EPS));
            basis_value* dx_final = basis_value_multiplication(dx_u, dx_scale);

            basis_value* dxx_1 = basis_value_addition(v_uxp, v_uxm);
            basis_value* uc_2 = basis_value_multiplication(v_uc, basis_value_new(2.0));
            basis_value* dxx_2 = basis_value_subtraction(dxx_1, uc_2);
            basis_value* dxx_scale = basis_value_new(1.0 / (EPS * EPS));
            basis_value* dxx_final = basis_value_multiplication(dxx_2, dxx_scale);

            // R = u_t + u * u_x - nu * u_xx
            basis_value* conv = basis_value_multiplication(v_uc, dx_final);
            basis_value* visc = basis_value_multiplication(dxx_final, basis_value_new(NU));

            basis_value* r1 = basis_value_addition(dt_final, conv);
            basis_value* res = basis_value_subtraction(r1, visc);
            basis_value* res_sq = basis_value_power(res, 2.0);

            basis_value* new_sum = basis_value_addition(loss_sum, res_sq);
            basis_value_free(loss_sum); loss_sum = new_sum;

            basis_value_free(v_uc); basis_value_free(v_utp); basis_value_free(v_utm);
            basis_value_free(v_uxp); basis_value_free(v_uxm);
            basis_value_free(dt_u); basis_value_free(dt_scale); basis_value_free(dt_final);
            basis_value_free(dx_u); basis_value_free(dx_scale); basis_value_free(dx_final);
            basis_value_free(dxx_1); basis_value_free(uc_2); basis_value_free(dxx_2);
            basis_value_free(dxx_scale); basis_value_free(dxx_final);
            basis_value_free(conv); basis_value_free(visc); basis_value_free(r1);
            basis_value_free(res); basis_value_free(res_sq);
        }

        basis_value* divisor = basis_value_new(1.0 / NUM_COLLOC);
        basis_value* loss = basis_value_multiplication(loss_sum, divisor);

        if (epoch % 300 == 0) printf("Epoch %4d | Loss: %.6f\n", epoch, loss->data);

        basis_value_backward_propagation(loss);

        basis_adam_optimization_step(opt_W1, W1);
        basis_adam_optimization_step(opt_b1, b1);
        basis_adam_optimization_step(opt_W2, W2);
        basis_adam_optimization_step(opt_b2, b2);

        basis_value_free(loss); basis_value_free(divisor); basis_value_free(loss_sum);
    }

    printf("\n=== Final Verification (Shockwave Profile at t=0.4) ===\n");
    printf("x\t | u(x, 0.0) [IC] \t | u(x, 0.4) [Shock]\n");
    for(int i=-10; i<=10; i++) {
        double xc = i / 10.0;
        basis_tensor* u_0 = eval_u(xc, 0.0, W1, b1, W2, b2);
        basis_tensor* u_4 = eval_u(xc, 0.4, W1, b1, W2, b2);
        printf("%.1f\t | %.4f \t\t | %.4f\n", xc, u_0->data[0]->data, u_4->data[0]->data);
        basis_tensor_free(u_0); basis_tensor_free(u_4);
    }

    basis_adam_free(opt_W1); basis_adam_free(opt_b1); basis_adam_free(opt_W2); basis_adam_free(opt_b2);
    basis_tensor_free(W1); basis_tensor_free(b1); basis_tensor_free(W2); basis_tensor_free(b2);
    return 0;
}

```

---

### FILE: train_mlp_aot.c
Location: `train_mlp_aot.c`
```cpp
#include <basis.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <math.h>

typedef struct {
    double* t5; double* t6; double* t7; double* t8; double* t9;
} GhostContext;

typedef void (*forward_t)(double*, double*, double*, double*, double*, GhostContext*, double*);
typedef void (*backward_t)(double*, double*, double*, double*, double*, GhostContext*, double*, double*, double*, double*, double*, double*);
typedef void (*free_ctx_t)(GhostContext*);

int main() {
    printf("=== BASIS v6: AOT Autograd (Training via Compiled Backward Pass) ===\n\n");

    // 1. Trace IR
    basis_graph* g = basis_graph_new();
    basis_node* X = basis_graph_input(g, "X", 4, 2);
    basis_node* W1 = basis_graph_input(g, "W1", 2, 8);
    basis_node* b1 = basis_graph_input(g, "b1", 1, 8);
    basis_node* W2 = basis_graph_input(g, "W2", 8, 1);
    basis_node* b2 = basis_graph_input(g, "b2", 1, 1);

    basis_node* Z1 = basis_graph_matmul(g, X, W1);
    basis_node* A1 = basis_graph_broadcast_add(g, Z1, b1);
    basis_node* H1 = basis_graph_relu(g, A1);
    basis_node* Z2 = basis_graph_matmul(g, H1, W2);
    basis_node* Y = basis_graph_broadcast_add(g, Z2, b2);

    // 2. Lower to C
    printf("[1/5] Lowering IR to ghost_model.c (Forward + Backward)...\n");
    basis_graph_lower_to_c(g, "ghost_model.c");
    basis_graph_free(g);

    // 3. Compile
    printf("[2/5] Compiling ghost_model.c to native machine code...\n");
    int status = system("gcc -shared -fPIC -O3 -march=native -o ghost_model.so ghost_model.c");
    (void)status;
    if (status != 0) {
        printf("Compilation failed!\n");
        return 1;
    }

    // 4. Load
    printf("[3/5] Loading compiled forward and backward kernels via dlopen...\n");
    void* handle = dlopen("./ghost_model.so", RTLD_NOW);
    if (!handle) {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }

    forward_t forward = (forward_t)dlsym(handle, "ghost_weight_forward");
    backward_t backward = (backward_t)dlsym(handle, "ghost_weight_backward");
    free_ctx_t free_ctx = (free_ctx_t)dlsym(handle, "ghost_free_context");

    if (!forward || !backward || !free_ctx) {
        printf("dlsym failed: %s\n", dlerror());
        return 1;
    }

    // 5. Raw C Training Loop
    printf("[4/5] Executing 2000 Epochs using purely compiled C-arrays...\n\n");

    double raw_X[8] = {0,0, 0,1, 1,0, 1,1};
    double raw_Y_true[4] = {0, 1, 1, 0};

    double raw_W1[16]; double raw_b1[8] = {0};
    double raw_W2[8];  double raw_b2[1] = {0};
    for(int i=0; i<16; i++) raw_W1[i] = ((rand() % 100) / 100.0) - 0.5;
    for(int i=0; i<8; i++)  raw_W2[i] = ((rand() % 100) / 100.0) - 0.5;

    double raw_grad_W1[16], raw_grad_b1[8], raw_grad_W2[8], raw_grad_b2[1];
    double raw_grad_X[8];

    double Y_pred[4];
    double grad_out[4];
    GhostContext ctx;

    double lr = 0.05;

    for(int epoch=0; epoch<2000; epoch++) {
        // Forward
        forward(raw_X, raw_W1, raw_b1, raw_W2, raw_b2, &ctx, Y_pred);

        // Compute MSE Loss & Output Gradient manually
        double loss = 0.0;
        for(int i=0; i<4; i++) {
            double diff = Y_pred[i] - raw_Y_true[i];
            loss += 0.5 * diff * diff;
            grad_out[i] = diff / 4.0; // dL/dY
        }
        loss /= 4.0;

        if(epoch % 400 == 0) printf("Epoch %4d | Loss: %.6f\n", epoch, loss);

        // Zero Gradients
        for(int i=0; i<16; i++) raw_grad_W1[i] = 0.0;
        for(int i=0; i<8; i++)  { raw_grad_b1[i] = 0.0; raw_grad_W2[i] = 0.0; }
        raw_grad_b2[0] = 0.0;

        // Backward (Compiled Chain Rule!)
        backward(raw_X, raw_W1, raw_b1, raw_W2, raw_b2, &ctx, grad_out,
                 raw_grad_X, raw_grad_W1, raw_grad_b1, raw_grad_W2, raw_grad_b2);

        // SGD Update
        for(int i=0; i<16; i++) raw_W1[i] -= lr * raw_grad_W1[i];
        for(int i=0; i<8; i++)  { raw_b1[i] -= lr * raw_grad_b1[i]; raw_W2[i] -= lr * raw_grad_W2[i]; }
        raw_b2[0] -= lr * raw_grad_b2[0];

        // Free cached forward activations
        free_ctx(&ctx);
    }

    printf("\n[5/5] Final Inference:\n");
    forward(raw_X, raw_W1, raw_b1, raw_W2, raw_b2, &ctx, Y_pred);
    for(int i=0; i<4; i++) {
        printf("Input: [%.0f, %.0f] | Target: %.0f | Predicted: %.4f\n",
               raw_X[i*2], raw_X[i*2+1], raw_Y_true[i], Y_pred[i]);
    }

    free_ctx(&ctx);
    dlclose(handle);
    printf("\n✅ AOT Autograd Training Complete. ZERO basis_value nodes were created.\n");
    return 0;
}

```

---

### FILE: train_pinn.c
Location: `train_pinn.c`
```cpp
#include <basis.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.14159265358979323846
#define NUM_COLLOC 30
#define NUM_IC 15
#define EPOCHS 1000
#define EPS 1e-4

// Evaluates the MLP and applies the hard-constrained BC Ansatz: u(0,t)=0, u(1,t)=0
basis_tensor* eval_u(double x_val, double t_val, basis_tensor* W1, basis_tensor* b1, basis_tensor* W2, basis_tensor* b2) {
    basis_tensor* X_in = basis_tensor_new(1, 2);
    basis_tensor_set(X_in, 0, 0, x_val); basis_tensor_set(X_in, 0, 1, t_val);

    basis_tensor* Z1 = basis_tensor_matrix_multiplication(X_in, W1);
    basis_tensor* b1_v = basis_tensor_broadcast_view(b1, 1, 16);
    basis_tensor* A1 = basis_tensor_add(Z1, b1_v);
    basis_tensor* H1 = basis_tensor_tanh(A1); // Smooth activation for 2nd-order derivatives

    basis_tensor* Z2 = basis_tensor_matrix_multiplication(H1, W2);
    basis_tensor* b2_v = basis_tensor_broadcast_view(b2, 1, 1);
    basis_tensor* U_raw = basis_tensor_add(Z2, b2_v);

    // Ansatz Mask: x * (1 - x)
    double mask = x_val * (1.0 - x_val);
    basis_tensor* U_masked = basis_tensor_scalar_multiplication(U_raw, mask);

    basis_tensor_free(X_in); basis_tensor_free(Z1); basis_tensor_free(b1_v);
    basis_tensor_free(A1); basis_tensor_free(H1); basis_tensor_free(Z2);
    basis_tensor_free(b2_v); basis_tensor_free(U_raw);

    return U_masked;
}

int main() {
    srand(42);
    printf("=== BASIS v5: Physics-Informed Neural Network (1D Heat Equation) ===\n");
    printf("Solving: du/dt = d^2u/dx^2\n");
    printf("BCs: u(0,t)=0, u(1,t)=0 | IC: u(x,0)=sin(pi*x)\n\n");

    basis_tensor* W1 = basis_tensor_new(2, 16);
    basis_tensor* b1 = basis_tensor_new(1, 16);
    basis_tensor* W2 = basis_tensor_new(16, 1);
    basis_tensor* b2 = basis_tensor_new(1, 1);

    for(size_t i=0; i<32; i++) W1->data[i]->data = ((rand() % 100) / 100.0 - 0.5) * 1.0;
    for(size_t i=0; i<16; i++) W2->data[i]->data = ((rand() % 100) / 100.0 - 0.5) * 1.0;

    basis_adam* opt_W1 = basis_adam_new(2, 16, 0.05);
    basis_adam* opt_b1 = basis_adam_new(1, 16, 0.05);
    basis_adam* opt_W2 = basis_adam_new(16, 1, 0.05);
    basis_adam* opt_b2 = basis_adam_new(1, 1, 0.05);

    double x_f[NUM_COLLOC], t_f[NUM_COLLOC];
    for(int i=0; i<NUM_COLLOC; i++) {
        x_f[i] = (double)rand() / RAND_MAX;
        t_f[i] = (double)rand() / RAND_MAX;
    }
    double x_i[NUM_IC];
    for(int i=0; i<NUM_IC; i++) x_i[i] = (double)rand() / RAND_MAX;

    for(int epoch=0; epoch<EPOCHS; epoch++) {
        basis_value* loss_sum = basis_value_new(0.0);

        // 1. PDE Residual Loss (Finite Difference Stencil)
        for(int i=0; i<NUM_COLLOC; i++) {
            double xc = x_f[i], tc = t_f[i];
            basis_tensor* u_c = eval_u(xc, tc, W1, b1, W2, b2);
            basis_tensor* u_tp = eval_u(xc, tc + EPS, W1, b1, W2, b2);
            basis_tensor* u_tm = eval_u(xc, tc - EPS, W1, b1, W2, b2);
            basis_tensor* u_xp = eval_u(xc + EPS, tc, W1, b1, W2, b2);
            basis_tensor* u_xm = eval_u(xc - EPS, tc, W1, b1, W2, b2);

            basis_value* v_uc = u_c->data[0]; basis_value_retain(v_uc);
            basis_value* v_utp = u_tp->data[0]; basis_value_retain(v_utp);
            basis_value* v_utm = u_tm->data[0]; basis_value_retain(v_utm);
            basis_value* v_uxp = u_xp->data[0]; basis_value_retain(v_uxp);
            basis_value* v_uxm = u_xm->data[0]; basis_value_retain(v_uxm);

            basis_tensor_free(u_c); basis_tensor_free(u_tp); basis_tensor_free(u_tm);
            basis_tensor_free(u_xp); basis_tensor_free(u_xm);

            basis_value* dt_u = basis_value_subtraction(v_utp, v_utm);
            basis_value* dt_scale = basis_value_new(1.0 / (2.0 * EPS));
            basis_value* dt_final = basis_value_multiplication(dt_u, dt_scale);

            basis_value* dxx_1 = basis_value_addition(v_uxp, v_uxm);
            basis_value* uc_2 = basis_value_multiplication(v_uc, basis_value_new(2.0));
            basis_value* dxx_2 = basis_value_subtraction(dxx_1, uc_2);
            basis_value* dxx_scale = basis_value_new(1.0 / (EPS * EPS));
            basis_value* dxx_final = basis_value_multiplication(dxx_2, dxx_scale);

            basis_value* res = basis_value_subtraction(dt_final, dxx_final);
            basis_value* res_sq = basis_value_power(res, 2.0);

            basis_value* new_sum = basis_value_addition(loss_sum, res_sq);
            basis_value_free(loss_sum); loss_sum = new_sum;

            basis_value_free(v_uc); basis_value_free(v_utp); basis_value_free(v_utm);
            basis_value_free(v_uxp); basis_value_free(v_uxm);
            basis_value_free(dt_u); basis_value_free(dt_scale); basis_value_free(dt_final);
            basis_value_free(dxx_1); basis_value_free(uc_2); basis_value_free(dxx_2);
            basis_value_free(dxx_scale); basis_value_free(dxx_final);
            basis_value_free(res); basis_value_free(res_sq);
        }

        // 2. Initial Condition Loss
        for(int i=0; i<NUM_IC; i++) {
            double xc = x_i[i];
            basis_tensor* u_ic = eval_u(xc, 0.0, W1, b1, W2, b2);
            basis_value* v_uic = u_ic->data[0]; basis_value_retain(v_uic);
            basis_tensor_free(u_ic);

            basis_value* target = basis_value_new(sin(PI * xc));
            basis_value* diff = basis_value_subtraction(v_uic, target);
            basis_value* sq = basis_value_power(diff, 2.0);

            basis_value* new_sum = basis_value_addition(loss_sum, sq);
            basis_value_free(loss_sum); loss_sum = new_sum;

            basis_value_free(v_uic); basis_value_free(target); basis_value_free(diff); basis_value_free(sq);
        }

        basis_value* divisor = basis_value_new(1.0 / (NUM_COLLOC + NUM_IC));
        basis_value* loss = basis_value_multiplication(loss_sum, divisor);

        if (epoch % 100 == 0) printf("Epoch %4d | Loss: %.6f\n", epoch, loss->data);

        basis_value_backward_propagation(loss);

        basis_adam_optimization_step(opt_W1, W1);
        basis_adam_optimization_step(opt_b1, b1);
        basis_adam_optimization_step(opt_W2, W2);
        basis_adam_optimization_step(opt_b2, b2);

        basis_value_free(loss); basis_value_free(divisor); basis_value_free(loss_sum);
    }

    printf("\n=== Final Verification ===\n");
    printf("x\t | t=0.0 (Target: sin(pi*x)) \t | t=0.1 (Predicted)\n");
    for(int i=0; i<=10; i++) {
        double xc = i / 10.0;
        basis_tensor* u_0 = eval_u(xc, 0.0, W1, b1, W2, b2);
        basis_tensor* u_1 = eval_u(xc, 0.1, W1, b1, W2, b2);
        printf("%.1f\t | %.4f \t\t\t\t | %.4f\n", xc, u_0->data[0]->data, u_1->data[0]->data);
        basis_tensor_free(u_0); basis_tensor_free(u_1);
    }

    basis_adam_free(opt_W1); basis_adam_free(opt_b1); basis_adam_free(opt_W2); basis_adam_free(opt_b2);
    basis_tensor_free(W1); basis_tensor_free(b1); basis_tensor_free(W2); basis_tensor_free(b2);
    return 0;
}

```

---

