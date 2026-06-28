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
