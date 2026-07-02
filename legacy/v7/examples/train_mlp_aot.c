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
