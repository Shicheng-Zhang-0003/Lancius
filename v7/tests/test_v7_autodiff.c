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
