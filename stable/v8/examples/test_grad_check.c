#include <basis.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define EPS 1e-5
#define TOL 1e-4

double compute_loss(v8_schedule* sched, v8_arena* scratch, v8_node* loss_node) {
    for(uint32_t w=0; w<sched->wave_count; w++) {
        for(uint32_t k=0; k<sched->waves[w].node_count; k++) {
            v8_node* n = sched->waves[w].nodes[k];
            if (n->op != V8_OP_INPUT && n->op != V8_OP_CONST) n->runtime_data = NULL;
        }
    }
    v8_schedule_execute(sched, scratch);
    double l = loss_node->runtime_data[0];
    v8_arena_reset(scratch);
    return l;
}

int main() {
    printf("================================================================\n");
    printf("  BASIS V8: FINITE DIFFERENCE GRADIENT CHECKER (RED TEAM)    \n");
    printf("================================================================\n\n");

    v8_graph* g = v8_graph_create();
    v8_node* X = v8_input_4d(g, 2, 3, 8, 8);
    v8_node* W1 = v8_input_4d(g, 4, 3, 3, 3);
    v8_node* C1 = v8_conv2d(g, X, W1, 1, 0);
    v8_node* R1 = v8_relu(g, C1);
    v8_node* F1 = v8_flatten(g, R1);
    v8_node* W2 = v8_input(g, 144, 5);
    v8_node* Z1 = v8_matmul(g, F1, W2);
    v8_node* Y = v8_input(g, 2, 5);
    v8_node* loss = v8_cross_entropy(g, Z1, Y);

    size_t x_sz = v8_node_elements(X);
    size_t w1_sz = v8_node_elements(W1);
    size_t w2_sz = v8_node_elements(W2);
    size_t y_sz = v8_node_elements(Y);

    X->runtime_data = (double*)calloc(x_sz, sizeof(double));
    W1->runtime_data = (double*)calloc(w1_sz, sizeof(double));
    W2->runtime_data = (double*)calloc(w2_sz, sizeof(double));
    Y->runtime_data = (double*)calloc(y_sz, sizeof(double));

    srand(42);
    for(size_t i=0; i<x_sz; i++) X->runtime_data[i] = ((double)rand()/RAND_MAX) - 0.5;
    for(size_t i=0; i<w1_sz; i++) W1->runtime_data[i] = ((double)rand()/RAND_MAX) - 0.5;
    for(size_t i=0; i<w2_sz; i++) W2->runtime_data[i] = ((double)rand()/RAND_MAX) - 0.5;

    memset(Y->runtime_data, 0, y_sz * sizeof(double));
    Y->runtime_data[2] = 1.0;
    Y->runtime_data[5 + 4] = 1.0;

    v8_schedule* fwd_sched = v8_ir_schedule(g);
    v8_arena* scratch = v8_arena_create(64 * 1024 * 1024);

    printf("[1/3] Computing Analytical Gradients via Autodiff...\n");
    v8_training_graph* tg = v8_ir_autodiff(g, loss);
    v8_schedule* bwd_sched = v8_ir_schedule(tg->graph);

    for(uint32_t w=0; w<bwd_sched->wave_count; w++) {
        for(uint32_t k=0; k<bwd_sched->waves[w].node_count; k++) {
            v8_node* n = bwd_sched->waves[w].nodes[k];
            if (n->op != V8_OP_INPUT && n->op != V8_OP_CONST) n->runtime_data = NULL;
        }
    }
    v8_schedule_execute(bwd_sched, scratch);
    v8_arena_reset(scratch);

    double* ag_w1 = tg->grad_nodes[W1->id] ? tg->grad_nodes[W1->id]->runtime_data : NULL;
    double* ag_w2 = tg->grad_nodes[W2->id] ? tg->grad_nodes[W2->id]->runtime_data : NULL;

    if (!ag_w1 || !ag_w2) {
        printf("FATAL: Autodiff failed to produce gradients.\n");
        return 1;
    }

    printf("[2/3] Computing Finite Differences (Perturbing Weights)...\n");
    double* fd_w1 = (double*)calloc(w1_sz, sizeof(double));
    double* fd_w2 = (double*)calloc(w2_sz, sizeof(double));

    for(size_t i=0; i<w1_sz; i++) {
        W1->runtime_data[i] += EPS;
        double l_plus = compute_loss(fwd_sched, scratch, loss);
        W1->runtime_data[i] -= 2*EPS;
        double l_minus = compute_loss(fwd_sched, scratch, loss);
        W1->runtime_data[i] += EPS;
        fd_w1[i] = (l_plus - l_minus) / (2*EPS);
    }

    for(size_t i=0; i<w2_sz; i++) {
        W2->runtime_data[i] += EPS;
        double l_plus = compute_loss(fwd_sched, scratch, loss);
        W2->runtime_data[i] -= 2*EPS;
        double l_minus = compute_loss(fwd_sched, scratch, loss);
        W2->runtime_data[i] += EPS;
        fd_w2[i] = (l_plus - l_minus) / (2*EPS);
    }

    printf("[3/3] Comparing Analytical vs. Numerical...\n");
    double max_err_w1 = 0, max_err_w2 = 0;
    for(size_t i=0; i<w1_sz; i++) {
        double err = fabs(ag_w1[i] - fd_w1[i]) / (fabs(ag_w1[i]) + fabs(fd_w1[i]) + 1e-8);
        if (err > max_err_w1) max_err_w1 = err;
    }
    for(size_t i=0; i<w2_sz; i++) {
        double err = fabs(ag_w2[i] - fd_w2[i]) / (fabs(ag_w2[i]) + fabs(fd_w2[i]) + 1e-8);
        if (err > max_err_w2) max_err_w2 = err;
    }

    printf("\n================================================================\n");
    printf("  GRADIENT CHECK RESULTS:\n");
    printf("  Conv2D Weights (W1) Max Relative Error: %e\n", max_err_w1);
    if (max_err_w1 < TOL) printf("  ✅ PASS (Threshold: < %e)\n", TOL);
    else printf("  ❌ FAIL\n");

    printf("  MatMul Weights (W2) Max Relative Error: %e\n", max_err_w2);
    if (max_err_w2 < TOL) printf("  ✅ PASS (Threshold: < %e)\n", TOL);
    else printf("  ❌ FAIL\n");
    printf("================================================================\n");

    return 0;
}
