#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define EPS 1e-5
#define TOL 1e-4

double compute_loss(lancius_schedule* sched, lancius_arena* scratch, lancius_node* loss_node) {
    for(uint32_t w=0; w<sched->wave_count; w++) {
        for(uint32_t k=0; k<sched->waves[w].node_count; k++) {
            lancius_node* n = sched->waves[w].nodes[k];
            if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) n->runtime_data = NULL;
        }
    }
    lancius_schedule_execute(sched, scratch);
    if (!loss_node->runtime_data) {
        fprintf(stderr, "FATAL: loss node has no data\n");
        exit(1);
    }
    double l = loss_node->runtime_data[0];
    lancius_arena_reset(scratch);
    return l;
}

int main() {
    printf("================================================================\n");
    printf("  Lancius: FINITE DIFFERENCE GRADIENT CHECKER (RED TEAM)    \n");
    printf("================================================================\n\n");

    lancius_graph* g = lancius_graph_create();
    if (!g) { fprintf(stderr, "FATAL: graph create failed\n"); return 1; }
    lancius_node* X = lancius_input_4d(g, 2, 3, 8, 8);
    lancius_node* W1 = lancius_input_4d(g, 4, 3, 3, 3);
    lancius_node* C1 = lancius_conv2d(g, X, W1, 1, 0);
    lancius_node* R1 = lancius_relu(g, C1);
    lancius_node* F1 = lancius_flatten(g, R1);
    lancius_node* W2 = lancius_input(g, 144, 5);
    lancius_node* Z1 = lancius_matmul(g, F1, W2);
    lancius_node* Y = lancius_input(g, 2, 5);
    lancius_node* loss = lancius_cross_entropy(g, Z1, Y);
    if (!X || !W1 || !C1 || !R1 || !F1 || !W2 || !Z1 || !Y || !loss) {
        fprintf(stderr, "FATAL: graph build failed\n");
        lancius_graph_destroy(g);
        return 1;
    }

    size_t x_sz = lancius_node_elements(X);
    size_t w1_sz = lancius_node_elements(W1);
    size_t w2_sz = lancius_node_elements(W2);
    size_t y_sz = lancius_node_elements(Y);

    X->runtime_data = (double*)calloc(x_sz, sizeof(double));
    W1->runtime_data = (double*)calloc(w1_sz, sizeof(double));
    W2->runtime_data = (double*)calloc(w2_sz, sizeof(double));
    Y->runtime_data = (double*)calloc(y_sz, sizeof(double));
    if (!X->runtime_data || !W1->runtime_data || !W2->runtime_data || !Y->runtime_data) {
        fprintf(stderr, "FATAL: OOM allocating dummy data\n");
        free(X->runtime_data); free(W1->runtime_data); free(W2->runtime_data); free(Y->runtime_data);
        lancius_graph_destroy(g);
        return 1;
    }

    srand(42);
    for(size_t i=0; i<x_sz; i++) X->runtime_data[i] = ((double)rand()/RAND_MAX) - 0.5;
    for(size_t i=0; i<w1_sz; i++) W1->runtime_data[i] = ((double)rand()/RAND_MAX) - 0.5;
    for(size_t i=0; i<w2_sz; i++) W2->runtime_data[i] = ((double)rand()/RAND_MAX) - 0.5;

    memset(Y->runtime_data, 0, y_sz * sizeof(double));
    Y->runtime_data[2] = 1.0;
    Y->runtime_data[5 + 4] = 1.0;

    lancius_schedule* fwd_sched = lancius_ir_schedule(g);
    if (!fwd_sched) {
        fprintf(stderr, "FATAL: forward schedule failed\n");
        free(X->runtime_data); free(W1->runtime_data); free(W2->runtime_data); free(Y->runtime_data);
        lancius_graph_destroy(g);
        return 1;
    }
    lancius_arena* scratch = lancius_arena_create(64 * 1024 * 1024);
    if (!scratch) {
        fprintf(stderr, "FATAL: OOM scratch arena\n");
        lancius_schedule_destroy(fwd_sched);
        free(X->runtime_data); free(W1->runtime_data); free(W2->runtime_data); free(Y->runtime_data);
        lancius_graph_destroy(g);
        return 1;
    }

    printf("[1/3] Computing Analytical Gradients via Autodiff...\n");
    lancius_training_graph* tg = lancius_ir_autodiff(g, loss);
    if (!tg) {
        fprintf(stderr, "FATAL: autodiff failed\n");
        lancius_schedule_destroy(fwd_sched);
        lancius_arena_destroy(scratch);
        free(X->runtime_data); free(W1->runtime_data); free(W2->runtime_data); free(Y->runtime_data);
        lancius_graph_destroy(g);
        return 1;
    }
    lancius_schedule* bwd_sched = lancius_ir_schedule(tg->graph);
    if (!bwd_sched) {
        fprintf(stderr, "FATAL: backward schedule failed\n");
        lancius_training_graph_destroy(tg);
        lancius_schedule_destroy(fwd_sched);
        lancius_arena_destroy(scratch);
        free(X->runtime_data); free(W1->runtime_data); free(W2->runtime_data); free(Y->runtime_data);
        lancius_graph_destroy(g);
        return 1;
    }

    for(uint32_t w=0; w<bwd_sched->wave_count; w++) {
        for(uint32_t k=0; k<bwd_sched->waves[w].node_count; k++) {
            lancius_node* n = bwd_sched->waves[w].nodes[k];
            if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) n->runtime_data = NULL;
        }
    }
    lancius_schedule_execute(bwd_sched, scratch);

    double* ag_w1_ptr = tg->grad_nodes[W1->id] ? tg->grad_nodes[W1->id]->runtime_data : NULL;
    double* ag_w2_ptr = tg->grad_nodes[W2->id] ? tg->grad_nodes[W2->id]->runtime_data : NULL;

    if (!ag_w1_ptr || !ag_w2_ptr) {
        printf("FATAL: Autodiff failed to produce gradients.\n");
        lancius_schedule_destroy(bwd_sched);
        lancius_training_graph_destroy(tg);
        lancius_schedule_destroy(fwd_sched);
        lancius_arena_destroy(scratch);
        free(X->runtime_data); free(W1->runtime_data); free(W2->runtime_data); free(Y->runtime_data);
        lancius_graph_destroy(g);
        return 1;
    }
    /* Despot audit: grads live in the scratch arena; lancius_arena_reset()
     * below invalidates them and compute_loss() reuses the arena, so the
     * old code compared freed memory by allocation-order luck. Copy out. */
    double* ag_w1 = (double*)malloc(w1_sz * sizeof(double));
    double* ag_w2 = (double*)malloc(w2_sz * sizeof(double));
    if (!ag_w1 || !ag_w2) {
        fprintf(stderr, "FATAL: OOM copying analytic grads\n");
        free(ag_w1); free(ag_w2);
        lancius_schedule_destroy(bwd_sched);
        lancius_training_graph_destroy(tg);
        lancius_schedule_destroy(fwd_sched);
        lancius_arena_destroy(scratch);
        free(X->runtime_data); free(W1->runtime_data); free(W2->runtime_data); free(Y->runtime_data);
        lancius_graph_destroy(g);
        return 1;
    }
    memcpy(ag_w1, ag_w1_ptr, w1_sz * sizeof(double));
    memcpy(ag_w2, ag_w2_ptr, w2_sz * sizeof(double));
    lancius_arena_reset(scratch);

    printf("[2/3] Computing Finite Differences (Perturbing Weights)...\n");
    double* fd_w1 = (double*)calloc(w1_sz, sizeof(double));
    double* fd_w2 = (double*)calloc(w2_sz, sizeof(double));
    if (!fd_w1 || !fd_w2) {
        fprintf(stderr, "FATAL: OOM allocating FD buffers\n");
        free(fd_w1); free(fd_w2);
        lancius_schedule_destroy(bwd_sched);
        lancius_training_graph_destroy(tg);
        lancius_schedule_destroy(fwd_sched);
        lancius_arena_destroy(scratch);
        free(X->runtime_data); free(W1->runtime_data); free(W2->runtime_data); free(Y->runtime_data);
        lancius_graph_destroy(g);
        return 1;
    }

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
    if (max_err_w1 < TOL) printf("  PASS (Threshold: < %e)\n", TOL);
    else printf("  FAIL\n");

    printf("  MatMul Weights (W2) Max Relative Error: %e\n", max_err_w2);
    if (max_err_w2 < TOL) printf("  PASS (Threshold: < %e)\n", TOL);
    else printf("  FAIL\n");
    printf("================================================================\n");

    int ok = (max_err_w1 < TOL) && (max_err_w2 < TOL);

    free(fd_w1); free(fd_w2); free(ag_w1); free(ag_w2);
    lancius_schedule_destroy(bwd_sched);
    lancius_training_graph_destroy(tg);
    lancius_schedule_destroy(fwd_sched);
    lancius_arena_destroy(scratch);
    free(X->runtime_data); free(W1->runtime_data); free(W2->runtime_data); free(Y->runtime_data);
    lancius_graph_destroy(g);

    return ok ? 0 : 1;
}
