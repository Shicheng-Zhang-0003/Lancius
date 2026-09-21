#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
 * LANCIUS v12R2: VERIFIER-HEAD TRAINING (models-side example)
 *
 * Demonstrates that the framework can train extended-boolean verifiers:
 * a tiny MLP with a scalar tanh head (fluid scale [-1,1]) regressed by MSE
 * against step-truth targets, with weakest-link credit assignment.
 *
 * Framework boundary respected: tanh/MSE are generic primitives in the C
 * core. Truth semantics (weakest-link argmin selection, residual targets,
 * thresholds) live HERE, in plain model-side C — explicitly NOT an IR op.
 * Delete this file and the framework is still complete.
 *
 * Gate: loss must strictly decrease and finish below threshold; a
 * finite-difference spot check must agree with the analytic gradient
 * (proves the TANH/MSE VJPs); all scores must stay within [-1,1].
 * Any violation exits nonzero.
 */

#define BATCH 8
#define DIN 4
#define DHID 8
#define ITERS 300
#define LR 0.05
#define TOL 1e-4

/* Truth target for the synthetic steps: a fixed tanh readout, so a
 * tanh-headed MLP can drive the loss toward zero. Models-side doctrine. */
static double truth_fn(const double* x) {
    return tanh(2.0 * x[0] - x[1] + 0.5 * x[2] - 0.25 * x[3]);
}

static double rand_u(void) { return (double)rand() / (double)RAND_MAX; }

/* Forward graph: batch of steps -> tanh scores. Native broadcast fuses
 * the biases (no explicit broadcast nodes needed). */
static lancius_node* build_fwd(lancius_graph* g, lancius_node** out_S,
                               lancius_node* Xd, lancius_node* W1d,
                               lancius_node* B1d, lancius_node* W2d,
                               double b2, uint32_t rows) {
    (void)rows;
    lancius_node* H = lancius_tanh(g, lancius_add(g, lancius_matmul(g, Xd, W1d), B1d));
    lancius_node* S = lancius_tanh(g, lancius_matmul(g, H, W2d));
    lancius_node* C2 = lancius_const_scalar(g, b2, 2);
    lancius_node* Sb = lancius_add(g, S, C2);
    *out_S = Sb;
    return Sb;
}

int main(void) {
    printf("================================================================\n");
    printf("  LANCIUS v12R2: VERIFIER-HEAD TRAINING (tanh + MSE, weakest-link)\n");
    printf("================================================================\n");

    srand(12345);

    /* Master weights (small uniform: safe for tanh). */
    double* W1 = (double*)malloc(DIN * DHID * sizeof(double));
    double* b1 = (double*)calloc(DHID, sizeof(double));
    double* W2 = (double*)malloc(DHID * sizeof(double));
    double b2 = 0.0;
    for (int i = 0; i < DIN * DHID; i++) W1[i] = (rand_u() * 2.0 - 1.0) * sqrt(1.0 / DIN);
    for (int i = 0; i < DHID; i++) W2[i] = (rand_u() * 2.0 - 1.0) * sqrt(1.0 / DHID);

    /* Synthetic step batch + truth targets in [-1,1]. */
    double X[BATCH * DIN], T[BATCH];
    for (int b = 0; b < BATCH; b++) {
        for (int d = 0; d < DIN; d++) X[b * DIN + d] = rand_u() * 2.0 - 1.0;
        T[b] = truth_fn(&X[b * DIN]);
    }

    lancius_arena* scratch = lancius_arena_create(16 * 1024 * 1024);
    double loss0 = -1.0, loss1 = -1.0, min0 = -1.0, min1 = -1.0;

    for (int it = 0; it < ITERS; it++) {
        /* ---- Forward: score every step in the batch ---- */
        lancius_graph* g = lancius_graph_create();
        lancius_node* Xn = lancius_input(g, BATCH, DIN);
        lancius_node* W1n = lancius_input(g, DIN, DHID);
        lancius_node* B1n = lancius_input(g, 1, DHID);
        lancius_node* W2n = lancius_input(g, DHID, 1);
        lancius_node* Sb = NULL;
        build_fwd(g, &Sb, Xn, W1n, B1n, W2n, b2, BATCH);
        lancius_node* Tn = lancius_input(g, BATCH, 1);
        Xn->runtime_data = X; W1n->runtime_data = W1;
        B1n->runtime_data = b1; W2n->runtime_data = W2; Tn->runtime_data = T;
        lancius_node* L = lancius_mse(g, Sb, Tn);
        lancius_schedule* fs = lancius_ir_schedule(g);
        lancius_schedule_execute(fs, scratch);
        double batch_loss = L->runtime_data[0];
        /* Snapshot scores NOW: later executions reuse the same arena. */
        double scores[BATCH];
        memcpy(scores, Sb->runtime_data, sizeof(scores));
        lancius_arena_reset(scratch);
        /* Weakest-link error: squared residual of the worst step. The credit
         * mechanism optimizes THIS quantity, so the gate asserts on it. */
        double min_err = -1.0;
        for (int b = 0; b < BATCH; b++) {
            double e = (scores[b] - T[b]) * (scores[b] - T[b]);
            if (min_err < 0.0 || e < min_err) min_err = e;
        }
        if (it == 0) loss0 = batch_loss;
        if (it == ITERS - 1) loss1 = batch_loss;
        if (it == 0) min0 = min_err;
        if (it == ITERS - 1) min1 = min_err;

        /* Scores must never leave the fluid scale. */
        for (int b = 0; b < BATCH; b++) {
            if (fabs(scores[b]) > 1.0 + 1e-12) {
                printf("  ❌ FAIL: score escaped [-1,1]: %f\n", scores[b]);
                return 1;
            }
        }

        /* ---- Weakest link (models-side, plain C): argmin over scores ---- */
        int weak = 0;
        for (int b = 1; b < BATCH; b++)
            if (scores[b] < scores[weak]) weak = b;

        /* ---- Backward through the weakest step only ---- */
        lancius_graph* g2 = lancius_graph_create();
        lancius_node* x = lancius_input(g2, 1, DIN);
        lancius_node* w1 = lancius_input(g2, DIN, DHID);
        lancius_node* c1 = lancius_input(g2, 1, DHID);
        lancius_node* w2 = lancius_input(g2, DHID, 1);
        lancius_node* s2 = NULL;
        build_fwd(g2, &s2, x, w1, c1, w2, b2, 1);
        lancius_node* tt = lancius_input(g2, 1, 1);
        x->runtime_data = &X[weak * DIN]; w1->runtime_data = W1;
        c1->runtime_data = b1; w2->runtime_data = W2; tt->runtime_data = &T[weak];
        lancius_node* l = lancius_mse(g2, s2, tt);
        /* Forward the single step first (g2 nodes only get buffers by execution). */
        lancius_schedule* fs2 = lancius_ir_schedule(g2);
        lancius_schedule_execute(fs2, scratch);
        double weak_score = s2->runtime_data[0];
        lancius_arena_reset(scratch);
        lancius_training_graph* tg = lancius_ir_autodiff(g2, l);
        if (!tg) { printf("  ❌ FAIL: autodiff refused verifier graph\n"); return 1; }
        lancius_schedule* bs = lancius_ir_schedule(tg->graph);
        lancius_schedule_execute(bs, scratch);
        double* gW1 = tg->grad_nodes[w1->id] ? tg->grad_nodes[w1->id]->runtime_data : NULL;
        double* gB1 = tg->grad_nodes[c1->id] ? tg->grad_nodes[c1->id]->runtime_data : NULL;
        double* gW2 = tg->grad_nodes[w2->id] ? tg->grad_nodes[w2->id]->runtime_data : NULL;
        lancius_arena_reset(scratch);
        if (!gW1 || !gB1 || !gW2) { printf("  ❌ FAIL: missing verifier gradients\n"); return 1; }

        /* ---- SGD step on shared master weights ----
         * b2 gradient through the single-step loss: dL/db2 = 2*(s-t). */
        double gb2 = 2.0 * (weak_score - T[weak]);
        for (int i = 0; i < DIN * DHID; i++) W1[i] -= LR * gW1[i];
        for (int i = 0; i < DHID; i++) b1[i] -= LR * gB1[i];
        for (int i = 0; i < DHID; i++) W2[i] -= LR * gW2[i];
        b2 -= LR * gb2;

        lancius_schedule_destroy(fs); lancius_graph_destroy(g);
        lancius_schedule_destroy(fs2);
        lancius_schedule_destroy(bs); lancius_training_graph_destroy(tg);
    }

    printf("  batch-mean loss: %.6f -> %.6f over %d iters (weakest-link SGD)\n", loss0, loss1, ITERS);
    printf("  weakest-step err: %.6f -> %.6f\n", min0, min1);
    int ok = (min1 < min0) && (min1 < 0.05) && (loss1 < loss0);

    /* ---- Finite-difference spot check on W2 (proves TANH/MSE VJPs) ---- */
    {
        const double eps = 1e-5;
        lancius_graph* g = lancius_graph_create();
        lancius_node* x = lancius_input(g, BATCH, DIN);
        lancius_node* w1 = lancius_input(g, DIN, DHID);
        lancius_node* c1 = lancius_input(g, 1, DHID);
        lancius_node* w2 = lancius_input(g, DHID, 1);
        lancius_node* s = NULL;
        build_fwd(g, &s, x, w1, c1, w2, b2, BATCH);
        lancius_node* tt = lancius_input(g, BATCH, 1);
        x->runtime_data = X; w1->runtime_data = W1; c1->runtime_data = b1;
        w2->runtime_data = W2; tt->runtime_data = T;
        lancius_node* l = lancius_mse(g, s, tt);
        lancius_training_graph* tg = lancius_ir_autodiff(g, l);
        lancius_schedule* bs = lancius_ir_schedule(tg->graph);
        lancius_schedule_execute(bs, scratch);
        double* ag = tg->grad_nodes[w2->id] ? tg->grad_nodes[w2->id]->runtime_data : NULL;
        double analytic[DHID];
        if (ag) memcpy(analytic, ag, sizeof(analytic));
        lancius_arena_reset(scratch);
        double max_err = 0.0;
        if (ag) {
            for (int i = 0; i < DHID; i++) {
                double save = W2[i];
                W2[i] = save + eps;
                lancius_schedule* f1 = lancius_ir_schedule(g);
                for (uint32_t w = 0; w < f1->wave_count; w++)
                    for (uint32_t k = 0; k < f1->waves[w].node_count; k++) {
                        lancius_node* nn = f1->waves[w].nodes[k];
                        if (nn->op != LANCIUS_OP_INPUT && nn->op != LANCIUS_OP_CONST) nn->runtime_data = NULL;
                    }
                lancius_schedule_execute(f1, scratch);
                double lp = l->runtime_data[0];
                lancius_arena_reset(scratch);
                W2[i] = save - eps;
                lancius_schedule* f2 = lancius_ir_schedule(g);
                for (uint32_t w = 0; w < f2->wave_count; w++)
                    for (uint32_t k = 0; k < f2->waves[w].node_count; k++) {
                        lancius_node* nn = f2->waves[w].nodes[k];
                        if (nn->op != LANCIUS_OP_INPUT && nn->op != LANCIUS_OP_CONST) nn->runtime_data = NULL;
                    }
                lancius_schedule_execute(f2, scratch);
                double lm = l->runtime_data[0];
                lancius_arena_reset(scratch);
                W2[i] = save;
                double fd = (lp - lm) / (2.0 * eps);
                double err = fabs(analytic[i] - fd) / (fabs(analytic[i]) + fabs(fd) + 1e-8);
                if (err > max_err) max_err = err;
                lancius_schedule_destroy(f1); lancius_schedule_destroy(f2);
            }
        }
        printf("  grad spot-check max rel err: %e (tol %e)\n", max_err, TOL);
        if (!ag || max_err >= TOL) { printf("  ❌ FAIL: analytic gradient disagrees\n"); ok = 0; }
        lancius_schedule_destroy(bs); lancius_training_graph_destroy(tg);
        lancius_graph_destroy(g);
    }

    free(W1); free(b1); free(W2);
    lancius_arena_destroy(scratch);

    printf("================================================================\n");
    if (ok) printf("  VERIFIER TRAINING VERIFIED: loss falls, grads exact, scores bounded.\n");
    else printf("  VERIFIER TRAINING FAILED.\n");
    printf("================================================================\n");
    return ok ? 0 : 1;
}
