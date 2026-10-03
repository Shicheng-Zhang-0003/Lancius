/* R3-3 gate: an actual framework trains. A 2-4-1 tanh MLP solves XOR
 * end-to-end through graph build + autodiff + train-lib SGD (not a hand
 * loop): loss must fall below threshold with 4/4 correct, reruns from the
 * same seed must agree, and a v2 checkpoint at half-time must resume to
 * the same final loss. Exits 1 on any divergence (no false-green).
 */
#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lancius/lancius_train.h"

static int checks = 0;
static int failures = 0;

#define CHECK(cond, msg) \
    do { \
        checks++; \
        if (!(cond)) { \
            printf("  FAIL: %s (err=%d)\n", msg, (int)lancius_get_error()); \
            failures++; \
        } \
    } while (0)

#define N_ITERS 1500
#define CKPT_AT 750
#define LR 0.3
#define LOSS_GATE 0.01

static unsigned long long rng_state;
static double rnd(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((double)(rng_state >> 33) / (double)(1ULL << 31)) - 0.5;
}

typedef struct {
    lancius_graph* g;
    lancius_arena* sc;
    lancius_schedule* fs;
    lancius_schedule* bs;
    lancius_training_graph* tg;
    lancius_node *X, *W1, *B1, *W2, *B2, *Y, *P, *L;
} xor_net;

static const double XD[8] = {0,0, 0,1, 1,0, 1,1};
static const double YD[4] = {0,1,1,0};

static int xor_build(xor_net* n, unsigned long long seed) {
    memset(n, 0, sizeof(*n));
    rng_state = seed;
    n->g = lancius_graph_create();
    if (!n->g) return 0;
    n->X = lancius_input(n->g, 4, 2);
    n->W1 = lancius_input(n->g, 2, 4);
    n->B1 = lancius_input(n->g, 1, 4);
    n->W2 = lancius_input(n->g, 4, 1);
    n->B2 = lancius_input(n->g, 1, 1);
    n->Y = lancius_input(n->g, 4, 1);
    if (!n->X || !n->W1 || !n->B1 || !n->W2 || !n->B2 || !n->Y) return 0;
    n->W1->runtime_data = malloc(8 * sizeof(double));
    n->B1->runtime_data = malloc(4 * sizeof(double));
    n->W2->runtime_data = malloc(4 * sizeof(double));
    n->B2->runtime_data = malloc(1 * sizeof(double));
    n->X->runtime_data = malloc(8 * sizeof(double));
    n->Y->runtime_data = malloc(4 * sizeof(double));
    if (!n->W1->runtime_data || !n->B1->runtime_data || !n->W2->runtime_data ||
        !n->B2->runtime_data || !n->X->runtime_data || !n->Y->runtime_data) return 0;
    for (int i = 0; i < 8; i++) n->W1->runtime_data[i] = rnd() * sqrt(2.0 / 2.0);
    for (int i = 0; i < 4; i++) n->B1->runtime_data[i] = 0.0;
    for (int i = 0; i < 4; i++) n->W2->runtime_data[i] = rnd() * sqrt(2.0 / 4.0);
    n->B2->runtime_data[0] = 0.0;
    memcpy(n->X->runtime_data, XD, sizeof(XD));
    memcpy(n->Y->runtime_data, YD, sizeof(YD));
    lancius_node* H = lancius_tanh(n->g, lancius_add(n->g, lancius_matmul(n->g, n->X, n->W1), n->B1));
    n->P = lancius_add(n->g, lancius_matmul(n->g, H, n->W2), n->B2);
    n->L = lancius_mse(n->g, n->P, n->Y);
    if (!H || !n->P || !n->L) return 0;
    n->sc = lancius_arena_create(64 * 1024 * 1024);
    if (!n->sc) return 0;
    n->fs = lancius_ir_schedule(n->g);
    if (!n->fs) return 0;
    n->tg = lancius_ir_autodiff(n->g, n->L);
    if (!n->tg) return 0;
    n->bs = lancius_ir_schedule(n->tg->graph);
    if (!n->bs) return 0;
    return 1;
}

static void xor_free(xor_net* n) {
    if (!n) return;
    if (n->X) free(n->X->runtime_data);
    if (n->W1) free(n->W1->runtime_data);
    if (n->B1) free(n->B1->runtime_data);
    if (n->W2) free(n->W2->runtime_data);
    if (n->B2) free(n->B2->runtime_data);
    if (n->Y) free(n->Y->runtime_data);
    /* NOTE: weight buffers are shared with the backward graph by pointer
     * (autodiff clones INPUTs by reference); free once, via forward nodes. */
    if (n->bs) lancius_schedule_destroy(n->bs);
    if (n->tg) lancius_training_graph_destroy(n->tg);
    if (n->fs) lancius_schedule_destroy(n->fs);
    if (n->sc) lancius_arena_destroy(n->sc);
    if (n->g) lancius_graph_destroy(n->g);
    memset(n, 0, sizeof(*n));
}

static void clear_non_input(lancius_schedule* s) {
    for (uint32_t w = 0; w < s->wave_count; w++)
        for (uint32_t k = 0; k < s->waves[w].node_count; k++) {
            lancius_node* n = s->waves[w].nodes[k];
            if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) n->runtime_data = NULL;
        }
}

/* One full-batch SGD step. Returns current loss, or <0 on failure. */
static double xor_step(xor_net* n) {
    clear_non_input(n->fs);
    lancius_schedule_execute(n->fs, n->sc);
    if (lancius_get_error() != LANCIUS_ERROR_OK || !n->L->runtime_data) return -1.0;
    double loss = n->L->runtime_data[0];
    lancius_arena_reset(n->sc);
    clear_non_input(n->bs);
    lancius_schedule_execute(n->bs, n->sc);
    if (lancius_get_error() != LANCIUS_ERROR_OK) { lancius_arena_reset(n->sc); return -1.0; }
    double* gW1 = n->tg->grad_nodes[n->W1->id] ? n->tg->grad_nodes[n->W1->id]->runtime_data : NULL;
    double* gB1 = n->tg->grad_nodes[n->B1->id] ? n->tg->grad_nodes[n->B1->id]->runtime_data : NULL;
    double* gW2 = n->tg->grad_nodes[n->W2->id] ? n->tg->grad_nodes[n->W2->id]->runtime_data : NULL;
    double* gB2 = n->tg->grad_nodes[n->B2->id] ? n->tg->grad_nodes[n->B2->id]->runtime_data : NULL;
    if (!gW1 || !gB1 || !gW2 || !gB2) { lancius_arena_reset(n->sc); return -1.0; }
    lancius_sgd_step(n->W1->runtime_data, gW1, 8, LR);
    lancius_sgd_step(n->B1->runtime_data, gB1, 4, LR);
    lancius_sgd_step(n->W2->runtime_data, gW2, 4, LR);
    lancius_sgd_step(n->B2->runtime_data, gB2, 1, LR);
    lancius_arena_reset(n->sc);
    return loss;
}

static double xor_train(xor_net* n, int iters) {
    double loss = -1.0;
    for (int it = 0; it < iters; it++) {
        loss = xor_step(n);
        if (loss < 0) return -1.0;
    }
    return loss;
}

static int xor_score(xor_net* n) {
    clear_non_input(n->fs);
    lancius_schedule_execute(n->fs, n->sc);
    int ok = 1;
    if (!n->P->runtime_data) ok = 0;
    for (int i = 0; ok && i < 4; i++) {
        int pred = n->P->runtime_data[i] > 0.5 ? 1 : 0;
        if (pred != (int)YD[i]) ok = 0;
    }
    lancius_arena_reset(n->sc);
    return ok;
}

int main(void) {
    printf("TRAIN CONVERGENCE AUDIT (R3-3)\n");

    /* 1. Uninterrupted run solves XOR. */
    xor_net a;
    CHECK(xor_build(&a, 0x12345ULL), "xor net builds + autodiff");
    double lossA = xor_train(&a, N_ITERS);
    printf("  uninterrupted loss: %.6f\n", lossA);
    CHECK(lossA >= 0.0 && lossA < LOSS_GATE, "loss falls below gate");
    CHECK(xor_score(&a) == 1, "4/4 correct");

    /* 2. Same seed reruns agree (deterministic training). */
    xor_net b;
    CHECK(xor_build(&b, 0x12345ULL), "second net builds");
    double lossB = xor_train(&b, N_ITERS);
    printf("  rerun loss: %.6f\n", lossB);
    CHECK(lossB >= 0.0 && fabs(lossA - lossB) < 1e-12, "same-seed rerun agrees");
    xor_free(&b);

    /* 3. Checkpoint at half-time resumes to the same loss. */
    xor_net c;
    CHECK(xor_build(&c, 0x12345ULL), "third net builds");
    double half = xor_train(&c, CKPT_AT);
    CHECK(half >= 0.0, "half run completes");
    CHECK(lancius_graph_save(c.g, "/tmp/opencode/xor_ckpt.lancius") == 0, "checkpoint saves");
    /* Reload into a fresh graph; remap weight nodes by id. */
    lancius_graph* g2 = lancius_graph_load("/tmp/opencode/xor_ckpt.lancius");
    CHECK(g2 != NULL, "checkpoint reloads");
    xor_net d;
    memset(&d, 0, sizeof(d));
    if (g2) {
        /* Remap by node id: the v2 loader preserves ids, and c is still
         * alive so its ids are the ground truth (no shape guessing). */
        for (uint32_t i = 0; i < g2->node_count; i++) {
            lancius_node* n = g2->nodes[i];
            if (n->id == c.X->id) d.X = n;
            else if (n->id == c.W1->id) d.W1 = n;
            else if (n->id == c.B1->id) d.B1 = n;
            else if (n->id == c.W2->id) d.W2 = n;
            else if (n->id == c.B2->id) d.B2 = n;
            else if (n->id == c.Y->id) d.Y = n;
        }
        CHECK(d.X && d.W1 && d.B1 && d.W2 && d.B2 && d.Y, "checkpoint weight remap by id");
        if (d.X && d.W1 && d.B1 && d.W2 && d.B2 && d.Y) {
            /* Rebuild compute edges on the reloaded weights. Payloads came
             * back as owned-heap copies, independent of run c. */
            d.g = g2;
            lancius_node* H = lancius_tanh(g2, lancius_add(g2, lancius_matmul(g2, d.X, d.W1), d.B1));
            d.P = lancius_add(g2, lancius_matmul(g2, H, d.W2), d.B2);
            d.L = lancius_mse(g2, d.P, d.Y);
            d.sc = lancius_arena_create(64 * 1024 * 1024);
            d.fs = (H && d.P && d.L && d.sc) ? lancius_ir_schedule(g2) : NULL;
            d.tg = d.fs ? lancius_ir_autodiff(g2, d.L) : NULL;
            d.bs = d.tg ? lancius_ir_schedule(d.tg->graph) : NULL;
            CHECK(d.bs != NULL, "resumed graph trains");
            if (d.bs) {
                double lossC = xor_train(&d, N_ITERS - CKPT_AT);
                printf("  resumed loss: %.6f\n", lossC);
                CHECK(lossC >= 0.0 && fabs(lossA - lossC) < 1e-9, "resume matches uninterrupted");
            }
        }
        /* d borrows g2 nodes; payloads are heap-owned by the graph and
         * released by graph destroy. Schedules/arena freed explicitly. */
        if (d.bs) lancius_schedule_destroy(d.bs);
        if (d.tg) lancius_training_graph_destroy(d.tg);
        if (d.fs) lancius_schedule_destroy(d.fs);
        if (d.sc) lancius_arena_destroy(d.sc);
        lancius_graph_destroy(g2);
        memset(&d, 0, sizeof(d));
    }
    xor_free(&a);
    xor_free(&c);

    printf("TRAIN CONVERGENCE AUDIT COMPLETE: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
