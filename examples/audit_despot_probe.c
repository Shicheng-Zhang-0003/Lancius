#include <lancius.h>
#include "lancius/lancius_memory_planner.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * LANCIUS v12R1 DESPOT PROBE
 *
 * Pins the four load-bearing truth fixes to executable checks. If any of
 * these regress, the failure must be loud (nonzero exit), never green.
 *
 * [1/4] N-dim broadcast values + incompatible-shape rejection (no OOB).
 * [2/4] Cross-rank trailing broadcast ([1,2,2,2] + [2,2]).
 * [3/4] Memory-planner diamond reuse: distinct offsets + pooled parity.
 * [4/4] Cross-entropy degenerate denominator is NUMERICAL, not a sentinel.
 */
int main(void) {
    int fails = 0;

    /* [1/4] Broadcast same-rank [2,2] + [1,2] -> [[11,22],[13,24]] */
    {
        lancius_graph* g = lancius_graph_create();
        lancius_node* A = lancius_input(g, 2, 2);
        lancius_node* B = lancius_input(g, 1, 2);
        lancius_node* C = lancius_add(g, A, B);
        if (!C) { printf("  ❌ FAIL: broadcast builder rejected [2,2]+[1,2]\n"); fails++; }
        else {
            double* ad = (double*)malloc(4 * sizeof(double));
            double* bd = (double*)malloc(2 * sizeof(double));
            ad[0] = 1; ad[1] = 2; ad[2] = 3; ad[3] = 4; bd[0] = 10; bd[1] = 20;
            A->runtime_data = ad; B->runtime_data = bd;
            lancius_schedule* s = lancius_ir_schedule(g);
            lancius_arena* ar = lancius_arena_create(1 << 20);
            lancius_schedule_execute(s, ar);
            double exp[4] = {11, 22, 13, 24};
            for (int i = 0; i < 4; i++) {
                if (fabs(C->runtime_data[i] - exp[i]) > 1e-12) {
                    printf("  ❌ FAIL: broadcast val %d got %f exp %f\n", i, C->runtime_data[i], exp[i]);
                    fails++;
                }
            }
            if (fails == 0) printf("  ✅ PASS: broadcast [2,2]+[1,2]\n");
            /* Incompatible [1,3] vs [1,5] must be rejected, never OOB-read. */
            lancius_node* X = lancius_input(g, 1, 3);
            lancius_node* Y = lancius_input(g, 1, 5);
            lancius_node* Z = lancius_add(g, X, Y);
            if (Z != NULL) { printf("  ❌ FAIL: incompatible [1,3]+[1,5] accepted (OOB risk)\n"); fails++; }
            else printf("  ✅ PASS: incompatible broadcast rejected\n");
            lancius_schedule_destroy(s); lancius_arena_destroy(ar);
            lancius_graph_destroy(g);
            free(ad); free(bd);
        }
    }

    /* [2/4] Cross-rank trailing: 4D [1,2,2,2] + 2D [2,2] */
    {
        lancius_graph* g = lancius_graph_create();
        lancius_node* A = lancius_input_4d(g, 1, 2, 2, 2);
        lancius_node* B = lancius_input(g, 2, 2);
        lancius_node* C = lancius_add(g, A, B);
        if (!C) { printf("  ❌ FAIL: cross-rank builder rejected [1,2,2,2]+[2,2]\n"); fails++; }
        else if (C->ndim != 4 || C->shape[0] != 1 || C->shape[1] != 2 ||
                 C->shape[2] != 2 || C->shape[3] != 2) {
            printf("  ❌ FAIL: cross-rank output shape wrong\n"); fails++;
            lancius_graph_destroy(g);
        } else {
            double* ad = (double*)calloc(8, sizeof(double));
            double* bd = (double*)calloc(4, sizeof(double));
            for (size_t i = 0; i < 8; i++) ad[i] = (double)i;
            bd[0] = 10; bd[1] = 20; bd[2] = 30; bd[3] = 40;
            A->runtime_data = ad; B->runtime_data = bd;
            lancius_schedule* s = lancius_ir_schedule(g);
            lancius_arena* ar = lancius_arena_create(1 << 20);
            lancius_schedule_execute(s, ar);
            if (fabs(C->runtime_data[0] - 10.0) > 1e-12 ||
                fabs(C->runtime_data[7] - 47.0) > 1e-12) {
                printf("  ❌ FAIL: cross-rank vals %f %f\n", C->runtime_data[0], C->runtime_data[7]);
                fails++;
            } else printf("  ✅ PASS: cross-rank trailing broadcast\n");
            lancius_schedule_destroy(s); lancius_arena_destroy(ar);
            lancius_graph_destroy(g);
            free(ad); free(bd);
        }
    }

    /* [3/4] Diamond memory planner: A->B,C->D, pooled must match direct */
    {
        lancius_graph* g = lancius_graph_create();
        lancius_node* A = lancius_input(g, 4, 4);
        double* ad = (double*)malloc(16 * sizeof(double));
        for (int i = 0; i < 16; i++) ad[i] = (double)(i + 1);
        A->runtime_data = ad;
        lancius_node* B = lancius_relu(g, A);
        lancius_node* Cc = lancius_relu(g, A);
        lancius_node* D = lancius_add(g, B, Cc);
        lancius_schedule* s = lancius_ir_schedule(g);
        lancius_arena* ar = lancius_arena_create(1 << 20);
        lancius_schedule_execute(s, ar);
        double direct[16];
        for (int i = 0; i < 16; i++) direct[i] = D->runtime_data[i];
        lancius_memory_plan* plan = lancius_build_memory_plan(s, g);
        if (!plan) { printf("  ❌ FAIL: planner returned NULL on diamond\n"); fails++; }
        else {
            size_t ob = plan->offsets[B->id], oc = plan->offsets[Cc->id];
            if (ob == oc) { printf("  ❌ FAIL: planner overlap B==C offset %zu\n", ob); fails++; }
            else printf("  ✅ PASS: planner offsets distinct B=%zu C=%zu peak=%zu\n", ob, oc, plan->peak_memory);
            void* buf = malloc(plan->peak_memory);
            if (!buf) { printf("  ❌ FAIL: probe flat-buffer OOM\n"); fails++; }
            else {
                lancius_schedule_attach_pool(s, buf, plan);
                lancius_schedule_execute(s, NULL);
                for (int i = 0; i < 16; i++) {
                    if (fabs(D->runtime_data[i] - direct[i]) > 1e-12) {
                        printf("  ❌ FAIL: diamond pooled mismatch %d %f vs %f\n",
                               i, D->runtime_data[i], direct[i]);
                        fails++;
                        break;
                    }
                }
                if (fails == 0) printf("  ✅ PASS: diamond pooled parity\n");
                free(buf);
            }
            lancius_memory_plan_destroy(plan);
        }
        lancius_schedule_destroy(s); lancius_arena_destroy(ar);
        lancius_graph_destroy(g); free(ad);
    }

    /* [4/4] CE degenerate denominator must be NUMERICAL, not a sentinel */
    {
        lancius_graph* g = lancius_graph_create();
        lancius_node* X = lancius_input(g, 1, 2);
        lancius_node* Y = lancius_input(g, 1, 2);
        lancius_node* L = lancius_cross_entropy(g, X, Y);
        (void)L;
        double* xd = (double*)malloc(2 * sizeof(double));
        double* yd = (double*)malloc(2 * sizeof(double));
        xd[0] = -INFINITY; xd[1] = -INFINITY; yd[0] = 1; yd[1] = 0;
        X->runtime_data = xd; Y->runtime_data = yd;
        lancius_schedule* s = lancius_ir_schedule(g);
        lancius_arena* ar = lancius_arena_create(1 << 20);
        lancius_clear_error();
        lancius_schedule_execute(s, ar);
        lancius_error e = lancius_get_error();
        if (e == LANCIUS_ERROR_NUMERICAL) printf("  ✅ PASS: CE degenerate is NUMERICAL\n");
        else { printf("  ❌ FAIL: CE degenerate err=%d (want NUMERICAL=%d)\n", e, LANCIUS_ERROR_NUMERICAL); fails++; }
        lancius_schedule_destroy(s); lancius_arena_destroy(ar);
        lancius_graph_destroy(g); free(xd); free(yd);
    }

    printf("================================================================\n");
    if (fails == 0) printf("  DESPOT PROBE: ALL TRUTH HOLDS\n");
    else printf("  DESPOT PROBE: %d FAILURES\n", fails);
    printf("================================================================\n");
    return fails ? 1 : 0;
}
