/* R2-5 gate: sandbox caps + replay + abstention. Exits 1 on failure. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "lancius/lancius_sandbox.h"
#include "lancius/lancius_error.h"
#include <lancius.h>

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); fails++; } else { printf("  PASS: %s\n", m); } } while (0)

int main(void) {
    printf("=== R2-5 sandbox audit ===\n");
    /* Accept small graph */
    {
        lancius_graph *g = lancius_graph_create();
        lancius_node *a = lancius_input(g, 2, 2);
        lancius_node *b = lancius_input(g, 2, 2);
        lancius_node *c = lancius_add(g, a, b);
        (void)c;
        lancius_sandbox_caps caps = {1024*1024, 1000, 1000};
        CHECK(lancius_sandbox_check_graph(g, &caps) == 0, "accept small graph");
        CHECK(lancius_sandbox_check_weights(g) == 0, "weights ok (no data)");
        lancius_graph_destroy(g);
        lancius_clear_error();
    }
    /* Reject oversized node_count */
    {
        lancius_graph *g = lancius_graph_create();
        lancius_node *a = lancius_input(g, 2, 2);
        (void)a;
        lancius_sandbox_caps caps = {1024*1024, 0, 1000};
        CHECK(lancius_sandbox_check_graph(g, &caps) != 0, "reject max_nodes=0");
        lancius_graph_destroy(g);
        lancius_clear_error();
    }
    /* Reject peak over budget */
    {
        lancius_graph *g = lancius_graph_create();
        lancius_node *a = lancius_input(g, 100, 100);
        lancius_node *b = lancius_input(g, 100, 100);
        lancius_node *c = lancius_add(g, a, b);
        (void)c;
        lancius_sandbox_caps caps = {100, 1000, 1000};
        CHECK(lancius_sandbox_check_graph(g, &caps) != 0, "reject peak over budget");
        lancius_graph_destroy(g);
        lancius_clear_error();
    }
    /* Abstain on NaN weights */
    {
        lancius_graph *g = lancius_graph_create();
        lancius_node *a = lancius_input(g, 2, 2);
        size_t n = 4;
        double *d = (double*)calloc(n, sizeof(double));
        d[0] = NAN;
        a->runtime_data = d;
        lancius_node_bind_external(a, d);
        CHECK(lancius_sandbox_check_weights(g) != 0, "abstain on NaN weights");
        free(d);
        a->runtime_data = NULL;
        lancius_graph_destroy(g);
        lancius_clear_error();
    }
    /* Deterministic replay: same graph twice bit-identical */
    {
        lancius_graph *g = lancius_graph_create();
        lancius_node *a = lancius_input(g, 2, 2);
        lancius_node *b = lancius_relu(g, a);
        double ad[4] = {1.0, -2.0, 3.0, -4.0};
        lancius_node_bind_external(a, ad);
        lancius_schedule *s1 = lancius_ir_schedule(g);
        lancius_arena *ar1 = lancius_arena_create(1024*1024);
        lancius_schedule_execute(s1, ar1);
        double out1[4]; memcpy(out1, b->runtime_data, sizeof(out1));
        /* reset and rerun */
        b->runtime_data = NULL;
        lancius_arena_reset(ar1);
        lancius_schedule_execute(s1, ar1);
        double out2[4]; memcpy(out2, b->runtime_data, sizeof(out2));
        CHECK(memcmp(out1, out2, sizeof(out1)) == 0, "replay bit-identical");
        CHECK(out1[0]==1.0 && out1[1]==0.0 && out1[2]==3.0 && out1[3]==0.0, "relu values");
        lancius_schedule_destroy(s1);
        lancius_arena_destroy(ar1);
        lancius_graph_destroy(g);
        lancius_clear_error();
    }
    if (fails) { printf("SANDBOX AUDIT: %d FAILURES\n", fails); return 1; }
    printf("SANDBOX AUDIT: ALL PROOFS HOLD\n");
    return 0;
}
