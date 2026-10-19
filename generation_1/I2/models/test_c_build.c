#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    const char* out_path = argc > 1 ? argv[1] : "/tmp/c_built.lancius";

    /* Build graph: 10x20 · 20x5 → MATMUL → RELU */
    lancius_graph* g = lancius_graph_create();
    if (!g) { printf("FAIL: graph_create\n"); return 1; }

    lancius_node* X = lancius_input(g, 10, 20);       /* id 0 */
    lancius_node* W = lancius_input(g, 20, 5);        /* id 1 */
    /* Bind weights to W */
    double* wdata = (double*)malloc(20 * 5 * sizeof(double));
    for (int i = 0; i < 20 * 5; i++) wdata[i] = 0.1;
    lancius_node_bind_external(W, wdata);

    lancius_node* M = lancius_matmul(g, X, W);        /* id 2 */
    lancius_node* R = lancius_relu(g, M);             /* id 3 */

    if (!X || !W || !M || !R) {
        printf("FAIL: node creation (err=%d)\n", (int)lancius_get_error());
        return 1;
    }

    printf("Graph built: %u nodes\n", g->node_count);
    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        printf("  node %u: op=%d ndim=%u shape=[%zu,%zu,%zu,%zu] inputs=%u\n",
            n->id, n->op, n->ndim, n->shape[0], n->shape[1], n->shape[2], n->shape[3],
            n->input_count);
    }

    if (lancius_graph_save(g, out_path) != 0) {
        printf("FAIL: save (err=%d: %s)\n", (int)lancius_get_error(),
               lancius_error_string(lancius_get_error()));
        return 1;
    }
    printf("Saved to %s\n", out_path);

    /* Load it back */
    lancius_graph_destroy(g);
    g = lancius_graph_load(out_path);
    if (!g) {
        printf("FAIL: load (err=%d: %s)\n", (int)lancius_get_error(),
               lancius_error_string(lancius_get_error()));
        return 1;
    }
    printf("Loaded back OK: %u nodes\n", g->node_count);

    lancius_graph_destroy(g);
    free(wdata);
    printf("SUCCESS\n");
    return 0;
}
