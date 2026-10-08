/* Leak + soundness probe for the autodiff abort paths.
 *
 * Every failing lancius_ir_autodiff call used to leak tg->grad_nodes.
 * This drives the SAME failing path (GQA forward -> unsupported backward)
 * 2000 times and additionally exercises the shapes that hit the other
 * abort branches, then requires a clean LeakSanitizer report.
 */
#include <lancius.h>
#include <lancius/lancius_autodiff.h>
#include <stdio.h>
#include <stdlib.h>

static int drive_unsupported(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* q = lancius_input_3d(g, 2, 4, 4);
    lancius_node* k = lancius_input_3d(g, 2, 2, 4);
    lancius_node* v = lancius_input_3d(g, 2, 2, 4);
    lancius_node* a = lancius_gqa(g, q, k, v, 4, 2);
    lancius_node* l = a ? lancius_sum(g, a) : NULL;
    lancius_training_graph* tg = l ? lancius_ir_autodiff(g, l) : NULL;
    if (tg) lancius_training_graph_destroy(tg);
    lancius_graph_destroy(g);
    return tg != NULL;
}

static int drive_missing_node(void) {
    /* loss_node == NULL: early return, nothing allocated */
    lancius_graph* g = lancius_graph_create();
    lancius_training_graph* tg = lancius_ir_autodiff(g, NULL);
    if (tg) lancius_training_graph_destroy(tg);
    lancius_graph_destroy(g);
    return tg != NULL;
}

static int drive_foreign_loss(void) {
    /* loss node from another graph -> id out of range abort branch */
    lancius_graph* g1 = lancius_graph_create();
    lancius_graph* g2 = lancius_graph_create();
    lancius_node* a = lancius_input(g2, 3, 3);
    lancius_node* b = lancius_input(g2, 3, 3);
    lancius_node* s = lancius_add(g2, a, b);
    (void)g1;
    lancius_training_graph* tg = lancius_ir_autodiff(g1, s);
    if (tg) lancius_training_graph_destroy(tg);
    lancius_graph_destroy(g1);
    lancius_graph_destroy(g2);
    return tg != NULL;
}

int main(void) {
    int leaked_behaviour = 0;
    for (int i = 0; i < 2000; i++) {
        if (drive_unsupported()) leaked_behaviour++;   /* must stay 0: GQA bwd fails loud */
        if (drive_missing_node())  leaked_behaviour++;
        if (drive_foreign_loss())   leaked_behaviour++;
    }
    printf("autodiff_leak_probe: 6000 failing/edge autodiff calls, "
           "unexpected successes=%d\n", leaked_behaviour);
    return leaked_behaviour ? 1 : 0;
}