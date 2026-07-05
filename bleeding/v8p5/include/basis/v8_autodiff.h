#ifndef BASIS_V8_AUTODIFF_H
#define BASIS_V8_AUTODIFF_H
#include "basis/v8_ir.h"
typedef struct {
    v8_graph* graph;
    v8_node* loss_node;
    v8_node** grad_nodes;
    uint32_t max_id;
} v8_training_graph;

v8_training_graph* v8_ir_autodiff(v8_graph* fwd_g, v8_node* loss_node);
void v8_training_graph_destroy(v8_training_graph* tg);
#endif
