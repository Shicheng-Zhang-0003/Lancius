#ifndef Lancius_LANCIUS_AUTODIFF_H
#define Lancius_LANCIUS_AUTODIFF_H
#include "lancius/lancius_ir.h"
typedef struct {
    lancius_graph* graph;
    lancius_node* loss_node;
    lancius_node** grad_nodes;
    uint32_t max_id;
} lancius_training_graph;

lancius_training_graph* lancius_ir_autodiff(lancius_graph* fwd_g, lancius_node* loss_node);
void lancius_training_graph_destroy(lancius_training_graph* tg);
#endif
