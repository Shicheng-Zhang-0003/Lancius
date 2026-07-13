#ifndef BASIS_V7_AUTODIFF_H
#define BASIS_V7_AUTODIFF_H
#include "basis/core/ir.h"

typedef struct {
    basis_ir_graph* graph;
    basis_ir_node* loss_node;
    basis_ir_node** grad_nodes; // Mapped by forward input ID
    uint32_t max_id;
} basis_training_graph;

basis_training_graph* basis_ir_autodiff(basis_ir_graph* fwd_g, basis_ir_node* loss_node);
void basis_training_graph_destroy(basis_training_graph* tg);
#endif
