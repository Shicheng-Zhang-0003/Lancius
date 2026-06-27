#ifndef BASIS_STAGE6_IR_LOWERING_H
#define BASIS_STAGE6_IR_LOWERING_H

#include "basis/stage6_ir/graph.h"

// Lowers the IR graph into a monolithic C source file containing the forward pass.
// The generated function signature will be:
// void ghost_weight_forward(double* input_0, double* input_1, ..., double* out)
void basis_graph_lower_to_c(basis_graph* g, const char* output_filename);

#endif
