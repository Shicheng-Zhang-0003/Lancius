#ifndef BASIS_V7_PASS_MANAGER_H
#define BASIS_V7_PASS_MANAGER_H

#include "basis/core/ir.h"

// Runs optimization passes (Constant Folding, Identity Elimination)
// and returns a newly optimized, immutable graph.
basis_ir_graph* basis_ir_run_passes(basis_ir_graph* old_g);

#endif // BASIS_V7_PASS_MANAGER_H
