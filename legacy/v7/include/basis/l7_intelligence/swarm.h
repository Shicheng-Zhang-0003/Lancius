#ifndef BASIS_V7_SWARM_H
#define BASIS_V7_SWARM_H

#include "basis/core/ir.h"
#include "basis/l5_runtime/scheduler.h"
#include "basis/core/memory.h"

typedef struct {
    const char* name;
    basis_schedule* schedule;
    basis_arena* scratch;      // Isolated memory domain

    double** input_buffers;    // Array of persistent double* pointers
    size_t* input_sizes;       // Array of sizes in bytes
    uint32_t input_count;

    basis_ir_node* out_node;   // Reference to the output node to fetch active scratch data
    size_t output_size;        // Size of output in bytes
} basis_expert;

typedef struct {
    basis_expert** experts;
    uint32_t expert_count;
    uint32_t expert_cap;
} basis_swarm;

basis_swarm* basis_swarm_create(void);
void basis_swarm_register(basis_swarm* swarm, const char* name, basis_schedule* sched, basis_ir_graph* g);
basis_expert* basis_swarm_get(basis_swarm* swarm, const char* name);

void basis_swarm_dispatch(basis_expert* expert, double* in_data, double* out_data);
void basis_swarm_destroy(basis_swarm* swarm);

#endif
