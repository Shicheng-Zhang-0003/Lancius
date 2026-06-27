#ifndef BASIS_V7_SCHEDULER_H
#define BASIS_V7_SCHEDULER_H

#include "basis/core/ir.h"
#include <stdint.h>

// A Wave contains a list of nodes that have NO dependencies on each other.
// They can be executed safely in parallel.
typedef struct {
    basis_ir_node** nodes;
    uint32_t node_count;
} basis_wave;

// The Execution Schedule
typedef struct {
    basis_wave* waves;
    uint32_t wave_count;
} basis_schedule;

// Analyzes the IR and groups nodes into deterministic execution waves.
basis_schedule* basis_ir_schedule(basis_ir_graph* g);

// Executes the schedule sequentially (Deterministic baseline).
// Future: basis_schedule_execute_parallel(schedule, thread_pool);
void basis_schedule_execute(basis_schedule* schedule, basis_arena* scratch);

void basis_schedule_print(basis_schedule* schedule);
void basis_schedule_destroy(basis_schedule* schedule);

#endif // BASIS_V7_SCHEDULER_H
