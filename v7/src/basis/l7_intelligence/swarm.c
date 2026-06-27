#include "basis/l7_intelligence/swarm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

basis_swarm* basis_swarm_create(void) {
    basis_swarm* s = (basis_swarm*)calloc(1, sizeof(basis_swarm));
    s->expert_cap = 16;
    s->experts = (basis_expert**)malloc(sizeof(basis_expert*) * s->expert_cap);
    return s;
}

void basis_swarm_register(basis_swarm* swarm, const char* name, basis_schedule* sched, basis_ir_graph* g) {
    if (swarm->expert_count >= swarm->expert_cap) {
        swarm->expert_cap *= 2;
        swarm->experts = (basis_expert**)realloc(swarm->experts, sizeof(basis_expert*) * swarm->expert_cap);
    }

    basis_expert* exp = (basis_expert*)calloc(1, sizeof(basis_expert));
    exp->name = name;
    exp->schedule = sched;
    exp->scratch = basis_arena_create(1024 * 1024);

    // Map Inputs
    exp->input_count = 0;
    for(uint32_t i=0; i<g->node_count; i++) if(g->nodes[i]->op == BASIS_IR_OP_INPUT) exp->input_count++;

    exp->input_buffers = (double**)calloc(exp->input_count, sizeof(double*));
    exp->input_sizes = (size_t*)calloc(exp->input_count, sizeof(size_t));

    uint32_t idx = 0;
    for(uint32_t i=0; i<g->node_count; i++) {
        if(g->nodes[i]->op == BASIS_IR_OP_INPUT) {
            size_t bytes = g->nodes[i]->rows * g->nodes[i]->cols * sizeof(double);
            g->nodes[i]->runtime_data = (double*)malloc(bytes);
            exp->input_buffers[idx] = g->nodes[i]->runtime_data;
            exp->input_sizes[idx] = bytes;
            idx++;
        }
    }

    // Map Output (Last Node)
    basis_ir_node* out_node = g->nodes[g->node_count - 1];
    exp->out_node = out_node;
    exp->output_size = out_node->rows * out_node->cols * sizeof(double);

    swarm->experts[swarm->expert_count++] = exp;
}

basis_expert* basis_swarm_get(basis_swarm* swarm, const char* name) {
    for(uint32_t i=0; i<swarm->expert_count; i++) {
        if(strcmp(swarm->experts[i]->name, name) == 0) return swarm->experts[i];
    }
    return NULL;
}

void basis_swarm_dispatch(basis_expert* expert, double* in_data, double* out_data) {
    if (!expert || !expert->schedule || expert->input_count == 0) return;

    // 1. Inject data
    memcpy(expert->input_buffers[0], in_data, expert->input_sizes[0]);

    // 2. Nullify intermediate nodes (including output) to force scratch allocation
    for(uint32_t w=0; w<expert->schedule->wave_count; w++) {
        for(uint32_t i=0; i<expert->schedule->waves[w].node_count; i++) {
            basis_ir_node* n = expert->schedule->waves[w].nodes[i];
            if (n->op != BASIS_IR_OP_INPUT && n->op != BASIS_IR_OP_CONST) {
                n->runtime_data = NULL;
            }
        }
    }

    // 3. Execute
    basis_schedule_execute(expert->schedule, expert->scratch);

    // 4. Extract result directly from the node's newly allocated scratch buffer
    memcpy(out_data, expert->out_node->runtime_data, expert->output_size);

    // 5. O(1) Teardown
    basis_arena_reset(expert->scratch);
}

void basis_swarm_destroy(basis_swarm* swarm) {
    if (!swarm) return;
    for(uint32_t i=0; i<swarm->expert_count; i++) {
        basis_expert* e = swarm->experts[i];
        basis_arena_destroy(e->scratch);
        basis_schedule_destroy(e->schedule);

        for(uint32_t j=0; j<e->input_count; j++) free(e->input_buffers[j]);
        free(e->input_buffers);
        free(e->input_sizes);
        free(e);
    }
    free(swarm->experts);
    free(swarm);
}
