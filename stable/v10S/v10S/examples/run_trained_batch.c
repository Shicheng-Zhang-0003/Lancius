#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    lancius_graph* g = lancius_graph_load("trained_lenet.lancius");
    if(!g) { printf("FATAL: Failed to load trained_lenet.lancius\n"); return 1; }

    lancius_node* in_node = NULL;
    lancius_node* out_node = NULL;
    for(uint32_t i=0; i<g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        if (n->op == LANCIUS_OP_INPUT && n->ndim == 4 && n->shape[1] == 3 && n->runtime_data == NULL) in_node = n;
        if (n->ndim == 2 && n->shape[1] == 10 && n->op != LANCIUS_OP_INPUT) out_node = n;
    }
    if(!in_node || !out_node) { printf("FATAL: Could not find I/O nodes\n"); return 1; }

    size_t in_elems = 1 * 3 * 32 * 32;
    float* temp_in = (float*)malloc(in_elems * sizeof(float));
    FILE* f_in = fopen("test_batch.bin", "rb");

    lancius_schedule* sched = lancius_ir_schedule(g);
    lancius_arena* scratch = lancius_arena_create(16 * 1024 * 1024);

    FILE* f_out = fopen("lancius_preds.bin", "wb");
    int32_t pred;

    in_node->runtime_data = (double*)malloc(in_elems * sizeof(double));

    for(int img=0; img<100; img++) {
        fread(temp_in, sizeof(float), in_elems, f_in);
        for(size_t i=0; i<in_elems; i++) in_node->runtime_data[i] = (double)temp_in[i];

        for(uint32_t w=0; w<sched->wave_count; w++) {
            for(uint32_t k=0; k<sched->waves[w].node_count; k++) {
                lancius_node* n = sched->waves[w].nodes[k];
                if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) n->runtime_data = NULL;
            }
        }

        lancius_schedule_execute(sched, scratch);

        double max_logit = -1e9;
        pred = 0;
        for(int c=0; c<10; c++) {
            if(out_node->runtime_data[c] > max_logit) {
                max_logit = out_node->runtime_data[c];
                pred = c;
            }
        }
        fwrite(&pred, sizeof(int32_t), 1, f_out);
        lancius_arena_reset(scratch);
    }

    fclose(f_in); fclose(f_out);
    free(temp_in);

    for(uint32_t i=0; i<g->node_count; i++) {
        if (g->nodes[i]->op == LANCIUS_OP_INPUT) {
            if (g->nodes[i]->runtime_data) free(g->nodes[i]->runtime_data);
            if (g->nodes[i]->runtime_data_int8) free(g->nodes[i]->runtime_data_int8);
        }
    }

    lancius_schedule_destroy(sched);
    lancius_arena_destroy(scratch);
    lancius_graph_destroy(g);
    return 0;
}
