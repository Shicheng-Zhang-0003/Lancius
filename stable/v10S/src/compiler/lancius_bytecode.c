#include "lancius/lancius_bytecode.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

lancius_program* lancius_compile_graph(lancius_graph* g) {
    if (!g || g->node_count == 0) return NULL;
    lancius_program* prog = (lancius_program*)calloc(1, sizeof(lancius_program));
    if (!prog) return NULL;

    uint32_t* reg_map = (uint32_t*)calloc(g->next_id, sizeof(uint32_t));
    if (!reg_map) { free(prog); return NULL; }

    prog->num_regs = g->node_count;
    prog->code = (uint32_t*)malloc(g->node_count * 5 * sizeof(uint32_t));
    prog->rows = (size_t*)calloc(prog->num_regs, sizeof(size_t));
    prog->cols = (size_t*)calloc(prog->num_regs, sizeof(size_t));
    prog->input_regs = (uint32_t*)malloc(g->node_count * sizeof(uint32_t));
    prog->input_count = 0;

    // v10S GUARD: Bytecode VM only supports 2D tensors
    for (uint32_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i]->ndim > 2) {
            fprintf(stderr, "[BYTECODE FATAL] ndim > 2 not supported in v10S VM.\n");
            free(reg_map); free(prog->code); free(prog->rows); free(prog->cols); free(prog->input_regs); free(prog);
            return NULL;
        }
    }

    size_t pc = 0;
    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        reg_map[n->id] = i;
        prog->rows[i] = n->shape[0];
        prog->cols[i] = n->shape[1];
        if (n->op == LANCIUS_OP_INPUT) prog->input_regs[prog->input_count++] = i;
    }

    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST) continue;

        uint32_t out_r = reg_map[n->id];
        if (n->op == LANCIUS_OP_MATMUL) {
            prog->code[pc++] = LANCIUS_BC_MATMUL; prog->code[pc++] = out_r;
            prog->code[pc++] = reg_map[n->inputs[0]->id]; prog->code[pc++] = reg_map[n->inputs[1]->id];
        } else if (n->op == LANCIUS_OP_ADD) {
            prog->code[pc++] = LANCIUS_BC_ADD; prog->code[pc++] = out_r;
            prog->code[pc++] = reg_map[n->inputs[0]->id]; prog->code[pc++] = reg_map[n->inputs[1]->id];
        } else if (n->op == LANCIUS_OP_SUB) {
            prog->code[pc++] = LANCIUS_BC_SUB; prog->code[pc++] = out_r;
            prog->code[pc++] = reg_map[n->inputs[0]->id]; prog->code[pc++] = reg_map[n->inputs[1]->id];
        } else if (n->op == LANCIUS_OP_MUL) {
            prog->code[pc++] = LANCIUS_BC_MUL; prog->code[pc++] = out_r;
            prog->code[pc++] = reg_map[n->inputs[0]->id]; prog->code[pc++] = reg_map[n->inputs[1]->id];
        } else if (n->op == LANCIUS_OP_RELU) {
            prog->code[pc++] = LANCIUS_BC_RELU; prog->code[pc++] = out_r;
            prog->code[pc++] = reg_map[n->inputs[0]->id];
        } else if (n->op == LANCIUS_OP_BROADCAST) {
            prog->code[pc++] = LANCIUS_BC_BROADCAST; prog->code[pc++] = out_r;
            prog->code[pc++] = reg_map[n->inputs[0]->id];
        } else if (n->op == LANCIUS_OP_SOFTMAX) {
            prog->code[pc++] = LANCIUS_BC_SOFTMAX; prog->code[pc++] = out_r;
            prog->code[pc++] = reg_map[n->inputs[0]->id];
        } else if (n->op == LANCIUS_OP_SUM) {
            prog->code[pc++] = LANCIUS_BC_SUM; prog->code[pc++] = out_r;
            prog->code[pc++] = reg_map[n->inputs[0]->id];
        }
    }
    prog->code[pc++] = LANCIUS_BC_HALT;
    prog->code_len = pc;
    prog->out_reg = reg_map[g->nodes[g->node_count - 1]->id];

    free(reg_map);
    return prog;
}

void lancius_vm_execute(lancius_program* prog, double** inputs, double* out, lancius_arena* scratch) {
    if (!prog || !scratch || !out) return;

    double** regs = (double**)lancius_arena_alloc(scratch, prog->num_regs * sizeof(double*), 8);
    if (!regs) return;

    for (uint32_t i = 0; i < prog->input_count; i++) regs[prog->input_regs[i]] = inputs[i];

    size_t pc = 0;
    while (pc < prog->code_len) {
        uint32_t op = prog->code[pc++];
        if (op == LANCIUS_BC_HALT) break;

        uint32_t r_out = prog->code[pc++];
        uint32_t r_a = prog->code[pc++];
        uint32_t r_b = 0;

        bool is_unary = (op == LANCIUS_BC_RELU || op == LANCIUS_BC_BROADCAST || op == LANCIUS_BC_SOFTMAX || op == LANCIUS_BC_SUM);
        if (!is_unary) r_b = prog->code[pc++];

        size_t elements = prog->rows[r_out] * prog->cols[r_out];
        regs[r_out] = (double*)lancius_arena_alloc(scratch, elements * sizeof(double), 32);
        if (!regs[r_out]) continue;

        double* a = regs[r_a];
        double* b = is_unary ? NULL : regs[r_b];
        double* o = regs[r_out];

        if (!a) continue;

        if (op == LANCIUS_BC_MATMUL) {
            size_t M = prog->rows[r_a]; size_t K = prog->cols[r_a]; size_t N = prog->cols[r_b];
            for(size_t r=0; r<M; r++) for(size_t c=0; c<N; c++) {
                double sum = 0.0; for(size_t k=0; k<K; k++) sum += a[r*K + k] * b[k*N + c];
                o[r*N + c] = sum;
            }
        } else if (op == LANCIUS_BC_ADD) {
            for(size_t k=0; k<elements; k++) o[k] = a[k] + b[k];
        } else if (op == LANCIUS_BC_SUB) {
            for(size_t k=0; k<elements; k++) o[k] = a[k] - b[k];
        } else if (op == LANCIUS_BC_MUL) {
            for(size_t k=0; k<elements; k++) o[k] = a[k] * b[k];
        } else if (op == LANCIUS_BC_RELU) {
            for(size_t k=0; k<elements; k++) o[k] = a[k] > 0.0 ? a[k] : 0.0;
        } else if (op == LANCIUS_BC_BROADCAST) {
            size_t cols = prog->cols[r_out]; size_t rows = prog->rows[r_out];
            size_t in_elems = prog->rows[r_a] * prog->cols[r_a];
            if (in_elems == 1) {
                double val = a[0];
                for(size_t k=0; k<rows*cols; k++) o[k] = val;
            } else {
                for(size_t r=0; r<rows; r++) for(size_t c=0; c<cols; c++) o[r*cols + c] = a[c];
            }
        } else if (op == LANCIUS_BC_SOFTMAX) {
            size_t R = prog->rows[r_out]; size_t C = prog->cols[r_out];
            for(size_t r=0; r<R; r++) {
                double max_val = a[r*C];
                for(size_t c=1; c<C; c++) if(a[r*C+c] > max_val) max_val = a[r*C+c];
                double sum = 0.0;
                for(size_t c=0; c<C; c++) { o[r*C+c] = exp(a[r*C+c] - max_val); sum += o[r*C+c]; }
                for(size_t c=0; c<C; c++) o[r*C+c] /= sum;
            }
        } else if (op == LANCIUS_BC_SUM) {
            size_t elems = prog->rows[r_a] * prog->cols[r_a];
            double sum = 0.0; for(size_t k=0; k<elems; k++) sum += a[k];
            o[0] = sum;
        }
    }

    size_t out_elements = prog->rows[prog->out_reg] * prog->cols[prog->out_reg];
    if (regs[prog->out_reg]) memcpy(out, regs[prog->out_reg], out_elements * sizeof(double));
}

void lancius_program_destroy(lancius_program* prog) {
    if (!prog) return;
    if (prog->code) free(prog->code);
    if (prog->input_regs) free(prog->input_regs);
    if (prog->rows) free(prog->rows);
    if (prog->cols) free(prog->cols);
    free(prog);
}
