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
    prog->is_const = (uint8_t*)calloc(prog->num_regs, sizeof(uint8_t));
    prog->const_val = (double*)calloc(prog->num_regs, sizeof(double));
    if (!prog->code || !prog->rows || !prog->cols || !prog->input_regs || !prog->is_const || !prog->const_val) {
        free(reg_map); free(prog->code); free(prog->rows); free(prog->cols); free(prog->input_regs); free(prog->is_const); free(prog->const_val); free(prog);
        return NULL;
    }
    prog->input_count = 0;

    // v10S GUARD: Bytecode VM only supports 2D tensors
    // Despot V6 truth: ndim!=2 rejected (was >2, so 0D/1D compiled to Rx0).
    for (uint32_t i = 0; i < g->node_count; i++) {
        if (!g->nodes[i] || g->nodes[i]->id >= g->next_id) {
            lancius_set_error(LANCIUS_ERROR_INTERNAL);
            free(reg_map); free(prog->code); free(prog->rows); free(prog->cols); free(prog->input_regs); free(prog->is_const); free(prog->const_val); free(prog);
            return NULL;
        }
        if (g->nodes[i]->ndim != 2) {
            lancius_set_error(LANCIUS_ERROR_INVALID_RANK);
            free(reg_map); free(prog->code); free(prog->rows); free(prog->cols); free(prog->input_regs); free(prog->is_const); free(prog->const_val); free(prog);
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
        else if (n->op == LANCIUS_OP_CONST) { prog->is_const[i] = 1; prog->const_val[i] = n->attr_val; }
    }

    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        if (n->op == LANCIUS_OP_INPUT || n->op == LANCIUS_OP_CONST) continue;

        /* Despot V6 truth: validate inputs before reg_map deref (was NULL+OOB). */
        uint32_t need = 0;
        if (n->op == LANCIUS_OP_MATMUL || n->op == LANCIUS_OP_ADD || n->op == LANCIUS_OP_SUB || n->op == LANCIUS_OP_MUL) need = 2;
        else if (n->op == LANCIUS_OP_RELU || n->op == LANCIUS_OP_BROADCAST || n->op == LANCIUS_OP_SOFTMAX || n->op == LANCIUS_OP_SUM) need = 1;
        else need = 0;
        if (need > 0) {
            if (!n->inputs || n->input_count < need) {
                lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID);
                free(reg_map); free(prog->code); free(prog->rows); free(prog->cols); free(prog->input_regs); free(prog->is_const); free(prog->const_val); free(prog);
                return NULL;
            }
            for (uint32_t k = 0; k < need; k++) {
                if (!n->inputs[k] || n->inputs[k]->id >= g->next_id) {
                    lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID);
                    free(reg_map); free(prog->code); free(prog->rows); free(prog->cols); free(prog->input_regs); free(prog->is_const); free(prog->const_val); free(prog);
                    return NULL;
                }
            }
        }
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
        } else {
            lancius_set_error(LANCIUS_ERROR_UNSUPPORTED_OP);
            free(reg_map); free(prog->code); free(prog->rows); free(prog->cols); free(prog->input_regs); free(prog->is_const); free(prog->const_val); free(prog);
            return NULL;
        }
    }
    prog->code[pc++] = LANCIUS_BC_HALT;
    prog->code_len = pc;
    prog->out_reg = reg_map[g->nodes[g->node_count - 1]->id];

    free(reg_map);
    return prog;
}

int lancius_vm_execute(lancius_program* prog, double** inputs, double* out, lancius_arena* scratch) {
    if (!prog || !scratch || !out) return -1;
    /* Despot V6 truth: program invariants validated (was OOB out_reg). */
    if (!prog->code || !prog->rows || !prog->cols) return -1;
    if (prog->num_regs == 0 || prog->code_len == 0) return -1;
    if (prog->out_reg >= prog->num_regs) return -1;
    /* Despot truth: inputs deref was unchecked (NULL + OOB reg). */
    if (prog->input_count > 0 && !inputs) return -1;
    if (prog->input_count > prog->num_regs) return -1;

    double** regs = (double**)lancius_arena_alloc(scratch, prog->num_regs * sizeof(double*), 8);
    if (!regs) return -1;
    memset(regs, 0, prog->num_regs * sizeof(double*));

    for (uint32_t i = 0; i < prog->input_count; i++) {
        if (prog->input_regs[i] >= prog->num_regs) return -1;
        regs[prog->input_regs[i]] = inputs[i];
    }
    /* Despot truth: materialize CONST regs (were uninitialized garbage). */
    if (prog->is_const && prog->const_val) {
        for (uint32_t r = 0; r < prog->num_regs; r++) {
            if (!prog->is_const[r]) continue;
            size_t ce = 0;
            if (prog->rows[r] && prog->cols[r] > SIZE_MAX / prog->rows[r]) return -1;
            ce = prog->rows[r] * prog->cols[r];
            if (ce && ce > SIZE_MAX / sizeof(double)) return -1;
            regs[r] = (double*)lancius_arena_alloc(scratch, ce ? ce * sizeof(double) : 32, 32);
            if (!regs[r]) return -1;
            for (size_t k = 0; k < ce; k++) regs[r][k] = prog->const_val[r];
        }
    }

    size_t pc = 0;
    while (pc < prog->code_len) {
        /* Despot V6 truth: tape OOB read guarded (was 3-word overread). */
        if (pc + 3 > prog->code_len) return -1;
        uint32_t op = prog->code[pc++];
        if (op == LANCIUS_BC_HALT) break;

        bool is_unary_peek = (op == LANCIUS_BC_RELU || op == LANCIUS_BC_BROADCAST || op == LANCIUS_BC_SOFTMAX || op == LANCIUS_BC_SUM);
        size_t need = is_unary_peek ? 3u : 4u;
        /* op already consumed; need (need-1) more words */
        if (pc + (need - 1) > prog->code_len) return -1;
        uint32_t r_out = prog->code[pc++];
        uint32_t r_a = prog->code[pc++];
        uint32_t r_b = 0;

        bool is_unary = (op == LANCIUS_BC_RELU || op == LANCIUS_BC_BROADCAST || op == LANCIUS_BC_SOFTMAX || op == LANCIUS_BC_SUM);
        if (!is_unary) r_b = prog->code[pc++];

        if (r_out >= prog->num_regs || r_a >= prog->num_regs || (!is_unary && r_b >= prog->num_regs)) return -1;
        size_t elements = 0;
        if (prog->rows[r_out] && prog->cols[r_out] > SIZE_MAX / prog->rows[r_out]) return -1;
        elements = prog->rows[r_out] * prog->cols[r_out];
        if (elements && elements > SIZE_MAX / sizeof(double)) return -1;
        regs[r_out] = (double*)lancius_arena_alloc(scratch, elements ? elements * sizeof(double) : 32, 32);
        if (!regs[r_out]) return -1; /* v11S C1 fix: OOM is fatal, not silent */

        double* a = regs[r_a];
        double* b = is_unary ? NULL : regs[r_b];
        double* o = regs[r_out];

        if (!a || (!is_unary && !b)) return -1;

        if (op == LANCIUS_BC_MATMUL) {
            size_t M = prog->rows[r_a]; size_t K = prog->cols[r_a]; size_t N = prog->cols[r_b];
            if (prog->rows[r_out] != M || prog->cols[r_out] != N || prog->rows[r_b] != K) return -1;
            for(size_t r=0; r<M; r++) for(size_t c=0; c<N; c++) {
                double sum = 0.0; for(size_t k=0; k<K; k++) sum += a[r*K + k] * b[k*N + c];
                o[r*N + c] = sum;
            }
        } else if (op == LANCIUS_BC_ADD || op == LANCIUS_BC_SUB || op == LANCIUS_BC_MUL) {
            // Hostile fix: broadcast-aware (was flat a[k] OP b[k] => wrong + OOB on [2,2] vs [1,2])
            size_t R = prog->rows[r_out], Cc = prog->cols[r_out];
            size_t aR = prog->rows[r_a], aC = prog->cols[r_a];
            size_t bR = prog->rows[r_b], bC = prog->cols[r_b];
            size_t aE = (aR && aC && aC <= SIZE_MAX / (aR ? aR : 1)) ? aR*aC : 0;
            size_t bE = (bR && bC && bC <= SIZE_MAX / (bR ? bR : 1)) ? bR*bC : 0;
            if (aE != R*Cc && aE != 1 && !(aR == 1 && aC == Cc) && !(aC == 1 && aR == R)) return -1;
            if (bE != R*Cc && bE != 1 && !(bR == 1 && bC == Cc) && !(bC == 1 && bR == R)) return -1;
            for (size_t r = 0; r < R; r++) for (size_t c = 0; c < Cc; c++) {
                size_t oi = r*Cc + c;
                size_t ai = (aE == 1) ? 0 : ((aR == 1 && aC == Cc) ? c : ((aC == 1 && aR == R) ? r : oi));
                size_t bi = (bE == 1) ? 0 : ((bR == 1 && bC == Cc) ? c : ((bC == 1 && bR == R) ? r : oi));
                if (op == LANCIUS_BC_ADD) o[oi] = a[ai] + b[bi];
                else if (op == LANCIUS_BC_SUB) o[oi] = a[ai] - b[bi];
                else o[oi] = a[ai] * b[bi];
            }
        } else if (op == LANCIUS_BC_RELU) {
            /* Despot V6 truth: input/output shapes must match (was OOB). */
            if (prog->rows[r_a] != prog->rows[r_out] || prog->cols[r_a] != prog->cols[r_out]) return -1;
            for(size_t k=0; k<elements; k++) o[k] = a[k] > 0.0 ? a[k] : 0.0;
        } else if (op == LANCIUS_BC_BROADCAST) {
            size_t cols = prog->cols[r_out]; size_t rows = prog->rows[r_out];
            size_t in_rows = prog->rows[r_a]; size_t in_cols = prog->cols[r_a];
            if (in_rows && in_cols > SIZE_MAX / in_rows) return -1;
            size_t in_elems = in_rows * in_cols;
            if (in_elems == 1) {
                double val = a[0];
                for(size_t k=0; k<rows*cols; k++) o[k] = val;
            } else if (in_rows == 1 && in_cols == cols) {
                for(size_t r=0; r<rows; r++) for(size_t c=0; c<cols; c++) o[r*cols + c] = a[c];
            } else if (in_cols == 1 && in_rows == rows) {
                for(size_t r=0; r<rows; r++) for(size_t c=0; c<cols; c++) o[r*cols + c] = a[r];
            } else if (in_rows == rows && in_cols == cols) {
                for(size_t k=0; k<rows*cols; k++) o[k] = a[k];
            } else {
                return -1;
            }
        } else if (op == LANCIUS_BC_SOFTMAX) {
            /* Despot V6 truth: input dims must equal output dims (was OOB). */
            if (prog->rows[r_a] != prog->rows[r_out] || prog->cols[r_a] != prog->cols[r_out]) return -1;
            size_t R = prog->rows[r_out]; size_t C = prog->cols[r_out];
            for(size_t r=0; r<R; r++) {
                double max_val = a[r*C];
                for(size_t c=1; c<C; c++) if(a[r*C+c] > max_val) max_val = a[r*C+c];
                double sum = 0.0;
                for(size_t c=0; c<C; c++) { o[r*C+c] = exp(a[r*C+c] - max_val); sum += o[r*C+c]; }
                if (!(sum > 0.0) || sum != sum) return -1;
                for(size_t c=0; c<C; c++) o[r*C+c] /= sum;
            }
        } else if (op == LANCIUS_BC_SUM) {
            /* Despot V6 truth: SUM out must be 1x1 (was uninit leak). */
            if (prog->rows[r_out] != 1 || prog->cols[r_out] != 1) return -1;
            if (prog->rows[r_a] && prog->cols[r_a] > SIZE_MAX / prog->rows[r_a]) return -1;
            size_t elems = prog->rows[r_a] * prog->cols[r_a];
            double sum = 0.0; for(size_t k=0; k<elems; k++) sum += a[k];
            o[0] = sum;
        }
    }

    if (prog->rows[prog->out_reg] && prog->cols[prog->out_reg] > SIZE_MAX / prog->rows[prog->out_reg]) return -1;
    size_t out_elements = prog->rows[prog->out_reg] * prog->cols[prog->out_reg];
    if (!regs[prog->out_reg]) return -1;
    memcpy(out, regs[prog->out_reg], out_elements * sizeof(double));
    return 0;
}

void lancius_program_destroy(lancius_program* prog) {
    if (!prog) return;
    if (prog->code) free(prog->code);
    if (prog->input_regs) free(prog->input_regs);
    if (prog->rows) free(prog->rows);
    if (prog->cols) free(prog->cols);
    if (prog->is_const) free(prog->is_const);
    if (prog->const_val) free(prog->const_val);
    free(prog);
}
