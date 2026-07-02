#include "basis/stage6_ir/lowering.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void get_val_buf(basis_node* n, char* buf, size_t buf_size) {
    if (n->op == BASIS_OP_INPUT) snprintf(buf, buf_size, "%s", n->name);
    else snprintf(buf, buf_size, "ctx->t%d", n->id);
}

static void get_grad_buf(basis_node* n, char* buf, size_t buf_size) {
    if (n->op == BASIS_OP_INPUT) snprintf(buf, buf_size, "grad_%s", n->name);
    else snprintf(buf, buf_size, "grad_t%d", n->id);
}

void basis_graph_lower_to_c(basis_graph* g, const char* output_filename) {
    FILE* f = fopen(output_filename, "w");
    if (!f) return;

    fprintf(f, "#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n\n");

    // 1. Emit Context Struct
    fprintf(f, "typedef struct {\n");
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op != BASIS_OP_INPUT) {
            fprintf(f, "    double* t%zu;\n", i);
        }
    }
    fprintf(f, "} GhostContext;\n\n");

    // 2. Emit Forward Pass
    fprintf(f, "void ghost_weight_forward(");
    int first = 1;
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op == BASIS_OP_INPUT) {
            if (!first) fprintf(f, ", ");
            fprintf(f, "double* %s", g->nodes[i].name);
            first = 0;
        }
    }
    fprintf(f, ", GhostContext* ctx, double* out) {\n");

    for (size_t i = 0; i < g->node_count; i++) {
        basis_node* n = &g->nodes[i];
        if (n->op == BASIS_OP_INPUT) continue;

        fprintf(f, "    ctx->t%zu = (double*)malloc(%zu * %zu * sizeof(double));\n", i, n->rows, n->cols);

        if (n->op == BASIS_OP_MATMUL) {
            basis_node* a = &g->nodes[n->input_ids[0]];
            basis_node* b = &g->nodes[n->input_ids[1]];
            char a_buf[64], b_buf[64];
            get_val_buf(a, a_buf, sizeof(a_buf)); get_val_buf(b, b_buf, sizeof(b_buf));

            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t j=0; j<%zu; j++) {\n", n->rows, n->cols);
            fprintf(f, "        double sum = 0.0;\n");
            fprintf(f, "        for(size_t k=0; k<%zu; k++) sum += %s[i*%zu + k] * %s[k*%zu + j];\n", a->cols, a_buf, a->cols, b_buf, b->cols);
            fprintf(f, "        ctx->t%zu[i*%zu + j] = sum;\n    }\n", i, n->cols);
        } else if (n->op == BASIS_OP_BROADCAST_ADD) {
            basis_node* a = &g->nodes[n->input_ids[0]];
            basis_node* b = &g->nodes[n->input_ids[1]];
            char a_buf[64], b_buf[64];
            get_val_buf(a, a_buf, sizeof(a_buf)); get_val_buf(b, b_buf, sizeof(b_buf));
            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t j=0; j<%zu; j++) ctx->t%zu[i*%zu + j] = %s[i*%zu + j] + %s[j];\n", n->rows, n->cols, i, n->cols, a_buf, a->cols, b_buf);
        } else if (n->op == BASIS_OP_RELU) {
            basis_node* a = &g->nodes[n->input_ids[0]];
            char a_buf[64]; get_val_buf(a, a_buf, sizeof(a_buf));
            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t j=0; j<%zu; j++) { double v = %s[i*%zu + j]; ctx->t%zu[i*%zu + j] = v > 0.0 ? v : 0.0; }\n", n->rows, n->cols, a_buf, a->cols, i, n->cols);
        }
    }

    basis_node* last = &g->nodes[g->node_count - 1];
    char last_buf[64]; get_val_buf(last, last_buf, sizeof(last_buf));
    fprintf(f, "    memcpy(out, %s, %zu * %zu * sizeof(double));\n}\n\n", last_buf, last->rows, last->cols);

    // 3. Emit Backward Pass
    fprintf(f, "void ghost_weight_backward(");
    first = 1;
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op == BASIS_OP_INPUT) {
            if (!first) fprintf(f, ", ");
            fprintf(f, "double* %s", g->nodes[i].name);
            first = 0;
        }
    }
    fprintf(f, ", GhostContext* ctx, double* grad_out, ");
    first = 1;
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op == BASIS_OP_INPUT) {
            if (!first) fprintf(f, ", ");
            fprintf(f, "double* grad_%s", g->nodes[i].name);
            first = 0;
        }
    }
    fprintf(f, ") {\n");

    // Allocate intermediate gradients (EXCEPT the last node, which maps to grad_out)
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op != BASIS_OP_INPUT && i != g->node_count - 1) {
            fprintf(f, "    double* grad_t%zu = (double*)calloc(%zu * %zu, sizeof(double));\n", i, g->nodes[i].rows, g->nodes[i].cols);
        }
    }

    // Map last node's gradient to grad_out
    fprintf(f, "    double* grad_t%zu = grad_out;\n", g->node_count - 1);

    // Reverse topological traversal
    for (int idx = (int)g->node_count - 1; idx >= 0; idx--) {
        basis_node* n = &g->nodes[idx];
        if (n->op == BASIS_OP_INPUT) continue;

        char grad_n[64]; get_grad_buf(n, grad_n, sizeof(grad_n));

        if (n->op == BASIS_OP_MATMUL) {
            basis_node* a = &g->nodes[n->input_ids[0]];
            basis_node* b = &g->nodes[n->input_ids[1]];
            char a_buf[64], b_buf[64], grad_a[64], grad_b[64];
            get_val_buf(a, a_buf, sizeof(a_buf)); get_val_buf(b, b_buf, sizeof(b_buf));
            get_grad_buf(a, grad_a, sizeof(grad_a)); get_grad_buf(b, grad_b, sizeof(grad_b));

            // dA = dC * B^T
            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t k=0; k<%zu; k++) { double s=0; for(size_t j=0; j<%zu; j++) s += %s[i*%zu+j] * %s[k*%zu+j]; %s[i*%zu+k] += s; }\n",
                    a->rows, a->cols, b->cols, grad_n, n->cols, b_buf, b->cols, grad_a, a->cols);
            // dB = A^T * dC
            fprintf(f, "    for(size_t k=0; k<%zu; k++) for(size_t j=0; j<%zu; j++) { double s=0; for(size_t i=0; i<%zu; i++) s += %s[i*%zu+k] * %s[i*%zu+j]; %s[k*%zu+j] += s; }\n",
                    a->cols, b->cols, a->rows, a_buf, a->cols, grad_n, n->cols, grad_b, b->cols);
        } else if (n->op == BASIS_OP_BROADCAST_ADD) {
            basis_node* a = &g->nodes[n->input_ids[0]];
            basis_node* b = &g->nodes[n->input_ids[1]];
            char grad_a[64], grad_b[64];
            get_grad_buf(a, grad_a, sizeof(grad_a)); get_grad_buf(b, grad_b, sizeof(grad_b));

            // dA = dC
            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t j=0; j<%zu; j++) %s[i*%zu+j] += %s[i*%zu+j];\n", n->rows, n->cols, grad_a, n->cols, grad_n, n->cols);
            // dB = sum_rows(dC)
            fprintf(f, "    for(size_t j=0; j<%zu; j++) { double s=0; for(size_t i=0; i<%zu; i++) s += %s[i*%zu+j]; %s[j] += s; }\n", n->cols, n->rows, grad_n, n->cols, grad_b);
        } else if (n->op == BASIS_OP_RELU) {
            basis_node* a = &g->nodes[n->input_ids[0]];
            char grad_a[64];
            get_grad_buf(a, grad_a, sizeof(grad_a));

            // dA = dC * (out > 0). We use ctx->t{idx} (the output of RELU) as the mask!
            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t j=0; j<%zu; j++) %s[i*%zu+j] += %s[i*%zu+j] * (ctx->t%zu[i*%zu+j] > 0.0 ? 1.0 : 0.0);\n",
                    n->rows, n->cols, grad_a, n->cols, grad_n, n->cols, idx, n->cols);
        }
    }

    // Free intermediate gradients
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op != BASIS_OP_INPUT && i != g->node_count - 1) {
            fprintf(f, "    free(grad_t%zu);\n", i);
        }
    }
    fprintf(f, "}\n\n");

    // 4. Emit Free Context
    fprintf(f, "void ghost_free_context(GhostContext* ctx) {\n");
    for (size_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].op != BASIS_OP_INPUT) fprintf(f, "    free(ctx->t%zu);\n", i);
    }
    fprintf(f, "}\n");

    fclose(f);
}
