#include "basis/l4_compiler/lowering.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void basis_ir_lower_to_c(basis_ir_graph* g, const char* output_filename) {
    FILE* f = fopen(output_filename, "w");
    if (!f) return;
    fprintf(f, "#include <stdlib.h>\n#include <string.h>\n\n");
    int input_count = 0;
    char** c_vars = (char**)calloc(g->next_id, sizeof(char*));
    for (uint32_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i]->op == BASIS_IR_OP_INPUT) {
            c_vars[g->nodes[i]->id] = (char*)malloc(16);
            sprintf(c_vars[g->nodes[i]->id], "in%d", input_count++);
        }
    }
    fprintf(f, "void v7_compiled_kernel(double** inputs, double* out) {\n");
    for (uint32_t i = 0; i < g->node_count; i++) {
        basis_ir_node* n = g->nodes[i];
        if (n->op != BASIS_IR_OP_INPUT) {
            c_vars[n->id] = (char*)malloc(16);
            sprintf(c_vars[n->id], "buf%u", n->id);
            fprintf(f, "    double* buf%u = (double*)malloc(%zu * sizeof(double));\n", n->id, n->rows * n->cols);
        }
    }
    fprintf(f, "\n");
    for (uint32_t i = 0; i < g->node_count; i++) {
        basis_ir_node* n = g->nodes[i];
        size_t elements = n->rows * n->cols;
        if (n->op == BASIS_IR_OP_INPUT) continue;
        if (n->op == BASIS_IR_OP_CONST) {
            fprintf(f, "    for(size_t i=0; i<%zu; i++) buf%u[i] = %f;\n", elements, n->id, n->attr_val);
        } else if (n->op == BASIS_IR_OP_ADD) {
            fprintf(f, "    for(size_t i=0; i<%zu; i++) buf%u[i] = %s[i] + %s[i];\n", elements, n->id, c_vars[n->inputs[0]->id], c_vars[n->inputs[1]->id]);
        } else if (n->op == BASIS_IR_OP_MATMUL) {
            size_t M = n->inputs[0]->rows, K = n->inputs[0]->cols, N = n->inputs[1]->cols;
            fprintf(f, "    for(size_t i=0; i<%zu; i++) for(size_t j=0; j<%zu; j++) { double sum=0; for(size_t k=0; k<%zu; k++) sum+=%s[i*%zu+k]*%s[k*%zu+j]; buf%u[i*%zu+j]=sum; }\n", M, N, K, c_vars[n->inputs[0]->id], K, c_vars[n->inputs[1]->id], N, n->id, N);
        } else if (n->op == BASIS_IR_OP_RELU) {
            fprintf(f, "    for(size_t i=0; i<%zu; i++) { double v=%s[i]; buf%u[i]=v>0.0?v:0.0; }\n", elements, c_vars[n->inputs[0]->id], n->id);
        }
    }
    basis_ir_node* final_node = g->nodes[g->node_count - 1];
    fprintf(f, "\n    memcpy(out, buf%u, %zu * sizeof(double));\n", final_node->id, final_node->rows * final_node->cols);
    fprintf(f, "\n");
    for (uint32_t i = 0; i < g->node_count; i++) if (g->nodes[i]->op != BASIS_IR_OP_INPUT) fprintf(f, "    free(buf%u);\n", g->nodes[i]->id);
    fprintf(f, "}\n");
    fclose(f);
    for (uint32_t i = 0; i < g->next_id; i++) if (c_vars[i]) free(c_vars[i]);
    free(c_vars);
}
