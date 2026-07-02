#ifndef BASIS_STAGE6_IR_GRAPH_H
#define BASIS_STAGE6_IR_GRAPH_H

#include <stdlib.h>
#include <stdio.h>

typedef enum {
    BASIS_OP_INPUT,
    BASIS_OP_MATMUL,
    BASIS_OP_ADD,
    BASIS_OP_RELU,
    BASIS_OP_TANH,
    BASIS_OP_BROADCAST_ADD
} basis_op_type;

typedef struct basis_node {
    int id;
    basis_op_type op;
    int input_ids[2];
    size_t rows;
    size_t cols;
    char* name;
} basis_node;

typedef struct basis_graph {
    basis_node* nodes;
    size_t node_count;
    size_t node_cap;
} basis_graph;

basis_graph* basis_graph_new(void);
basis_node* basis_graph_input(basis_graph* g, const char* name, size_t r, size_t c);
basis_node* basis_graph_matmul(basis_graph* g, basis_node* a, basis_node* b);
basis_node* basis_graph_add(basis_graph* g, basis_node* a, basis_node* b);
basis_node* basis_graph_broadcast_add(basis_graph* g, basis_node* a, basis_node* bias);
basis_node* basis_graph_relu(basis_graph* g, basis_node* a);
basis_node* basis_graph_tanh(basis_graph* g, basis_node* a);
void basis_graph_print(basis_graph* g);
void basis_graph_free(basis_graph* g);

#endif
