#include <basis.h>
#include <stdio.h>

int main() {
    printf("=== BASIS v6: Static Graph Tracing (No Execution) ===\n\n");

    // 1. Initialize the Graph IR
    basis_graph* g = basis_graph_new();

    // 2. Define Inputs (Metadata only, no memory allocated for data)
    basis_node* X = basis_graph_input(g, "X", 4, 2);      // Batch of 4, 2 features
    basis_node* W1 = basis_graph_input(g, "W1", 2, 8);    // Weights
    basis_node* b1 = basis_graph_input(g, "b1", 1, 8);    // Bias
    basis_node* W2 = basis_graph_input(g, "W2", 8, 1);
    basis_node* b2 = basis_graph_input(g, "b2", 1, 1);

    // 3. Trace the Forward Pass (Building the IR DAG)
    basis_node* Z1 = basis_graph_matmul(g, X, W1);
    basis_node* A1 = basis_graph_broadcast_add(g, Z1, b1);
    basis_node* H1 = basis_graph_relu(g, A1);

    basis_node* Z2 = basis_graph_matmul(g, H1, W2);
    basis_node* Y = basis_graph_broadcast_add(g, Z2, b2);

    // 4. Print the IR
    basis_graph_print(g);

    printf("\nNotice: We just defined a complete Neural Network architecture.\n");
    printf("However, ZERO bytes of weight memory were allocated.\n");
    printf("ZERO autograd scalar nodes were created.\n");
    printf("This IR is now ready to be fed to the v6 Lowering Compiler.\n");

    basis_graph_free(g);
    return 0;
}
