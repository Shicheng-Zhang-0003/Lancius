#include <lancius/lancius_stable_api.h>
#include <stdio.h>

int main() {
    lancius_context ctx = lancius_create_context();
    lancius_graph_handle g = lancius_graph_create_stable(ctx);

    lancius_tensor_handle A = lancius_add_input(g, 2, 3);
    lancius_tensor_handle B = lancius_add_input(g, 2, 3);

    // THIS SHOULD FAIL - shapes don't match for MatMul
    lancius_tensor_handle C = lancius_add_matmul(g, A, B);

    if (lancius_get_last_error() != LANCIUS_OK) {
        const char* err = lancius_get_error_string(lancius_get_last_error());
        printf("Error caught: %s\n", err ? err : "NULL (SEGFAULT RISK)");
    } else {
        printf("ERROR: Should have failed but didn't!\n");
    }

    lancius_graph_destroy_stable(g);
    lancius_destroy_context(ctx);
    return 0;
}
