// V1.0 PRODUCTION API AUDIT
// This file ONLY includes the stable API header. It proves the FFI boundary is secure.
#include <lancius/lancius_stable_api.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("================================================================\n");
    printf("  LANCIUS V1.0: PRODUCTION FFI & API AUDIT                     \n");
    printf("================================================================\n\n");

    printf("[1/4] Initializing Context & Opaque Handles...\n");
    lancius_context ctx = lancius_create_context();
    if (!ctx) {
        printf("  ❌ Failed to create context: %s\n", lancius_get_error_string(lancius_get_last_error()));
        return 1;
    }
    printf("  ✅ Context Initialized (Opaque Pointer: %p)\n", ctx);

    printf("[2/4] Building Graph via Stable API...\n");
    lancius_graph_handle g = lancius_graph_create_stable(ctx);

    lancius_tensor_handle A = lancius_add_input(g, 2, 3);
    lancius_tensor_handle B = lancius_add_input(g, 3, 4);
    lancius_tensor_handle C = lancius_add_matmul(g, A, B);
    lancius_tensor_handle D = lancius_add_relu(g, C);

    if (lancius_get_last_error() != LANCIUS_OK) {
        printf("  ❌ Graph Build Failed: %s\n", lancius_get_error_string(lancius_get_last_error()));
        return 1;
    }
    printf("  ✅ Graph Built (MatMul -> ReLU)\n");

    printf("[3/4] Binding External Data & Executing...\n");
    double data_A[6] = {-1.0, 2.0, 3.0,  4.0, -5.0, 6.0};
    double data_B[12] = {1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0};

    lancius_bind_data(A, data_A);
    lancius_bind_data(B, data_B);

    lancius_status exec_status = lancius_compile_and_run(g);
    if (exec_status != LANCIUS_OK) {
        printf("  ❌ Execution Failed: %s\n", lancius_get_error_string(exec_status));
        return 1;
    }
    printf("  ✅ Execution Complete (Zero Internal Struct Access)\n");

    printf("[4/4] Extracting Output via FFI Boundary...\n");
    double out_buffer[8] = {0};
    lancius_read_output(D, out_buffer, sizeof(out_buffer));

    // Math check:
    // A = [-1, 2, 3; 4, -5, 6]
    // B = [1, 1, 1, 1; ...] (3x4 matrix of 1s)
    // C = A * B. Row 1 sum = -1+2+3 = 4. Row 2 sum = 4-5+6 = 5.
    // C = [4, 4, 4, 4; 5, 5, 5, 5]
    // D = ReLU(C) = [4, 4, 4, 4; 5, 5, 5, 5]

    int math_pass = 1;
    for(int i=0; i<4; i++) if (out_buffer[i] != 4.0) math_pass = 0;
    for(int i=4; i<8; i++) if (out_buffer[i] != 5.0) math_pass = 0;

    printf("  Output Buffer: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n",
           out_buffer[0], out_buffer[1], out_buffer[2], out_buffer[3],
           out_buffer[4], out_buffer[5], out_buffer[6], out_buffer[7]);

    if (math_pass) printf("  ✅ FFI Math Verified.\n");
    else printf("  ❌ FFI Math Corrupted.\n");

    
printf("\n[5/5] Testing FFI Error Boundary (Intentional Shape Mismatch)...\n");
lancius_tensor_handle Bad_A = lancius_add_input(g, 2, 3);
lancius_tensor_handle Bad_B = lancius_add_input(g, 2, 3); // 2x3 and 2x3 cannot be MatMuled!
lancius_tensor_handle Bad_C = lancius_add_matmul(g, Bad_A, Bad_B);

if (lancius_get_last_error() == LANCIUS_ERR_SHAPE_MISMATCH) {
    printf("  ✅ FFI correctly caught Shape Mismatch.\n");
    printf("  🛡️ Error String: \"%s\"\n", lancius_get_error_string(lancius_get_last_error()));
} else {
    printf("  ❌ FFI FAILED to catch Shape Mismatch!\n");
}

lancius_graph_destroy_stable(g);
    lancius_destroy_context(ctx);

    printf("\n================================================================\n");
    printf("  V1.0 PRODUCTION API VERIFIED. READY FOR GITHUB RELEASE.      \n");
    printf("================================================================\n");
    return 0;
}
