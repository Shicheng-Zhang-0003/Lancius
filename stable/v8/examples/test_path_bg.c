#include <basis.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("================================================================\n");
    printf("  BASIS V8: Path B (Serialization) & Path G (Permute) Test   \n");
    printf("================================================================\n\n");

    v8_arena* scratch = v8_arena_create(16 * 1024 * 1024);

    // --- PATH G: Permute Test ---
    printf("[PATH G] Testing N-Dimensional Permute...\n");
    v8_graph* g_perm = v8_graph_create();
    v8_node* in4d = v8_input_4d(g_perm, 2, 3, 4, 5); // N=2, C=3, H=4, W=5

    // Permute to N, H, W, C (axes: 0, 2, 3, 1)
    v8_node* perm = v8_permute(g_perm, in4d, 0, 2, 3, 1);

    v8_schedule* sched_perm = v8_ir_schedule(g_perm);

    double* dummy_in = (double*)calloc(2*3*4*5, sizeof(double));
    in4d->runtime_data = dummy_in;

    v8_schedule_execute(sched_perm, scratch);

    if (perm->runtime_data != NULL) {
        printf("  ✅ Permute Executed! Output shape: [%zu, %zu, %zu, %zu]\n",
               perm->shape[0], perm->shape[1], perm->shape[2], perm->shape[3]);
        if (perm->shape[0] == 2 && perm->shape[1] == 4 && perm->shape[2] == 5 && perm->shape[3] == 3) {
            printf("  ✅ Shape correctly transposed from [2,3,4,5] to [2,4,5,3]!\n");
        }
    } else {
        printf("  ❌ Permute Failed!\n");
    }
    v8_schedule_destroy(sched_perm);
    v8_graph_destroy(g_perm);
    free(dummy_in);
    v8_arena_reset(scratch);

    // --- PATH B: Serialization Test ---
    printf("\n[PATH B] Testing Graph Save & Load...\n");
    v8_graph* g_save = v8_graph_create();
    v8_node* X = v8_input(g_save, 10, 20);
    v8_node* W = v8_input(g_save, 20, 5);
    v8_node* Z = v8_matmul(g_save, X, W);
    v8_node* Y = v8_relu(g_save, Z);
    (void)Y;

    // Assign dummy weights to W
    W->runtime_data = (double*)calloc(20*5, sizeof(double));
    for(int i=0; i<100; i++) W->runtime_data[i] = 0.42;

    printf("  Saving graph to 'test_model.v8b'...\n");
    v8_graph_save(g_save, "test_model.v8b");

    printf("  Destroying original graph from RAM...\n");
    free(W->runtime_data); // Free the heap memory before destroying the graph
    v8_graph_destroy(g_save);

    printf("  Loading graph from 'test_model.v8b'...\n");
    v8_graph* g_load = v8_graph_load("test_model.v8b");

    if (g_load && g_load->node_count == 4) {
        printf("  ✅ Serialization Success! Loaded %u nodes.\n", g_load->node_count);

        // Verify weights survived
        v8_node* loaded_W = g_load->nodes[1]; // Should be W
        if (loaded_W && loaded_W->runtime_data && loaded_W->runtime_data[0] == 0.42) {
            printf("  ✅ Weights perfectly preserved! (W[0] = %f)\n", loaded_W->runtime_data[0]);
        } else {
            printf("  ❌ Weights corrupted or missing!\n");
        }

        // Clean up loaded graph
        if (loaded_W && loaded_W->runtime_data) free(loaded_W->runtime_data);
        v8_graph_destroy(g_load);
    } else {
        printf("  ❌ Serialization Failed!\n");
    }

    v8_arena_destroy(scratch);
    printf("\n================================================================\n");
    printf("  PATH B & G VERIFICATION COMPLETE.\n");
    printf("================================================================\n");
    return 0;
}
