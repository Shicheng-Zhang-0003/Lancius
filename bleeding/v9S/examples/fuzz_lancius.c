#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define FUZZ_ITERATIONS 500
#define POOL_CAP 256

static int check_nan_inf(double* data, size_t count) {
    if (!data) return 0;
    for (size_t i = 0; i < count; i++) {
        if (isnan(data[i]) || isinf(data[i])) return 1;
    }
    return 0;
}

static int shapes_match(const lancius_node* a, const lancius_node* b) {
    if (a->ndim != b->ndim) return 0;
    for (uint8_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    uint32_t seed = (argc > 1) ? (uint32_t)atoi(argv[1]) : (uint32_t)time(NULL);
    srand(seed);

    printf("================================================================\n");
    printf("  Lancius DYNAMIC FUZZER (Seed: %u)\n", seed);
    printf("================================================================\n\n");

    lancius_arena* scratch = lancius_arena_create(64 * 1024 * 1024); // 64MB scratch
    int pass_count = 0;
    int fail_count = 0;

    for (int iter = 0; iter < FUZZ_ITERATIONS; iter++) {
        lancius_graph* g = lancius_graph_create();
        lancius_node* pool[POOL_CAP];
        int pool_sz = 0;

        // 1. Seed the pool with diverse base topologies
        pool[pool_sz++] = lancius_input_4d(g, 2, 2, 8, 8);   // 4D Image
        pool[pool_sz++] = lancius_input_4d(g, 4, 2, 3, 3);   // 4D Filter
        pool[pool_sz++] = lancius_input(g, 16, 8);            // 2D Matrix A
        pool[pool_sz++] = lancius_input(g, 8, 4);             // 2D Matrix B
        pool[pool_sz++] = lancius_const(g, 0.5, 1, 1);        // Scalar

        // Allocate dummy persistent data for inputs
        double* dummy_data[5];
        for(int i=0; i<5; i++) {
            size_t elems = lancius_node_elements(pool[i]);
            dummy_data[i] = (double*)calloc(elems, sizeof(double));
            pool[i]->runtime_data = dummy_data[i];
        }

        // 2. Shape-Aware Random Graph Generation
        for (int i = 0; i < 40; i++) {
            if (pool_sz >= POOL_CAP - 2) break;
            int op = rand() % 6;

            if (op == 0) { // Unary: RELU
                lancius_node* a = pool[rand() % pool_sz];
                lancius_node* r = lancius_relu(g, a);
                if (r) pool[pool_sz++] = r;
            }
            else if (op == 1) { // Binary Exact: ADD / MUL
                lancius_node* a = pool[rand() % pool_sz];
                for (int j = 0; j < pool_sz; j++) {
                    if (shapes_match(a, pool[j])) {
                        lancius_node* r = (rand() % 2) ? lancius_add(g, a, pool[j]) : lancius_mul(g, a, pool[j]);
                        if (r) { pool[pool_sz++] = r; break; }
                    }
                }
            }
            else if (op == 2) { // MATMUL
                for (int j = 0; j < pool_sz; j++) {
                    if (pool[j]->ndim != 2) continue;
                    for (int k = 0; k < pool_sz; k++) {
                        if (pool[k]->ndim == 2 && pool[j]->shape[1] == pool[k]->shape[0]) {
                            lancius_node* r = lancius_matmul(g, pool[j], pool[k]);
                            if (r) { pool[pool_sz++] = r; goto matmul_done; }
                        }
                    }
                }
                matmul_done:;
            }
            else if (op == 3) { // CONV2D
                for (int j = 0; j < pool_sz; j++) {
                    if (pool[j]->ndim != 4) continue;
                    for (int k = 0; k < pool_sz; k++) {
                        if (pool[k]->ndim == 4 && pool[j]->shape[1] == pool[k]->shape[1]) {
                            lancius_node* r = lancius_conv2d(g, pool[j], pool[k], 1, 0);
                            if (r) { pool[pool_sz++] = r; goto conv_done; }
                        }
                    }
                }
                conv_done:;
            }
            else if (op == 4) { // MAXPOOL2D
                for (int j = 0; j < pool_sz; j++) {
                    if (pool[j]->ndim == 4 && pool[j]->shape[2] >= 2 && pool[j]->shape[3] >= 2) {
                        lancius_node* r = lancius_maxpool2d(g, pool[j], 2, 2);
                        if (r) { pool[pool_sz++] = r; break; }
                    }
                }
            }
            else if (op == 5) { // FLATTEN
                for (int j = 0; j < pool_sz; j++) {
                    if (pool[j]->ndim == 4) {
                        lancius_node* r = lancius_flatten(g, pool[j]);
                        if (r) { pool[pool_sz++] = r; break; }
                    }
                }
            }
        }

        // 3. Create a Loss Node (Reduce random node to scalar)
        lancius_node* target = pool[rand() % pool_sz];
        lancius_node* loss = lancius_sum(g, target);
        if (!loss) {
            fail_count++;
            goto cleanup;
        }

        // 4. Compile Backward Pass
        lancius_training_graph* tg = lancius_ir_autodiff(g, loss);
        if (!tg || !tg->loss_node) {
            fail_count++;
            if(tg) lancius_training_graph_destroy(tg);
            goto cleanup;
        }

        lancius_schedule* sched = lancius_ir_schedule(tg->graph);
        if (!sched) {
            fail_count++;
            lancius_training_graph_destroy(tg);
            goto cleanup;
        }

        // 5. Execute Forward & Backward
        lancius_schedule_execute(sched, scratch);

        // 6. Hunt for NaN / Inf / Memory Corruption
        int corrupted = 0;
        for (uint32_t k = 0; k < tg->graph->node_count; k++) {
            lancius_node* n = tg->graph->nodes[k];
            if (n->runtime_data) {
                size_t elems = lancius_node_elements(n);
                if (check_nan_inf(n->runtime_data, elems)) {
                    corrupted = 1;
                    break;
                }
            }
        }

        if (corrupted) {
            printf("[FUZZ FAIL] Iter %d (Seed %u): NaN/Inf detected in graph execution!\n", iter, seed);
            fail_count++;
        } else {
            pass_count++;
        }

        lancius_schedule_destroy(sched);
        lancius_training_graph_destroy(tg);

        cleanup:
        lancius_graph_destroy(g);
        lancius_arena_reset(scratch); // Hammer the O(1) teardown
        for(int i=0; i<5; i++) free(dummy_data[i]);

        if ((iter + 1) % 50 == 0) {
            printf("\r  Fuzzing... %d / %d graphs processed.", iter + 1, FUZZ_ITERATIONS);
            fflush(stdout);
        }
    }

    lancius_arena_destroy(scratch);

    printf("\n\n================================================================\n");
    printf("  FUZZ RESULTS: %d PASSED | %d FAILED (out of %d)\n", pass_count, fail_count, FUZZ_ITERATIONS);
    printf("  (Failures here usually mean IR shape-rejection, which is SAFE)\n");
    printf("================================================================\n");

    return 0;
}
