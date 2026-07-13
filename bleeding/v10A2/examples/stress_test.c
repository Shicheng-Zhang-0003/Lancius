
#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// =====================================================================
// STRESS TEST 1: ARENA HAMMERING (Fragmentation & Alignment)
// =====================================================================
int test_arena_hammering() {
    printf("[STRESS 1] Arena Hammering (100,000 reset cycles)...\n");
    lancius_arena* arena = lancius_arena_create(4096); // Small blocks to force chaining
    int pass = 1;

    for(int i=0; i<100000; i++) {
        void* p1 = lancius_arena_alloc(arena, 1, 8);
        void* p2 = lancius_arena_alloc(arena, 7, 32);
        void* p3 = lancius_arena_alloc(arena, 13, 64);
        void* p4 = lancius_arena_alloc(arena, SIZE_MAX, 8); // Should fail gracefully

        if (!p1 || !p2 || !p3) {
            printf("  ❌ FAIL: Allocation returned NULL at cycle %d (p1=%p, p2=%p, p3=%p)\n", i, p1, p2, p3);
            pass = 0; break;
        }
        if (p4 != NULL) {
            printf("  ❌ FAIL: SIZE_MAX alloc succeeded at cycle %d (p4=%p)\n", i, p4);
            pass = 0; break;
        }

        if (((uintptr_t)p1 & 7) != 0) { printf("  ❌ FAIL: p1 alignment failed at cycle %d (addr=%p, mod=%zu)\n", i, p1, (uintptr_t)p1 & 7); pass = 0; break; }
        if (((uintptr_t)p2 & 31) != 0) { printf("  ❌ FAIL: p2 alignment failed at cycle %d (addr=%p, mod=%zu)\n", i, p2, (uintptr_t)p2 & 31); pass = 0; break; }
        if (((uintptr_t)p3 & 63) != 0) { printf("  ❌ FAIL: p3 alignment failed at cycle %d (addr=%p, mod=%zu)\n", i, p3, (uintptr_t)p3 & 63); pass = 0; break; }

        if (i == 0) {
            printf("  [DEBUG] Cycle 0 Pointers: p1=%p (mod8=%zu), p2=%p (mod32=%zu), p3=%p (mod64=%zu)\n",
                   p1, (uintptr_t)p1 & 7, p2, (uintptr_t)p2 & 31, p3, (uintptr_t)p3 & 63);
        }

        lancius_arena_reset(arena);
    }
    lancius_arena_destroy(arena);

    if (pass) printf("  ✅ PASS: Arena survived 100k fragmentation cycles. (Alignments verified)\n");
    return pass;
}

// =====================================================================
// STRESS TEST 2: AUTODIFF BROADCAST NIGHTMARE
// =====================================================================
int test_autodiff_broadcast() {
    printf("[STRESS 2] Autodiff Broadcast & Reduction Nightmare...\n");
    lancius_graph* g = lancius_graph_create();

    // [1, 10] Bias + [32, 10] Matrix -> [32, 10] -> SUM -> [1, 1] Loss
    lancius_node* bias = lancius_input(g, 1, 10);
    lancius_node* matrix = lancius_input(g, 32, 10);
    lancius_node* bcast_bias = lancius_broadcast(g, bias, 32, 10);
    lancius_node* add = lancius_add(g, matrix, bcast_bias);
    lancius_node* loss = lancius_sum(g, add);

    bias->runtime_data = (double*)calloc(10, sizeof(double));
    matrix->runtime_data = (double*)calloc(320, sizeof(double));
    for(int i=0; i<10; i++) bias->runtime_data[i] = 0.5;
    for(int i=0; i<320; i++) matrix->runtime_data[i] = 0.1;

    lancius_training_graph* tg = lancius_ir_autodiff(g, loss);
    if (!tg) {
        printf("  ❌ FAIL: Autodiff returned NULL.\n");
        return 0;
    }

    lancius_schedule* fwd = lancius_ir_schedule(g);
    lancius_schedule* bwd = lancius_ir_schedule(tg->graph);
    lancius_arena* scratch = lancius_arena_create(1024 * 1024);

    lancius_schedule_execute(fwd, scratch);
    lancius_schedule_execute(bwd, scratch);

    // The gradient of the bias should be exactly 32.0 (summed across the batch of 32)
    lancius_node* bias_grad = tg->grad_nodes[bias->id];
    int pass = 1;
    if (bias_grad && bias_grad->runtime_data) {
        printf("  [DEBUG] Bias Gradients (Expected 32.0): [");
        for(int i=0; i<10; i++) {
            printf("%.4f ", bias_grad->runtime_data[i]);
            if (bias_grad->runtime_data[i] < 31.99 || bias_grad->runtime_data[i] > 32.01) {
                pass = 0;
            }
        }
        printf("]\n");
    } else {
        printf("  ❌ FAIL: Bias gradient node or data is NULL.\n");
        pass = 0;
    }

    if (pass) printf("  ✅ PASS: Autodiff correctly reduced 32x10 gradient to 1x10 bias.\n");
    else printf("  ❌ FAIL: Autodiff broadcast reduction failed.\n");

    free(bias->runtime_data); free(matrix->runtime_data);
    lancius_schedule_destroy(fwd); lancius_schedule_destroy(bwd);
    lancius_training_graph_destroy(tg); lancius_graph_destroy(g);
    lancius_arena_destroy(scratch);
    return pass;
}

// =====================================================================
// STRESS TEST 3: DEEP GRAPH SCHEDULING (Stack/Queue limits)
// =====================================================================
int test_deep_scheduler() {
    printf("[STRESS 3] Deep Graph Scheduling (10,000 node chain)...\n");
    lancius_graph* g = lancius_graph_create();
    lancius_node* prev = lancius_input(g, 10, 10);
    prev->runtime_data = (double*)calloc(100, sizeof(double));

    int actual_depth = 0;
    for(int i=0; i<10000; i++) {
        prev = lancius_relu(g, prev);
        if (!prev) {
            printf("  ❌ FAIL: IR rejected deep chain at node %d.\n", i);
            return 0;
        }
        actual_depth++;
    }
    printf("  [DEBUG] Successfully built chain of %d ReLU nodes.\n", actual_depth);

    lancius_schedule* sched = lancius_ir_schedule(g);
    int pass = (sched != NULL && sched->wave_count > 0);

    if (pass) {
        printf("  [DEBUG] Scheduler produced %u waves for %u total nodes.\n", sched->wave_count, g->node_count);
        printf("  ✅ PASS: Scheduler handled 10k node depth without deadlock.\n");
    } else {
        printf("  ❌ FAIL: Scheduler deadlocked or failed (sched=%p, waves=%u).\n", (void*)sched, sched ? sched->wave_count : 0);
    }

    free(g->nodes[0]->runtime_data);
    if(sched) lancius_schedule_destroy(sched);
    lancius_graph_destroy(g);
    return pass;
}

// =====================================================================
// STRESS TEST 4: MALICIOUS DESERIALIZATION
// =====================================================================
int test_malicious_load() {
    printf("[STRESS 4] Malicious Deserialization (Garbage & Truncation)...\n");
    int pass = 1;

    // Test 1: Pure garbage
    FILE* f = fopen("garbage.lancius", "wb");
    fwrite("THIS IS NOT A MODEL", 1, 19, f);
    fclose(f);
    lancius_graph* g1 = lancius_graph_load("garbage.lancius");
    printf("  [DEBUG] Garbage file load returned: %p\n", (void*)g1);
    if (g1 != NULL) { pass = 0; lancius_graph_destroy(g1); }

    // Test 2: Valid magic, truncated header
    f = fopen("trunc.lancius", "wb");
    uint32_t magic = 0x21434E41;
    fwrite(&magic, 4, 1, f);
    fclose(f);
    lancius_graph* g2 = lancius_graph_load("trunc.lancius");
    printf("  [DEBUG] Truncated file load returned: %p\n", (void*)g2);
    if (g2 != NULL) { pass = 0; lancius_graph_destroy(g2); }

    remove("garbage.lancius"); remove("trunc.lancius");

    if (pass) printf("  ✅ PASS: Serializer safely rejected malicious payloads.\n");
    else printf("  ❌ FAIL: Serializer crashed or loaded garbage.\n");
    return pass;
}

int main() {
    printf("================================================================\n");
    printf("  LANCIUS ULTIMATE STRESS TEST (DEEP INSPECTION MODE)         \n");
    printf("================================================================\n\n");

    int total = 0, passed = 0;

    total++; passed += test_arena_hammering();
    total++; passed += test_autodiff_broadcast();
    total++; passed += test_deep_scheduler();
    total++; passed += test_malicious_load();

    printf("\n================================================================\n");
    printf("  STRESS TEST COMPLETE: %d / %d PASSED\n", passed, total);
    if (passed == total) printf("  🏆 LANCIUS IS BULLETPROOF. NO HIDDEN BUGS FOUND.\n");
    printf("================================================================\n");

    return (passed == total) ? 0 : 1;
}
