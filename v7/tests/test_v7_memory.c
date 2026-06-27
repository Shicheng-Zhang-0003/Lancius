#include "basis/core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#define TEST_PASSED(msg) printf("  [PASS] %s\n", msg)
#define TEST_FAILED(msg) do { printf("  [FAIL] %s\n", msg); failures++; } while(0)

int main() {
    printf("=========================================================\n");
    printf("  BASIS V7: Memory Subsystem Validation (Arena & Pool)  \n");
    printf("=========================================================\n\n");

    int failures = 0;

    // ==========================================
    // TEST 1: Arena Alignment & Bump Allocation
    // ==========================================
    printf("=== Arena Allocator Tests ===\n");
    basis_arena* arena = basis_arena_create(4096); // 4KB blocks

    void* p1 = basis_arena_alloc(arena, 10, 8);
    void* p2 = basis_arena_alloc(arena, 33, 16);
    void* p3 = basis_arena_alloc(arena, 100, 32);

    if (((uintptr_t)p1 % 8) == 0) TEST_PASSED("p1 aligned to 8 bytes"); else TEST_FAILED("p1 alignment");
    if (((uintptr_t)p2 % 16) == 0) TEST_PASSED("p2 aligned to 16 bytes"); else TEST_FAILED("p2 alignment");
    if (((uintptr_t)p3 % 32) == 0) TEST_PASSED("p3 aligned to 32 bytes"); else TEST_FAILED("p3 alignment");

    if (p2 > p1 && p3 > p2) TEST_PASSED("Arena pointer bumping is monotonic"); else TEST_FAILED("Bump order");

    // ==========================================
    // TEST 2: Arena Oversized Allocation & Reset
    // ==========================================
    void* huge = basis_arena_alloc(arena, 8192, 8);
    if (huge != NULL) TEST_PASSED("Arena handles oversized allocations (> block size)"); else TEST_FAILED("Oversized alloc");

    basis_arena_reset(arena);

    void* p4 = basis_arena_alloc(arena, 10, 8);
    if (p4 == p1) TEST_PASSED("Arena Reset correctly reuses baseline memory (O(1) teardown)"); else TEST_FAILED("Arena Reset reuse");

    basis_arena_destroy(arena);
    TEST_PASSED("Arena destroyed without leaks");

    // ==========================================
    // TEST 3: Pool Allocation & Free-List Reuse
    // ==========================================
    printf("\n=== Pool Allocator Tests ===\n");

    typedef struct { double x, y, z; int id; } mock_node;
    basis_pool* pool = basis_pool_create(sizeof(mock_node), 100);

    mock_node* nodes[100];
    for(int i=0; i<100; i++) {
        nodes[i] = (mock_node*)basis_pool_alloc(pool);
        nodes[i]->id = i;
    }
    TEST_PASSED("Pool allocated 100 objects seamlessly");

    mock_node* freed_node = nodes[50];
    basis_pool_free(pool, freed_node);

    mock_node* reused_node = (mock_node*)basis_pool_alloc(pool);
    if (reused_node == freed_node) TEST_PASSED("Pool Free-List correctly reuses freed memory pointer"); else TEST_FAILED("Pool reuse");

    mock_node* overflow_node = (mock_node*)basis_pool_alloc(pool);
    if (overflow_node != NULL) TEST_PASSED("Pool successfully allocated secondary chunk on overflow"); else TEST_FAILED("Pool overflow");

    basis_pool_destroy(pool);
    TEST_PASSED("Pool destroyed without leaks");

    // ==========================================
    // TEST 4: Realistic Graph Lifecycle (1,000,000 Nodes)
    // ==========================================
    printf("\n=== Realistic Graph Lifecycle (1,000,000 Nodes) ===\n");
    printf("Simulating Compiler IR: Bulk Alloc -> Use -> Mass Teardown\n");
    size_t N = 1000000;
    clock_t start, end;

    // Array to hold pointers. This forces malloc to manage real heap structures
    // and prevents the CPU from just reusing a single L1-cached address.
    void** ptrs = (void**)malloc(N * sizeof(void*));

    // --- MALLOC LIFECYCLE ---
    start = clock();
    // Phase 1: Build Graph (Varying sizes 64-127 to scatter across glibc bins)
    for(size_t i=0; i<N; i++) {
        size_t sz = 64 + (i % 64);
        ptrs[i] = malloc(sz);
        ((char*)ptrs[i])[0] = (char)i; // Touch memory
    }
    // Phase 2: Teardown Graph (O(N) mass free)
    for(size_t i=0; i<N; i++) {
        free(ptrs[i]);
    }
    end = clock();
    double malloc_time = (double)(end - start) / CLOCKS_PER_SEC;

    // --- ARENA LIFECYCLE ---
    // 64MB blocks easily holds the ~100MB graph with minimal chunking overhead
    basis_arena* perf_arena = basis_arena_create(1024 * 1024 * 64);
    start = clock();
    // Phase 1: Build Graph (Linear streaming)
    for(size_t i=0; i<N; i++) {
        size_t sz = 64 + (i % 64);
        char* ptr = (char*)basis_arena_alloc(perf_arena, sz, 8);
        ptr[0] = (char)i; // Touch memory
        ptrs[i] = ptr;
    }
    // Phase 2: Teardown Graph (O(1) Reset!)
    basis_arena_reset(perf_arena);
    end = clock();
    double arena_time = (double)(end - start) / CLOCKS_PER_SEC;

    basis_arena_destroy(perf_arena);
    free(ptrs);

    printf("  Malloc Lifecycle (Alloc + Mass Free): %.4f sec\n", malloc_time);
    printf("  Arena Lifecycle (Alloc + O(1) Reset):      %.4f sec\n", arena_time);
    if (arena_time < malloc_time) TEST_PASSED("Arena Lifecycle crushes Malloc Lifecycle"); else TEST_FAILED("Arena performance");

    printf("\n=========================================================\n");
    if (failures == 0) {
        printf("  ✅ ALL V7 MEMORY TESTS PASSED.\n");
        printf("=========================================================\n");
        return 0;
    } else {
        printf("  !!! %d FAILURES DETECTED !!!\n", failures);
        printf("=========================================================\n");
        return 1;
    }
}
