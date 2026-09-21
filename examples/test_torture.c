
#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int test_malicious_serialization() {
    printf("[TORTURE] Testing Malicious Serialization (Huge Node Count)...\n");
    FILE* f = fopen("malicious.lancius", "wb");
    uint32_t magic = 0x21434E41; // "LANC!"
    uint32_t huge_nodes = 0xFFFFFFFF; // Attempt to trigger calloc overflow
    fwrite(&magic, 4, 1, f);
    fwrite(&huge_nodes, 4, 1, f);
    fclose(f);

    lancius_graph* g = lancius_graph_load("malicious.lancius");
    if (g == NULL) {
        printf("  ✅ PASS: Malicious model rejected gracefully.\n");
        remove("malicious.lancius");
        return 1;
    } else {
        printf("  ❌ FAIL: Malicious model loaded! Memory exhausted.\n");
        lancius_graph_destroy(g);
        remove("malicious.lancius");
        return 0;
    }
}

int test_arena_invariants() {
    printf("[TORTURE] Testing Arena Invariants (0-byte and SIZE_MAX)...\n");
    lancius_arena* arena = lancius_arena_create(1024);
    int pass = 1;

    // Test 0-byte alloc
    void* p1 = lancius_arena_alloc(arena, 0, 8);
    void* p2 = lancius_arena_alloc(arena, 0, 8);
    if (p1 != p2) {
        printf("  ✅ PASS: 0-byte allocs do not overlap.\n");
    } else {
        printf("  ❌ FAIL: 0-byte allocs overlap!\n");
        pass = 0;
    }

    // Test SIZE_MAX
    void* p_huge = lancius_arena_alloc(arena, SIZE_MAX, 8);
    if (p_huge == NULL) {
        printf("  ✅ PASS: SIZE_MAX alloc rejected.\n");
    } else {
        printf("  ❌ FAIL: SIZE_MAX alloc succeeded!\n");
        pass = 0;
    }

    lancius_arena_destroy(arena);
    return pass;
}

int test_cyclic_graph() {
    printf("[TORTURE] Testing Cyclic Graph Scheduling...\n");
    /* Despot honesty: previously a printf-stub. Now prove scheduler rejects
       a genuinely disconnected/cyclic structure instead of claiming SSA prevents it. */
    lancius_graph* g = lancius_graph_create();
    if (!g) { printf("  ❌ FAIL: graph create OOM.\n"); return 0; }
    lancius_node* a = lancius_input(g, 2, 2);
    lancius_node* b = lancius_relu(g, a);
    if (!a || !b) { printf("  ❌ FAIL: builder rejected valid chain.\n"); lancius_graph_destroy(g); return 0; }
    /* Corrupt connectivity: point b back at itself to simulate a cycle. */
    const lancius_node* saved = b->inputs[0];
    b->inputs[0] = b;
    lancius_schedule* s = lancius_ir_schedule(g);
    b->inputs[0] = saved;
    if (s == NULL) {
        printf("  ✅ PASS: scheduler refused cyclic/disconnected graph.\n");
        lancius_graph_destroy(g);
        return 1;
    }
    /* If scheduler accepted, it must still execute without crashing; but that is a FAIL. */
    printf("  ❌ FAIL: scheduler accepted a cyclic graph.\n");
    lancius_schedule_destroy(s);
    lancius_graph_destroy(g);
    return 0;
}

int main() {
    printf("================================================================\n");
    printf("  LANCIUS TORTURE SUITE (INVARIANT VALIDATION)                 \n");
    printf("================================================================\n\n");

    int pass = 0;

    pass += test_malicious_serialization();
    pass += test_arena_invariants();
    pass += test_cyclic_graph();

    printf("\n================================================================\n");
    printf("  TORTURE SUITE COMPLETE: %d PASSED | %d FAILED\n", pass, 3 - pass);
    printf("================================================================\n");

    return (pass == 3) ? 0 : 1;
}
