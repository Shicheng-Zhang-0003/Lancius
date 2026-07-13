#ifndef BASIS_V7_MEMORY_H
#define BASIS_V7_MEMORY_H

#include <stddef.h>
#include <stdint.h>

/* ==========================================================================
 * Arena Allocator (L0 Platform)
 * Used for: Graphs, ASTs, Scratch buffers.
 * Properties: O(1) allocation (pointer bump), O(1) teardown (reset).
 * ========================================================================== */
typedef struct basis_arena basis_arena;

basis_arena* basis_arena_create(size_t default_block_size);
void* basis_arena_alloc(basis_arena* arena, size_t size, size_t alignment);
void basis_arena_reset(basis_arena* arena); // O(1) "Free"
void basis_arena_destroy(basis_arena* arena);

/* ==========================================================================
 * Pool Allocator (L0 Platform)
 * Used for: Fixed-size Tensors, Graph Nodes.
 * Properties: O(1) alloc/free via free-list, zero fragmentation.
 * ========================================================================== */
typedef struct basis_pool basis_pool;

basis_pool* basis_pool_create(size_t object_size, size_t objects_per_chunk);
void* basis_pool_alloc(basis_pool* pool);
void basis_pool_free(basis_pool* pool, void* ptr);
void basis_pool_destroy(basis_pool* pool);

#endif // BASIS_V7_MEMORY_H
