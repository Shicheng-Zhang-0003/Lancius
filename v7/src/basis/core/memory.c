#include "basis/core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align) - 1))

/* ==========================================================================
 * Arena Implementation
 * ========================================================================== */
typedef struct basis_arena_block {
    struct basis_arena_block* next;
    size_t size;
    size_t used;
    uint8_t* memory;
} basis_arena_block;

struct basis_arena {
    basis_arena_block* current;
    basis_arena_block* first;
    size_t default_block_size;
};

static basis_arena_block* arena_block_create(size_t size) {
    basis_arena_block* block = (basis_arena_block*)malloc(sizeof(basis_arena_block));
    if (!block) return NULL;
    block->memory = (uint8_t*)malloc(size);
    if (!block->memory) { free(block); return NULL; }
    block->size = size;
    block->used = 0;
    block->next = NULL;
    return block;
}

basis_arena* basis_arena_create(size_t default_block_size) {
    basis_arena* arena = (basis_arena*)calloc(1, sizeof(basis_arena));
    if (!arena) return NULL;
    arena->default_block_size = default_block_size > 0 ? default_block_size : 65536; // 64KB default
    arena->first = arena_block_create(arena->default_block_size);
    arena->current = arena->first;
    return arena;
}

void* basis_arena_alloc(basis_arena* arena, size_t size, size_t alignment) {
    if (!arena || size == 0) return NULL;
    if (alignment == 0) alignment = sizeof(void*);

    // CRITICAL FIX: Enforce power-of-2 alignment to prevent SIMD SIGBUS crashes
    if (alignment & (alignment - 1)) {
        fprintf(stderr, "[BASIS ARENA FATAL] Alignment %zu is not a power of 2!\n", alignment);
        return NULL;
    }

    basis_arena_block* block = arena->current;
    uintptr_t current_ptr = (uintptr_t)(block->memory + block->used);
    uintptr_t aligned_ptr = ALIGN_UP(current_ptr, alignment);
    size_t padding = aligned_ptr - current_ptr;

    if (block->used + padding + size > block->size) {
        // Need a new block
        size_t new_block_size = arena->default_block_size;
        if (size + alignment > new_block_size) {
            new_block_size = size + alignment; // Oversized allocation
        }
        basis_arena_block* new_block = arena_block_create(new_block_size);
        if (!new_block) return NULL;

        // Link and switch
        block->next = new_block;
        arena->current = new_block;
        block = new_block;

        current_ptr = (uintptr_t)(block->memory + block->used);
        aligned_ptr = ALIGN_UP(current_ptr, alignment);
        padding = aligned_ptr - current_ptr;
    }

    block->used += padding + size;
    return (void*)aligned_ptr;
}

void basis_arena_reset(basis_arena* arena) {
    if (!arena) return;
    // Free all blocks except the first one to maintain a baseline capacity
    basis_arena_block* block = arena->first->next;
    while (block) {
        basis_arena_block* next = block->next;
        free(block->memory);
        free(block);
        block = next;
    }
    arena->first->next = NULL;
    arena->first->used = 0;
    arena->current = arena->first;
}

void basis_arena_destroy(basis_arena* arena) {
    if (!arena) return;
    basis_arena_block* block = arena->first;
    while (block) {
        basis_arena_block* next = block->next;
        free(block->memory);
        free(block);
        block = next;
    }
    free(arena);
}

/* ==========================================================================
 * Pool Implementation
 * ========================================================================== */
typedef struct pool_slot {
    struct pool_slot* next;
} pool_slot;

struct basis_pool {
    size_t object_size;      // Actual usable size
    size_t slot_size;        // Size including free-list pointer overhead + alignment
    size_t objects_per_chunk;
    pool_slot* free_list;

    void** chunks;           // Array of raw malloc pointers for teardown
    size_t chunk_count;
    size_t chunk_cap;
};

basis_pool* basis_pool_create(size_t object_size, size_t objects_per_chunk) {
    basis_pool* pool = (basis_pool*)calloc(1, sizeof(basis_pool));
    if (!pool) return NULL;

    // Ensure slot is large enough to hold a pointer for the free list, and aligned
    size_t min_size = object_size < sizeof(pool_slot*) ? sizeof(pool_slot*) : object_size;
    pool->object_size = object_size;
    pool->slot_size = ALIGN_UP(min_size, sizeof(void*));
    pool->objects_per_chunk = objects_per_chunk > 0 ? objects_per_chunk : 256;
    pool->free_list = NULL;

    pool->chunk_cap = 16;
    pool->chunks = (void**)malloc(sizeof(void*) * pool->chunk_cap);
    pool->chunk_count = 0;

    return pool;
}

static void pool_allocate_chunk(basis_pool* pool) {
    size_t chunk_bytes = pool->slot_size * pool->objects_per_chunk;
    uint8_t* raw_chunk = (uint8_t*)malloc(chunk_bytes);
    if (!raw_chunk) return;

    // Track chunk for destruction
    if (pool->chunk_count >= pool->chunk_cap) {
        pool->chunk_cap *= 2;
        pool->chunks = (void**)realloc(pool->chunks, sizeof(void*) * pool->chunk_cap);
    }
    pool->chunks[pool->chunk_count++] = raw_chunk;

    // Carve chunk into slots and prepend to free list
    for (size_t i = 0; i < pool->objects_per_chunk; i++) {
        pool_slot* slot = (pool_slot*)(raw_chunk + (i * pool->slot_size));
        slot->next = pool->free_list;
        pool->free_list = slot;
    }
}

void* basis_pool_alloc(basis_pool* pool) {
    if (!pool) return NULL;
    if (!pool->free_list) {
        pool_allocate_chunk(pool);
        if (!pool->free_list) return NULL; // OOM
    }
    pool_slot* slot = pool->free_list;
    pool->free_list = slot->next;
    return (void*)slot;
}

void basis_pool_free(basis_pool* pool, void* ptr) {
    if (!pool || !ptr) return;
    pool_slot* slot = (pool_slot*)ptr;
    slot->next = pool->free_list;
    pool->free_list = slot;
}

void basis_pool_destroy(basis_pool* pool) {
    if (!pool) return;
    for (size_t i = 0; i < pool->chunk_count; i++) {
        free(pool->chunks[i]);
    }
    free(pool->chunks);
    free(pool);
}
